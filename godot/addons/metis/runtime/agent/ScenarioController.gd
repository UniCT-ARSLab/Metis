extends Node
class_name ScenarioController
signal episode_reset_started(seed:int)
signal episode_reset_completed(seed:int)
signal episode_step_completed(step:int)
signal scenario_configured(config:Dictionary)

@export var controlled_agents: Array[Node] = []
@export var scenario_reward_system_path: NodePath = NodePath("ScenarioRewardSystem")
@export var progress_provider_path: NodePath = NodePath("ProgressProvider")
@export var event_system_path: NodePath = NodePath("ScenarioEventSystem")

@export_category("Reset Randomization")
@export var randomize_reset := false
@export var reset_use_progress_provider := false
@export_range(0.0, 1.0, 0.001) var reset_progress_min := 0.0
@export_range(0.0, 1.0, 0.001) var reset_progress_max := 0.03
@export var reset_lateral_jitter := 0.25
@export var reset_position_jitter := Vector3.ZERO
@export var reset_yaw_jitter_degrees := 8.0
@export var reset_align_to_progress := true
@export var use_agent_specific_reset_seed := true

@export_category("RL Info")
@export var max_steps:= 500 # Zero disables step-based truncation.
@export_range(1, 16, 1) var physics_frames_per_step := 1
@export var manage_agent_cameras := true
@export var continue_after_success := false
@export var training_mode := false

@export_category("Training Optimization")
@export var auto_optimize_in_headless := true
@export var disable_camera_manager_in_headless := true
@export var disable_debug_ui_in_headless := true
@export var deactivate_done_agents := true

@export_category("Agents Replication")
@export var agent_to_replicate : Node
@export var number_of_replications:int = 0
@export var agents_container:Node
@export var debug_total_agents_label_path: NodePath = NodePath("../DEBUG/totalAgentsLabel")

@export_category("Recording")
@export var recording_mode := false
@export var recording_agent_id := ""
@export var disable_replication_in_recording := true

var step_count := 0
## Last scenario reward computed for an agent this step (exposed for debug HUDs / tooling).
var last_reward := 0.0
var _agents: Array[Node] = []
var _replicated_agents: Array[Node] = []
var _original_agent_transforms := {}
var _last_reset_info := {}
var _done_agents := {}
var _cached_done_step_results := {}
var _is_headless_runtime := false
var _scenario_reward_system: Node
var _progress_provider: Node
var _event_system: Node
var _training_episode := 0

func _ready() -> void:
	_is_headless_runtime = _is_headless()
	_scenario_reward_system = get_node_or_null(scenario_reward_system_path)
	_progress_provider = get_node_or_null(progress_provider_path)
	_event_system = get_node_or_null(event_system_path)
	_apply_user_args()
	if auto_optimize_in_headless and _is_headless_runtime:
		if disable_camera_manager_in_headless:
			manage_agent_cameras = false
		if disable_debug_ui_in_headless:
			debug_total_agents_label_path = NodePath("")
	_spawn_replicated_agents()
	_refresh_agents()
	_apply_training_optimizations()
	_update_debug_agents_label()
	
	for agent in _agents:
		var agent_id := _agent_id(agent)
		if agent is Node3D or agent is Node2D:
			_original_agent_transforms[agent_id] = agent.transform


func _spawn_replicated_agents() -> void:
	_replicated_agents.clear()

	if recording_mode and disable_replication_in_recording:
		number_of_replications = 0
		return

	if agent_to_replicate == null or number_of_replications <= 0:
		return

	var parent := agents_container
	if parent == null:
		parent = agent_to_replicate.get_parent()
	if parent == null:
		push_warning("Cannot replicate agent without a parent/container")
		return

	var packed_scene := _agent_packed_scene(agent_to_replicate)
	for idx in range(number_of_replications):
		var new_agent := _create_agent_replica(agent_to_replicate, packed_scene)
		if new_agent == null:
			continue

		new_agent.name = _unique_child_name(parent, "%s%d" % [str(agent_to_replicate.name), idx + 2])
		_copy_stored_root_properties(agent_to_replicate, new_agent)
		parent.add_child(new_agent, true)
		if agent_to_replicate is Node3D and new_agent is Node3D:
			new_agent.global_transform = agent_to_replicate.global_transform
		_replicated_agents.append(new_agent)

	if _replicated_agents.size() != number_of_replications:
		push_warning(
			"Requested %d replicated agents, but only spawned %d"
			% [number_of_replications, _replicated_agents.size()]
		)


func _agent_packed_scene(agent:Node) -> PackedScene:
	var scene_path := agent.scene_file_path
	if scene_path.is_empty():
		return null

	var resource := load(scene_path)
	if resource is PackedScene:
		return resource
	return null


func _create_agent_replica(agent:Node, packed_scene:PackedScene) -> Node:
	if packed_scene != null:
		var instance := packed_scene.instantiate()
		if instance is Node:
			return instance

	var duplicate_flags := DUPLICATE_SIGNALS | DUPLICATE_GROUPS | DUPLICATE_SCRIPTS
	var duplicated := agent.duplicate(duplicate_flags)
	if duplicated is Node:
		return duplicated
	return null


func _copy_stored_root_properties(source:Node, target_node:Node) -> void:
	for property in source.get_property_list():
		var property_name := str(property.get("name", ""))
		if _should_skip_replica_property(property_name):
			continue

		var usage := int(property.get("usage", 0))
		if (usage & PROPERTY_USAGE_STORAGE) == 0:
			continue

		target_node.set(property_name, source.get(property_name))


func _should_skip_replica_property(property_name:String) -> bool:
	return property_name in [
		"name",
		"owner",
		"script",
		"unique_name_in_owner",
		"scene_file_path"
	]


func _apply_user_args() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg == "--recording-mode":
			recording_mode = true
		elif arg == "--disable-agent-replication":
			disable_replication_in_recording = true
		elif arg == "--allow-agent-replication":
			disable_replication_in_recording = false
		elif arg.begins_with("--recording-agent-id="):
			recording_agent_id = str(arg.split("=", true, 1)[1])


func _unique_child_name(parent:Node, desired_name:String) -> String:
	var candidate := desired_name
	var suffix := 2
	while parent.get_node_or_null(NodePath(candidate)) != null:
		candidate = "%s_%d" % [desired_name, suffix]
		suffix += 1
	return candidate


func compute_reward_scenario(agent:Node, context:Dictionary = {}) -> float:
	if _scenario_reward_system == null or not _scenario_reward_system.has_method("compute_reward_scenario"):
		return 0.0
	var reward_context := context if not context.is_empty() else _build_scenario_context(agent)
	return float(_scenario_reward_system.compute_reward_scenario(agent, reward_context))


func step(actions:Variant):
	_refresh_agents()
	step_count += 1

	var applied_actions := {}
	for agent in _agents:
		if deactivate_done_agents and _is_agent_done(agent):
			continue
		var agent_action: Variant = _get_action_for_agent(actions, agent)
		var applied_action: Variant = agent.apply_action(agent_action)
		applied_actions[_agent_id(agent)] = applied_action

	await _advance_physics_frames(physics_frames_per_step)

	var truncated := max_steps > 0 and step_count >= max_steps
	var channels := []
	for agent in _agents:
		if deactivate_done_agents and _is_agent_done(agent):
			channels.append(_cached_done_step_result(agent, truncated))
			continue

		var channel := _build_agent_step_result(agent, truncated, applied_actions.get(_agent_id(agent), 0))
		channels.append(channel)
		if bool(channel.get("done", false)):
			_mark_agent_done(agent, channel)
	_update_current_agent_camera()
	episode_step_completed.emit(step_count)

	if _should_return_single_agent_response(actions):
		if channels.is_empty():
			return {
				"obs": [],
				"reward": 0.0,
				"done": true,
				"terminated": true,
				"truncated": truncated,
				"agents": [],
				"info": {
					"step": step_count,
					"error": "No controlled agents configured"
				}
			}

		var single: Dictionary = channels[0].duplicate(true)
		single["agents"] = channels
		return single

	return {
		"agents": channels,
		"done": _all_agents_done(channels),
		"terminated": _all_agents_terminated(channels),
		"truncated": truncated,
		"info": {
			"step": step_count
		}
	}
	

func reset_episode_with_request(request:Dictionary):
	var _seed := int(request.get("seed", 0))
	var preserve_state := bool(request.get("preserve_state", false))
	return await reset_episode(_seed, preserve_state)


## Lets a scenario attach task-specific reset diagnostics without coupling the controller
## to that task. Values survive the controller's own transform-randomization bookkeeping
## and are returned in the reset info sent to Python.
func merge_agent_reset_info(agent:Node, values:Dictionary) -> void:
	if agent == null or values.is_empty():
		return
	var agent_id := _agent_id(agent)
	var reset_info := Dictionary(_last_reset_info.get(agent_id, {})).duplicate(true)
	reset_info.merge(values, true)
	_last_reset_info[agent_id] = reset_info


func configure(config:Dictionary) -> Dictionary:
	_scenario_reward_system = get_node_or_null(scenario_reward_system_path)
	_progress_provider = get_node_or_null(progress_provider_path)
	_event_system = get_node_or_null(event_system_path)
	if config.has("max_steps"):
		max_steps = maxi(0, int(config["max_steps"]))
	if config.has("physics_frames_per_step"):
		physics_frames_per_step = maxi(1, int(config["physics_frames_per_step"]))
	if config.has("training_episode"):
		_training_episode = max(0, int(config["training_episode"]))
	if config.has("reset_progress_min"):
		reset_progress_min = clampf(float(config["reset_progress_min"]), 0.0, 1.0)
	elif config.has("reset_track_progress_min"):
		reset_progress_min = clampf(float(config["reset_track_progress_min"]), 0.0, 1.0)
	if config.has("reset_progress_max"):
		reset_progress_max = clampf(float(config["reset_progress_max"]), 0.0, 1.0)
	elif config.has("reset_track_progress_max"):
		reset_progress_max = clampf(float(config["reset_track_progress_max"]), 0.0, 1.0)
	if config.has("reset_lateral_jitter"):
		reset_lateral_jitter = maxf(float(config["reset_lateral_jitter"]), 0.0)
	if config.has("reset_yaw_jitter_degrees"):
		reset_yaw_jitter_degrees = maxf(float(config["reset_yaw_jitter_degrees"]), 0.0)
	if config.has("recording_mode"):
		recording_mode = bool(config["recording_mode"])
	if config.has("recording_agent_id"):
		recording_agent_id = str(config["recording_agent_id"])
	if config.has("disable_replication_in_recording"):
		disable_replication_in_recording = bool(config["disable_replication_in_recording"])
	if config.has("manage_agent_cameras"):
		manage_agent_cameras = bool(config["manage_agent_cameras"])
	if config.has("continue_after_success"):
		continue_after_success = bool(config["continue_after_success"])
	if config.has("training_mode"):
		training_mode = bool(config["training_mode"])
	if _scenario_reward_system != null:
		_apply_config_to_node_tree(_scenario_reward_system, config)
	_update_current_agent_camera()
	scenario_configured.emit(config.duplicate(true))

	return {
		"ok": true,
		"training_episode": _training_episode,
		"max_steps": max_steps,
		"physics_frames_per_step": physics_frames_per_step,
		"reset_progress_min": reset_progress_min,
		"reset_progress_max": reset_progress_max,
		"reset_track_progress_min": reset_progress_min,
		"reset_track_progress_max": reset_progress_max,
		"reset_lateral_jitter": reset_lateral_jitter,
		"reset_yaw_jitter_degrees": reset_yaw_jitter_degrees,
		"recording_mode": recording_mode,
		"recording_agent_id": recording_agent_id,
		"disable_replication_in_recording": disable_replication_in_recording,
		"manage_agent_cameras": manage_agent_cameras,
		"continue_after_success": continue_after_success,
		"training_mode": training_mode
	}


func reset_episode(_seed := 0, preserve_state := false):
	_refresh_agents()
	_scenario_reward_system = get_node_or_null(scenario_reward_system_path)
	_progress_provider = get_node_or_null(progress_provider_path)
	_event_system = get_node_or_null(event_system_path)
	step_count = 0
	_last_reset_info.clear()
	_done_agents.clear()
	_cached_done_step_results.clear()
	if not preserve_state:
		episode_reset_started.emit(_seed)
	if _scenario_reward_system != null and _scenario_reward_system.has_method("reset_rewards"):
		_scenario_reward_system.reset_rewards()
	if _progress_provider != null and _progress_provider.has_method("reset_provider"):
		_progress_provider.reset_provider()
	if _event_system != null and _event_system.has_method("reset_events"):
		_event_system.reset_events()

	var shared_rng := RandomNumberGenerator.new()
	shared_rng.seed = int(_seed)

	var channels := []
	for agent in _agents:
		var agent_id := _agent_id(agent)
		var rng := _reset_rng_for_agent(int(_seed), agent_id) if use_agent_specific_reset_seed else shared_rng

		_done_agents[agent_id] = false
		_set_agent_training_active(agent, true)
		if preserve_state:
			_last_reset_info[agent_id] = {
				"mode": "current_state",
				"preserved": true
			}
			if agent.has_method("initialize_episode_from_current_state"):
				agent.initialize_episode_from_current_state()
		else:
			var original_transform: Variant = null
			if agent is Node3D or agent is Node2D:
				original_transform = agent.transform
			if _original_agent_transforms.has(agent_id):
				original_transform = _original_agent_transforms[agent_id]
			var reset_transform: Variant = _build_reset_transform(agent_id, original_transform, rng)
			agent.reset_all(reset_transform, false)
		var reset_context := {"agent_id": agent_id, "step": step_count}
		if _event_system != null and _event_system.has_method("reset_agent"):
			_event_system.reset_agent(agent, reset_context)
		reset_context = _build_scenario_context(agent)
		if _progress_provider != null and _progress_provider.has_method("reset_agent"):
			_progress_provider.reset_agent(agent, reset_context)
		if _scenario_reward_system != null and _scenario_reward_system.has_method("reset_agent"):
			_scenario_reward_system.reset_agent(agent_id, reset_context)

	await _advance_physics_frames(1)
	for agent in _agents:
		_refresh_agent_sensors(agent)

	await _advance_physics_frames(1)
	for agent in _agents:
		_refresh_agent_sensors(agent)
		_reset_agent_reward(agent)

	for agent in _agents:
		channels.append(_build_agent_reset_result(agent))
	_update_current_agent_camera()
	episode_reset_completed.emit(_seed)
	
	if channels.size() == 1:
		var single: Dictionary = channels[0].duplicate(true)
		single["agents"] = channels
		var single_info: Dictionary = single.get("info", {})
		single_info["preserve_state"] = preserve_state
		single["info"] = single_info
		return single

	return {
		"agents": channels,
		"info": {
			"step": step_count,
			"reset": _last_reset_info.duplicate(true),
			"preserve_state": preserve_state
		}
	}


func get_spec() -> Dictionary:
	_refresh_agents()
	if _agents.is_empty():
		var error_message := (
			"ScenarioController has no runtime agents. Assign controlled_agents or " +
			"agent_to_replicate in the scene before starting Metis."
		)
		push_error(error_message)
		return {
			"ok": false,
			"error": error_message,
			"agents": [],
			"multi_agent": false,
			"physics_frames_per_step": physics_frames_per_step
		}

	var agent_specs := []
	for agent in _agents:
		var action_space := _get_agent_action_space(agent)
		var action_type := _infer_action_type(action_space)
		var action_names := _flatten_action_names(action_space)
		var agent_spec := {
			"id": _agent_id(agent),
			"team_id": _agent_team_id(agent),
			"policy_id": _agent_policy_id(agent),
			"obs_dim": _agent_observation_size(agent),
			"observation_names": _agent_observation_names(agent),
			"action_names": action_names,
			"action_type": action_type,
			"action_space": action_space
		}
		if action_type == "continuous":
			agent_spec["action_size"] = _continuous_action_size(action_space)
			agent_spec["action_low"] = _continuous_action_bounds(action_space, "low", -1.0)
			agent_spec["action_high"] = _continuous_action_bounds(action_space, "high", 1.0)
			agent_spec["num_actions"] = int(agent_spec["action_size"])
		elif action_type == "discrete":
			agent_spec["num_actions"] = _single_discrete_action_size(action_space)
			agent_spec["action_size"] = int(agent_spec["num_actions"])
		else:
			agent_spec["action_size"] = _continuous_action_size(action_space)
			agent_spec["num_actions"] = _total_discrete_action_size(action_space)
		agent_specs.append(agent_spec)

	return {
		"ok": true,
		"agents": agent_specs,
		"multi_agent": agent_specs.size() > 1,
		"physics_frames_per_step": physics_frames_per_step
	}


func _advance_physics_frames(frame_count:int) -> void:
	# A lockstep tree must be running before physics-frame signals can advance.
	var was_paused := get_tree().paused
	if was_paused:
		get_tree().paused = false
		
	for _frame in range(maxi(frame_count, 1)):
		# Wait for both physics and process frames so Godot completes the full tick.
		await get_tree().physics_frame
		await get_tree().process_frame

	# Restore the pause state; BridgeServer will manage it after the request.
	if was_paused:
		get_tree().paused = true


func get_progress(agent:Node, context:Dictionary = {}) -> float:
	if _progress_provider != null and _progress_provider.has_method("get_progress"):
		return float(_progress_provider.get_progress(agent, context))
	if agent.has_method("get_progress"):
		return float(agent.get_progress())
	return 0.0


func rebase_agent_tracking(agent:Node) -> void:
	# An exogenous goal change is not backward movement by the agent. Rebase progress deltas,
	# stall tracking, and one-shot goal rewards while preserving the physical episode and the
	# agent-local action history.
	if not agent:
		return
	var context := _build_scenario_context(agent)
	if _progress_provider != null and _progress_provider.has_method("reset_agent"):
		_progress_provider.reset_agent(agent, context)
	if _scenario_reward_system != null and _scenario_reward_system.has_method("reset_agent"):
		_scenario_reward_system.reset_agent(_agent_id(agent), context)


func _get_agent_action_space(agent:Node) -> Dictionary:
	var interface := _agent_interface(agent)
	if interface == null:
		push_warning("Agent body %s has no Agent child or RL interface" % agent.name)
		return {}
	if interface != null and interface.has_method("get_action_space"):
		var action_space: Variant = interface.get_action_space()
		if typeof(action_space) == TYPE_DICTIONARY:
			return _normalize_action_space(action_space)

	var action_type := "discrete"
	if interface != null and interface.has_method("get_action_type"):
		action_type = str(interface.get_action_type())

	if action_type == "continuous":
		var action_names: Array = interface.get_action_names()
		var lows: Array = []
		var highs: Array = []
		if interface.has_method("get_action_low"):
			lows = interface.get_action_low()
		if interface.has_method("get_action_high"):
			highs = interface.get_action_high()
		var result := {}
		for idx in range(int(interface.get_action_size())):
			var action_name := str(idx)
			if idx < action_names.size():
				action_name = str(action_names[idx])
			result[action_name] = {
				"size": 1,
				"action_type": "continuous",
				"low": float(lows[idx]) if idx < lows.size() else -1.0,
				"high": float(highs[idx]) if idx < highs.size() else 1.0
			}
		return result

	return {
		"action": {
			"size": interface.get_action_count(),
			"action_type": "discrete",
			"names": interface.get_action_names()
		}
	}


func _normalize_action_space(action_space:Dictionary) -> Dictionary:
	var result := {}
	for action_name in action_space.keys():
		var raw_component: Variant = action_space[action_name]
		var component := {}
		if typeof(raw_component) == TYPE_DICTIONARY:
			component = raw_component.duplicate(true)

		var component_type := str(component.get("action_type", component.get("type", "discrete")))
		var size := int(component.get("size", 1))
		component["action_type"] = component_type
		component["size"] = max(size, 1)
		if component_type == "continuous":
			component["low"] = component.get("low", -1.0)
			component["high"] = component.get("high", 1.0)
		result[str(action_name)] = component
	return result


func _infer_action_type(action_space:Dictionary) -> String:
	var continuous_count := 0
	var discrete_count := 0
	for component in action_space.values():
		var component_type := str(component.get("action_type", "discrete"))
		if component_type == "continuous":
			continuous_count += 1
		elif component_type == "discrete":
			discrete_count += 1

	if continuous_count > 0 and discrete_count == 0:
		return "continuous"
	if discrete_count == 1 and continuous_count == 0 and action_space.size() == 1:
		return "discrete"
	return "hybrid"


func _flatten_action_names(action_space:Dictionary) -> Array:
	var names := []
	for action_name in action_space.keys():
		var component: Dictionary = action_space[action_name]
		var component_type := str(component.get("action_type", "discrete"))
		var size := int(component.get("size", 1))
		if component_type == "discrete" and component.has("names"):
			names.append_array(component["names"])
		elif size <= 1:
			names.append(str(action_name))
		else:
			for idx in range(size):
				names.append("%s_%d" % [str(action_name), idx])
	return names


func _continuous_action_size(action_space:Dictionary) -> int:
	var total := 0
	for component in action_space.values():
		if str(component.get("action_type", "discrete")) == "continuous":
			total += int(component.get("size", 1))
	return total


func _continuous_action_bounds(action_space:Dictionary, key:String, default_value:float) -> Array:
	var result := []
	for component in action_space.values():
		if str(component.get("action_type", "discrete")) != "continuous":
			continue
		var size := int(component.get("size", 1))
		var value: Variant = component.get(key, default_value)
		if typeof(value) == TYPE_ARRAY or typeof(value) == TYPE_PACKED_FLOAT32_ARRAY or typeof(value) == TYPE_PACKED_FLOAT64_ARRAY:
			var values: Array = Array(value)
			for idx in range(size):
				result.append(float(values[idx]) if idx < values.size() else default_value)
		else:
			for idx in range(size):
				result.append(float(value))
	return result


func _single_discrete_action_size(action_space:Dictionary) -> int:
	for component in action_space.values():
		if str(component.get("action_type", "discrete")) == "discrete":
			return int(component.get("size", 1))
	return 0


func _total_discrete_action_size(action_space:Dictionary) -> int:
	var total := 0
	for component in action_space.values():
		if str(component.get("action_type", "discrete")) == "discrete":
			total += int(component.get("size", 1))
	return total


func _build_reset_transform(agent_id:String, original_transform:Variant, rng:RandomNumberGenerator) -> Variant:
	var reset_transform: Variant = original_transform
	var reset_info := Dictionary(_last_reset_info.get(agent_id, {})).duplicate(true)
	reset_info.merge({
		"randomized": false,
		"mode": "original"
	}, false)

	if not randomize_reset:
		_last_reset_info[agent_id] = reset_info
		return reset_transform

	reset_info["randomized"] = true

	if reset_use_progress_provider and _progress_provider != null and _progress_provider.has_method("build_reset_transform"):
		var provider_result:Variant = _progress_provider.build_reset_transform(original_transform, rng, {
			"progress_min": reset_progress_min,
			"progress_max": reset_progress_max,
			"lateral_jitter": reset_lateral_jitter,
			"align_to_progress": reset_align_to_progress
		})
		if typeof(provider_result) == TYPE_DICTIONARY and provider_result.has("transform"):
			reset_transform = provider_result["transform"]
			reset_info.merge(provider_result, true)
			reset_info.erase("transform")
			if reset_info.has("progress"):
				reset_info["track_progress"] = reset_info["progress"]
		else:
			reset_info["mode"] = "jitter"
	else:
		reset_info["mode"] = "jitter"

	if typeof(reset_transform) == TYPE_TRANSFORM3D:
		reset_transform = _apply_position_jitter(reset_transform, rng, reset_info)
		reset_transform = _apply_yaw_jitter(reset_transform, rng, reset_info)
	elif typeof(reset_transform) == TYPE_TRANSFORM2D:
		reset_transform = _apply_2d_jitter(reset_transform, rng, reset_info)
	_last_reset_info[agent_id] = reset_info
	return reset_transform

func _reset_rng_for_agent(_seed:int, agent_id:String) -> RandomNumberGenerator:
	var rng := RandomNumberGenerator.new()
	var mixed_seed := int(_seed) & 0x7fffffff
	for idx in range(agent_id.length()):
		mixed_seed = int((mixed_seed * 1103515245 + agent_id.unicode_at(idx) + 12345) & 0x7fffffff)
	rng.seed = mixed_seed
	return rng

func _apply_position_jitter(reset_transform:Transform3D, rng:RandomNumberGenerator, reset_info:Dictionary) -> Transform3D:
	if reset_position_jitter == Vector3.ZERO:
		return reset_transform

	var jitter := Vector3(
		rng.randf_range(-reset_position_jitter.x, reset_position_jitter.x),
		rng.randf_range(-reset_position_jitter.y, reset_position_jitter.y),
		rng.randf_range(-reset_position_jitter.z, reset_position_jitter.z)
	)
	reset_transform.origin += reset_transform.basis * jitter
	reset_info["position_jitter"] = jitter
	return reset_transform


func _apply_yaw_jitter(reset_transform:Transform3D, rng:RandomNumberGenerator, reset_info:Dictionary) -> Transform3D:
	if reset_yaw_jitter_degrees <= 0.0:
		return reset_transform

	var yaw_jitter := deg_to_rad(rng.randf_range(-reset_yaw_jitter_degrees, reset_yaw_jitter_degrees))
	reset_transform.basis = Basis().rotated(Vector3.UP, yaw_jitter) * reset_transform.basis
	reset_info["yaw_jitter_degrees"] = rad_to_deg(yaw_jitter)
	return reset_transform


func _apply_2d_jitter(reset_transform:Transform2D, rng:RandomNumberGenerator, reset_info:Dictionary) -> Transform2D:
	var jitter := Vector2(
		rng.randf_range(-reset_position_jitter.x, reset_position_jitter.x),
		rng.randf_range(-reset_position_jitter.y, reset_position_jitter.y)
	)
	reset_transform.origin += reset_transform.basis_xform(jitter)
	if jitter != Vector2.ZERO:
		reset_info["position_jitter"] = jitter

	if reset_yaw_jitter_degrees > 0.0:
		var rotation_jitter := deg_to_rad(rng.randf_range(-reset_yaw_jitter_degrees, reset_yaw_jitter_degrees))
		reset_transform = reset_transform.rotated_local(rotation_jitter)
		reset_info["rotation_jitter_degrees"] = rad_to_deg(rotation_jitter)
	return reset_transform

func _refresh_agents() -> void:
	_agents.clear()

	for agent in controlled_agents:
		_append_runtime_agent(agent)

	if agent_to_replicate != null:
		_append_runtime_agent(agent_to_replicate)

	for agent in _replicated_agents:
		_append_runtime_agent(agent)

func _apply_training_optimizations() -> void:
	if not auto_optimize_in_headless or not _is_headless_runtime:
		return
	for agent in _agents:
		if agent.has_method("set_training_optimized"):
			agent.set_training_optimized(true)


func _append_runtime_agent(agent:Node) -> void:
	if agent == null:
		return
	if _agents.has(agent):
		return
	_agents.append(agent)


func _update_debug_agents_label() -> void:
	var text := (
		"Agents runtime: %d\nConfigured: %d\nReplicas: %d/%d"
		% [_agents.size(), controlled_agents.size(), _replicated_agents.size(), number_of_replications]
	)
	print("[ScenarioController] %s" % text.replace("\n", " | "))

	if debug_total_agents_label_path.is_empty():
		return

	var debug_label := get_node_or_null(debug_total_agents_label_path)
	if debug_label != null:
		debug_label.set("text", text)



func _build_agent_reset_result(agent:Node) -> Dictionary:
	var agent_id := _agent_id(agent)
	var progress := get_progress(agent)
	return {
		"id": agent_id,
		"obs": _agent_observation_vector(agent),
		"info": {
			"step": step_count,
			"reset": _last_reset_info.get(agent_id, {}),
			"progress": progress,
			"track_progress": progress
		}
	}


func _build_agent_step_result(agent:Node, truncated:bool, applied_action:Variant = 0) -> Dictionary:
	var agent_id := _agent_id(agent)
	_update_agent_events(agent)
	var context := _build_scenario_context(agent, truncated)
	var scenario_reward := compute_reward_scenario(agent, context)
	var scenario_terminal_reason := _get_scenario_terminal_reason(agent_id)
	var scenario_terminal := not scenario_terminal_reason.is_empty()
	if scenario_terminal:
		context["scenario_terminal"] = true
		context["scenario_terminal_reason"] = scenario_terminal_reason
		context["terminated"] = true
		context["done"] = true
		context["episode_done"] = true
	var local_reward := _agent_reward(agent, context)
	# The reward actually sent to Python is agent-local + scenario (see the return below). Expose
	# that full per-step total for debug HUDs / tooling, not just the scenario part.
	last_reward = local_reward + scenario_reward
	var target_reached := bool(context.get("target_reached", false))
	var finish_reached := bool(context.get("finish_reached", target_reached))
	var agent_terminal := bool(context.get("agent_terminal", false))
	var event_terminal_reason := _get_event_terminal_reason(agent_id)
	var progress_stalled := _is_progress_stalled(agent_id)
	var events := _get_agent_events(agent_id)
	var terminated := not event_terminal_reason.is_empty() or agent_terminal or scenario_terminal
	var done := terminated or truncated
	var progress := float(context.get("progress", 0.0))

	var info := {
		"step": step_count,
		"local_term_rewards": _agent_reward_terms(agent),
		"scenario_reward": scenario_reward,
		"scenario_terms": _get_scenario_terms(agent_id),
		"events": events,
		"target_reached": target_reached,
		"finish_reached": finish_reached,
		"agent_terminal": agent_terminal,
		"progress_stalled": progress_stalled,
		"scenario_terminal": scenario_terminal,
		"terminal_reason": _terminal_reason(event_terminal_reason, agent_terminal, scenario_terminal_reason, truncated),
		"target_first_seen": bool(context.get("target_first_seen", false)),
		"progress": progress,
		"track_progress": progress,
		"applied_action": applied_action
	}
	# Generic hook: any agent body may surface extra per-step diagnostics (e.g. max_joint_speed,
	# hold_frames) for the trainer logs, without the controller knowing the scenario specifics.
	if agent.has_method("get_debug_metrics"):
		var debug_metrics: Variant = agent.get_debug_metrics()
		if typeof(debug_metrics) == TYPE_DICTIONARY:
			info.merge(debug_metrics, true)

	return {
		"id": agent_id,
		"obs": _agent_observation_vector(agent),
		"reward": local_reward + scenario_reward,
		"done": done,
		"terminated": terminated,
		"truncated": truncated,
		"info": info
	}


func _update_agent_events(agent:Node) -> void:
	if _event_system != null and _event_system.has_method("update_agent"):
		_event_system.update_agent(agent, {
			"agent_id": _agent_id(agent),
			"step": step_count
		})


func _build_scenario_context(agent:Node, truncated := false) -> Dictionary:
	var agent_id := _agent_id(agent)
	var agent_terminal := bool(agent.is_terminal()) if agent.has_method("is_terminal") else false
	var event_terminal_reason := _get_event_terminal_reason(agent_id)
	var event_terminal := not event_terminal_reason.is_empty()
	var terminated := agent_terminal or event_terminal
	var episode_done := terminated or truncated
	var context := {
		"agent_id": agent_id,
		"step": step_count,
		"agent_terminal": agent_terminal,
		"event_terminal": event_terminal,
		"event_terminal_reason": event_terminal_reason,
		"terminated": terminated,
		"truncated": truncated,
		"done": episode_done,
		"episode_done": episode_done
	}
	if agent.has_method("has_succeeded"):
		context["agent_succeeded"] = bool(agent.has_succeeded())
	if _event_system != null and _event_system.has_method("get_agent_context"):
		var event_context:Variant = _event_system.get_agent_context(agent_id)
		if typeof(event_context) == TYPE_DICTIONARY:
			context.merge(event_context, true)
	var progress := get_progress(agent, context)
	context["progress"] = progress
	context["track_progress"] = progress
	return context


func _mark_agent_done(agent:Node, channel:Dictionary) -> void:
	var agent_id := _agent_id(agent)
	_done_agents[agent_id] = true
	_cached_done_step_results[agent_id] = channel.duplicate(true)
	_set_agent_training_active(agent, false)


func _cached_done_step_result(agent:Node, truncated:bool) -> Dictionary:
	var agent_id := _agent_id(agent)
	var cached: Variant = _cached_done_step_results.get(agent_id, {})
	var channel: Dictionary = cached.duplicate(true) if typeof(cached) == TYPE_DICTIONARY else {}
	if channel.is_empty():
		channel = {
			"id": agent_id,
			"obs": _agent_observation_vector(agent),
			"reward": 0.0,
			"done": true,
			"terminated": true,
			"truncated": truncated,
			"info": {
				"step": step_count,
				"terminal_reason": "cached_done"
			}
		}
	channel["reward"] = 0.0
	channel["done"] = true
	channel["terminated"] = bool(channel.get("terminated", true))
	channel["truncated"] = bool(channel.get("truncated", false)) or truncated
	var info: Dictionary = channel.get("info", {})
	info["step"] = step_count
	info["cached_done"] = true
	channel["info"] = info
	return channel


func _set_agent_training_active(agent:Node, enabled:bool) -> void:
	if agent.has_method("set_training_active"):
		agent.set_training_active(enabled)


func _is_agent_done(agent:Node) -> bool:
	return bool(_done_agents.get(_agent_id(agent), false))


func _terminal_reason(event_reason:String, agent_terminal:bool, scenario_reason:String, truncated:bool) -> String:
	if not event_reason.is_empty():
		return event_reason
	if agent_terminal:
		return "agent_terminal"
	if not scenario_reason.is_empty():
		return scenario_reason
	if truncated:
		return "truncated"
	return ""


func _is_progress_stalled(agent_id:String) -> bool:
	if _scenario_reward_system != null and _scenario_reward_system.has_method("is_agent_stalled"):
		return bool(_scenario_reward_system.is_agent_stalled(agent_id))
	return false


func _get_scenario_terminal_reason(agent_id:String) -> String:
	if _scenario_reward_system != null and _scenario_reward_system.has_method("get_terminal_reason"):
		return str(_scenario_reward_system.get_terminal_reason(agent_id))
	return ""


func _get_event_terminal_reason(agent_id:String) -> String:
	if _event_system != null and _event_system.has_method("get_terminal_reason"):
		return str(_event_system.get_terminal_reason(agent_id))
	return ""


func _get_scenario_terms(agent_id:String) -> Dictionary:
	if _scenario_reward_system != null and _scenario_reward_system.has_method("get_agent_terms"):
		var terms: Variant = _scenario_reward_system.get_agent_terms(agent_id)
		if typeof(terms) == TYPE_DICTIONARY:
			return terms
	return {}


func _get_agent_events(agent_id:String) -> Dictionary:
	if _event_system != null and _event_system.has_method("get_agent_context"):
		var events:Variant = _event_system.get_agent_context(agent_id)
		if typeof(events) == TYPE_DICTIONARY:
			return events
	return {}


func _apply_config_to_node_tree(node:Node, config:Dictionary) -> void:
	if node.has_method("apply_scenario_config"):
		node.apply_scenario_config(config)
	for child in node.get_children():
		_apply_config_to_node_tree(child, config)


func _node_has_property(node:Node, property_name:String) -> bool:
	for property in node.get_property_list():
		if str(property.get("name", "")) == property_name:
			return true
	return false


func _agent_interface(agent_body:Node) -> Node:
	var child := agent_body.get_node_or_null(NodePath("Agent"))
	if child != null and child.has_method("get_observation_vector"):
		return child
	if agent_body.has_method("get_observation_vector"):
		return agent_body
	return null


func _agent_observation_vector(agent_body:Node) -> Array:
	var interface := _agent_interface(agent_body)
	if interface != null:
		return interface.get_observation_vector()
	return []


func _agent_observation_size(agent_body:Node) -> int:
	var interface := _agent_interface(agent_body)
	if interface != null:
		return int(interface.get_observation_size())
	return 0


func _agent_observation_names(agent_body:Node) -> Array:
	var interface := _agent_interface(agent_body)
	if interface != null and interface.has_method("get_observation_names"):
		return interface.get_observation_names()
	return []


func _agent_reward(agent_body:Node, context:Dictionary = {}) -> float:
	var interface := _agent_interface(agent_body)
	if interface != null and interface.has_method("get_reward"):
		var reward_context := context.duplicate(true)
		reward_context["body"] = agent_body
		return float(interface.get_reward(reward_context))
	return 0.0


func _agent_reward_terms(agent_body:Node) -> Dictionary:
	var interface := _agent_interface(agent_body)
	if interface != null and interface.has_method("get_reward_terms"):
		return interface.get_reward_terms()
	return {}


func _reset_agent_reward(agent_body:Node) -> void:
	var interface := _agent_interface(agent_body)
	if interface != null and interface.has_method("reset_reward"):
		interface.reset_reward({"body": agent_body})


func _refresh_agent_sensors(agent_body:Node) -> void:
	var interface := _agent_interface(agent_body)
	if interface != null and interface.has_method("refresh_observation_sources"):
		interface.refresh_observation_sources()
	elif agent_body.has_method("refresh_sensors"):
		agent_body.refresh_sensors()


func _get_action_for_agent(actions:Variant, agent:Node) -> Variant:
	if typeof(actions) == TYPE_DICTIONARY:
		var action_map: Dictionary = actions
		return action_map.get(_agent_id(agent), 0)

	return actions


func _should_return_single_agent_response(actions:Variant) -> bool:
	return _agents.size() == 1 and typeof(actions) != TYPE_DICTIONARY


func _agent_id(agent:Node) -> String:
	return str(agent.name)


func _agent_team_id(agent:Node) -> Variant:
	if agent.has_method("get_team_id"):
		return agent.get_team_id()

	var interface := _agent_interface(agent)
	if interface != null and interface.has_method("get_team_id"):
		return interface.get_team_id()

	if _node_has_property(agent, "team_id"):
		return agent.get("team_id")
	if interface != null and _node_has_property(interface, "team_id"):
		return interface.get("team_id")
	return null


func _agent_policy_id(agent:Node) -> String:
	if agent.has_method("get_policy_id"):
		return str(agent.get_policy_id())

	var interface := _agent_interface(agent)
	if interface != null and interface.has_method("get_policy_id"):
		return str(interface.get_policy_id())

	if _node_has_property(agent, "policy_id"):
		return str(agent.get("policy_id"))
	if interface != null and _node_has_property(interface, "policy_id"):
		return str(interface.get("policy_id"))
	return "shared"


func _all_agents_done(channels:Array) -> bool:
	if channels.is_empty():
		return true

	for channel in channels:
		if not bool(channel.get("done", false)):
			return false
	return true


func _all_agents_terminated(channels:Array) -> bool:
	if channels.is_empty():
		return true

	for channel in channels:
		if not bool(channel.get("terminated", false)):
			return false
	return true


func _update_current_agent_camera() -> void:
	if not manage_agent_cameras:
		return

	var selected_agent: Node = null
	if recording_mode and not recording_agent_id.is_empty():
		for agent in _agents:
			if _agent_id(agent) == recording_agent_id and not _is_agent_terminal(agent):
				selected_agent = agent
				break

	for agent in _agents:
		if selected_agent != null:
			break
		if _is_agent_terminal(agent):
			continue
		selected_agent = agent

	for agent in _agents:
		if agent.has_method("set_camera_current"):
			agent.set_camera_current(agent == selected_agent)


func _is_agent_terminal(agent:Node) -> bool:
	var agent_id := _agent_id(agent)
	if _is_agent_done(agent):
		return true
	if not _get_event_terminal_reason(agent_id).is_empty():
		return true
	if not _get_scenario_terminal_reason(agent_id).is_empty():
		return true
	if agent.has_method("is_terminal"):
		return bool(agent.is_terminal())
	return false


func _is_headless() -> bool:
	return DisplayServer.get_name().to_lower() == "headless" or OS.has_feature("headless")







	
