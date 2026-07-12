extends Node

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
@export var max_steps:= 500
@export var manage_agent_cameras := true

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

	await get_tree().physics_frame

	var truncated := step_count >= max_steps
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
	var seed := int(request.get("seed", 0))
	return await reset_episode(seed)


func configure(config:Dictionary) -> Dictionary:
	_scenario_reward_system = get_node_or_null(scenario_reward_system_path)
	_progress_provider = get_node_or_null(progress_provider_path)
	_event_system = get_node_or_null(event_system_path)
	if config.has("max_steps"):
		max_steps = int(config["max_steps"])
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
	if _scenario_reward_system != null:
		_apply_config_to_node_tree(_scenario_reward_system, config)
	_update_current_agent_camera()

	return {
		"ok": true,
		"max_steps": max_steps,
		"reset_progress_min": reset_progress_min,
		"reset_progress_max": reset_progress_max,
		"reset_track_progress_min": reset_progress_min,
		"reset_track_progress_max": reset_progress_max,
		"reset_lateral_jitter": reset_lateral_jitter,
		"reset_yaw_jitter_degrees": reset_yaw_jitter_degrees,
		"recording_mode": recording_mode,
		"recording_agent_id": recording_agent_id,
		"disable_replication_in_recording": disable_replication_in_recording,
		"manage_agent_cameras": manage_agent_cameras
	}


func reset_episode(seed := 0):
	_refresh_agents()
	_scenario_reward_system = get_node_or_null(scenario_reward_system_path)
	_progress_provider = get_node_or_null(progress_provider_path)
	_event_system = get_node_or_null(event_system_path)
	step_count = 0
	_last_reset_info.clear()
	_done_agents.clear()
	_cached_done_step_results.clear()
	if _scenario_reward_system != null and _scenario_reward_system.has_method("reset_rewards"):
		_scenario_reward_system.reset_rewards()
	if _progress_provider != null and _progress_provider.has_method("reset_provider"):
		_progress_provider.reset_provider()
	if _event_system != null and _event_system.has_method("reset_events"):
		_event_system.reset_events()

	var shared_rng := RandomNumberGenerator.new()
	shared_rng.seed = int(seed)

	var channels := []
	for agent in _agents:
		var agent_id := _agent_id(agent)
		var rng := _reset_rng_for_agent(int(seed), agent_id) if use_agent_specific_reset_seed else shared_rng
		
		_done_agents[agent_id] = false

		var original_transform: Transform3D = agent.transform
		if _original_agent_transforms.has(agent_id):
			original_transform = _original_agent_transforms[agent_id]
		var reset_transform := _build_reset_transform(agent_id, original_transform, rng)
		_set_agent_training_active(agent, true)
		agent.reset_all(reset_transform, false)
		var reset_context := {"agent_id": agent_id, "step": step_count}
		if _event_system != null and _event_system.has_method("reset_agent"):
			_event_system.reset_agent(agent, reset_context)
		reset_context = _build_scenario_context(agent)
		if _progress_provider != null and _progress_provider.has_method("reset_agent"):
			_progress_provider.reset_agent(agent, reset_context)
		if _scenario_reward_system != null and _scenario_reward_system.has_method("reset_agent"):
			_scenario_reward_system.reset_agent(agent_id, reset_context)

	await get_tree().physics_frame
	for agent in _agents:
		agent.refresh_sensors()

	await get_tree().physics_frame
	for agent in _agents:
		agent.refresh_sensors()
		agent.reset_reward()

	for agent in _agents:
		channels.append(_build_agent_reset_result(agent))
	_update_current_agent_camera()
	
	if channels.size() == 1:
		var single: Dictionary = channels[0].duplicate(true)
		single["agents"] = channels
		return single

	return {
		"agents": channels,
		"info": {
			"step": step_count,
			"reset": _last_reset_info.duplicate(true)
		}
	}


func get_spec() -> Dictionary:
	_refresh_agents()

	var agent_specs := []
	for agent in _agents:
		var action_space := _get_agent_action_space(agent)
		var action_type := _infer_action_type(action_space)
		var action_names := _flatten_action_names(action_space)
		var agent_spec := {
			"id": _agent_id(agent),
			"obs_dim": agent.get_observation_size(),
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
		"multi_agent": agent_specs.size() > 1
	}


func get_progress(agent:Node, context:Dictionary = {}) -> float:
	if _progress_provider != null and _progress_provider.has_method("get_progress"):
		return float(_progress_provider.get_progress(agent, context))
	if agent.has_method("get_progress"):
		return float(agent.get_progress())
	return 0.0


func _get_agent_action_space(agent:Node) -> Dictionary:
	if agent.has_method("get_action_space"):
		var action_space: Variant = agent.get_action_space()
		if typeof(action_space) == TYPE_DICTIONARY:
			return _normalize_action_space(action_space)

	var action_type := "discrete"
	if agent.has_method("get_action_type"):
		action_type = str(agent.get_action_type())

	if action_type == "continuous":
		var action_names: Array = agent.get_action_names()
		var lows: Array = []
		var highs: Array = []
		if agent.has_method("get_action_low"):
			lows = agent.get_action_low()
		if agent.has_method("get_action_high"):
			highs = agent.get_action_high()
		var result := {}
		for idx in range(int(agent.get_action_size())):
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
			"size": agent.get_action_count(),
			"action_type": "discrete",
			"names": agent.get_action_names()
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


func _build_reset_transform(agent_id:String, original_transform:Transform3D, rng:RandomNumberGenerator) -> Transform3D:
	var reset_transform := original_transform
	var reset_info := {
		"randomized": false,
		"mode": "original"
	}

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

	reset_transform = _apply_position_jitter(reset_transform, rng, reset_info)
	reset_transform = _apply_yaw_jitter(reset_transform, rng, reset_info)
	_last_reset_info[agent_id] = reset_info
	return reset_transform

func _reset_rng_for_agent(seed:int, agent_id:String) -> RandomNumberGenerator:
	var rng := RandomNumberGenerator.new()
	var mixed_seed := int(seed) & 0x7fffffff
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
		"obs": agent.get_observation_vector(),
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
	var context := _build_scenario_context(agent)
	var local_reward := float(agent.get_reward())
	var scenario_reward := compute_reward_scenario(agent, context)
	var target_reached := bool(context.get("target_reached", false))
	var finish_reached := bool(context.get("finish_reached", target_reached))
	var agent_terminal := bool(context.get("agent_terminal", false))
	var event_terminal_reason := _get_event_terminal_reason(agent_id)
	var scenario_terminal_reason := _get_scenario_terminal_reason(agent_id)
	var scenario_terminal := not scenario_terminal_reason.is_empty()
	var progress_stalled := _is_progress_stalled(agent_id)
	var terminated := not event_terminal_reason.is_empty() or agent_terminal or scenario_terminal
	var done := terminated or truncated
	var progress := float(context.get("progress", 0.0))

	return {
		"id": agent_id,
		"obs": agent.get_observation_vector(),
		"reward": local_reward + scenario_reward,
		"done": done,
		"terminated": terminated,
		"truncated": truncated,
		"info": {
			"step": step_count,
			"local_term_rewards": agent.get_reward_terms(),
			"scenario_reward": scenario_reward,
			"scenario_terms": _get_scenario_terms(agent_id),
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
	}


func _update_agent_events(agent:Node) -> void:
	if _event_system != null and _event_system.has_method("update_agent"):
		_event_system.update_agent(agent, {
			"agent_id": _agent_id(agent),
			"step": step_count
		})


func _build_scenario_context(agent:Node) -> Dictionary:
	var agent_id := _agent_id(agent)
	var agent_terminal := bool(agent.is_terminal()) if agent.has_method("is_terminal") else false
	var context := {
		"agent_id": agent_id,
		"step": step_count,
		"agent_terminal": agent_terminal
	}
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
			"obs": agent.get_observation_vector(),
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


func _apply_config_to_node_tree(node:Node, config:Dictionary) -> void:
	_apply_config_to_node(node, config)
	for child in node.get_children():
		_apply_config_to_node_tree(child, config)


func _apply_config_to_node(node:Node, config:Dictionary) -> void:
	for key in config.keys():
		var property_name := str(key)
		if _node_has_property(node, property_name):
			node.set(property_name, config[key])


func _node_has_property(node:Node, property_name:String) -> bool:
	for property in node.get_property_list():
		if str(property.get("name", "")) == property_name:
			return true
	return false


func _get_action_for_agent(actions:Variant, agent:Node) -> Variant:
	if typeof(actions) == TYPE_DICTIONARY:
		var action_map: Dictionary = actions
		return action_map.get(_agent_id(agent), 0)

	return actions


func _should_return_single_agent_response(actions:Variant) -> bool:
	return _agents.size() == 1 and typeof(actions) != TYPE_DICTIONARY


func _agent_id(agent:Node) -> String:
	return str(agent.name)


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







	
