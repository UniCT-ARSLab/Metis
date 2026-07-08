extends Node

@export var controlled_agents: Array[Node] = []
@export var target:Area3D

@export var track_path: Path3D
@export var progress_reward_scale := 10.0
@export var backward_penalty_scale := 10.0

var _previous_progress := {}

@export_category("Reset Randomization")
@export var randomize_reset := false
@export var reset_use_track_path := false
@export_range(0.0, 1.0, 0.001) var reset_track_progress_min := 0.0
@export_range(0.0, 1.0, 0.001) var reset_track_progress_max := 0.03
@export var reset_lateral_jitter := 0.25
@export var reset_position_jitter := Vector3.ZERO
@export var reset_yaw_jitter_degrees := 8.0
@export var reset_align_to_track := true

@export_category("RL Info")
@export var reward_target_reached = 5.0
@export var max_steps:= 500

var step_count := 0
var _agents: Array[Node] = []
var _target_reached := {}
var _target_first_seen := {}
var _original_agent_transforms := {}
var _last_reset_info := {}

func _ready() -> void:
	_refresh_agents()
	if target != null:
		target.body_entered.connect(on_target_body_entered)
	
	for agent in _agents:
		var agent_id := _agent_id(agent)
		_original_agent_transforms[agent_id] = agent.transform


func on_target_body_entered(body):
	if _is_controlled_agent(body):
		_target_reached[_agent_id(body)] = true


func compute_reward_scenario(agent:Node) -> float:
	var agent_id := _agent_id(agent)
	var reward := 0.0

	if agent is Node3D and track_path != null:
		var progress := get_track_progress(agent)
		var previous := float(_previous_progress.get(agent_id, progress))
		var delta := progress - previous

		if delta >= 0.0:
			reward += delta * progress_reward_scale
		else:
			reward += delta * backward_penalty_scale

		_previous_progress[agent_id] = progress

	if agent.has_method("get_observations"):
		var observations: Dictionary = agent.get_observations()
		var target_visible := float(observations.get("target_visible", 0.0))
		if target_visible > 0.5 and not bool(_target_first_seen.get(agent_id, false)):
			_target_first_seen[agent_id] = true

	if bool(_target_reached.get(agent_id, false)):
		reward += reward_target_reached

	return reward


func step(actions:Variant):
	_refresh_agents()
	step_count += 1

	var applied_actions := {}
	for agent in _agents:
		var agent_action: Variant = _get_action_for_agent(actions, agent)
		var applied_action: Variant = agent.apply_action(agent_action)
		applied_actions[_agent_id(agent)] = applied_action

	await get_tree().physics_frame

	var truncated := step_count >= max_steps
	var channels := []
	for agent in _agents:
		channels.append(_build_agent_step_result(agent, truncated, applied_actions.get(_agent_id(agent), 0)))

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


func reset_episode(seed := 0):
	_refresh_agents()
	step_count = 0
	_target_reached.clear()
	_target_first_seen.clear()
	_previous_progress.clear()
	_last_reset_info.clear()

	var rng := RandomNumberGenerator.new()
	rng.seed = int(seed)
	

	var channels := []
	for agent in _agents:
		var agent_id := _agent_id(agent)
		
		_target_reached[agent_id] = false
		_target_first_seen[agent_id] = false

		var original_transform: Transform3D = agent.transform
		if _original_agent_transforms.has(agent_id):
			original_transform = _original_agent_transforms[agent_id]
		var reset_transform := _build_reset_transform(agent_id, original_transform, rng)
		agent.reset_all(reset_transform, false)
		if agent is Node3D and track_path != null:
			_previous_progress[agent_id] = get_track_progress(agent)

	await get_tree().physics_frame
	for agent in _agents:
		agent.refresh_sensors()

	await get_tree().physics_frame
	for agent in _agents:
		agent.refresh_sensors()
		agent.reset_reward()

	for agent in _agents:
		channels.append(_build_agent_reset_result(agent))
	
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
		var agent_spec := {
			"id": _agent_id(agent),
			"obs_dim": agent.get_observation_size(),
			"action_names": agent.get_action_names()
		}
		var action_type := "discrete"
		if agent.has_method("get_action_type"):
			action_type = str(agent.get_action_type())
		agent_spec["action_type"] = action_type
		if action_type == "continuous":
			agent_spec["action_size"] = int(agent.get_action_size())
			agent_spec["action_low"] = agent.get_action_low()
			agent_spec["action_high"] = agent.get_action_high()
			agent_spec["num_actions"] = int(agent.get_action_size())
		else:
			agent_spec["num_actions"] = agent.get_action_count()
		agent_specs.append(agent_spec)

	return {
		"ok": true,
		"agents": agent_specs,
		"multi_agent": agent_specs.size() > 1
	}


func get_track_progress(agent: Node3D) -> float:
	if track_path == null:
		return 0.0

	var curve := track_path.curve
	if curve == null:
		return 0.0

	var total_length := curve.get_baked_length()
	if total_length <= 0.001:
		return 0.0

	var local_position := track_path.to_local(agent.global_position)
	var offset := curve.get_closest_offset(local_position)

	return clampf(offset / total_length, 0.0, 1.0)


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

	if reset_use_track_path and track_path != null and track_path.curve != null:
		reset_transform = _build_track_reset_transform(original_transform, rng, reset_info)
	else:
		reset_info["mode"] = "jitter"

	reset_transform = _apply_position_jitter(reset_transform, rng, reset_info)
	reset_transform = _apply_yaw_jitter(reset_transform, rng, reset_info)
	_last_reset_info[agent_id] = reset_info
	return reset_transform


func _build_track_reset_transform(original_transform:Transform3D, rng:RandomNumberGenerator, reset_info:Dictionary) -> Transform3D:
	var curve := track_path.curve
	var total_length := curve.get_baked_length()
	if total_length <= 0.001:
		reset_info["mode"] = "jitter"
		return original_transform

	var progress_min := clampf(minf(reset_track_progress_min, reset_track_progress_max), 0.0, 1.0)
	var progress_max := clampf(maxf(reset_track_progress_min, reset_track_progress_max), 0.0, 1.0)
	var progress := rng.randf_range(progress_min, progress_max)
	var offset := progress * total_length
	var local_position := curve.sample_baked(offset)
	var global_position := track_path.to_global(local_position)
	global_position.y = original_transform.origin.y

	var tangent := _track_tangent_at_offset(curve, offset, total_length)
	var yaw := _yaw_from_forward(tangent)
	var reset_transform := Transform3D(Basis().rotated(Vector3.UP, yaw), global_position)

	if not reset_align_to_track:
		reset_transform.basis = original_transform.basis

	if reset_lateral_jitter > 0.0:
		var lateral := rng.randf_range(-reset_lateral_jitter, reset_lateral_jitter)
		var right := Vector3.UP.cross(tangent).normalized()
		reset_transform.origin += right * lateral
		reset_info["lateral_offset"] = lateral

	reset_info["mode"] = "track_path"
	reset_info["track_progress"] = progress
	return reset_transform


func _track_tangent_at_offset(curve:Curve3D, offset:float, total_length:float) -> Vector3:
	var sample_delta := minf(0.5, maxf(total_length * 0.005, 0.05))
	var before := curve.sample_baked(clampf(offset - sample_delta, 0.0, total_length))
	var after := curve.sample_baked(clampf(offset + sample_delta, 0.0, total_length))
	var tangent := track_path.to_global(after) - track_path.to_global(before)
	tangent.y = 0.0
	if tangent.length() <= 0.000001:
		return Vector3.FORWARD
	return tangent.normalized()


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


func _yaw_from_forward(forward:Vector3) -> float:
	var flat_forward := forward
	flat_forward.y = 0.0
	if flat_forward.length() <= 0.000001:
		return 0.0
	flat_forward = flat_forward.normalized()
	return atan2(flat_forward.x, flat_forward.z)

func _refresh_agents() -> void:
	_agents.clear()

	if not controlled_agents.is_empty():
		for agent in controlled_agents:
			if agent != null:
				_agents.append(agent)



func _build_agent_reset_result(agent:Node) -> Dictionary:
	var agent_id := _agent_id(agent)
	return {
		"id": agent_id,
		"obs": agent.get_observation_vector(),
		"info": {
			"step": step_count,
			"reset": _last_reset_info.get(agent_id, {})
		}
	}


func _build_agent_step_result(agent:Node, truncated:bool, applied_action:Variant = 0) -> Dictionary:
	var agent_id := _agent_id(agent)
	var local_reward := float(agent.get_reward())
	var scenario_reward := compute_reward_scenario(agent)
	var target_reached := bool(_target_reached.get(agent_id, false))
	var agent_terminal := false
	if agent.has_method("is_terminal"):
		agent_terminal = bool(agent.is_terminal())
	var terminated := target_reached or agent_terminal
	var done := terminated or truncated

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
			"target_reached": target_reached,
			"finish_reached": target_reached,
			"agent_terminal": agent_terminal,
			"target_first_seen": bool(_target_first_seen.get(agent_id, false)),
			"track_progress": get_track_progress(agent) if agent is Node3D else 0.0,
			"applied_action": applied_action
		}
	}


func _get_action_for_agent(actions:Variant, agent:Node) -> Variant:
	if typeof(actions) == TYPE_DICTIONARY:
		var action_map: Dictionary = actions
		return action_map.get(_agent_id(agent), 0)

	return actions


func _should_return_single_agent_response(actions:Variant) -> bool:
	return _agents.size() == 1 and typeof(actions) != TYPE_DICTIONARY


func _is_controlled_agent(body:Node) -> bool:
	for agent in _agents:
		if agent == body:
			return true
	return false


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







	
