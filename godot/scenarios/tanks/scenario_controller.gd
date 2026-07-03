extends Node

@export var controlled_agents: Array[Tank] = []
@export var target:Area3D

@export_category("RL Info")
@export var reward_target_reached = 1.0
@export var first_seen_reward = 0.1
@export var max_steps:= 500

var step_count := 0
var _agents: Array[Tank] = []
var _target_reached := {}
var _target_first_seen := {}
var _original_agent_transforms := {}

func _ready() -> void:
	_refresh_agents()
	if target != null:
		target.body_entered.connect(on_target_body_entered)
	
	for agent in _agents:
		var agent_id := _agent_id(agent)
		_original_agent_transforms[agent_id] = agent.transform


func on_target_body_entered(body):
	if body is Tank and _is_controlled_agent(body):
		_target_reached[_agent_id(body)] = true


func compute_reward_scenario(agent:Tank) -> float:
	var agent_id := _agent_id(agent)
	var reward := 0.0
	
	var observations := agent.get_observations()
	var target_visible := float(observations.get("target_visible", 0.0))
	if target_visible > 0.5 and not bool(_target_first_seen.get(agent_id, false)):
		_target_first_seen[agent_id] = true
		reward += first_seen_reward
	
	if bool(_target_reached.get(agent_id, false)):
		reward += reward_target_reached
	
	return reward


func step(actions:Variant):
	_refresh_agents()
	step_count += 1

	for agent in _agents:
		var agent_action := _get_action_for_agent(actions, agent)
		agent.apply_action(agent_action)

	await get_tree().physics_frame

	var truncated := step_count >= max_steps
	var channels := []
	for agent in _agents:
		channels.append(_build_agent_step_result(agent, truncated))

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
	

func reset_episode():
	_refresh_agents()
	step_count = 0
	_target_reached.clear()
	_target_first_seen.clear()

	var channels := []
	for agent in _agents:
		var agent_id := _agent_id(agent)
		_target_reached[agent_id] = false
		_target_first_seen[agent_id] = false

		var original_transform: Transform3D = agent.transform
		if _original_agent_transforms.has(agent_id):
			original_transform = _original_agent_transforms[agent_id]
		agent.reset_all(original_transform)
		channels.append(_build_agent_reset_result(agent))
	
	if channels.size() == 1:
		var single: Dictionary = channels[0].duplicate(true)
		single["agents"] = channels
		return single

	return {
		"agents": channels,
		"info": {
			"step": step_count
		}
	}


func get_spec() -> Dictionary:
	_refresh_agents()

	var agent_specs := []
	for agent in _agents:
		agent_specs.append({
			"id": _agent_id(agent),
			"obs_dim": agent.get_observation_size(),
			"num_actions": agent.get_action_count(),
			"action_names": agent.get_action_names()
		})

	return {
		"ok": true,
		"agents": agent_specs,
		"multi_agent": agent_specs.size() > 1
	}


func _refresh_agents() -> void:
	_agents.clear()

	if not controlled_agents.is_empty():
		for agent in controlled_agents:
			if agent != null:
				_agents.append(agent)



func _build_agent_reset_result(agent:Tank) -> Dictionary:
	return {
		"id": _agent_id(agent),
		"obs": agent.get_observation_vector(),
		"info": {
			"step": step_count
		}
	}


func _build_agent_step_result(agent:Tank, truncated:bool) -> Dictionary:
	var agent_id := _agent_id(agent)
	var local_reward := float(agent.get_reward())
	var scenario_reward := compute_reward_scenario(agent)
	var terminated := bool(_target_reached.get(agent_id, false))
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
			"target_reached": terminated,
			"target_first_seen": bool(_target_first_seen.get(agent_id, false))
		}
	}


func _get_action_for_agent(actions:Variant, agent:Tank) -> int:
	if typeof(actions) == TYPE_DICTIONARY:
		var action_map: Dictionary = actions
		return int(action_map.get(_agent_id(agent), 0))

	return int(actions)


func _should_return_single_agent_response(actions:Variant) -> bool:
	return _agents.size() == 1 and typeof(actions) != TYPE_DICTIONARY


func _is_controlled_agent(body:Tank) -> bool:
	for agent in _agents:
		if agent == body:
			return true
	return false


func _agent_id(agent:Tank) -> String:
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







	
