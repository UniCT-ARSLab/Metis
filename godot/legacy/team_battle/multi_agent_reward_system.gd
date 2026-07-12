extends Node

@export var controller_path: NodePath

var controller
var prev_hp := {}
var prev_alive := {}

func _ready() -> void:
	controller = get_node(controller_path)
	reset_reward()

func reset_reward() -> void:
	prev_hp.clear()
	prev_alive.clear()
	for unit in controller.get_all_units():
		prev_hp[unit.name] = unit.hp
		prev_alive[unit.name] = unit.alive

func compute_per_agent_rewards(step_events: Dictionary, winner: int) -> Dictionary:
	var rewards := {}
	for unit in controller.get_all_units():
		rewards[unit.name] = -0.002

	for agent_name in step_events.get("damage_dealt", {}).keys():
		rewards[agent_name] = float(rewards.get(agent_name, 0.0)) + float(step_events.damage_dealt[agent_name]) * 0.02

	for agent_name in step_events.get("kills", {}).keys():
		rewards[agent_name] = float(rewards.get(agent_name, 0.0)) + float(step_events.kills[agent_name]) * 1.0

	for agent_name in step_events.get("boundary_hits", {}).keys():
		rewards[agent_name] = float(rewards.get(agent_name, 0.0)) - float(step_events.boundary_hits[agent_name]) * 0.02

	for unit in controller.get_all_units():
		var old_hp := float(prev_hp.get(unit.name, unit.max_hp))
		var old_alive := bool(prev_alive.get(unit.name, true))
		if unit.hp < old_hp:
			rewards[unit.name] = float(rewards.get(unit.name, 0.0)) - (old_hp - unit.hp) * 0.015
		if old_alive and not unit.alive:
			rewards[unit.name] = float(rewards.get(unit.name, 0.0)) - 1.0
		prev_hp[unit.name] = unit.hp
		prev_alive[unit.name] = unit.alive

	if winner == 0:
		for unit in controller.get_all_units():
			var team_a_bonus := 3.0 if unit.team_id == 0 else -3.0
			rewards[unit.name] = float(rewards.get(unit.name, 0.0)) + team_a_bonus
	elif winner == 1:
		for unit in controller.get_all_units():
			var team_b_bonus := 3.0 if unit.team_id == 1 else -3.0
			rewards[unit.name] = float(rewards.get(unit.name, 0.0)) + team_b_bonus

	return rewards
