extends "res://addons/metis/runtime/agent/scenario_reward_components/ScenarioRewardComponent.gd"
class_name TargetDistanceDeltaScenarioReward

## Method exposed by the controlled agent that returns its current target distance.
@export var distance_method: StringName = &"get_target_distance"
## Reward per normalized distance unit gained while approaching the target.
@export var approach_reward_scale := 20.0
## Penalty per normalized distance unit lost while moving away from the target.
@export var retreat_penalty_scale := 25.0
## Converts the distance unit returned by distance_method into a normalized delta.
## Keep this at 1.0 for methods returning metres.
@export var normalization_distance := 1.0

var _previous_distance := {}
var _last_terms := {}


func reset_rewards() -> void:
	_previous_distance.clear()
	_last_terms.clear()


func reset_agent(agent_id:String, _context:Dictionary = {}) -> void:
	_previous_distance.erase(agent_id)
	_last_terms[agent_id] = {}


func compute_reward(agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", agent.name))
	if not agent or not agent.has_method(distance_method):
		_last_terms[agent_id] = {"available": false}
		return 0.0

	var distance := maxf(float(agent.call(distance_method)), 0.0)
	if not is_finite(distance):
		_last_terms[agent_id] = {"available": false}
		return 0.0
	if not _previous_distance.has(agent_id):
		_previous_distance[agent_id] = distance
		_last_terms[agent_id] = {
			"available": true,
			"distance": distance,
			"distance_delta": 0.0,
		}
		return 0.0

	var previous := float(_previous_distance[agent_id])
	var improvement := previous - distance
	_previous_distance[agent_id] = distance
	var normalized_delta := improvement / maxf(normalization_distance, 0.000001)
	var scale := (
		approach_reward_scale
		if normalized_delta >= 0.0
		else retreat_penalty_scale)
	var value := normalized_delta * scale * weight
	_last_terms[agent_id] = {
		"available": true,
		"distance": distance,
		"previous_distance": previous,
		"distance_delta": improvement,
		"reward": value,
	}
	return value


func get_agent_terms(agent_id:String) -> Dictionary:
	return Dictionary(_last_terms.get(agent_id, {})).duplicate(true)
