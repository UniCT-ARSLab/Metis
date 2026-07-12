extends "res://scripts/agent/scenario_reward_components/ScenarioRewardComponent.gd"
class_name ProgressDeltaScenarioReward

@export var progress_reward_scale := 10.0
@export var backward_penalty_scale := 10.0
@export var forward_term_name := "progress"
@export var backward_term_name := "progress_backward"

var _previous_progress := {}
var _last_terms := {}


func reset_rewards() -> void:
	_previous_progress.clear()
	_last_terms.clear()


func reset_agent(agent_id:String, context:Dictionary = {}) -> void:
	_previous_progress[agent_id] = _context_progress(context)
	_last_terms[agent_id] = {}


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", ""))
	var progress := _context_progress(context)
	var previous := float(_previous_progress.get(agent_id, progress))
	var delta := progress - previous
	_previous_progress[agent_id] = progress

	var value := delta * (progress_reward_scale if delta >= 0.0 else backward_penalty_scale) * weight
	var term := forward_term_name if delta >= 0.0 else backward_term_name
	_last_terms[agent_id] = {
		term: value,
		"previous_progress": previous,
		"current_progress": progress
	}
	return value


func get_agent_terms(agent_id:String) -> Dictionary:
	return Dictionary(_last_terms.get(agent_id, {})).duplicate(true)


func _context_progress(context:Dictionary) -> float:
	return float(context.get("progress", context.get("track_progress", 0.0)))
