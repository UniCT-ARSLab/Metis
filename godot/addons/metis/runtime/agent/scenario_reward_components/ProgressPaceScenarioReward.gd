extends ScenarioRewardComponent
class_name ProgressPaceScenarioReward

@export_range(0.001, 1.0, 0.001) var progress_pace_bucket_size := 0.025
@export var progress_pace_steps_per_bucket := 80
@export var progress_pace_penalty := -0.01
@export var terminate_on_progress_pace_failure := false

var _anchor_progress := {}
var _anchor_step := {}
var _pace_failed := {}


func reset_rewards() -> void:
	_anchor_progress.clear()
	_anchor_step.clear()
	_pace_failed.clear()


func reset_agent(agent_id:String, context:Dictionary = {}) -> void:
	var progress := float(context.get("progress", context.get("track_progress", 0.0)))
	var step := int(context.get("step", 0))
	_anchor_progress[agent_id] = progress
	_anchor_step[agent_id] = step
	_pace_failed[agent_id] = false


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", ""))
	var progress := float(context.get("progress", context.get("track_progress", 0.0)))
	var step := int(context.get("step", 0))

	if not _anchor_progress.has(agent_id):
		reset_agent(agent_id, context)
		return 0.0

	var anchor := float(_anchor_progress.get(agent_id, progress))
	if progress >= anchor + progress_pace_bucket_size:
		_anchor_progress[agent_id] = progress
		_anchor_step[agent_id] = step
		_pace_failed[agent_id] = false
		return 0.0

	var start_step := int(_anchor_step.get(agent_id, step))
	if step - start_step <= max(1, progress_pace_steps_per_bucket):
		return 0.0

	_pace_failed[agent_id] = true
	return progress_pace_penalty * weight


func is_agent_stalled(agent_id:String) -> bool:
	if not terminate_on_progress_pace_failure:
		return false
	return bool(_pace_failed.get(agent_id, false))


func get_terminal_reason(agent_id:String) -> String:
	return "progress_pace_failed" if is_agent_stalled(agent_id) else ""


func get_agent_terms(agent_id:String) -> Dictionary:
	return {
		"pace_failed": bool(_pace_failed.get(agent_id, false)),
		"pace_anchor_progress": float(_anchor_progress.get(agent_id, 0.0)),
		"pace_anchor_step": int(_anchor_step.get(agent_id, 0))
	}
