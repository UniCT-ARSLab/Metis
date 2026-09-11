extends ScenarioRewardComponent
class_name TerminalProgressScenarioReward

## Base cost shared by every failed terminal state.
@export var base_failure_penalty := -50.0
## Additional cost when the episode ends far from completion. At progress 0 this value is
## applied in full; at progress 1 it contributes nothing.
@export var remaining_progress_penalty := -50.0
## Empty means every non-success terminal reason. Otherwise only these reasons are penalized.
@export var failure_terminal_reasons := PackedStringArray()
@export var success_terminal_reasons := PackedStringArray(
	["target_reached", "finish_reached", "level_cleared", "goal_scored", "success"])
@export var success_context_keys := PackedStringArray(
	[
		"agent_succeeded",
		"target_acquired",
		"target_reached",
		"finish_reached",
		"level_cleared",
		"goal_scored",
		"success",
	])
@export var penalize_truncation := true
@export var penalize_unclassified_terminal := true
@export var only_once := true

var _penalized_agents := {}


func reset_rewards() -> void:
	_penalized_agents.clear()


func reset_agent(agent_id: String, _context: Dictionary = {}) -> void:
	_penalized_agents[agent_id] = false


func is_episode_end_only() -> bool:
	return true


func compute_reward(_agent: Node, context: Dictionary = {}) -> float:
	if not bool(context.get("episode_done", context.get("done", false))):
		return 0.0

	var agent_id := str(context.get("agent_id", ""))
	if only_once and bool(_penalized_agents.get(agent_id, false)):
		return 0.0

	var terminal_reason := _terminal_reason(context)
	if _is_success(context, terminal_reason):
		return 0.0
	if not _is_failure(context, terminal_reason):
		return 0.0

	_penalized_agents[agent_id] = true
	var progress := clampf(
		float(context.get("progress", context.get("track_progress", 0.0))),
		0.0,
		1.0)
	return (
		base_failure_penalty
		+ remaining_progress_penalty * (1.0 - progress)
	) * weight


func _terminal_reason(context: Dictionary) -> String:
	for key in [
		"scenario_terminal_reason",
		"event_terminal_reason",
		"terminal_reason",
	]:
		var reason := str(context.get(key, "")).strip_edges().to_lower()
		if not reason.is_empty():
			return reason
	return "truncated" if bool(context.get("truncated", false)) else ""


func _is_success(context: Dictionary, terminal_reason: String) -> bool:
	if terminal_reason in success_terminal_reasons:
		return true
	# A historical success flag must not hide a later explicit failure. This matters for
	# continuous tasks that keep running after reaching the goal: reach -> collision is a
	# failed terminal transition, while reach -> time-limit truncation may still count.
	if not terminal_reason.is_empty() and terminal_reason != "truncated":
		return false
	for key in success_context_keys:
		if bool(context.get(key, false)):
			return true
	return false


func _is_failure(context: Dictionary, terminal_reason: String) -> bool:
	if bool(context.get("truncated", false)):
		return penalize_truncation
	if terminal_reason.is_empty():
		return penalize_unclassified_terminal
	return (
		failure_terminal_reasons.is_empty()
		or terminal_reason in failure_terminal_reasons
	)
