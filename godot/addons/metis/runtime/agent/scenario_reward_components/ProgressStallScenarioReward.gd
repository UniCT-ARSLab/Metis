extends "res://addons/metis/runtime/agent/scenario_reward_components/ScenarioRewardComponent.gd"
class_name ProgressStallScenarioReward

@export var terminate_on_stalled_progress := false
@export var stalled_progress_window_steps := 90
@export var stalled_progress_min_delta := 0.01
@export var stalled_progress_penalty := -10.0
## Optional penalty applied on every step spent without a meaningful new best progress. It begins
## after no_progress_penalty_grace_steps and stops whenever progress advances or reaches the
## near-goal exemption. Zero preserves the original terminal-only behavior.
@export_range(-10.0, 0.0, 0.001, "or_less") var no_progress_step_penalty := 0.0
@export_range(0, 10000, 1) var no_progress_penalty_grace_steps := 0
## When the agent is at/near the target (progress at or above this), holding position is the goal
## -- not a stall -- so the stall timer keeps resetting and never fires. Default 1.0 keeps the old
## behavior (only a perfectly-maxed progress is exempt). Lower it (e.g. 0.85) for reach+hold tasks
## where the arm should stay on target without being penalized for "no progress".
@export_range(0.0, 1.0, 0.01) var ignore_stall_above_progress := 1.0

var _best_progress := {}
var _last_progress_step := {}
var _progress_stalled := {}
var _penalty_applied := {}
var _last_terms := {}


func reset_rewards() -> void:
	_best_progress.clear()
	_last_progress_step.clear()
	_progress_stalled.clear()
	_penalty_applied.clear()
	_last_terms.clear()


func reset_agent(agent_id:String, context:Dictionary = {}) -> void:
	var progress := float(context.get("progress", context.get("track_progress", 0.0)))
	var step := int(context.get("step", 0))
	_best_progress[agent_id] = progress
	_last_progress_step[agent_id] = step
	_progress_stalled[agent_id] = false
	_penalty_applied[agent_id] = false
	_last_terms[agent_id] = {}


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", ""))
	var progress := float(context.get("progress", context.get("track_progress", 0.0)))
	var step := int(context.get("step", 0))

	_update_progress_stall(agent_id, progress, step)
	var last_progress_step := int(_last_progress_step.get(agent_id, step))
	var no_progress_steps := maxi(step - last_progress_step, 0)
	var step_penalty := 0.0
	if (
		progress < ignore_stall_above_progress
		and no_progress_steps >= no_progress_penalty_grace_steps
	):
		step_penalty = no_progress_step_penalty * weight

	var terminal_penalty := 0.0
	if (
		bool(_progress_stalled.get(agent_id, false))
		and not bool(_penalty_applied.get(agent_id, false))
	):
		_penalty_applied[agent_id] = true
		terminal_penalty = stalled_progress_penalty * weight

	_last_terms[agent_id] = {
		"progress_stalled": bool(_progress_stalled.get(agent_id, false)),
		"best_progress": float(_best_progress.get(agent_id, progress)),
		"no_progress_steps": no_progress_steps,
		"step_penalty": step_penalty,
		"terminal_penalty": terminal_penalty,
	}
	return step_penalty + terminal_penalty


func is_agent_stalled(agent_id:String) -> bool:
	if not terminate_on_stalled_progress:
		return false
	return bool(_progress_stalled.get(agent_id, false))


func get_terminal_reason(agent_id:String) -> String:
	return "progress_stalled" if is_agent_stalled(agent_id) else ""


func get_agent_terms(agent_id:String) -> Dictionary:
	var terms := Dictionary(_last_terms.get(agent_id, {})).duplicate(true)
	terms.merge({
		"progress_stalled": bool(_progress_stalled.get(agent_id, false)),
		"best_progress": float(_best_progress.get(agent_id, 0.0))
	}, true)
	return terms


func _update_progress_stall(agent_id:String, progress:float, step:int) -> void:
	if bool(_progress_stalled.get(agent_id, false)):
		return

	if not _best_progress.has(agent_id):
		_best_progress[agent_id] = progress
		_last_progress_step[agent_id] = step
		return

	# At/near the target, holding is the objective: keep the stall timer alive so staying put is
	# never flagged as a stall (the far-from-target stall guard below still applies otherwise).
	if progress >= ignore_stall_above_progress:
		_last_progress_step[agent_id] = step
		if progress > float(_best_progress.get(agent_id, progress)):
			_best_progress[agent_id] = progress
		return

	var best := float(_best_progress.get(agent_id, progress))
	if progress > best + stalled_progress_min_delta:
		_best_progress[agent_id] = progress
		_last_progress_step[agent_id] = step
		return

	var last_progress_step := int(_last_progress_step.get(agent_id, step))
	if step - last_progress_step >= max(1, stalled_progress_window_steps):
		_progress_stalled[agent_id] = true
