extends SceneTree

const TerminalProgressReward = preload(
	"res://addons/metis/runtime/agent/scenario_reward_components/TerminalProgressScenarioReward.gd")


func _initialize() -> void:
	var component := TerminalProgressReward.new()
	component.base_failure_penalty = -50.0
	component.remaining_progress_penalty = -50.0
	component.failure_terminal_reasons = PackedStringArray(
		["collision", "progress_stalled"])
	var failures: Array[String] = []

	component.reset_agent("arm")
	var collision_reward := component.compute_reward(null, {
		"agent_id": "arm",
		"episode_done": true,
		"event_terminal_reason": "collision",
		"progress": 0.4,
	})
	if not is_equal_approx(collision_reward, -80.0):
		failures.append("collision_progress_penalty")
	if not is_zero_approx(component.compute_reward(null, {
		"agent_id": "arm",
		"episode_done": true,
		"event_terminal_reason": "collision",
		"progress": 0.4,
	})):
		failures.append("failure_penalty_repeated")

	component.reset_agent("arm")
	var near_stall_reward := component.compute_reward(null, {
		"agent_id": "arm",
		"episode_done": true,
		"scenario_terminal_reason": "progress_stalled",
		"progress": 0.9,
	})
	if not is_equal_approx(near_stall_reward, -55.0):
		failures.append("near_stall_progress_penalty")

	component.reset_agent("arm")
	if not is_equal_approx(component.compute_reward(null, {
		"agent_id": "arm",
		"episode_done": true,
		"event_terminal_reason": "collision",
		"agent_succeeded": true,
		"progress": 0.95,
	}), -52.5):
		failures.append("collision_after_success_not_penalized")

	component.reset_agent("arm")
	if not is_zero_approx(component.compute_reward(null, {
		"agent_id": "arm",
		"episode_done": true,
		"event_terminal_reason": "target_reached",
		"target_reached": true,
		"progress": 1.0,
	})):
		failures.append("success_penalized")

	component.reset_agent("arm")
	if not is_zero_approx(component.compute_reward(null, {
		"agent_id": "arm",
		"episode_done": true,
		"truncated": true,
		"agent_succeeded": true,
		"progress": 0.95,
	})):
		failures.append("latched_success_penalized")

	component.reset_agent("arm")
	if not is_equal_approx(component.compute_reward(null, {
		"agent_id": "arm",
		"episode_done": true,
		"truncated": true,
		"progress": 0.0,
	}), -100.0):
		failures.append("truncation_not_penalized")

	if not component.is_episode_end_only():
		failures.append("not_episode_end_only")

	if not failures.is_empty():
		push_error("Terminal progress reward test failed: %s" % failures)
		component.free()
		quit(1)
		return

	print("Terminal progress reward test passed")
	component.free()
	quit(0)
