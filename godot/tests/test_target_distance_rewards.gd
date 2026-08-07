extends SceneTree


class DistanceProbe:
	extends Node

	var distance := 1.0

	func get_target_distance() -> float:
		return distance


func _initialize() -> void:
	var failures: Array[String] = []
	var probe := DistanceProbe.new()
	probe.name = "Probe"
	root.add_child(probe)

	var distance_reward = load(
		"res://addons/metis/runtime/agent/scenario_reward_components/TargetDistanceDeltaScenarioReward.gd"
	).new()
	root.add_child(distance_reward)
	distance_reward.approach_reward_scale = 20.0
	distance_reward.retreat_penalty_scale = 25.0
	distance_reward.reset_agent("Probe")

	if not is_zero_approx(float(distance_reward.compute_reward(
			probe, {"agent_id": "Probe"}))):
		failures.append("distance_initial_sample")
	probe.distance = 0.9
	if not is_equal_approx(float(distance_reward.compute_reward(
			probe, {"agent_id": "Probe"})), 2.0):
		failures.append("distance_approach")
	probe.distance = 1.0
	if not is_equal_approx(float(distance_reward.compute_reward(
			probe, {"agent_id": "Probe"})), -2.5):
		failures.append("distance_retreat")
	distance_reward.reset_agent("Probe")
	if not is_zero_approx(float(distance_reward.compute_reward(
			probe, {"agent_id": "Probe"}))):
		failures.append("distance_reset")

	var precision_reward = load(
		"res://addons/metis/runtime/agent/scenario_reward_components/TargetDistancePotentialScenarioReward.gd"
	).new()
	root.add_child(precision_reward)
	precision_reward.potential_distance = 0.06
	precision_reward.approach_reward_scale = 5.0
	precision_reward.retreat_penalty_multiplier = 1.25
	probe.distance = 0.12
	precision_reward.reset_agent("Probe")
	if not is_zero_approx(float(precision_reward.compute_reward(
			probe, {"agent_id": "Probe"}))):
		failures.append("precision_initial_sample")
	var far_potential := exp(-0.12 / 0.06)
	var near_potential := exp(-0.06 / 0.06)
	probe.distance = 0.06
	var approach_value := float(precision_reward.compute_reward(
		probe, {"agent_id": "Probe"}))
	if not is_equal_approx(
			approach_value, (near_potential - far_potential) * 5.0):
		failures.append("precision_approach")
	if approach_value <= 0.0:
		failures.append("precision_approach_sign")
	probe.distance = 0.12
	var retreat_value := float(precision_reward.compute_reward(
		probe, {"agent_id": "Probe"}))
	if not is_equal_approx(
			retreat_value, (far_potential - near_potential) * 5.0 * 1.25):
		failures.append("precision_retreat")
	if retreat_value >= -approach_value:
		failures.append("precision_retreat_multiplier")
	precision_reward.reset_agent("Probe")
	if not is_zero_approx(float(precision_reward.compute_reward(
			probe, {"agent_id": "Probe"}))):
		failures.append("precision_reset")

	var stall_reward = load(
		"res://addons/metis/runtime/agent/scenario_reward_components/ProgressStallScenarioReward.gd"
	).new()
	root.add_child(stall_reward)
	stall_reward.terminate_on_stalled_progress = true
	stall_reward.stalled_progress_window_steps = 3
	stall_reward.stalled_progress_penalty = -40.0
	stall_reward.no_progress_step_penalty = -0.05
	stall_reward.no_progress_penalty_grace_steps = 2
	stall_reward.reset_agent("Probe", {"progress": 0.2, "step": 0})
	if not is_zero_approx(float(stall_reward.compute_reward(
			probe, {"agent_id": "Probe", "progress": 0.2, "step": 1}))):
		failures.append("stall_grace")
	if not is_equal_approx(float(stall_reward.compute_reward(
			probe, {"agent_id": "Probe", "progress": 0.2, "step": 2})), -0.05):
		failures.append("stall_step_penalty")
	if not is_equal_approx(float(stall_reward.compute_reward(
			probe, {"agent_id": "Probe", "progress": 0.2, "step": 3})), -40.05):
		failures.append("stall_terminal_penalty")
	if not stall_reward.is_agent_stalled("Probe"):
		failures.append("stall_terminal_state")

	if not failures.is_empty():
		push_error("Target distance reward test failed: %s" % [failures])
		quit(1)
		return

	print("Target distance reward test passed")
	quit(0)
