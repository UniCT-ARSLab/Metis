extends SceneTree


func _initialize() -> void:
	var reward = load("res://addons/metis/runtime/agent/reward_components/ObservationMarginReward.gd").new()
	reward.observation_name = "position"
	reward.target_value = 10.0
	reward.half_width = 2.0
	reward.scale = 3.0
	var failures: Array[String] = []
	for sample in [[10.0, 0.0], [9.0, -0.75], [11.0, -0.75], [8.0, -3.0], [12.0, -3.0], [14.0, -12.0]]:
		var actual := float(reward.compute_reward({"observations": {"position": sample[0]}}))
		if not is_equal_approx(actual, float(sample[1])):
			failures.append("quadratic_%s" % sample[0])
	reward.shape = 0
	if not is_equal_approx(float(reward.compute_reward({"observations": {"position": 11.0}})), -1.5):
		failures.append("linear")
	reward.weight = 2.0
	if not is_equal_approx(float(reward.compute_reward({"observations": {"position": 11.0}})), -3.0):
		failures.append("weight")
	if not is_zero_approx(float(reward.compute_reward({"observations": {}}))):
		failures.append("missing")
	if not is_zero_approx(float(reward.compute_reward({"observations": {"position": NAN}}))):
		failures.append("nonfinite")
	reward.gate_observation = "active"
	reward.gate_min_value = 0.5
	reward.gate_max_value = 1.0
	if not is_zero_approx(float(reward.compute_reward({"observations": {"position": 11.0}}))):
		failures.append("missing_gate")
	if not is_zero_approx(float(reward.compute_reward({"observations": {"position": 11.0, "active": 0.0}}))):
		failures.append("closed_gate")
	if not is_equal_approx(float(reward.compute_reward({"observations": {"position": 11.0, "active": 1.0}})), -3.0):
		failures.append("open_gate")
	reward.half_width = 0.0
	if not is_zero_approx(float(reward.compute_reward({"observations": {"position": 11.0, "active": 1.0}}))):
		failures.append("invalid_width")
	reward.free()
	if not failures.is_empty():
		push_error("Observation margin reward test failed: %s" % [failures])
		quit(1)
		return
	print("Observation margin reward test passed")
	quit(0)
