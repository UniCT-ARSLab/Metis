extends SceneTree
## M4 test: the reach-hold scene uses a DISCRETE 6-stage curriculum (A..F at levels
## 0.0/0.2/0.4/0.6/0.8/1.0). Verifies each stage's exact region weights, hold, joint-speed and
## per-region success gate; that the config depends only on the stage (not the episode number);
## that A-C sample Easy only, D-E Easy/Medium, F all three; that the reward orientation weight
## stays 0.10; and that bootstrap/jitter/tracking/relocation are always disabled.

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"

var scenario: Node3D
var arm
var failures: Array[String] = []


func _eq(a: float, b: float, tag: String) -> void:
	if not is_equal_approx(a, b):
		failures.append("%s=%.4f!=%.4f" % [tag, a, b])


func _config(level: float, episode: int) -> void:
	scenario.call("_on_scenario_configured",
		{"training_episode": episode, "training_mode": true, "curriculum_level": level})


func _check_stage(level: float, name: String, weights: Vector3, hold: int, speed: float, gates: Dictionary) -> void:
	_config(level, 0)
	scenario.call("_apply_adaptive_pose_curriculum")
	if arm.success_hold_physics_frames != hold:
		failures.append("%s_hold=%d!=%d" % [name, arm.success_hold_physics_frames, hold])
	_eq(float(arm.success_max_joint_speed), speed, "%s_speed" % name)
	var w: Vector3 = scenario.call("_adaptive_region_weights")
	if not w.is_equal_approx(weights):
		failures.append("%s_weights=%s!=%s" % [name, str(w), str(weights)])
	_eq(float(scenario.call("_curriculum_orientation_weight")), 0.10, "%s_orient" % name)
	for region in gates:
		scenario.call("_apply_per_region_gate", region)
		var g: Array = gates[region]
		_eq(float(arm.success_distance), float(g[0]), "%s_%s_dist" % [name, region])
		_eq(float(arm.success_angle_degrees), float(g[1]), "%s_%s_angle" % [name, region])


func _initialize() -> void:
	var packed := load(SCENE) as PackedScene
	scenario = packed.instantiate() as Node3D
	var bridge := scenario.get_node_or_null("BridgeServer")
	if bridge:
		scenario.remove_child(bridge)
		bridge.free()
	root.add_child(scenario)
	await process_frame

	arm = scenario.get_node("OpenarmAgent")
	if not bool(scenario.use_discrete_stage_curriculum):
		failures.append("discrete_curriculum_disabled")

	# Exact per-stage contract.
	_check_stage(0.0, "A", Vector3(1, 0, 0), 20, 0.30, {"easy": [0.04, 45.0]})
	_check_stage(0.2, "B", Vector3(1, 0, 0), 30, 0.25, {"easy": [0.03, 30.0]})
	_check_stage(0.4, "C", Vector3(1, 0, 0), 60, 0.15, {"easy": [0.02, 20.0]})
	_check_stage(0.6, "D", Vector3(0.5, 0.5, 0), 40, 0.20, {"easy": [0.02, 20.0], "medium": [0.04, 45.0]})
	_check_stage(0.8, "E", Vector3(0.5, 0.5, 0), 60, 0.15, {"easy": [0.02, 20.0], "medium": [0.03, 30.0]})
	_check_stage(1.0, "F", Vector3(0.2, 0.4, 0.4), 60, 0.15, {"easy": [0.02, 20.0], "medium": [0.03, 30.0], "hard": [0.04, 30.0]})

	# Config depends on the STAGE, not the episode number: same level, very different episodes.
	_config(0.4, 10)
	scenario.call("_apply_adaptive_pose_curriculum")
	var hold_ep10: int = arm.success_hold_physics_frames
	var speed_ep10 := float(arm.success_max_joint_speed)
	var w_ep10: Vector3 = scenario.call("_adaptive_region_weights")
	_config(0.4, 999999)
	scenario.call("_apply_adaptive_pose_curriculum")
	if arm.success_hold_physics_frames != hold_ep10 or not is_equal_approx(float(arm.success_max_joint_speed), speed_ep10):
		failures.append("stage_depends_on_episode")
	if not (scenario.call("_adaptive_region_weights") as Vector3).is_equal_approx(w_ep10):
		failures.append("weights_depend_on_episode")

	# Active regions: A-C -> Easy only; D-E -> Easy/Medium; F -> all three.
	for level in [0.0, 0.2, 0.4]:
		_config(level, 0)
		var w: Vector3 = scenario.call("_adaptive_region_weights")
		if w.y != 0.0 or w.z != 0.0:
			failures.append("stage_%.1f_not_easy_only=%s" % [level, str(w)])
	for level in [0.6, 0.8]:
		_config(level, 0)
		var w2: Vector3 = scenario.call("_adaptive_region_weights")
		if w2.x <= 0.0 or w2.y <= 0.0 or w2.z != 0.0:
			failures.append("stage_%.1f_not_easy_medium=%s" % [level, str(w2)])
	_config(1.0, 0)
	var wf: Vector3 = scenario.call("_adaptive_region_weights")
	if wf.x <= 0.0 or wf.y <= 0.0 or wf.z <= 0.0:
		failures.append("stage_F_not_all_three=%s" % str(wf))

	# Assist mechanisms must stay off in every stage.
	if scenario.reverse_curriculum_enabled:
		failures.append("reverse_enabled")
	if bool(scenario.call("_has_bootstrap_pose")):
		failures.append("bootstrap_enabled")
	if bool(scenario.continuous_pose_tracking) or bool(scenario.relocate_target_during_training):
		failures.append("tracking_or_relocation_enabled")
	if not is_zero_approx(float(scenario.target_yaw_randomization_degrees)):
		failures.append("yaw_randomization_enabled")
	if not is_zero_approx(float(scenario.intermediate_joint_jitter_degrees)) or not is_zero_approx(float(scenario.late_joint_jitter_degrees)):
		failures.append("joint_jitter_enabled")

	if failures.is_empty():
		print("test_openarm_reach_hold_stages: PASS (6 stages A-F)")
		scenario.free()
		quit(0)
	else:
		for f in failures:
			push_error("FAIL: " + f)
		print("test_openarm_reach_hold_stages: FAIL (%d) %s" % [failures.size(), str(failures)])
		scenario.free()
		quit(1)
