extends SceneTree
## M3 reward-contract tests for the minimal reach-hold reward on the stable scene.
## Covers: complete configuration (exact values + removed components), progress-delta sign,
## distance-potential RETREAT penalised >= approach (1.25x anti-pump), no free dense reward while
## parked far, orientation cannot create progress while far (POSITION_GATED), hold-gate reset,
## env-collision -25 once, self-collision -35 once, neither double-penalised by TerminalFailure,
## stall = -10 via progress_stalled, single +50 success, one time penalty per decision step, an
## approach->return cycle nets <= 0, no false first-step delta, dense terms never outweigh +50.

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"

var scenario: Node3D
var arm
var controller
var robot
var target: Node3D
var tool_pose: Node3D
var failures: Array[String] = []
var _zero: Array[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


func _eq(a: float, b: float, tag: String) -> void:
	if not is_equal_approx(a, b):
		failures.append("%s=%.4f!=%.4f" % [tag, a, b])


func _terms(result: Dictionary) -> Dictionary:
	var ch := _channel(result)
	var info: Dictionary = ch.get("info", {})
	var out := {}
	for k in info.get("scenario_terms", {}).get("component_values", {}):
		out[k] = float(info["scenario_terms"]["component_values"][k])
	for k in info.get("local_term_rewards", {}):
		out[k] = float(info["local_term_rewards"][k])
	out["_reward"] = float(ch.get("reward", 0.0))
	out["_terminated"] = bool(ch.get("terminated", false))
	out["_reason"] = str(info.get("terminal_reason", ""))
	return out


func _step(action: Array) -> Dictionary:
	return _terms(await controller.step({str(arm.name): action}))


func _place_target_at_tool() -> void:
	target.global_transform = tool_pose.global_transform
	arm.notify_target_pose_relocated()


func _place_target_far() -> void:
	target.global_transform = Transform3D(Basis.IDENTITY, tool_pose.global_position + Vector3(0.3, 0.0, 0.0))
	arm.notify_target_pose_relocated()


func _rs(path: String):
	return scenario.get_node_or_null("ScenarioController/ScenarioRewardSystem/" + path)


func _verify_config() -> void:
	var prog = _rs("Progress")
	_eq(float(prog.progress_reward_scale), 10.0, "progress_fwd")
	_eq(float(prog.backward_penalty_scale), 10.0, "progress_back")
	var prec = _rs("NearTargetPrecision")
	_eq(float(prec.approach_reward_scale), 5.0, "potential_scale")
	_eq(float(prec.potential_distance), 0.06, "potential_dist")
	_eq(float(prec.retreat_penalty_multiplier), 1.25, "potential_retreat")
	_eq(float(_rs("GoalReward").reward), 50.0, "goal")
	_eq(float(_rs("CollisionPenalty").reward), -25.0, "collision_cfg")
	_eq(float(_rs("SelfCollisionPenalty").reward), -35.0, "self_collision_cfg")
	var tf = _rs("TerminalFailure")
	_eq(float(tf.base_failure_penalty), -10.0, "tf_base")
	_eq(float(tf.remaining_progress_penalty), 0.0, "tf_remaining")
	if Array(tf.failure_terminal_reasons) != ["progress_stalled"]:
		failures.append("tf_reasons=%s" % str(tf.failure_terminal_reasons))
	if bool(tf.penalize_truncation) or bool(tf.penalize_unclassified_terminal):
		failures.append("tf_penalize_flags_true")
	var noprog = _rs("NoProgress")
	_eq(float(noprog.stalled_progress_penalty), 0.0, "noprog_stall_pen")
	_eq(float(noprog.no_progress_step_penalty), 0.0, "noprog_step_pen")
	if not bool(noprog.terminate_on_stalled_progress):
		failures.append("noprog_not_terminating")
	# agent reward: only joint_limit(0.01) + smoothness(-0.001) + time(-0.001) remain
	var jl = arm.get_node_or_null("Agent/RewardSystem/JointLimit")
	_eq(float(jl.weight), 0.01, "joint_limit_w")
	_eq(float(arm.get_node("Agent/RewardSystem/Smoothness").penalty_scale), -0.001, "smoothness")
	_eq(float(arm.get_node("Agent/RewardSystem/Time").penalty), -0.001, "time_cfg")
	for removed in ["JointMotion", "HoldStillness", "NearTargetSpeed", "PoseTracking", "HoldProgress", "NearTargetDistancePenalty"]:
		if arm.get_node_or_null("Agent/RewardSystem/" + removed) != null:
			failures.append("agent_component_present_%s" % removed)
	for removed in ["DistanceApproach", "Proximity"]:
		if _rs(removed) != null:
			failures.append("scenario_component_present_%s" % removed)


func _initialize() -> void:
	var packed := load(SCENE) as PackedScene
	scenario = packed.instantiate() as Node3D
	var bridge := scenario.get_node_or_null("BridgeServer")
	if bridge:
		scenario.remove_child(bridge)
		bridge.free()
	root.add_child(scenario)
	await process_frame
	await physics_frame
	arm = scenario.get_node("OpenarmAgent")
	controller = scenario.get_node("ScenarioController")
	robot = arm.get_node("openarm")
	target = scenario.get_node("Target")
	tool_pose = arm.get_node("EndEffector/ToolPose")
	scenario.call("_on_scenario_configured", {"training_episode": 0, "training_mode": true, "curriculum_level": 0.0})

	_verify_config()

	# orientation weight is applied by the curriculum on reset -> 0.10 at Stage A (level 0).
	await controller.reset_episode(6100)
	_eq(float(arm.orientation_progress_weight), 0.10, "orientation_weight")

	# 9) first step after reset: no false progress/potential delta.
	var first := await _step(_zero)
	if absf(first.get("target_progress", 0.0)) > 1e-4 or absf(first.get("target_distance_precision", 0.0)) > 1e-4:
		failures.append("first_step_false_delta")

	# 1+2) toward -> progress>0; away -> progress<0; potential RETREAT penalised >= approach.
	await controller.reset_episode(6200)
	_place_target_at_tool()
	await _step(_zero)
	var away := await _step([0.3, 0, 0, 0, 0, 0, 0])
	var toward := await _step([-0.3, 0, 0, 0, 0, 0, 0])
	if away.get("target_progress", 0.0) >= 0.0:
		failures.append("away_progress>=0")
	if toward.get("target_progress", 0.0) <= 0.0:
		failures.append("toward_progress<=0")
	# retreat multiplier 1.25: |away potential| must be >= |toward potential|.
	if absf(away.get("target_distance_precision", 0.0)) < absf(toward.get("target_distance_precision", 0.0)) - 1e-3:
		failures.append("retreat_not_penalised_more(%.3f<%.3f)" % [absf(away.get("target_distance_precision", 0.0)), absf(toward.get("target_distance_precision", 0.0))])
	# 8) approach->return cycle nets <= 0.
	if float(away.get("_reward", 0.0)) + float(toward.get("_reward", 0.0)) > 1e-3:
		failures.append("cycle_reward>0")

	# 3) parked far + still: no positive dense accumulation.
	await controller.reset_episode(6300)
	_place_target_far()
	await _step(_zero)
	var parked := 0.0
	for _i in range(8):
		var t := await _step(_zero)
		parked += maxf(t.get("target_progress", 0.0), 0.0) + maxf(t.get("target_distance_precision", 0.0), 0.0)
	if parked > 1e-3:
		failures.append("parked_far_positive=%.4f" % parked)

	# 11-orientation) rotating the wrist while far cannot SUBSTITUTE positional approach
	# (POSITION_GATED): any incidental TCP shift stays far below a real approach delta (~0.048).
	var rot := await _step([0, 0, 0, 0, 0.3, 0.3, 0.3])
	if rot.get("target_progress", 0.0) > 0.02:
		failures.append("orientation_far_created_progress=%.4f" % rot.get("target_progress", 0.0))

	# 10) one time penalty per decision step (not x3 for 3 physics frames).
	_eq(float((await _step(_zero)).get("time", 0.0)), -0.001, "time_per_step")

	# 5) entering then leaving the success gate resets hold.
	await controller.reset_episode(6350)
	_place_target_at_tool()
	for _i in range(4):
		await _step(_zero)
	if arm._success_frames <= 0:
		failures.append("hold_not_accumulated")
	_place_target_far()
	await _step(_zero)
	if arm._success_frames != 0:
		failures.append("hold_not_reset=%d" % arm._success_frames)

	# 12) dense budget over a real return-to-target trajectory stays well under +50.
	await controller.reset_episode(6360)
	_place_target_at_tool()
	await _step(_zero)
	for _i in range(3):
		await _step([0.3, 0, 0, 0, 0, 0, 0])  # leave
	var dense := 0.0
	for _i in range(12):
		var t := await _step([-0.3, 0, 0, 0, 0, 0, 0])  # approach back
		dense += maxf(t.get("target_progress", 0.0), 0.0) + maxf(t.get("target_distance_precision", 0.0), 0.0)
		if t.get("_terminated", false):
			break
	if dense >= 50.0:
		failures.append("dense_budget>=50=%.1f" % dense)

	# 6+11) env collision -25 once, terminal, NO TerminalFailure double-hit.
	await controller.reset_episode(6400)
	arm.report_obstacle_collision()
	var col := await _step(_zero)
	_eq(col.get("collision", 0.0), -25.0, "env_collision")
	if not is_zero_approx(col.get("terminal_failure", 0.0)):
		failures.append("tf_on_env_collision=%.1f" % col.get("terminal_failure", 0.0))
	if not col.get("_terminated", false):
		failures.append("env_collision_not_terminal")
	if not is_zero_approx((await _step(_zero)).get("_reward", 0.0)):
		failures.append("env_collision_twice")

	# self-collision -35 once, terminal, NO TerminalFailure double-hit.
	await controller.reset_episode(6450)
	scenario.self_collision_event.trigger(str(arm.name))
	var sc := await _step(_zero)
	_eq(sc.get("self_collision", 0.0), -35.0, "self_collision")
	if not is_zero_approx(sc.get("terminal_failure", 0.0)):
		failures.append("tf_on_self_collision=%.1f" % sc.get("terminal_failure", 0.0))
	if not sc.get("_terminated", false) or sc.get("_reason", "") != "self_collision":
		failures.append("self_collision_not_terminal(%s)" % sc.get("_reason", ""))

	# stall = -10 via progress_stalled (short window so the test stays fast).
	var noprog = _rs("NoProgress")
	noprog.stalled_progress_window_steps = 4
	noprog.no_progress_penalty_grace_steps = 0
	await controller.reset_episode(6500)
	_place_target_far()
	var stall_tf := 0.0
	var stall_reason := ""
	for _i in range(12):
		var t := await _step(_zero)
		if t.get("_terminated", false):
			stall_tf = t.get("terminal_failure", 0.0)
			stall_reason = t.get("_reason", "")
			break
	_eq(stall_tf, -10.0, "stall_penalty")
	if stall_reason != "progress_stalled":
		failures.append("stall_reason=%s" % stall_reason)

	# 7) success bonus +50 exactly once.
	noprog.stalled_progress_window_steps = 120
	await controller.reset_episode(6600)
	_place_target_at_tool()
	var goal_val := 0.0
	var succeeded := false
	for _i in range(int(arm.success_hold_physics_frames) + 6):
		var t := await _step(_zero)
		if t.get("_terminated", false):
			goal_val = t.get("GoalReward", 0.0)
			succeeded = t.get("_reason", "") == "target_reached"
			break
	_eq(goal_val, 50.0, "success_bonus")
	if not succeeded:
		failures.append("no_success")
	if float((await _step(_zero)).get("_reward", 0.0)) > 1.0:
		failures.append("reward_after_terminal")

	if failures.is_empty():
		print("test_openarm_reach_hold_reward: PASS")
		scenario.free()
		quit(0)
	else:
		for f in failures:
			push_error("FAIL: " + f)
		print("test_openarm_reach_hold_reward: FAIL (%d) %s" % [failures.size(), str(failures)])
		scenario.free()
		quit(1)


func _channel(result: Dictionary) -> Dictionary:
	if result.has("reward"):
		return result
	var agents: Array = result.get("agents", [])
	if agents.is_empty() or typeof(agents[0]) != TYPE_DICTIONARY:
		return {}
	return agents[0]
