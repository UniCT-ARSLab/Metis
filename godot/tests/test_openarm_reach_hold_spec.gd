extends SceneTree
## M2 spec/regression test for the stable reach-and-hold scene
## (openarm_reach_hold_scenario.tscn). Locks the base task contract:
##   - obs_dim 27, action_size 7, declared observation order, finite obs/actions;
##   - 7-joint order + finite URDF limits; TCP/ToolPose resolved;
##   - initial success thresholds (distance / orientation / speed / hold) present and sane;
##   - continue_after_success == false (hold terminates, contact alone does not);
##   - success is emitted ONLY after the full hold (not before), EXACTLY at hold completion,
##     with terminated=true, truncated=false, terminal_reason "target_reached";
##   - reset clears _success_frames, terminal state and previous action;
##   - the target stays static for the whole episode.

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
const EXPECTED_JOINTS := [
	"openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
	"openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
	"openarm_right_joint7",
]


func _all_finite(values: Array) -> bool:
	for v in values:
		if not is_finite(float(v)):
			return false
	return true


func _initialize() -> void:
	var failures: Array[String] = []
	var packed := load(SCENE) as PackedScene
	if packed == null:
		push_error("reach-hold spec: scene not found")
		quit(1)
		return
	var scenario := packed.instantiate() as Node3D
	var bridge := scenario.get_node_or_null("BridgeServer")
	if bridge:
		scenario.remove_child(bridge)
		bridge.free()
	root.add_child(scenario)
	await process_frame
	await physics_frame

	var arm = scenario.get_node("OpenarmAgent")
	var controller = scenario.get_node("ScenarioController")
	var robot = arm.get_node("openarm")
	var target: Node3D = scenario.get_node("Target")
	var tool_pose: Node3D = arm.get_node("EndEffector/ToolPose")

	# --- Contract: dims + order ---
	if arm.agent.get_action_size() != 7:
		failures.append("action_size=%d" % arm.agent.get_action_size())
	if arm.agent.get_observation_size() != 27:
		failures.append("obs_size=%d" % arm.agent.get_observation_size())
	if Array(arm.get_controlled_joint_names()) != EXPECTED_JOINTS:
		failures.append("joint_order=%s" % str(arm.get_controlled_joint_names()))

	# --- Contract: finite URDF joint limits, lower < upper ---
	for jname in EXPECTED_JOINTS:
		var jnode = robot.get_joint_node(jname)
		if jnode == null or jnode.joint == null or jnode.joint.limit == null:
			failures.append("joint_limit_missing_%s" % jname)
			continue
		var lo := float(jnode.joint.limit.lower)
		var hi := float(jnode.joint.limit.upper)
		if not is_finite(lo) or not is_finite(hi) or lo >= hi:
			failures.append("joint_limit_bad_%s(%f,%f)" % [jname, lo, hi])

	# --- Contract: TCP / ToolPose resolved ---
	if robot.get_link_node(arm.tcp_link_name) == null:
		failures.append("tcp_link_unresolved_%s" % arm.tcp_link_name)
	if tool_pose == null or not is_finite(tool_pose.global_position.x):
		failures.append("tool_pose_unresolved")

	# --- Contract: hold terminates, contact alone does not ---
	if controller.continue_after_success:
		failures.append("continue_after_success_true")
	if not arm.terminate_on_success:
		failures.append("terminate_on_success_false")

	# --- Configure the base contract (Easy, level 0) + read initial thresholds ---
	scenario.call("_on_scenario_configured",
		{"training_episode": 0, "training_mode": true, "curriculum_level": 0.0})
	scenario.call("_apply_adaptive_pose_curriculum")
	scenario.call("_apply_per_region_gate", "easy")  # discrete stages set distance/angle per region
	var thr_dist := float(arm.success_distance)
	var thr_ang := float(arm.success_angle_degrees)
	var thr_hold := int(arm.success_hold_physics_frames)
	var thr_speed := float(arm.success_max_joint_speed)
	# M2.1: Stage A gate == 4 cm / 45 deg / 20 frames / 0.30 rad/s at level 0.
	if not is_equal_approx(thr_dist, 0.04):
		failures.append("stage_a_distance=%.3f" % thr_dist)
	if not is_equal_approx(thr_ang, 45.0):
		failures.append("stage_a_angle=%.1f" % thr_ang)
	if thr_hold != 20:
		failures.append("stage_a_hold=%d" % thr_hold)
	if not is_equal_approx(thr_speed, 0.30):
		failures.append("stage_a_speed=%.2f" % thr_speed)
	if not bool(arm.success_require_still):
		failures.append("success_require_still_false")

	# M2.1: bootstrap + reverse fully disabled; max 300 decision steps.
	if scenario.reverse_curriculum_enabled:
		failures.append("reverse_enabled")
	if bool(scenario.call("_has_bootstrap_pose")):
		failures.append("bootstrap_enabled")
	if int(controller.max_steps) != 300:
		failures.append("max_steps=%d" % int(controller.max_steps))

	# M2.1: every reset is a regular HOME start (no bootstrap/reverse), target in Easy.
	for s in range(8):
		var rr: Dictionary = await controller.reset_episode(4500 + s)
		var rinfo: Dictionary = _channel(rr).get("info", {}).get("reset", {})
		if str(rinfo.get("task_reset_mode", "")) != "regular":
			failures.append("reset_mode_%d=%s" % [s, str(rinfo.get("task_reset_mode", ""))])
		var rregion := str(rinfo.get("target_region", ""))
		if rregion != "" and rregion != "easy":
			failures.append("reset_region_%d=%s" % [s, rregion])

	# --- Static target + no continuous tracking in the base contract ---
	if bool(scenario.continuous_target_sampling):
		failures.append("continuous_sampling_enabled")
	if bool(scenario.relocate_target_during_training):
		failures.append("relocate_enabled")
	if not is_zero_approx(float(scenario.target_yaw_randomization_degrees)):
		failures.append("yaw_randomization_enabled")

	# --- Observations finite (the 5 declared sources sum to 27) ---
	await controller.reset_episode(4100)
	await physics_frame
	var obs_parts := []
	obs_parts.append_array(arm.get_joint_position_observation())
	obs_parts.append_array(arm.get_joint_velocity_observation())
	var te = arm.get_target_error_observation(); obs_parts.append_array([te.x, te.y, te.z])
	var oe = arm.get_target_orientation_error_observation(); obs_parts.append_array([oe.x, oe.y, oe.z])
	obs_parts.append_array(arm.get_previous_action_observation())
	if obs_parts.size() != 27:
		failures.append("obs_compose=%d" % obs_parts.size())
	if not _all_finite(obs_parts):
		failures.append("obs_not_finite")

	# --- Success timing: hold ON target, from wherever reset placed the arm ---
	await controller.reset_episode(4200)
	await physics_frame
	# Put the target exactly on the current ToolPose so distance=0, orientation=0, still.
	target.global_transform = tool_pose.global_transform
	if arm.has_method("notify_target_pose_relocated"):
		arm.notify_target_pose_relocated()
	await physics_frame
	if arm.get_target_distance() > thr_dist:
		failures.append("on_target_setup_dist=%.3f" % arm.get_target_distance())
	var hold_needed := int(arm.success_hold_physics_frames)
	var pfps := int(controller.physics_frames_per_step)
	var zero_action: Array[float] = []
	zero_action.resize(7); zero_action.fill(0.0)
	var saved_target := target.global_transform
	var terminated := false
	var early_success := false
	var static_broken := false
	var last_terminal_reason := ""
	var last_truncated := true
	for step_i in range(1, hold_needed + 8):
		# success must NOT be flagged while fewer than hold_needed on-target frames elapsed
		if arm._success_frames < hold_needed and arm.has_succeeded():
			early_success = true
		var result: Dictionary = await controller.step({str(arm.name): zero_action})
		var ch := _channel(result)
		if target.global_transform != saved_target:
			static_broken = true
		if bool(ch.get("terminated", false)):
			terminated = true
			var info: Dictionary = ch.get("info", {})
			last_terminal_reason = str(info.get("terminal_reason", ""))
			last_truncated = bool(ch.get("truncated", true))
			# success must fire ONLY once the full hold is satisfied
			if arm._success_frames < hold_needed:
				failures.append("terminated_before_hold_%d<%d" % [arm._success_frames, hold_needed])
			break
	if early_success:
		failures.append("success_before_hold")
	if not terminated:
		failures.append("no_success_after_hold")
	if not arm.has_succeeded():
		failures.append("not_succeeded_flag")
	if last_terminal_reason != "target_reached":
		failures.append("terminal_reason='%s'" % last_terminal_reason)
	if last_truncated:
		failures.append("truncated_true")
	if static_broken:
		failures.append("target_moved_during_episode")

	# --- Reset clears _success_frames + terminal state ---
	# The arm is succeeded+terminal from the hold above; reset must clear it (checked directly,
	# isolated from reset_episode's settle frames which can legitimately re-accumulate on a new
	# nearby target).
	if not (arm.has_succeeded() and arm.is_terminal()):
		failures.append("precondition_not_terminal_before_reset")
	arm.reset_all(null, false)
	if arm._success_frames != 0:
		failures.append("success_frames_not_reset=%d" % arm._success_frames)
	if arm.is_terminal() or arm.has_succeeded() or arm.has_collided():
		failures.append("terminal_state_not_reset")

	# --- Reset clears previous action (apply a non-zero command, then reset -> zeros) ---
	await controller.reset_episode(4400)
	arm.apply_action([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2])
	await physics_frame
	arm.reset_all(null, false)
	if not _all_finite(Array(arm.get_previous_action_observation())):
		failures.append("prev_action_not_finite")
	for v in arm.get_previous_action_observation():
		if not is_zero_approx(float(v)):
			failures.append("prev_action_not_reset")
			break

	if failures.is_empty():
		print("test_openarm_reach_hold_spec: PASS (hold=%d frames, %d phys/step)" % [hold_needed, pfps])
		scenario.free()
		quit(0)
	else:
		for f in failures:
			push_error("FAIL: " + f)
		print("test_openarm_reach_hold_spec: FAIL (%d) %s" % [failures.size(), str(failures)])
		scenario.free()
		quit(1)


func _channel(result: Dictionary) -> Dictionary:
	# The per-agent channel carries info.terminal_reason; the top-level info only has "step".
	if result.has("reward"):
		return result
	var agents: Array = result.get("agents", [])
	if agents.is_empty() or typeof(agents[0]) != TYPE_DICTIONARY:
		return {}
	return agents[0]
