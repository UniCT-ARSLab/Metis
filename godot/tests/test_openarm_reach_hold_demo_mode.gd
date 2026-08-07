extends SceneTree
## M5 spec test for demo-generation mode on the stable reach-and-hold scene.
## Locks the IK-demo sampling contract used by the recorder:
##   - demo_mode forces the target region to the configured one (easy);
##   - demo_forced_cell pins the target to EXACTLY that allowed cell every reset;
##   - the reset info exposes a deterministic target_pose_id = "region:cell:seed";
##   - the same (cell, seed) reproduces the same pose_id, different seeds differ
##     (so train/val split by pose_id never shares a trajectory);
##   - the forced target actually lands inside the region volume.

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
const EASY_CELLS := [17, 18, 22, 25, 28, 29, 30, 31, 33, 34, 41]


func _reset_info(controller, seed: int) -> Dictionary:
	var rr: Dictionary = await controller.reset_episode(seed)
	return _channel(rr).get("info", {}).get("reset", {})


func _configure(scenario, cell: int) -> void:
	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
		"curriculum_level": 0.0,
		"demo_mode": true,
		"demo_forced_region": "easy",
		"demo_forced_cell": cell,
	})


func _initialize() -> void:
	var failures: Array[String] = []
	var packed := load(SCENE) as PackedScene
	if packed == null:
		push_error("demo-mode: scene not found")
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
	var easy_region = scenario.get_node("EasyArea")

	# 1) Force cell 17: region == easy, cell == 17, pose_id well-formed, target inside the region.
	_configure(scenario, 17)
	var info17: Dictionary = await _reset_info(controller, 900)
	if str(info17.get("target_region", "")) != "easy":
		failures.append("region=%s (want easy)" % str(info17.get("target_region", "")))
	if int(info17.get("target_cell", -1)) != 17:
		failures.append("cell=%d (want 17)" % int(info17.get("target_cell", -1)))
	var pid17 := str(info17.get("target_pose_id", ""))
	if not pid17.begins_with("easy:17:"):
		failures.append("pose_id='%s' (want easy:17:*)" % pid17)
	var tgt: Node3D = scenario.get_node("Target")
	if not easy_region.contains_global_position(tgt.global_position):
		failures.append("forced target not inside easy region")

	# 2) Determinism: same (cell, seed) -> identical pose_id.
	var info17b: Dictionary = await _reset_info(controller, 900)
	if str(info17b.get("target_pose_id", "")) != pid17:
		failures.append("pose_id not deterministic: '%s' != '%s'" % [str(info17b.get("target_pose_id", "")), pid17])

	# 3) Different seed on the same cell -> different pose_id (distinct pose; enables train/val split).
	var info17c: Dictionary = await _reset_info(controller, 901)
	if str(info17c.get("target_pose_id", "")) == pid17:
		failures.append("pose_id collided across seeds")

	# 4) Force a different allowed cell (41) -> exactly that cell.
	_configure(scenario, 41)
	var info41: Dictionary = await _reset_info(controller, 900)
	if int(info41.get("target_cell", -1)) != 41:
		failures.append("cell=%d (want 41)" % int(info41.get("target_cell", -1)))
	if not str(info41.get("target_pose_id", "")).begins_with("easy:41:"):
		failures.append("pose_id41='%s'" % str(info41.get("target_pose_id", "")))

	# 5) Every configured Easy cell is honoured exactly (coverage over the whole allowlist).
	for cell in EASY_CELLS:
		_configure(scenario, cell)
		var ci: Dictionary = await _reset_info(controller, 1000 + cell)
		if int(ci.get("target_cell", -1)) != cell:
			failures.append("coverage: cell %d -> %d" % [cell, int(ci.get("target_cell", -1))])

	# 6) Plan-follower fidelity: the emitted joint_velocity action is finite, in [-1,1], recorded
	# EXACTLY as applied (== previous_action), and it must NOT latch manual_control (else success
	# would never terminate the demo). A non-trivial waypoint must command real motion.
	_configure(scenario, 17)
	await controller.reset_episode(2222)
	var robot = arm.get_node("openarm")
	var jnames = arm.get_controlled_joint_names()
	var waypoint: Array = []
	for jn in jnames:
		waypoint.append(float(robot.get_joint_position(jn)) + 0.3)  # offset -> should command motion
	arm.set_demo_plan([waypoint], int(controller.physics_frames_per_step))
	var applied: Array = arm.apply_manual_action()
	var prev: Array = Array(arm.get_previous_action_observation())
	if applied.size() != 7:
		failures.append("plan-follower action size %d" % applied.size())
	var any_motion := false
	for i in range(applied.size()):
		var a := float(applied[i])
		if not is_finite(a) or a < -1.0001 or a > 1.0001:
			failures.append("plan action[%d]=%f out of [-1,1]" % [i, a])
		if i < prev.size() and absf(a - float(prev[i])) > 1e-6:
			failures.append("plan action[%d] recorded!=applied (%f vs %f)" % [i, float(prev[i]), a])
		if absf(a) > 0.01:
			any_motion = true
	if not any_motion:
		failures.append("plan-follower commanded no motion for an offset waypoint")
	if arm.manual_control:
		failures.append("plan-follower latched manual_control (would block success termination)")
	arm.set_demo_plan([], int(controller.physics_frames_per_step))  # clear

	if failures.is_empty():
		print("test_openarm_reach_hold_demo_mode: PASS (%d easy cells honoured)" % EASY_CELLS.size())
		scenario.free()
		quit(0)
	else:
		for f in failures:
			push_error("FAIL: " + f)
		print("test_openarm_reach_hold_demo_mode: FAIL (%d) %s" % [failures.size(), str(failures)])
		scenario.free()
		quit(1)


func _channel(result: Dictionary) -> Dictionary:
	if result.has("reward"):
		return result
	var agents: Array = result.get("agents", [])
	if agents.is_empty() or typeof(agents[0]) != TYPE_DICTIONARY:
		return {}
	return agents[0]
