extends SceneTree
## M5 Phase-1 diagnosis: is a PLANNER-DRIVEN frame-correct IK expert a viable Stage-A demo
## source on the stable reach-hold scene? For 2 Easy cells it (1) PLANS a goal joint config
## (position IK to ~2mm + frame-correct orientation refine, probe method), (2) validates the
## home->goal joint-space path is collision-free (else a via-point), then (3) EXECUTES from home
## by emitting proportional joint_velocity ACTIONS through the normal apply_action() path -- NO
## teleport, NO _ik.solve during execution -- and measures the real Stage-A gates
## (pos<=4cm, ori<=45deg, a >=20-frame hold window at speed<=0.30) + collisions + monotonic
## progress (so progress_stalled would not fire) + recorded-vs-applied action fidelity.
##
## Collision config is the scene's as-shipped runtime (self+env ON); nothing is forced.

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
const CELLS := [17, 18, 22, 25, 28, 29, 30, 31, 33, 34, 41]   # the 11 path-confirmed Easy cells
const ALT_SEEDS := 3
const PLAN_SPEED := 2.0           # rad/s IK cap for PLANNING only (fast convergence; not recorded)
const DEADZONE := 0.002
const DAMPING := 0.05
const ACC_TOL := 0.01             # planning "reached" tolerance (1cm) -> goal well inside the 4cm gate
const STAGE_A_DIST := 0.04
const STAGE_A_ANG := 45.0
const HOLD_FRAMES := 20
const HOLD_SPEED := 0.30

var _scenario: Node3D
var _arm
var _robot
var _scen_target: Node3D
var _grasp: Node3D
var _ik: URDFIKController
var _ori_target: Node3D
var _joint_names
var _b_off := Basis.IDENTITY
var _exec_max_speed := 0.5        # the AGENT's real action->velocity scale (max_joint_speed_override)
var _phys_per_step := 3


func _initialize() -> void:
	var packed: PackedScene = load(SCENE)
	_scenario = packed.instantiate()
	var bridge = _scenario.get_node_or_null("BridgeServer")
	if bridge:
		bridge.queue_free()
	var controller = _scenario.get_node_or_null("ScenarioController")
	if controller:
		controller.set_process(false)
		controller.set_physics_process(false)
		controller.set_process_input(false)
		_phys_per_step = int(controller.physics_frames_per_step)
	root.add_child(_scenario)
	await process_frame
	await physics_frame

	_arm = _scenario.get_node("OpenarmAgent")
	_robot = _arm.get_node("openarm")
	_scen_target = _scenario.get_node("Target")
	_grasp = _scenario.get_node("Target/GraspPose")
	_joint_names = _arm.get_controlled_joint_names()
	if "max_joint_speed_override" in _arm and float(_arm.max_joint_speed_override) > 0.0:
		_exec_max_speed = float(_arm.max_joint_speed_override)

	print("[diag] self_body_collision_enabled=%s check_links=%d" % [
		str(_arm.self_body_collision_enabled), Array(_arm.self_check_link_names).size()])
	print("[diag] exec_max_speed=%.2f phys_per_step=%d" % [_exec_max_speed, _phys_per_step])

	_ik = URDFIKController.new()
	_ik.name = "DiagIK"
	_ik.joint_names = _joint_names
	_ik.tcp_link_name = _arm.tcp_link_name
	_ik.tcp_local_offset = _arm.tcp_local_offset
	_ik.max_joint_speed = PLAN_SPEED
	_ik.track_orientation = false
	_ik.orientation_mode = "pointing"
	_ik.pointing_axis = Vector3(0, 1, 0)
	_ik.damping = DAMPING
	_ik.position_deadzone = DEADZONE
	_ik.active = false
	root.add_child(_ik)
	_ik.set_robot(_robot)

	_ori_target = Node3D.new()
	root.add_child(_ori_target)

	await _compute_frame_offset()

	var easy = _scenario.get_node("EasyArea")
	var reachable := 0
	var straight := 0
	var via_ok := 0
	var uncovered: Array = []
	for cell in CELLS:
		var pos: Vector3 = _cell_center_world(easy, cell)
		var r: Dictionary = await _scan_cell(cell, pos)
		print("[diag] cell=%d %s" % [cell, JSON.stringify(r)])
		if r["reached"]:
			reachable += 1
			if r["straight_clean"]:
				straight += 1
			elif r["via_clean"]:
				via_ok += 1
			else:
				uncovered.append(cell)
		else:
			uncovered.append(cell)
	print("[diag] SUMMARY reachable=%d/%d straight_joint_path=%d via_joint_path=%d uncovered=%s" % [
		reachable, CELLS.size(), straight, via_ok, str(uncovered)])
	quit(0)


# --- planning scan (may drive IK + teleport; NOT recorded) -------------------------------
## For one cell: can the frame-correct planner produce a Stage-A goal config, and is the
## home->goal path executable as collision-free JOINT-space straight segments (what the
## proportional velocity expert actually follows)? Reports straight vs via vs uncovered.
func _scan_cell(cell: int, pos: Vector3) -> Dictionary:
	_scen_target.global_transform = Transform3D(_grasp.global_transform.basis, pos)
	if _arm.has_method("notify_target_pose_relocated"):
		_arm.notify_target_pose_relocated()
	await physics_frame

	# 1) Goal config: try home, then alternate seeds until one reaches collision-free.
	var starts: Array = [_home_positions()]
	for s in range(ALT_SEEDS):
		starts.append(_seed_positions(s))
	var plan := {}
	var start_used := -1
	for i in range(starts.size()):
		plan = await _plan_goal(pos, starts[i])
		if plan["reached"]:
			start_used = i
			break
	if not plan.get("reached", false):
		return {"reached": false, "start": -1, "pos_err": plan.get("pos_err", -1.0),
			"straight_clean": false, "via_clean": false}
	var goal_joints: Array = plan["joints"]

	# 2) Executable path in JOINT space: straight home->goal, else a via whose BOTH joint
	# segments (home->via, via->goal) are collision-free (matches proportional execution).
	var home := _home_positions()
	var home_arr := _dict_to_array(home)
	var straight_clean: bool = await _check_segment(home_arr, goal_joints, 32)
	var via_clean := false
	if not straight_clean:
		var via: Dictionary = await _plan_via(pos)
		if via["ok"]:
			var vg: Array = via["goal_joints"]
			via_clean = (
				await _check_segment(home_arr, via["via_joints"], 24)
				and await _check_segment(via["via_joints"], vg, 24))
	return {
		"reached": true, "start": start_used,
		"pos_err": snappedf(float(plan["pos_err"]), 0.0001),
		"ori_deg": snappedf(float(plan["ori_deg"]), 0.01),
		"straight_clean": straight_clean, "via_clean": via_clean,
	}


func _plan_goal(pos: Vector3, start_positions: Dictionary) -> Dictionary:
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(start_positions)
	_arm.set_training_active(true)
	_ik.set_target(_grasp)
	_ik.track_orientation = false
	await physics_frame
	var best_pos := 1e9
	var collided := false
	for _s in range(240):
		_ik.solve(true)
		await physics_frame
		best_pos = minf(best_pos, _arm.get_target_distance())
		if _arm.has_collided():
			collided = true
			break
		if _arm.get_target_distance() <= DEADZONE + 0.001 and _arm.get_max_joint_speed() < 0.02:
			break
	var ori_deg := _orientation_error_deg()
	if not collided and _arm.get_target_distance() <= ACC_TOL:
		ori_deg = minf(ori_deg, await _refine_orientation(pos))
	return {
		"reached": (not collided) and _arm.get_target_distance() <= ACC_TOL,
		"pos_err": _arm.get_target_distance(),
		"ori_deg": ori_deg,
		"collided": collided,
		"joints": _current_joint_array(),
	}


func _plan_via(pos: Vector3) -> Dictionary:
	var seg_steps := 200
	for offset in [Vector3(0, 0.12, 0), Vector3(0.06, 0.12, 0.06)]:
		_arm.reset_all(null, false)
		_robot.reset_joint_positions(_home_positions())
		_arm.set_training_active(true)
		await physics_frame
		var s1: Dictionary = await _drive_ik_to(pos + offset, seg_steps)
		if s1["collided"] or not s1["reached"]:
			continue
		var via_joints: Array = _current_joint_array()
		var s2: Dictionary = await _drive_ik_to(pos, seg_steps)
		if s2["collided"] or _arm.get_target_distance() > ACC_TOL:
			continue
		return {"ok": true, "via_joints": via_joints, "goal_joints": _current_joint_array()}
	return {"ok": false}


func _drive_ik_to(point: Vector3, steps: int) -> Dictionary:
	_ori_target.global_transform = Transform3D(Basis.IDENTITY, point)
	_ik.set_target(_ori_target)
	_ik.track_orientation = false
	var collided := false
	var reached := false
	for _s in range(steps):
		_ik.solve(true)
		await physics_frame
		if _arm.has_collided():
			collided = true
			break
		if _tool_pos().distance_to(point) <= ACC_TOL:
			reached = true
			break
	_ik.set_target(_grasp)
	return {"reached": reached, "collided": collided}


func _check_joint_path(goal_joints: Array, steps: int) -> bool:
	if goal_joints.is_empty():
		return false
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(_home_positions())
	_arm.set_training_active(true)
	await physics_frame
	var home := _home_positions()
	for s in range(1, steps + 1):
		var t := float(s) / float(steps)
		var interp := {}
		var i := 0
		for n in _joint_names:
			interp[n] = lerpf(float(home.get(n, 0.0)), float(goal_joints[i]), t)
			i += 1
		_robot.reset_joint_positions(interp)
		await physics_frame
		if _arm.has_collided():
			return false
	return true


# --- execution: proportional joint_velocity actions through apply_action (RECORDED path) ---
func _execute_via_actions(waypoints: Array) -> Dictionary:
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(_home_positions())
	_arm.set_training_active(true)
	await physics_frame

	var dt_step := float(_phys_per_step) / 60.0
	var min_pos := 1e9
	var best_ori := 1e9
	var collided := false
	var max_fidelity_diff := 0.0
	var prev_pos: float = _arm.get_target_distance()
	var monotone_violations := 0
	var hold_run := 0
	var hold_ok := false
	var reached_pos := false
	var reached_ori := false
	var wp_index := 0
	var total_steps := 100

	for _step in range(total_steps):
		var goal: Array = waypoints[wp_index]
		# proportional joint_velocity command: full speed toward the waypoint, easing near it.
		var action: Array[float] = []
		action.resize(_joint_names.size())
		var max_joint_gap := 0.0
		for i in range(_joint_names.size()):
			var q: float = _robot.get_joint_position(_joint_names[i])
			var gap: float = float(goal[i]) - q
			max_joint_gap = maxf(max_joint_gap, absf(gap))
			action[i] = clampf(gap / (_exec_max_speed * dt_step), -1.0, 1.0)
		# advance to the next waypoint once close enough to the current one
		if max_joint_gap < 0.03 and wp_index < waypoints.size() - 1:
			wp_index += 1

		var applied = _arm.apply_action(action)
		for i in range(min(applied.size(), action.size())):
			max_fidelity_diff = maxf(max_fidelity_diff, absf(float(applied[i]) - float(action[i])))
		for _f in range(_phys_per_step):
			await physics_frame

		var d: float = _arm.get_target_distance()
		var ori := _orientation_error_deg()
		var spd: float = _arm.get_max_joint_speed()
		min_pos = minf(min_pos, d)
		if d <= STAGE_A_DIST:
			best_ori = minf(best_ori, ori)
		if _arm.has_collided():
			collided = true
			break
		if d > prev_pos + 0.005:
			monotone_violations += 1
		prev_pos = d
		# hold accounting: contiguous decision-steps fully inside the gate + still
		if d <= STAGE_A_DIST and ori <= STAGE_A_ANG and spd <= HOLD_SPEED:
			reached_pos = true
			reached_ori = true
			hold_run += _phys_per_step
			if hold_run >= HOLD_FRAMES:
				hold_ok = true
		else:
			hold_run = 0
		if d <= STAGE_A_DIST:
			reached_pos = true

	return {
		"min_pos": snappedf(min_pos, 0.0001),
		"best_ori_in_gate": snappedf(best_ori if best_ori < 1e8 else -1.0, 0.01),
		"reached_pos": reached_pos,
		"reached_ori": reached_ori,
		"hold_ok": hold_ok,
		"collided": collided,
		"monotone_ok": monotone_violations <= 3,
		"monotone_violations": monotone_violations,
		"fidelity_ok": max_fidelity_diff <= 1e-5,
		"max_fidelity_diff": snappedf(max_fidelity_diff, 0.000001),
	}


# --- helpers (ported from the probe) ------------------------------------------------------
func _compute_frame_offset() -> void:
	_arm.reset_all(null, false)
	_arm.set_training_active(true)
	await physics_frame
	var tcp: Node3D = _robot.get_link_node(_arm.tcp_link_name)
	var tool: Node3D = _arm.tool_pose if "tool_pose" in _arm and _arm.tool_pose else null
	if tcp != null and tool != null:
		_b_off = tcp.global_transform.basis.orthonormalized().inverse() * tool.global_transform.basis.orthonormalized()


func _refine_orientation(target_pos: Vector3) -> float:
	_ori_target.global_transform = Transform3D(_grasp.global_transform.basis * _b_off.inverse(), target_pos)
	_ik.set_target(_ori_target)
	_ik.track_orientation = true
	_ik.orientation_mode = "full"
	var best := _orientation_error_deg()
	var stall := 0
	for _s in range(90):
		_ik.solve(true)
		await physics_frame
		if _arm.has_collided():
			break
		if _arm.get_target_distance() <= ACC_TOL:
			var e := _orientation_error_deg()
			if e < best - 0.2:
				best = e
				stall = 0
			else:
				stall += 1
			if best <= 3.0 or stall >= 12:
				break
	_ik.track_orientation = false
	_ik.set_target(_grasp)
	return best


func _orientation_error_deg() -> float:
	if _arm.has_method("get_target_orientation_error_observation"):
		var v: Vector3 = _arm.get_target_orientation_error_observation()
		return rad_to_deg(v.length() * PI)
	return NAN


func _tool_pos() -> Vector3:
	var tool: Node3D = _arm.tool_pose if "tool_pose" in _arm and _arm.tool_pose else null
	return tool.global_position if tool else Vector3.INF


func _home_positions() -> Dictionary:
	if _arm.has_method("_build_home_positions"):
		return _arm._build_home_positions(true)
	var d := {}
	for n in _joint_names:
		d[n] = 0.0
	return d


func _current_joint_array() -> Array:
	var out := []
	for n in _joint_names:
		out.append(_robot.get_joint_position(n))
	return out


func _dict_to_array(d: Dictionary) -> Array:
	var out := []
	for n in _joint_names:
		out.append(float(d.get(n, 0.0)))
	return out


func _array_to_dict(a: Array) -> Dictionary:
	var d := {}
	var i := 0
	for n in _joint_names:
		d[n] = float(a[i]) if i < a.size() else 0.0
		i += 1
	return d


## Alternate IK seeds (probe method): home nudged on base-yaw/shoulder/elbow so a collision-free
## endpoint can be found when the straight home reach folds a link into the body.
func _seed_positions(index: int) -> Dictionary:
	var base := _home_positions()
	var perturb := [0.25, -0.25, 0.4]
	var mag: float = perturb[index % perturb.size()]
	var nudge := {0: mag, 1: -mag, 3: mag}
	var d := {}
	var j := 0
	for n in _joint_names:
		d[n] = float(base.get(n, 0.0)) + float(nudge.get(j, 0.0))
		j += 1
	return d


## Collision-check the straight JOINT-space interpolation between two configs (the path the
## proportional velocity expert actually executes). Teleport-based; planning only, not recorded.
func _check_segment(from_j: Array, to_j: Array, steps: int) -> bool:
	if from_j.is_empty() or to_j.is_empty():
		return false
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(_array_to_dict(from_j))
	_arm.set_training_active(true)
	await physics_frame
	for s in range(1, steps + 1):
		var t := float(s) / float(steps)
		var interp := {}
		var i := 0
		for n in _joint_names:
			interp[n] = lerpf(float(from_j[i]), float(to_j[i]), t)
			i += 1
		_robot.reset_joint_positions(interp)
		await physics_frame
		if _arm.has_collided():
			return false
	return true


func _cell_center_world(region: Node, cell: int) -> Vector3:
	var cs: CollisionShape3D = region.get_node_or_null(region.collision_shape_path)
	var box: BoxShape3D = cs.shape as BoxShape3D
	var grid: Vector3i = region.grid_size
	var half: Vector3 = box.size * 0.5
	var inset: Vector3 = region.sampling_inset
	var lmin: Vector3 = -half + inset
	var lmax: Vector3 = half - inset
	var usable: Vector3 = lmax - lmin
	var csz: Vector3 = Vector3(usable.x / grid.x, usable.y / grid.y, usable.z / grid.z)
	var x: int = cell % grid.x
	var yz: int = int(float(cell) / float(grid.x))
	var y: int = yz % grid.y
	var z: int = int(float(yz) / float(grid.y))
	var local: Vector3 = lmin + Vector3((x + 0.5) * csz.x, (y + 0.5) * csz.y, (z + 0.5) * csz.z)
	return cs.to_global(local)
