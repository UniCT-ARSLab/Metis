extends SceneTree
## M5 SEPARATE, GATED joint-space demo planner (offline; never in the RL path).
##
## For each (Easy cell, seed) it: reproduces the scenario's exact target for that (cell, seed),
## generates MULTIPLE frame-correct IK goal configs (position IK to ~mm from home + alt seeds,
## each refined to the task GraspPose orientation), runs a goal-biased RRT with FULL collision
## checking (self+env, teleport-interpolated local planner) toward the goal SET, then validated
## shortcut-smoothing. The resulting collision-free JOINT waypoint path (home -> goal) is written
## to JSON keyed by pose_id ("easy:cell:seed") for the Python generator to execute closed-loop
## as joint_velocity actions. Planning is seeded by the pose seed -> reproducible per pose_id.
##
## It does NOT touch task, reward or stall detection. Collision config = the scene's as-shipped
## runtime (self+env ON). Reports planning failures per pose.
##
##   godot --headless --path godot --script res://tools/openarm_ik_demo_planner.gd -- \
##       --out python/demos/openarm_reach_hold_m5/plans.json --seeds-per-cell 3 --seed-base 1000

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
const EASY_CELLS := [17, 18, 22, 25, 28, 29, 30, 31, 33, 34, 41]
const PLAN_TICKS := 60
const PLAN_SPEED := 2.0
const DEADZONE := 0.002
const DAMPING := 0.05
const ACC_TOL := 0.01           # goal "reached" tolerance (1cm) -> well inside the 4cm Stage-A gate
const SEG_RES := 0.06           # rad per collision sub-step along an edge
const RRT_STEP := 0.5           # rad max extension per RRT step
const RRT_MAX_ITER := 900
const RRT_RESTARTS := 1         # re-seeded RRT attempts before declaring rrt_no_path
const GOAL_BIAS := 0.2
const HARD_CELLS := []          # provisioning of hard cells is driven externally (parallel jobs)
const HARD_MULT := 1
const SMOOTH_ITERS := 50

var _scenario: Node3D
var _arm
var _robot
var _controller
var _scen_target: Node3D
var _grasp: Node3D
var _ik: URDFIKController
var _ori_target: Node3D
var _joint_names
var _b_off := Basis.IDENTITY
var _lo: Array = []
var _hi: Array = []
var _n := 7

var _out_path := "python/demos/openarm_reach_hold_m5/plans.json"
var _seeds_per_cell := 3
var _seed_base := 1000
var _cells: Array = EASY_CELLS
# Cache of successfully-solved goals {pos, cfg}: a new pose warm-starts its position IK from the
# nearest already-solved neighbour, which recovers cells whose straight home reach self-collides.
var _solved_goals: Array = []


var _in_path := ""

func _parse_args() -> void:
	var a := OS.get_cmdline_user_args()
	var i := 0
	while i < a.size():
		match a[i]:
			"--out": _out_path = str(a[i + 1]); i += 1
			"--in": _in_path = str(a[i + 1]); i += 1
		i += 1


func _initialize() -> void:
	# Offline planning does thousands of teleport-collision checks, each gated on a physics frame.
	# A headless SceneTree ticks physics at realtime (60/s) by default -> ~16ms/check. Crank the
	# tick rate so `await physics_frame` returns in ~1ms; collision checks are pure geometry so a
	# smaller dt does not change their result. (IK convergence is frame-counted, unaffected.)
	Engine.physics_ticks_per_second = PLAN_TICKS
	Engine.max_physics_steps_per_frame = 16
	_parse_args()
	var packed: PackedScene = load(SCENE)
	_scenario = packed.instantiate()
	var bridge = _scenario.get_node_or_null("BridgeServer")
	if bridge:
		bridge.queue_free()
	root.add_child(_scenario)
	await process_frame
	await physics_frame

	_arm = _scenario.get_node("OpenarmAgent")
	_robot = _arm.get_node("openarm")
	_controller = _scenario.get_node("ScenarioController")
	_scen_target = _scenario.get_node("Target")
	_grasp = _scenario.get_node("Target/GraspPose")
	_joint_names = _arm.get_controlled_joint_names()
	_n = _joint_names.size()
	for jn in _joint_names:
		var jnode = _robot.get_joint_node(jn)
		_lo.append(float(min(jnode.joint.limit.lower, jnode.joint.limit.upper)))
		_hi.append(float(max(jnode.joint.limit.lower, jnode.joint.limit.upper)))

	_ik = URDFIKController.new()
	_ik.name = "PlannerIK"
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

	# warm-start goal cache: M5 planner goals (last waypoint per pose) -> _solved_goals for the
	# nearby-goal fallback on the hard cells (25/31) whose cold home-IK fails.
	for cache_path in ["python/demos/openarm_reach_hold_m5/plan_train_full.json",
			"python/demos/openarm_reach_hold_m5/plan_val_full.json"]:
		var cp: String = (ProjectSettings.globalize_path("res://") + "../" + cache_path).simplify_path()
		if FileAccess.file_exists(cp):
			var cd: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(cp))
			for pid in cd.get("plans", {}):
				var pl: Dictionary = cd["plans"][pid]
				var wps: Array = pl.get("waypoints", [])
				var tp: Array = pl.get("target_pos", [])
				if wps.size() >= 1 and tp.size() == 3:
					_solved_goals.append({"pos": Vector3(tp[0], tp[1], tp[2]),
						"cfg": wps[wps.size() - 1]})
	print("[probe] loaded %d M5 cached goals" % _solved_goals.size())

	# M6.2A.1: warm-start IK from the clone's deviation config + REAL joint_velocity execution.
	var in_abs := _in_path if _in_path.begins_with("/") else (
		ProjectSettings.globalize_path("res://") + "../" + _in_path).simplify_path()
	var input: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(in_abs))
	var probe_poses: Array = input.get("poses", [])
	print("[probe] %d deviation states; warm-start + REAL execution" % probe_poses.size())

	var results := {}
	var n_reach := 0
	var n_unreach := 0
	for entry_v in probe_poses:
		var entry: Dictionary = entry_v
		var cell: int = int(entry["cell"])
		var seed: int = int(entry["seed"])
		var pose_id := str(entry.get("pose_id", "easy:%d:%d" % [cell, seed]))
		var start: Array = _denorm_joints(entry["norm_joints"])
		var in_limits := _within_limits(start)
		var placed: Dictionary = await _place_target(cell, seed)
		if not placed["ok"]:
			results[pose_id] = {"reachable": false, "reason": "target_place"}
			n_unreach += 1
			continue
		_robot.reset_joint_positions(_array_to_dict(start))
		await physics_frame
		var start_collides: bool = _arm.has_collided()
		# IK goal WARM-STARTED from the deviation config; fallback = nearby M5 cached goal
		var goals: Array = []
		await _collect_goal(placed["pos"], _array_to_dict(start), goals)
		if goals.is_empty():
			goals = await _generate_goals(placed["pos"])
		var ik_solved := not goals.is_empty()
		var path_ok := false
		if ik_solved:
			var goal_cfgs: Array = []
			for g in goals:
				goal_cfgs.append(g["cfg"])
			var rng := RandomNumberGenerator.new()
			rng.seed = seed
			var rrt: Dictionary = await _rrt(start, goal_cfgs, rng)
			path_ok = bool(rrt["ok"])
		# REAL recovery: drive from the deviation config with joint_velocity actions (NO teleport)
		var ex: Dictionary = await _execute_recovery(start, placed["pos"])
		var recovered: bool = bool(ex["executed"]) and not bool(ex["collided"])
		results[pose_id] = {
			"cell": cell, "seed": seed, "in_limits": in_limits, "start_collides": start_collides,
			"ik_solved": ik_solved, "path_planned": path_ok, "executed_successfully": recovered,
			"dist_final": ex["dist_final"], "dist_min": ex["dist_min"], "hold_max": ex["hold_max"],
			"exec_collided": bool(ex["collided"]), "reachable": recovered,
		}
		if recovered:
			n_reach += 1
		else:
			n_unreach += 1
		print("[probe] %s ik=%s path=%s executed=%s dist_final=%.4f hold=%d" % [
			pose_id, ik_solved, path_ok, recovered, float(ex["dist_final"]), int(ex["hold_max"])])

	var payload := {
		"schema": "openarm-recoverability-probe/1", "scene": SCENE,
		"n_probed": probe_poses.size(), "n_reachable": n_reach, "n_unreachable": n_unreach,
		"results": results,
	}
	# Absolute path -> use as-is; relative -> resolve against the repo root (res:// == <repo>/godot/).
	var abs_path := _out_path if _out_path.begins_with("/") else (
		ProjectSettings.globalize_path("res://") + "../" + _out_path)
	abs_path = abs_path.simplify_path()
	DirAccess.make_dir_recursive_absolute(abs_path.get_base_dir())
	var f := FileAccess.open(abs_path, FileAccess.WRITE)
	f.store_string(JSON.stringify(payload, "  "))
	f.close()
	print("[probe] wrote %s reachable=%d unreachable=%d" % [abs_path, n_reach, n_unreach])
	quit(0)


# --- target reproduction (exact scenario sampling for this cell+seed) --------------------
func _place_target(cell: int, seed: int) -> Dictionary:
	_scenario.call("_on_scenario_configured", {
		"training_episode": 0, "training_mode": true, "curriculum_level": 0.0,
		"demo_mode": true, "demo_forced_region": "easy", "demo_forced_cell": cell})
	var rr: Dictionary = await _controller.reset_episode(seed)
	var info: Dictionary = _channel(rr).get("info", {}).get("reset", {})
	var got_cell := int(info.get("target_cell", -1))
	if got_cell != cell:
		return {"ok": false, "got_cell": got_cell}
	return {"ok": true, "pos": _grasp.global_position, "basis": _grasp.global_transform.basis}


# --- planning ----------------------------------------------------------------------------
func _plan_pose(pos: Vector3, _basis: Basis, seed: int) -> Dictionary:
	var goals: Array = await _generate_goals(pos)
	if goals.is_empty():
		return {"ok": false, "fail_reason": "no_ik_goal"}
	var goal_meta: Dictionary = goals[0]["meta"]
	var goal_cfgs: Array = []
	for g in goals:
		goal_cfgs.append(g["cfg"])

	# RRT restarts on failure with a varied RNG (only hard poses that miss the first tree pay for
	# it). Reproducible: restart k reseeds deterministically from the pose seed.
	var home := _dict_to_array(_home_positions())
	var rrt := {"ok": false}
	for attempt in range(RRT_RESTARTS):
		var rng := RandomNumberGenerator.new()
		rng.seed = seed + attempt * 7919
		rrt = await _rrt(home, goal_cfgs, rng)
		if rrt["ok"]:
			break
	if not rrt["ok"]:
		return {"ok": false, "fail_reason": "rrt_no_path", "n_goals": goals.size()}
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var raw: Array = rrt["path"]
	var smoothed: Array = await _smooth(raw, rng)
	return {
		"ok": true, "waypoints": smoothed, "raw_len": raw.size(),
		"n_goals": goals.size(),
		"goal_cfg": smoothed[smoothed.size() - 1],
		"goal_pos_err": snappedf(float(goal_meta["pos_err"]), 0.0001),
		"goal_ori_deg": snappedf(float(goal_meta["ori_deg"]), 0.01),
	}


## Recoverability variant of _plan_pose: RRT from an ARBITRARY start config (the clone's deviation
## state) instead of home. Same IK-goal generation + RRT + collision checking.
func _plan_from(start: Array, pos: Vector3, seed: int) -> Dictionary:
	var goals: Array = await _generate_goals(pos)
	if goals.is_empty():
		return {"ok": false, "fail_reason": "no_ik_goal"}
	var goal_meta: Dictionary = goals[0]["meta"]
	var goal_cfgs: Array = []
	for g in goals:
		goal_cfgs.append(g["cfg"])
	var rrt := {"ok": false}
	for attempt in range(RRT_RESTARTS):
		var rng := RandomNumberGenerator.new()
		rng.seed = seed + attempt * 7919
		rrt = await _rrt(start, goal_cfgs, rng)
		if rrt["ok"]:
			break
	if not rrt["ok"]:
		return {"ok": false, "fail_reason": "rrt_no_path", "n_goals": goals.size()}
	return {"ok": true, "n_goals": goals.size(),
		"goal_pos_err": snappedf(float(goal_meta["pos_err"]), 0.0001)}


## Inverse of the obs[0:7] normalization (remap(pos, lo, hi, -1, 1)) using the live joint limits.
func _denorm_joints(nj: Array) -> Array:
	var out: Array = []
	for i in range(_n):
		var v := float(nj[i])
		out.append(_lo[i] + (v + 1.0) * 0.5 * (_hi[i] - _lo[i]))
	return out


func _within_limits(cfg: Array) -> bool:
	for i in range(_n):
		if float(cfg[i]) < _lo[i] - 1e-4 or float(cfg[i]) > _hi[i] + 1e-4:
			return false
	return true


## REAL recovery from the deviation config: closed-loop IK drives normalized joint-velocity actions
## (no teleport) toward the live target; success = the scenario's own target_reached-after-hold, and
## we separately log dist_min/final + the longest 4cm-still hold streak.
func _execute_recovery(start: Array, _pos: Vector3) -> Dictionary:
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(_array_to_dict(start))
	_arm.set_training_active(true)
	_ik.set_target(_grasp)
	_ik.track_orientation = false
	await physics_frame
	var collided := false
	var succeeded := false
	var hold := 0
	var hold_max := 0
	var dist_min := 9.0
	var dist_final := 9.0
	for _s in range(240):
		_ik.solve(true)
		await physics_frame
		if _arm.has_collided():
			collided = true
			break
		var d: float = _arm.get_target_distance()
		dist_final = d
		dist_min = minf(dist_min, d)
		if _arm.has_succeeded():
			succeeded = true
			break
		if d <= 0.04 and _arm.get_max_joint_speed() < 0.3:
			hold += 1
			hold_max = maxi(hold_max, hold)
		else:
			hold = 0
	return {"executed": succeeded, "collided": collided, "dist_final": dist_final,
		"dist_min": dist_min, "hold_max": hold_max}


## Up to k joint-config seeds from already-solved goals whose TCP target is within `radius` of pos,
## nearest first. These warm-start the position IK for hard poses (home reach self-collides).
func _nearby_seeds(pos: Vector3, k: int, radius: float) -> Array:
	var scored: Array = []
	for e in _solved_goals:
		var d: float = (e["pos"] as Vector3).distance_to(pos)
		if d <= radius:
			scored.append({"d": d, "cfg": e["cfg"]})
	scored.sort_custom(func(a, b): return float(a["d"]) < float(b["d"]))
	var out: Array = []
	for i in range(min(k, scored.size())):
		out.append(_array_to_dict(scored[i]["cfg"]))
	return out


## Frame-correct goal configs. PRIMARY: home + alt seeds (stops at 2 distinct goals -> reliable
## cells pay for ~2 IK solves, not 6). FALLBACK: nearby solved neighbours, only when the primary
## starts found too few (the hard cells whose straight home reach self-collides).
func _generate_goals(pos: Vector3) -> Array:
	var out: Array = []
	var primary: Array = [_home_positions()]
	for s in range(3):
		primary.append(_seed_positions(s))
	for st in primary:
		if out.size() >= 2:
			break
		await _collect_goal(pos, st, out)
	if out.size() < 2:
		for nb in _nearby_seeds(pos, 3, 0.22):
			if out.size() >= 2:
				break
			await _collect_goal(pos, nb, out)
	return out


func _collect_goal(pos: Vector3, start_positions: Dictionary, out: Array) -> void:
	var g: Dictionary = await _plan_goal(pos, start_positions)
	if not g["reached"]:
		return
	var cfg: Array = g["joints"]
	for e in out:
		if _config_dist(e["cfg"], cfg) < 0.05:
			return
	out.append({"cfg": cfg, "meta": {"pos_err": g["pos_err"], "ori_deg": g["ori_deg"]}})


func _plan_goal(pos: Vector3, start_positions: Dictionary) -> Dictionary:
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(start_positions)
	_arm.set_training_active(true)
	_ik.set_target(_grasp)
	_ik.track_orientation = false
	await physics_frame
	var collided := false
	for _s in range(240):
		_ik.solve(true)
		await physics_frame
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
		"pos_err": _arm.get_target_distance(), "ori_deg": ori_deg,
		"joints": _current_joint_array(),
	}


# --- goal-biased RRT with full collision checking + connect-to-goal ----------------------
func _rrt(start: Array, goals: Array, rng: RandomNumberGenerator) -> Dictionary:
	var cfg: Array = [start]
	var par: Array = [-1]
	# immediate straight shot(s) to a goal
	for gi in range(goals.size()):
		if await _segment_free(start, goals[gi]):
			return {"ok": true, "path": [start, goals[gi]]}
	for _it in range(RRT_MAX_ITER):
		var q_rand: Array = _sample_config(goals, rng)
		var near_i: int = _nearest(cfg, q_rand)
		var q_new: Array = _steer(cfg[near_i], q_rand)
		if not await _segment_free(cfg[near_i], q_new):
			continue
		cfg.append(q_new)
		par.append(near_i)
		var new_i: int = cfg.size() - 1
		# try to connect the new node straight to the nearest goal
		var gi2: int = _nearest(goals, q_new)
		if _config_dist(q_new, goals[gi2]) <= 2.0 and await _segment_free(q_new, goals[gi2]):
			var path: Array = _trace(cfg, par, new_i)
			path.append(goals[gi2])
			return {"ok": true, "path": path}
	return {"ok": false}


func _sample_config(goals: Array, rng: RandomNumberGenerator) -> Array:
	if rng.randf() < GOAL_BIAS and not goals.is_empty():
		return goals[rng.randi_range(0, goals.size() - 1)]
	var q: Array = []
	for i in range(_n):
		q.append(rng.randf_range(_lo[i], _hi[i]))
	return q


func _steer(from_q: Array, to_q: Array) -> Array:
	var d := _config_dist(from_q, to_q)
	if d <= RRT_STEP:
		return to_q.duplicate()
	var out: Array = []
	var t := RRT_STEP / d
	for i in range(_n):
		out.append(float(from_q[i]) + t * (float(to_q[i]) - float(from_q[i])))
	return out


func _nearest(nodes: Array, q: Array) -> int:
	var best := 0
	var best_d := 1e18
	for i in range(nodes.size()):
		var d := _config_dist(nodes[i], q)
		if d < best_d:
			best_d = d
			best = i
	return best


func _trace(cfg: Array, par: Array, index: int) -> Array:
	var path: Array = []
	var i := index
	while i >= 0:
		path.push_front(cfg[i])
		i = int(par[i])
	return path


func _config_dist(a: Array, b: Array) -> float:
	var s := 0.0
	for i in range(_n):
		var d := float(a[i]) - float(b[i])
		s += d * d
	return sqrt(s)


## Validated shortcut smoothing: replace sub-paths with a direct collision-free segment.
func _smooth(path: Array, rng: RandomNumberGenerator) -> Array:
	var p: Array = path.duplicate()
	for _it in range(SMOOTH_ITERS):
		if p.size() <= 2:
			break
		var i := rng.randi_range(0, p.size() - 2)
		var j := rng.randi_range(i + 1, p.size() - 1)
		if j - i < 2:
			continue
		if await _segment_free(p[i], p[j]):
			var np: Array = []
			for k in range(i + 1):
				np.append(p[k])
			for k in range(j, p.size()):
				np.append(p[k])
			p = np
	return p


## Collision-free straight JOINT segment? Teleport-interpolated (reliable), self+env collision.
func _segment_free(from_q: Array, to_q: Array) -> bool:
	var d := _config_dist(from_q, to_q)
	var steps: int = maxi(2, int(ceil(d / SEG_RES)))
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(_array_to_dict(from_q))
	_arm.set_training_active(true)
	await physics_frame
	if _arm.has_collided():
		return false
	for s in range(1, steps + 1):
		var t := float(s) / float(steps)
		var interp := {}
		for i in range(_n):
			interp[_joint_names[i]] = lerpf(float(from_q[i]), float(to_q[i]), t)
		_robot.reset_joint_positions(interp)
		await physics_frame
		if _arm.has_collided():
			return false
	return true


# --- helpers (ported from the probe/diag) ------------------------------------------------
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


func _home_positions() -> Dictionary:
	if _arm.has_method("_build_home_positions"):
		return _arm._build_home_positions(true)
	var d := {}
	for nm in _joint_names:
		d[nm] = 0.0
	return d


func _seed_positions(index: int) -> Dictionary:
	var base := _home_positions()
	var perturb := [0.25, -0.25, 0.4]
	var mag: float = perturb[index % perturb.size()]
	var nudge := {0: mag, 1: -mag, 3: mag}
	var d := {}
	var j := 0
	for nm in _joint_names:
		d[nm] = float(base.get(nm, 0.0)) + float(nudge.get(j, 0.0))
		j += 1
	return d


func _current_joint_array() -> Array:
	var out := []
	for nm in _joint_names:
		out.append(_robot.get_joint_position(nm))
	return out


func _dict_to_array(d: Dictionary) -> Array:
	var out := []
	for nm in _joint_names:
		out.append(float(d.get(nm, 0.0)))
	return out


func _array_to_dict(a: Array) -> Dictionary:
	var d := {}
	for i in range(_n):
		d[_joint_names[i]] = float(a[i]) if i < a.size() else 0.0
	return d


func _channel(result: Dictionary) -> Dictionary:
	if result.has("reward"):
		return result
	var agents: Array = result.get("agents", [])
	if agents.is_empty() or typeof(agents[0]) != TYPE_DICTIONARY:
		return {}
	return agents[0]
