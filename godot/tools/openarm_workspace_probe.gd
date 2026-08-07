extends Node3D
## Milestone 1 workspace probe for the OpenArm reach-and-hold task.
##
## Loads the real openarm_scenario (same URDF, IK controller, joint limits, collision
## geometry, table/support obstacles and self-collision contract), then for every cell of
## the Easy/Medium/Hard TargetSamplingRegion3D volumes it:
##   1. resets the arm to home,
##   2. drives the existing Jacobian/DLS IK from home toward the sampled target pose
##      (position + the fixed GraspPose orientation the task scores against),
##   3. checks collisions ALONG the interpolated home->target trajectory (not only at the
##      end) using the arm's own passive shape-query detection,
##   4. re-checks the settled endpoint pose,
##   5. records residual position/orientation, joint-limit margin and collision links,
##   6. retries from several deterministic IK seeds to tell a truly unreachable pose apart
##      from a home-trajectory failure.
##
## Self-collision is FORCED ON here (the training scene ships it disabled -> empty link
## lists) so the map surfaces poses that are physically invalid even though training never
## penalised them.
##
## Output: machine-readable JSON + a Markdown summary. Coloured cell markers are added to
## the tree (green valid / yellow ik-or-accuracy fail / red collision-or-limit) so a
## windowed run renders the map for screenshots.
##
## Run (headless):
##   godot --headless --path godot res://tools/openarm_workspace_probe.tscn -- \
##     --samples 20 --out ../.research/m1
## Render (for screenshots): drop --headless and add --render.

const SCENARIO_PATH := "res://scenarios/robotarms/openarm_scenario.tscn"
const SELF_BODY_LINKS := ["openarm_body_link0"]
const SELF_CHECK_LINKS := [
	"openarm_right_link3", "openarm_right_link4", "openarm_right_link5",
	"openarm_right_link6", "openarm_right_link7",
	"openarm_right_left_finger", "openarm_right_right_finger",
]

# Tunables (overridable via `-- --flag value`).
var samples_per_cell := 20
var ik_max_steps := 240          # physics steps allotted per IK solve; early-exits on convergence
var ik_speed := 2.0              # rad/s IK joint cap (probe drives faster than RL to converge in fewer frames)
var ik_damping := 0.04           # DLS lambda; lower = more precise near the target than the teleop default
var settle_steps := 12           # extra steps to let the endpoint settle before the final measurement
var alt_seeds := 3               # extra deterministic IK seeds when the home trajectory misses
var accuracy_tol := 0.02         # m; within this = "valid" reach (the M4 stage-C target)
var loose_tol := 0.06            # m; beyond this from every seed = ik_unreachable, else accuracy_failure
var limit_margin_tol := 0.02     # rad; below this to a joint limit = joint_limit failure
var holdout_stride := 5          # every Nth cell (by index) is a held-out validation cell
var train_pose_fraction := 0.8   # split of poses inside SEEN cells
var regions_to_probe := ["easy", "medium", "hard"]
var out_dir := "../.research/m1"
var scene_path := SCENARIO_PATH
var regions_config_path := ""
var render_mode := false
var verbose_poses := false
var diagnose_mode := false
var fast_mode := false            # skip orientation-refine + waypoint planner (position+greedy only)
var rng_seed := 10_000

var _scenario: Node = null
var _arm: Node = null
var _robot: Node = null
var _ik: URDFIKController = null
var _scen_target: Node3D = null   # the scenario's Target node, moved to each sampled pose
var _grasp: Node3D = null         # Target/GraspPose: the point get_target_distance() measures to
var _ori_target: Node3D = null    # separate IK target carrying the frame-corrected orientation basis
var _b_off := Basis.IDENTITY      # tcp_link.basis^-1 * ToolPose.basis (constant); ToolPose = tcp * B_off
var _regions := {}               # region_name -> TargetSamplingRegion3D
var _joint_names := PackedStringArray()
var _records: Array = []
var _marker_root: Node3D = null


func _ready() -> void:
	_parse_args()
	_build_world()
	if diagnose_mode:
		return   # _build_world already printed the config and quit
	if _arm == null:
		push_error("[probe] failed to resolve OpenarmAgent; aborting.")
		get_tree().quit(1)
		return
	if render_mode:
		_run_showcase()
	else:
		_run_probe()


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		var a := str(args[i])
		match a:
			"--samples": samples_per_cell = int(args[i + 1]); i += 1
			"--ik-steps": ik_max_steps = int(args[i + 1]); i += 1
			"--ik-speed": ik_speed = float(args[i + 1]); i += 1
			"--ik-damping": ik_damping = float(args[i + 1]); i += 1
			"--alt-seeds": alt_seeds = int(args[i + 1]); i += 1
			"--accuracy-tol": accuracy_tol = float(args[i + 1]); i += 1
			"--holdout-stride": holdout_stride = int(args[i + 1]); i += 1
			"--regions": regions_to_probe = str(args[i + 1]).split(","); i += 1
			"--out": out_dir = str(args[i + 1]); i += 1
			"--scene": scene_path = str(args[i + 1]); i += 1
			"--regions-config": regions_config_path = str(args[i + 1]); i += 1
			"--seed": rng_seed = int(args[i + 1]); i += 1
			"--render": render_mode = true
			"--verbose": verbose_poses = true
			"--diagnose": diagnose_mode = true
			"--fast": fast_mode = true
		i += 1


func _build_world() -> void:
	var packed: PackedScene = load(scene_path)
	_scenario = packed.instantiate()
	add_child(_scenario)

	# Neutralise the Python bridge + episode controller: the probe drives the arm itself.
	var bridge := _scenario.get_node_or_null("BridgeServer")
	if bridge:
		bridge.queue_free()
	var controller := _scenario.get_node_or_null("ScenarioController")
	if controller:
		controller.set_process(false)
		controller.set_physics_process(false)
		controller.set_process_input(false)
	var hud := _scenario.get_node_or_null("DebugHUD")
	if hud:
		hud.queue_free()

	_arm = _scenario.get_node_or_null("OpenarmAgent")
	if _arm == null:
		return
	_robot = _arm.get_node_or_null("openarm")
	_scen_target = _scenario.get_node_or_null("Target")
	_grasp = _scenario.get_node_or_null("Target/GraspPose")
	_regions = {
		"easy": _scenario.get_node_or_null("EasyArea"),
		"medium": _scenario.get_node_or_null("MediumArea"),
		"hard": _scenario.get_node_or_null("HardArea"),
	}
	_joint_names = _arm.get_controlled_joint_names()

	# Optional in-memory region relocation (never edits the scene file; safe alongside a
	# separate running eval process). JSON: {"medium":{"center":[x,y,z],"box":[x,y,z],"grid":[4,3,4]}}
	if regions_config_path != "":
		_apply_region_overrides(regions_config_path)

	# Record the arm's runtime self-collision config AS SHIPPED (before we override it), so we
	# can compare the training contract against the probe contract (M1.1 requirement).
	var shipped := {
		"self_body_collision_enabled": _arm.self_body_collision_enabled,
		"self_body_link_names": Array(_arm.self_body_link_names),
		"self_check_link_names": Array(_arm.self_check_link_names),
	}
	print("[probe] SHIPPED self-collision config (training runtime): %s" % JSON.stringify(shipped))
	print("[probe] PROBE self-collision config (forced): enabled=true body=%s check=%s" % [
		str(SELF_BODY_LINKS), str(SELF_CHECK_LINKS)])
	if diagnose_mode:
		var diag_path := ProjectSettings.globalize_path("res://") + out_dir + "/runtime_config.json"
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path("res://") + out_dir)
		var df := FileAccess.open(diag_path, FileAccess.WRITE)
		df.store_string(JSON.stringify({"shipped": shipped, "probe_forced": {
			"self_body_collision_enabled": true,
			"self_body_link_names": SELF_BODY_LINKS,
			"self_check_link_names": SELF_CHECK_LINKS}}, "  "))
		df.close()
		print("[probe] wrote %s ; diagnose-only, quitting." % diag_path)
		get_tree().quit(0)
		return

	# Force self-collision ON (training scene disables it).
	_arm.self_body_collision_enabled = true
	_arm.self_body_link_names = PackedStringArray(SELF_BODY_LINKS)
	_arm.self_check_link_names = PackedStringArray(SELF_CHECK_LINKS)
	if _arm.has_method("_cache_self_body_geometry"):
		_arm._cache_self_body_geometry()
	# Quiet the per-collision console spam (thousands of probe attempts collide by design).
	if "self_collision_debug" in _arm:
		_arm.self_collision_debug = false
	if "collision_debug" in _arm:
		_arm.collision_debug = false

	# Reuse the arm's exact TCP + joint contract; the IK aims at the scenario GraspPose so
	# get_target_distance() (ToolPose -> GraspPose) is the value we optimise and measure.
	_ik = URDFIKController.new()
	_ik.name = "ProbeIK"
	_ik.joint_names = _joint_names
	_ik.tcp_link_name = _arm.tcp_link_name
	_ik.tcp_local_offset = _arm.tcp_local_offset
	_ik.max_joint_speed = ik_speed
	# Position-only IK: the reachability map is a POSITION+collision question. The task's
	# orientation gate is deliberately loose (M0: 34 deg on successes) and the plan says not
	# to tighten it before the map exists, so we drive to position and RECORD the natural
	# orientation residual instead of fighting a position/orientation tradeoff that stalls IK.
	_ik.track_orientation = false
	_ik.orientation_mode = "pointing"
	_ik.pointing_axis = Vector3(0, 1, 0)
	_ik.damping = ik_damping
	_ik.position_deadzone = 0.002
	_ik.active = false                  # stepped manually
	add_child(_ik)
	_ik.set_robot(_robot)
	_ik.set_target(_grasp)

	_ori_target = Node3D.new()
	_ori_target.name = "ProbeOriTarget"
	add_child(_ori_target)

	_marker_root = Node3D.new()
	_marker_root.name = "ProbeMarkers"
	add_child(_marker_root)

	# Render mode: make the scenario's own camera active so we see the arm + workcell + lighting.
	if render_mode:
		var cam := _scenario.get_node_or_null("Environment/Camera3D") as Camera3D
		if cam:
			cam.current = true
		else:
			push_warning("[probe] no Environment/Camera3D found for render")


func _run_probe() -> void:
	await _compute_frame_offset()
	print("[probe] start | samples/cell=%d ik_steps=%d alt_seeds=%d regions=%s" % [
		samples_per_cell, ik_max_steps, alt_seeds, str(regions_to_probe)])
	for region_name in regions_to_probe:
		var region: Node = _regions.get(region_name)
		if region == null:
			print("[probe] region '%s' missing; skipped" % region_name)
			continue
		await _probe_region(region_name, region)
	_write_outputs()
	print("[probe] done | %d poses -> %s" % [_records.size(), out_dir])
	if not render_mode:
		get_tree().quit(0)


## Windowed visual pass: animate the IK from home to a few representative targets (a reachable
## Easy coverage-gap cell, a body-side Medium cell, a Hard cell), leave a TCP trail, show a status
## label with the colliding link pair, and save a PNG per stage. Holds the window at the end.
func _run_showcase() -> void:
	var label := _make_overlay()
	var showcase := [
		["easy", 17], ["medium", 16], ["hard", 57], ["hard", 20],
	]
	var shots_dir := ProjectSettings.globalize_path("res://") + out_dir + "/screens"
	DirAccess.make_dir_recursive_absolute(shots_dir)
	for entry in showcase:
		var region_name: String = entry[0]
		var cell: int = entry[1]
		var region: Node = _regions.get(region_name)
		if region == null:
			continue
		var pos := _cell_center_world(region, cell)
		var basis: Basis = _grasp.global_transform.basis
		_scen_target.global_transform = Transform3D(basis, pos)
		_place_target_marker(pos)
		_clear_trail()
		_arm.reset_all(null, false)
		_robot.reset_joint_positions(_home_positions())
		_arm.set_training_active(true)
		await get_tree().physics_frame
		label.text = "%s cell %d  |  target %.2v\nreaching..." % [region_name, cell, pos]
		await _screenshot(shots_dir, "%s_%d_start" % [region_name, cell])

		var collided := {}
		for step in range(ik_max_steps):
			_ik.solve(true)
			await get_tree().physics_frame
			if step % 5 == 0:
				_drop_trail_point()
			var d: float = _arm.get_target_distance()
			label.text = "%s cell %d  |  dist %.1f cm  ori %.0f deg" % [
				region_name, cell, d * 100.0, _orientation_error_deg()]
			if _arm.has_collided():
				collided = _arm.get_last_collision_details()
				break
			if d <= _ik.position_deadzone + 0.001 and _arm.get_max_joint_speed() < 0.02:
				break
		if collided.is_empty():
			label.text = "%s cell %d  |  REACHED %.1f cm (collision-free)" % [
				region_name, cell, _arm.get_target_distance() * 100.0]
			await _screenshot(shots_dir, "%s_%d_reached" % [region_name, cell])
		else:
			var src := str(collided.get("source", "?"))
			label.text = "%s cell %d  |  COLLISION [%s]\n%s <-> %s" % [
				region_name, cell, src,
				str(collided.get("checker_link", "")),
				str(collided.get("body_link", collided.get("collider", "")))]
			print("[showcase] %s/%d collision: %s" % [region_name, cell, JSON.stringify(collided)])
			await _screenshot(shots_dir, "%s_%d_collision" % [region_name, cell])
		for _h in range(90):     # ~1.5 s pause so the frame is watchable
			await get_tree().physics_frame
	print("[showcase] done; screenshots in %s . Window stays open (close it to quit)." % shots_dir)


func _make_overlay() -> Label:
	var layer := CanvasLayer.new()
	add_child(layer)
	var bg := ColorRect.new()
	bg.color = Color(0, 0, 0, 0.6)
	bg.offset_right = 640
	bg.offset_bottom = 70
	layer.add_child(bg)
	var label := Label.new()
	label.add_theme_color_override("font_color", Color(0.7, 1, 0.7))
	label.add_theme_font_size_override("font_size", 18)
	label.position = Vector2(12, 10)
	layer.add_child(label)
	return label


func _screenshot(dir_abs: String, name: String) -> void:
	await RenderingServer.frame_post_draw
	var img := get_viewport().get_texture().get_image()
	img.save_png("%s/%s.png" % [dir_abs, name])
	print("[showcase] shot %s.png" % name)


var _trail: Array = []

func _place_target_marker(pos: Vector3) -> void:
	if _marker_root == null:
		return
	var mesh := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.012
	sphere.height = 0.024
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1, 0, 0)
	mat.emission_enabled = true
	mat.emission = Color(1, 0, 0)
	mesh.mesh = sphere
	mesh.material_override = mat
	_marker_root.add_child(mesh)
	mesh.global_position = pos
	_trail.append(mesh)


func _drop_trail_point() -> void:
	var tcp: Node3D = _arm.tool_pose if "tool_pose" in _arm and _arm.tool_pose else null
	if tcp == null or _marker_root == null:
		return
	var mesh := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.004
	sphere.height = 0.008
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.2, 0.5, 1)
	mat.emission_enabled = true
	mat.emission = Color(0.2, 0.5, 1)
	mesh.mesh = sphere
	mesh.material_override = mat
	_marker_root.add_child(mesh)
	mesh.global_position = tcp.global_position
	_trail.append(mesh)


func _clear_trail() -> void:
	for m in _trail:
		if is_instance_valid(m):
			m.queue_free()
	_trail.clear()


## Relocate/resize region volumes in memory from a JSON config (never writes the scene).
func _apply_region_overrides(path: String) -> void:
	var abs_path := ProjectSettings.globalize_path("res://") + path
	var f := FileAccess.open(abs_path, FileAccess.READ)
	if f == null:
		push_error("[probe] regions-config not found: %s" % abs_path)
		return
	var cfg = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(cfg) != TYPE_DICTIONARY:
		push_error("[probe] regions-config not a JSON object")
		return
	for reg in ["easy", "medium", "hard"]:
		if not cfg.has(reg):
			continue
		var region: Node = _regions.get(reg)
		if region == null:
			continue
		var spec: Dictionary = cfg[reg]
		var c: Array = spec["center"]
		region.global_position = Vector3(c[0], c[1], c[2])
		var cs: CollisionShape3D = region.get_node(region.collision_shape_path)
		cs.position = Vector3.ZERO
		var box := BoxShape3D.new()
		var b: Array = spec["box"]
		box.size = Vector3(b[0], b[1], b[2])
		cs.shape = box
		if spec.has("grid"):
			var g: Array = spec["grid"]
			region.grid_size = Vector3i(int(g[0]), int(g[1]), int(g[2]))
		if spec.has("allowed_cells"):
			var ac := PackedInt32Array()
			for cc in spec["allowed_cells"]:
				ac.append(int(cc))
			region.allowed_cells = ac
		if region.has_method("reset_sampling_sequence"):
			region.reset_sampling_sequence()
		print("[probe] region %s relocated -> center %.3v box %.3v grid %s" % [
			reg, region.global_position, box.size, str(region.grid_size)])


func _cell_center_world(region: Node, cell: int) -> Vector3:
	# Reproduce TargetSamplingRegion3D cell geometry to get a cell's centre (deterministic).
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


func _probe_region(region_name: String, region: Node) -> void:
	# Sample samples_per_cell poses per ACTIVE cell (allowlist restricts which cells sample);
	# using total grid cells here would over-sample when an allowlist is set.
	var active_count := int(region.total_cells())
	if region.has_method("active_cells"):
		active_count = int(region.active_cells().size())
	var rng := RandomNumberGenerator.new()
	rng.seed = rng_seed + hash(region_name)
	if region.has_method("reset_sampling_sequence"):
		region.reset_sampling_sequence()

	# Draw samples_per_cell poses per active cell by running that many full stratified cycles.
	var draws := active_count * samples_per_cell
	var per_cell_index := {}
	var target_basis: Basis = _grasp.global_transform.basis if _grasp else Basis.IDENTITY
	var done := 0
	for d in range(draws):
		var sample: Dictionary = region.sample_transform(rng, target_basis)
		var cell := int(sample.get("cell", -1))
		var sampled_transform: Transform3D = sample["transform"]
		var pos: Vector3 = sampled_transform.origin
		var pose_basis := _task_orientation(rng, sampled_transform)
		var held_out := (cell % holdout_stride == 0)
		var idx_in_cell := int(per_cell_index.get(cell, 0))
		per_cell_index[cell] = idx_in_cell + 1
		# Seen-cell poses split 80/20 train/val; held-out cells are entirely validation.
		var split := "val_cell"
		if not held_out:
			split = "train" if float(idx_in_cell) < float(samples_per_cell) * train_pose_fraction else "val_seen_cell"

		var rec := await _probe_pose(region_name, cell, idx_in_cell, pos, pose_basis)
		rec["held_out_cell"] = held_out
		rec["split"] = split
		_records.append(rec)
		_spawn_marker(pos, rec["classification"])
		done += 1
		if verbose_poses:
			print("[probe] %s c%d s%d | home=%.1fcm best=%.1fcm ori=%.0fdeg %s" % [
				region_name, cell, idx_in_cell, float(rec["position_error_home_m"]) * 100.0,
				float(rec["position_error_m"]) * 100.0, float(rec["orientation_error_deg"]),
				str(rec["classification"])])
		elif done % 100 == 0:
			print("[probe] %s %d/%d" % [region_name, done, draws])


## Orientation the scenario itself will demand at this pose. Asking the scenario rather than
## recomputing the azimuth/pitch here is the whole point: a probe with its own copy of the
## convention silently stops measuring the task the moment the scene is retuned.
func _task_orientation(rng: RandomNumberGenerator, sampled: Transform3D) -> Basis:
	if _scenario != null and _scenario.has_method("_apply_target_orientation_randomization"):
		var oriented: Transform3D = _scenario._apply_target_orientation_randomization(rng, sampled)
		return oriented.basis
	return sampled.basis


## Drive one target pose from home, then retry from alternate seeds. Separates the ENDPOINT
## reachability question (does a collision-free IK solution exist at the target?) from the
## HOME-trajectory question (does the greedy straight IK path from home collide?).
func _probe_pose(region_name: String, cell: int, idx: int, pos: Vector3, basis: Basis) -> Dictionary:
	# Move the scenario target onto this sample, with the orientation training will demand there.
	_scen_target.global_transform = Transform3D(basis, pos)

	# Attempt from home (the real task start) + alternate seeds. Each attempt reports whether
	# it converged collision-free and its best position error.
	var attempts: Array = [await _run_ik_from(_home_positions())]
	var home: Dictionary = attempts[0]
	# Only spend alternate seeds when home did not already produce a clean, accurate reach.
	if not (home["collision"].is_empty() and home["best_pos_err"] <= accuracy_tol):
		for s in range(alt_seeds):
			attempts.append(await _run_ik_from(_seed_positions(s)))

	# Best collision-free solution across all attempts = the endpoint reachability answer.
	var best: Dictionary = {}
	var closest_any: Dictionary = attempts[0]      # closest approach regardless of collision
	var closest_colliding: Dictionary = {}          # closest approach that ended in a collision
	for a in attempts:
		if float(a["best_pos_err"]) < float(closest_any["best_pos_err"]):
			closest_any = a
		if a["collision"].is_empty():
			if best.is_empty() or float(a["best_pos_err"]) < float(best["best_pos_err"]):
				best = a
		else:
			if closest_colliding.is_empty() or float(a["best_pos_err"]) < float(closest_colliding["best_pos_err"]):
				closest_colliding = a

	var has_clean: bool = not best.is_empty()
	var best_pos_err: float = float(best["best_pos_err"]) if has_clean else float(closest_any["best_pos_err"])
	var min_limit_margin: float = float(best["min_limit_margin"]) if has_clean else float(closest_any["min_limit_margin"])
	var position_reachable: bool = has_clean and best_pos_err <= accuracy_tol

	# --- Orientation: from the best collision-free position solution, refine toward the task
	# GraspPose orientation and record the best angle achievable there. ---
	var ori_natural: float = float(best["best_ori_deg"]) if has_clean else float(closest_any["best_ori_deg"])
	var ori_best: float = ori_natural
	if position_reachable and not fast_mode:
		_arm.reset_all(null, false)
		_robot.reset_joint_positions(_dict_from_joints(best["joints"]))
		_arm.set_training_active(true)
		await get_tree().physics_frame
		ori_natural = _orientation_error_deg()
		ori_best = minf(ori_natural, await _refine_orientation(pos))

	# --- Path validity from home: greedy straight IK, else the waypoint planner. ---
	var home_collision: Dictionary = home["collision"]
	var home_reached: bool = home["collision"].is_empty() and float(home["best_pos_err"]) <= accuracy_tol
	var path_valid: bool = home_reached
	var path_via := "greedy" if home_reached else ""
	if position_reachable and not home_reached and not fast_mode:
		# 1) joint-space interpolation home->solution (cheap, usually best)
		if has_clean and await _check_joint_path(best["joints"], 32):
			path_valid = true
			path_via = "joint_lerp"
		else:
			# 2) Cartesian waypoint planner (approach from above / outside the body)
			var planned: Dictionary = await _try_planned_path(pos)
			if planned["ok"]:
				path_valid = true
				path_via = str(planned["via"])

	# --- Primary classification: endpoint reachability first ---
	var classification := "valid"
	var collision_src := ""
	var collision_checker := ""
	var collision_body := ""
	if has_clean and best_pos_err <= accuracy_tol:
		classification = "joint_limit" if min_limit_margin < limit_margin_tol else "valid"
	elif not closest_colliding.is_empty() and float(closest_colliding["best_pos_err"]) <= accuracy_tol:
		# The only way to the target we found ends in contact -> genuine endpoint collision.
		var c: Dictionary = closest_colliding["collision"]
		collision_src = str(c.get("source", "unknown"))
		collision_checker = str(c.get("checker_link", ""))
		collision_body = str(c.get("body_link", c.get("collider", "")))
		classification = "endpoint_collision:" + ("self" if collision_src == "self_body" else "env")
	elif best_pos_err <= loose_tol or float(closest_any["best_pos_err"]) <= loose_tol:
		classification = "accuracy_failure"
	else:
		classification = "ik_unreachable"

	var no_endpoint_coll: bool = not classification.begins_with("endpoint_collision")
	var joint_limits_valid: bool = has_clean and not is_nan(min_limit_margin) and min_limit_margin >= limit_margin_tol
	var task_valid_45: bool = position_reachable and path_valid and no_endpoint_coll and joint_limits_valid and ori_best <= 45.0
	var task_valid_30: bool = position_reachable and path_valid and no_endpoint_coll and joint_limits_valid and ori_best <= 30.0
	var task_valid_20: bool = position_reachable and path_valid and no_endpoint_coll and joint_limits_valid and ori_best <= 20.0

	# Path status (planner failure is path_not_found, NOT physical unreachability).
	var path_status := "n/a"
	if path_valid:
		path_status = path_via
	elif position_reachable and not fast_mode:
		path_status = "not_found"

	# Failure reason for the task_valid_30 gate, separated as requested.
	var fail_reason_30 := ""
	if not position_reachable:
		fail_reason_30 = "ik_failure"
	elif not no_endpoint_coll:
		fail_reason_30 = "collision:" + ("self" if collision_src == "self_body" else "env")
	elif not joint_limits_valid:
		fail_reason_30 = "joint_limit"
	elif not path_valid:
		fail_reason_30 = "planner_failure"
	elif ori_best > 30.0:
		fail_reason_30 = "orientation_failure"

	return {
		"region": region_name,
		"cell": cell,
		"sample_index": idx,
		"target_position": [pos.x, pos.y, pos.z],
		"target_basis": _basis_to_array(basis),
		"reachable_ik": position_reachable,
		"position_reachable": position_reachable,
		"joint_limits_valid": joint_limits_valid,
		"trajectory_valid": home_reached,
		"home_path_collision": not home_collision.is_empty(),
		"home_path_collision_source": str(home_collision.get("source", "")),
		"path_valid": path_valid,
		"path_via": path_via,
		"path_status": path_status,
		"pose_reachable_45": position_reachable and ori_best <= 45.0,
		"pose_reachable_30": position_reachable and ori_best <= 30.0,
		"pose_reachable_20": position_reachable and ori_best <= 20.0,
		"task_valid_45": task_valid_45,
		"task_valid_30": task_valid_30,
		"task_valid_20": task_valid_20,
		"task_fail_reason_30": fail_reason_30,
		"classification": classification,
		"position_error_m": best_pos_err,
		"position_error_home_m": float(home["best_pos_err"]),
		"orientation_error_deg": ori_best,
		"orientation_error_natural_deg": ori_natural,
		"min_limit_margin_rad": min_limit_margin,
		"collision_source": collision_src,
		"collision_checker_link": collision_checker,
		"collision_body_link": collision_body,
		"joint_solution": best.get("joints", []) if has_clean else [],
	}


func _dict_from_joints(joints: Array) -> Dictionary:
	var d := {}
	var i := 0
	for n in _joint_names:
		d[n] = float(joints[i]) if i < joints.size() else 0.0
		i += 1
	return d


## Reset to `start_positions`, then step the IK toward the probe target. Records the first
## collision seen along the way, the best (min) position error, and the settled endpoint.
func _run_ik_from(start_positions: Dictionary) -> Dictionary:
	_arm.reset_all(null, false)          # clears terminal/collision flags + returns to home
	_robot.reset_joint_positions(start_positions)
	_arm.set_training_active(true)
	await get_tree().physics_frame

	var collision := {}
	var best_pos_err := 1e9
	var steps_used := 0
	for step in range(ik_max_steps):
		_ik.solve(true)
		await get_tree().physics_frame
		steps_used = step + 1
		var d: float = _arm.get_target_distance()
		best_pos_err = minf(best_pos_err, d)
		if _arm.has_collided():
			collision = _arm.get_last_collision_details()
			break
		if d <= _ik.position_deadzone + 0.001 and _arm.get_max_joint_speed() < 0.02:
			break

	# Let the endpoint settle, then take the final measurements (unless we already collided).
	if collision.is_empty():
		for _s in range(settle_steps):
			_ik.solve(true)
			await get_tree().physics_frame
			best_pos_err = minf(best_pos_err, _arm.get_target_distance())
			if _arm.has_collided():
				collision = _arm.get_last_collision_details()
				break

	return {
		"collision": collision,
		"best_pos_err": best_pos_err,
		"best_ori_deg": _orientation_error_deg(),
		"min_limit_margin": _min_limit_margin(),
		"joints": _current_joint_array(),
		"steps": steps_used,
	}


func _tool_pos() -> Vector3:
	var tool: Node3D = _arm.tool_pose if "tool_pose" in _arm and _arm.tool_pose else null
	return tool.global_position if tool else Vector3.INF


## Drive position-only IK from the CURRENT config toward an arbitrary world point. Returns
## {reached, collided} after up to `steps` physics frames. Leaves _ik pointed back at _grasp.
func _drive_ik_to(point: Vector3, steps: int) -> Dictionary:
	_ori_target.global_transform = Transform3D(Basis.IDENTITY, point)
	_ik.set_target(_ori_target)
	_ik.track_orientation = false
	var collided := false
	var reached := false
	for _s in range(steps):
		_ik.solve(true)
		await get_tree().physics_frame
		if _arm.has_collided():
			collided = true
			break
		var d := _tool_pos().distance_to(point)
		if d <= accuracy_tol:
			reached = true
			break
	_ik.set_target(_grasp)
	return {"reached": reached, "collided": collided}


## Straight-line interpolation in JOINT space from home to the IK solution, collision-checked
## at each step. This is a real executable motion plan and usually succeeds where the greedy
## Cartesian IK path swings a link into the body; cheap (no per-step IK). Returns true if the
## whole interpolated path is collision-free.
func _check_joint_path(solution_joints: Array, steps: int) -> bool:
	if solution_joints.is_empty():
		return false
	_arm.reset_all(null, false)
	_robot.reset_joint_positions(_home_positions())
	_arm.set_training_active(true)
	await get_tree().physics_frame
	var home := _home_positions()
	for s in range(1, steps + 1):
		var t := float(s) / float(steps)
		var interp := {}
		var i := 0
		for n in _joint_names:
			interp[n] = lerpf(float(home.get(n, 0.0)), float(solution_joints[i]), t)
			i += 1
		_robot.reset_joint_positions(interp)
		await get_tree().physics_frame
		if _arm.has_collided():
			return false
	return true


## Joint-space waypoint planner for poses whose greedy straight home->target path collides.
## Tries a few heuristic via-points (approach from above / from outside the body), each a
## collision-checked home->via->target chain. Returns {ok, via} for the first clean chain.
func _try_planned_path(pos: Vector3) -> Dictionary:
	var seg_steps: int = mini(ik_max_steps, 200)
	var vias := {
		"over": pos + Vector3(0, 0.12, 0),                       # lift above, then descend
		"out_over": pos + Vector3(0.06, 0.12, 0.06),             # out from body + above
	}
	for via_name in vias:
		_arm.reset_all(null, false)
		_robot.reset_joint_positions(_home_positions())
		_arm.set_training_active(true)
		await get_tree().physics_frame
		var seg1: Dictionary = await _drive_ik_to(vias[via_name], seg_steps)
		if seg1["collided"] or not seg1["reached"]:
			continue
		var seg2: Dictionary = await _drive_ik_to(pos, seg_steps)
		if seg2["collided"]:
			continue
		# Confirm the final approach actually lands on the task target (ToolPose -> GraspPose).
		if _arm.get_target_distance() <= accuracy_tol:
			return {"ok": true, "via": via_name}
	return {"ok": false, "via": ""}


func _home_positions() -> Dictionary:
	if _arm.has_method("_build_home_positions"):
		return _arm._build_home_positions(true)
	var d := {}
	for n in _joint_names:
		d[n] = 0.0
	return d


## Deterministic alternate IK seeds: home nudged on base-yaw/shoulder/elbow (indices 0,1,3;
## joint 2 is skipped because it swings the forearm straight into the body). Gentle enough
## not to start already self-colliding.
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


## Orientation error in degrees, measured with the AGENT'S OWN metric (the one the reward and
## success gate use): ToolPose vs target GraspPose, via get_target_orientation_error_observation
## (normalized axis-angle vector; length * PI = radians). Falls back to a direct quaternion angle.
func _orientation_error_deg() -> float:
	if _arm.has_method("get_target_orientation_error_observation"):
		var v: Vector3 = _arm.get_target_orientation_error_observation()
		return rad_to_deg(v.length() * PI)
	var tool: Node3D = _arm.tool_pose if "tool_pose" in _arm and _arm.tool_pose else null
	if tool == null or _grasp == null:
		return NAN
	var q_cur := tool.global_transform.basis.get_rotation_quaternion()
	var q_des := _grasp.global_transform.basis.get_rotation_quaternion()
	return rad_to_deg(q_cur.angle_to(q_des))


## ToolPose.global_basis = tcp_link.global_basis * B_off (B_off constant). Measure it once at home
## so the orientation phase can target tcp_link.basis = GraspPose.basis * B_off^-1, which lands
## ToolPose exactly on the task's GraspPose orientation.
func _compute_frame_offset() -> void:
	_arm.reset_all(null, false)
	_arm.set_training_active(true)
	await get_tree().physics_frame
	var tcp: Node3D = _robot.get_link_node(_arm.tcp_link_name)
	var tool: Node3D = _arm.tool_pose if "tool_pose" in _arm and _arm.tool_pose else null
	if tcp != null and tool != null:
		_b_off = tcp.global_transform.basis.orthonormalized().inverse() * tool.global_transform.basis.orthonormalized()


## From the current (position-converged) config, drive full-pose IK toward the frame-corrected
## orientation and return the best ToolPose-vs-GraspPose angle achieved while staying on target and
## collision-free. Restores position-only tracking before returning.
func _refine_orientation(target_pos: Vector3) -> float:
	_ori_target.global_transform = Transform3D(_grasp.global_transform.basis * _b_off.inverse(), target_pos)
	_ik.set_target(_ori_target)
	_ik.track_orientation = true
	_ik.orientation_mode = "full"
	var best := _orientation_error_deg()
	var stall := 0
	for _s in range(90):
		_ik.solve(true)
		await get_tree().physics_frame
		if _arm.has_collided():
			break
		if _arm.get_target_distance() <= accuracy_tol:
			var e := _orientation_error_deg()
			if e < best - 0.2:
				best = e
				stall = 0
			else:
				stall += 1
			if best <= 3.0 or stall >= 12:
				break
	# Restore position-only tracking for the next pose.
	_ik.track_orientation = false
	_ik.set_target(_grasp)
	return best


func _min_limit_margin() -> float:
	var m := 1e9
	for n in _joint_names:
		var jnode = _robot.get_joint_node(n)
		if jnode == null or jnode.joint == null or jnode.joint.limit == null:
			continue
		var lower: float = minf(jnode.joint.limit.lower, jnode.joint.limit.upper)
		var upper: float = maxf(jnode.joint.limit.lower, jnode.joint.limit.upper)
		var pos: float = _robot.get_joint_position(n)
		m = minf(m, minf(pos - lower, upper - pos))
	return m if m < 1e8 else NAN


func _current_joint_array() -> Array:
	var out := []
	for n in _joint_names:
		out.append(_robot.get_joint_position(n))
	return out


func _basis_to_array(b: Basis) -> Array:
	return [b.x.x, b.x.y, b.x.z, b.y.x, b.y.y, b.y.z, b.z.x, b.z.y, b.z.z]


func _spawn_marker(pos: Vector3, classification: String) -> void:
	if _marker_root == null:
		return
	var color := Color(0.2, 0.9, 0.2)          # valid = green
	if classification.begins_with("path_collision") or classification.begins_with("endpoint_collision") or classification == "joint_limit":
		color = Color(0.9, 0.15, 0.15)          # collision / limit = red
	elif classification == "ik_unreachable" or classification == "accuracy_failure":
		color = Color(0.95, 0.8, 0.1)           # ik / accuracy = yellow
	var mesh := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.006
	sphere.height = 0.012
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.emission_enabled = true
	mat.emission = color
	mesh.mesh = sphere
	mesh.material_override = mat
	_marker_root.add_child(mesh)
	mesh.global_position = pos


func _write_outputs() -> void:
	var dir_abs := ProjectSettings.globalize_path("res://") + out_dir
	DirAccess.make_dir_recursive_absolute(dir_abs)
	var json_path := dir_abs + "/workspace_probe.json"
	var md_path := dir_abs + "/M1_REPORT.md"

	var summary := _aggregate()
	var payload := {
		"config": {
			"samples_per_cell": samples_per_cell,
			"ik_max_steps": ik_max_steps,
			"alt_seeds": alt_seeds,
			"accuracy_tol_m": accuracy_tol,
			"loose_tol_m": loose_tol,
			"limit_margin_tol_rad": limit_margin_tol,
			"holdout_stride": holdout_stride,
			"train_pose_fraction": train_pose_fraction,
			"seed": rng_seed,
			"self_collision_forced_on": true,
		},
		"summary": summary,
		"records": _records,
	}
	var jf := FileAccess.open(json_path, FileAccess.WRITE)
	jf.store_string(JSON.stringify(payload, "  "))
	jf.close()

	var mf := FileAccess.open(md_path, FileAccess.WRITE)
	mf.store_string(_markdown(summary))
	mf.close()
	print("[probe] wrote %s and %s" % [json_path, md_path])


func _new_bucket() -> Dictionary:
	return {
		"samples": 0, "valid": 0, "reachable": 0, "home_traj_valid": 0,
		"home_path_coll": 0, "endpoint_coll": 0,
		"pos_err_sum": 0.0, "pos_err_max": 0.0, "ori_err_sum": 0.0,
		"limit_margin_min": 1e9, "class": {}, "collision_links": {},
	}


func _accumulate(bucket: Dictionary, r: Dictionary) -> void:
	bucket["samples"] += 1
	var cls := str(r["classification"])
	bucket["valid"] += 1 if cls == "valid" else 0
	bucket["reachable"] += 1 if r["reachable_ik"] else 0
	bucket["home_traj_valid"] += 1 if r["trajectory_valid"] else 0
	bucket["home_path_coll"] += 1 if r["home_path_collision"] else 0
	bucket["endpoint_coll"] += 1 if cls.begins_with("endpoint_collision") else 0
	bucket["pos_err_sum"] += float(r["position_error_m"])
	bucket["pos_err_max"] = maxf(bucket["pos_err_max"], float(r["position_error_m"]))
	if not is_nan(float(r["orientation_error_deg"])):
		bucket["ori_err_sum"] += float(r["orientation_error_deg"])
	if not is_nan(float(r["min_limit_margin_rad"])):
		bucket["limit_margin_min"] = minf(bucket["limit_margin_min"], float(r["min_limit_margin_rad"]))
	bucket["class"][cls] = int(bucket["class"].get(cls, 0)) + 1
	var src := str(r["collision_source"])
	if src == "":
		src = str(r["home_path_collision_source"])
	if src != "":
		var link := "%s<->%s" % [str(r["collision_checker_link"]), str(r["collision_body_link"])]
		if str(r["collision_checker_link"]) == "":
			link = "home_path:" + src
		bucket["collision_links"][link] = int(bucket["collision_links"].get(link, 0)) + 1


func _finalize_bucket(b: Dictionary) -> void:
	var n := maxf(float(b["samples"]), 1.0)
	b["valid_pct"] = 100.0 * float(b["valid"]) / n
	b["reachable_pct"] = 100.0 * float(b["reachable"]) / n
	b["home_traj_valid_pct"] = 100.0 * float(b["home_traj_valid"]) / n
	b["home_path_coll_pct"] = 100.0 * float(b["home_path_coll"]) / n
	b["pos_err_mean"] = float(b["pos_err_sum"]) / n
	b["ori_err_mean"] = float(b["ori_err_sum"]) / n


func _aggregate() -> Dictionary:
	var by_region := {}
	var by_cell := {}
	for r in _records:
		var reg := str(r["region"])
		var cellkey := "%s/%d" % [reg, int(r["cell"])]
		if not by_region.has(reg):
			by_region[reg] = _new_bucket()
		if not by_cell.has(cellkey):
			by_cell[cellkey] = _new_bucket()
		_accumulate(by_region[reg], r)
		_accumulate(by_cell[cellkey], r)
	for k in by_region:
		_finalize_bucket(by_region[k])
	for k in by_cell:
		_finalize_bucket(by_cell[k])
		by_cell[k]["verdict"] = _cell_verdict(by_cell[k])
	return {"by_region": by_region, "by_cell": by_cell}


## Physical usability of a cell (endpoint reachability first; home-path is a policy concern,
## not a physics one). NEVER auto-mask: this only LABELS. The decision to mask/boost/keep is
## the reviewer's, per the M1 gate.
func _cell_verdict(b: Dictionary) -> String:
	var valid_pct := float(b["valid_pct"])
	var reachable_pct := float(b["reachable_pct"])
	if valid_pct >= 70.0:
		return "usable"
	if int(b["endpoint_coll"]) > int(b["samples"]) / 2:
		return "unsafe_endpoint_collision"
	if reachable_pct < 30.0:
		return "unreachable"
	if reachable_pct >= 70.0:
		# Physically reachable, but the plain IK often misses accuracy -> a policy/coverage gap.
		return "reachable_coverage_gap"
	return "marginal"


func _markdown(summary: Dictionary) -> String:
	var s := "# Milestone 1 report — OpenArm workspace validity\n\n"
	s += "Probe: position-only IK from home + home-path & endpoint collision + joint limits, "
	s += "self-collision FORCED ON (training scene ships it OFF). "
	s += "Accuracy tol %.0f cm, loose tol %.0f cm, %d samples/cell, %d IK seeds.\n\n" % [
		accuracy_tol * 100.0, loose_tol * 100.0, samples_per_cell, alt_seeds + 1]
	s += "Legend: **valid** = collision-free IK solution at target within accuracy & joint limits. "
	s += "**reachable** = a collision-free solution within accuracy exists (limit margin ignored). "
	s += "**home-traj valid** = the greedy straight IK path from home reaches it without colliding.\n\n"
	s += "## By region\n\n"
	s += "| region | samples | valid %% | reachable %% | home-traj valid %% | home-path coll %% | pos_err mean (cm) | ori mean (deg) | min limit margin (rad) |\n"
	s += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
	var br: Dictionary = summary["by_region"]
	for reg in ["easy", "medium", "hard"]:
		if not br.has(reg):
			continue
		var b: Dictionary = br[reg]
		s += "| %s | %d | %.1f | %.1f | %.1f | %.1f | %.2f | %.0f | %.3f |\n" % [
			reg, int(b["samples"]), float(b["valid_pct"]), float(b["reachable_pct"]),
			float(b["home_traj_valid_pct"]), float(b["home_path_coll_pct"]),
			float(b["pos_err_mean"]) * 100.0, float(b["ori_err_mean"]), float(b["limit_margin_min"])]
	s += "\n## Cells below 70%% valid (candidates for boosted sampling / mask — NOT auto-applied)\n\n"
	s += "| cell | samples | valid %% | reachable %% | home-traj %% | pos_err mean (cm) | dominant class | verdict |\n"
	s += "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
	var bc: Dictionary = summary["by_cell"]
	var keys := bc.keys()
	keys.sort()
	for k in keys:
		var b: Dictionary = bc[k]
		if float(b["valid_pct"]) >= 70.0:
			continue
		s += "| %s | %d | %.0f | %.0f | %.0f | %.2f | %s | %s |\n" % [
			k, int(b["samples"]), float(b["valid_pct"]), float(b["reachable_pct"]),
			float(b["home_traj_valid_pct"]), float(b["pos_err_mean"]) * 100.0,
			_dominant(b["class"]), str(b["verdict"])]
	s += "\n_Full per-cell + per-pose data (incl. joint solutions, collision links, train/val split) in workspace_probe.json._\n"
	return s


func _dominant(class_counts: Dictionary) -> String:
	var best := ""
	var best_n := -1
	for c in class_counts:
		if int(class_counts[c]) > best_n:
			best_n = int(class_counts[c])
			best = str(c)
	return best
