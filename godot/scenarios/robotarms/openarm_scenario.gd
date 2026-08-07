extends Node3D

@export var target_spawns: Array[Marker3D] = []
## Optional: assign a node whose Marker3D children are auto-collected as spawn points, so
## adding a marker in the editor is enough (no need to also wire it into target_spawns).
@export var target_spawns_root: Node3D
@export_category("Target Sampling Regions")
## Adaptive training samples continuous positions from these volumes. Markers remain available
## for bootstrap poses, non-adaptive runs, and deterministic regression tests.
@export var use_adaptive_target_regions := true
@export var easy_target_region: Area3D
@export var medium_target_region: Area3D
@export var hard_target_region: Area3D
## Easy is sampled alone first (mastered to 2cm), then Medium is introduced, then Hard.
@export_range(0.0, 1.0, 0.05) var adaptive_medium_region_start_level := 0.45
@export_range(0.0, 1.0, 0.05) var adaptive_medium_region_full_level := 0.55
@export_range(0.0, 1.0, 0.05) var adaptive_medium_region_initial_weight := 0.25
## Hard is introduced only after medium is mastered. The final distribution retains earlier
## regions to prevent catastrophic forgetting.
@export_range(0.0, 1.0, 0.05) var adaptive_hard_region_start_level := 0.70
@export_range(0.0, 1.0, 0.05) var adaptive_hard_region_full_level := 0.80
@export_range(0.0, 1.0, 0.05) var adaptive_hard_region_initial_weight := 0.20
@export var adaptive_final_region_weights := Vector3(0.20, 0.40, 0.40)
## Number of region choices in one balanced deck. Twenty matches the default frozen evaluation
## length, so every active region receives a predictable number of trials.
@export_range(3, 200, 1) var target_region_schedule_size := 20
## Per-region success gate. Tighten Easy/Medium (closer to the base, so higher achievable precision)
## to a smaller radius while keeping Hard at a looser radius the arm can physically reach at full
## extension. Applied only on regular (region-sampled) resets, and it propagates to frozen evaluation
## because the evaluator reuses this same reset path. Disabled by default (keeps the level-based gate).
@export var per_region_gate_enabled := false
## Per-region success radii (EE tip within this distance of the target). Easy and Medium tighten
## progressively 4cm -> 3cm -> 2cm as the curriculum level rises (mastery). Hard is fixed at the loose
## 4cm (full-extension reach; sub-4cm precision is not expected there). These are the FLOOR (tightest)
## radii each region converges to; the loose/mid stages below apply while the level is still low.
@export var easy_success_distance := 0.02
@export var medium_success_distance := 0.02
@export var hard_success_distance := 0.04
## Loose (start) and mid radii used before a region reaches its tightest gate.
@export var per_region_gate_loose := 0.04
@export var per_region_gate_mid := 0.03
## Level thresholds at which Easy tightens 4->3 (stage1) and 3->2 (stage2). Easy is learned alone
## early, so it tightens quickly.
@export_range(0.0, 1.0, 0.05) var easy_gate_stage1_level := 0.25
@export_range(0.0, 1.0, 0.05) var easy_gate_stage2_level := 0.38
## Level thresholds at which Medium tightens 4->3 (stage1) and 3->2 (stage2). Medium is introduced at
## adaptive_medium_region_start_level and tightens once it is stable.
@export_range(0.0, 1.0, 0.05) var medium_gate_stage1_level := 0.55
@export_range(0.0, 1.0, 0.05) var medium_gate_stage2_level := 0.65
@export_category("Target Curriculum")
@export var easy_target_count := 4
## Keep the first easy_target_count markers active initially, then add the remaining markers
## progressively. This avoids the old 4 -> all-markers difficulty jump at episode 300.
@export var target_pool_curriculum_start_episode := 800
@export var target_pool_curriculum_full_episode := 3200
@export_category("Bootstrap Curriculum")
## Optional known-good pose used only at the beginning of training. The reset starts close to this
## solved configuration, then smoothly becomes the regular home-to-random-target task. Once
## bootstrap_curriculum_full_episode is reached, this pose has no effect on training.
@export var bootstrap_target_pose: Marker3D
@export var bootstrap_joint_positions := PackedFloat32Array()
## Optional set of MULTIPLE bootstrap demo poses (reverse-training). Each entry of bootstrap_pose_joints
## is a full joint configuration; the matching entry of bootstrap_pose_targets is the world-space EE
## target the arm holds in that pose. When non-empty these take precedence over the single
## bootstrap_joint_positions / bootstrap_target_pose above: each bootstrap-reset episode picks one demo
## at random, so early successes cover the whole area (better critic anchoring) instead of one point.
@export var bootstrap_pose_joints: Array[PackedFloat32Array] = []
@export var bootstrap_pose_targets: Array[Vector3] = []
@export var bootstrap_curriculum_start_episode := 200
@export var bootstrap_curriculum_full_episode := 1800
@export var bootstrap_target_jitter_radius := 0.005
@export var bootstrap_joint_jitter_degrees := 1.0
## SPATIAL reverse curriculum. On assisted (bootstrap) resets the arm starts at an INTERPOLATED joint
## configuration between the demo pose (fraction 1.0 = EE on the target, a tiny reach) and home
## (fraction 0.0 = full reach). At curriculum level 0 the window is [1,1] (all resets at the demo);
## it widens toward home as the level approaches reverse_curriculum_full_level, so the arm learns the
## collision-free approach corridor from the target backward instead of blindly searching for it from
## home (which self-collides: finger -> body). Set <=0 to disable (reset exactly at the demo pose).
##
## The widening is driven by EPISODES (reverse_curriculum_start/full_episode below), NOT the promotion
## level: the region-promotion level is gated by the home-start (worst-region) success, which is exactly
## what the reverse curriculum has to TEACH, so tying reverse_t to the level deadlocks (the arm only
## ever practises near-demo reaches, never home-start, so the level never rises). Episode pacing breaks
## that loop - the arm always progresses toward home-start regardless of the stuck level.
@export_range(0.0, 1.0, 0.05) var reverse_curriculum_full_level := 0.45
@export var reverse_curriculum_start_episode := 500
@export var reverse_curriculum_full_episode := 5000
## false = the arm always resets at HOME (no near-target reverse reset). Use for
## demo-augmented training whose demos are home-start reaches -- keeps training and
## demos consistent and the render clean (always a full home->target reach).
@export var reverse_curriculum_enabled := true
## Minimum fraction of episodes that use a REGULAR (home-start, reach-the-target) reset even at the
## lowest curriculum level. Zero keeps the original behaviour (level 0 = 100% bootstrap = the arm only
## ever holds at the solved pose and never practises reaching, so it walls the moment the assist fades).
## A small positive value forces reach practice from the start so the arm learns to travel home->target
## gradually instead of hitting a cliff when the bootstrap assist disappears.
@export_range(0.0, 1.0, 0.05) var bootstrap_min_regular_reset_fraction := 0.0
@export_category("Target Curriculum")
## Reach accuracy tightens independently from target diversity.
@export var reach_curriculum_stage1_until := 1000
@export var reach_curriculum_stage2_until := 2400
@export var reach_curriculum_stage3_until := 4000
## Position dominates early. Orientation is introduced gradually once basic reaching has started.
@export_range(0.0, 1.0, 0.01) var orientation_progress_start_weight := 0.05
@export_range(0.0, 1.0, 0.01) var orientation_progress_final_weight := 0.25
@export var orientation_progress_curriculum_start_episode := 800
@export var orientation_progress_curriculum_full_episode := 3200
## Once the basic reach is learned, sample the complete axis-aligned volume delimited by the
## Marker3D nodes instead of memorizing a finite set of target coordinates.
@export var continuous_target_sampling := true
@export var continuous_target_sampling_start_episode := 800
## Spatial curriculum: rather than jumping straight to the full marker volume, sample within a radius
## around a discrete marker that GROWS from spatial_curriculum_start_radius (tight, essentially on the
## markers) to the full per-axis half-extent, linearly between the two episodes below. Keeps early
## continuous targets close to the learned discrete poses, then widens to cover the whole volume.
## Leave full<=start to disable (samples the full volume immediately, the original behaviour).
@export var spatial_curriculum_start_episode := 0
@export var spatial_curriculum_full_episode := 0
@export var spatial_curriculum_start_radius := 0.01
## Shrinks the marker-defined volume on each axis. Useful when the outer markers sit too close to
## a wall or to the physical edge of the robot workspace.
@export var target_sampling_inset := Vector3.ZERO
@export var intermediate_joint_jitter_degrees := 2.0
@export var late_joint_jitter_degrees := 10.0
@export_category("Continuous Pose Tracking")
@export var continuous_pose_tracking := true
## With adaptive curriculum, earlier levels terminate after a valid hold so the learner first
## masters reach-and-stop. Continuous post-success tracking is introduced at this level.
@export_range(0.0, 1.0, 0.05) var continuous_tracking_curriculum_level := 1.0
## Random yaw teaches the policy that the target is a pose, not only a point. The target's local
## +X axis remains the canonical approach direction.
@export var target_yaw_randomization_start_episode := 800
@export_range(0.0, 180.0, 1.0) var target_yaw_randomization_degrees := 30.0
## Yaw the demanded grasp pose to follow the target's azimuth around the arm base, so every
## position asks for the same wrist pose relative to the arm instead of one fixed world
## orientation. Without it a single absolute orientation has to absorb the whole azimuth spread of
## the sampling regions, which makes an orientation gate free on one side of the workspace and
## unreachable on the other.
@export var target_yaw_follows_azimuth := false
## Fraction of the azimuth applied; 1.0 tracks it fully, 0.0 reproduces the fixed orientation.
## Negative values mirror the sweep, for when the workspace reads the other way round.
@export_range(-2.0, 2.0, 0.05) var target_azimuth_yaw_gain := 1.0
## Azimuth (degrees, measured like the target's own) that maps to zero yaw. Use it to anchor the
## unrotated pose to a landmark such as the near edge of the table, so the sweep starts there
## instead of straight ahead of the base.
@export_range(-180.0, 180.0, 0.5) var target_azimuth_reference_degrees := 0.0
## Degrees the demanded approach axis tilts downwards, applied in the target's own frame after the
## yaw. Zero asks for a purely horizontal side approach, which is what an untransformed grasp pose
## happens to request; a small positive value asks the tool to come down onto the target instead.
@export_range(0.0, 90.0, 0.5) var target_pitch_degrees := 0.0
## Random spread around target_pitch_degrees, so the tool learns a family of downward approaches
## instead of memorising one. The pitch is clamped to stay downwards: a negative one would ask the
## tool to come at the target from underneath the table.
@export_range(0.0, 45.0, 0.5) var target_pitch_randomization_degrees := 0.0
## Random roll about the approach axis. Note this is charged in full against the orientation gate,
## which measures the complete 3-DOF angle, so a wide roll silently eats the tolerance budget.
@export_range(0.0, 45.0, 0.5) var target_roll_randomization_degrees := 0.0
## Once static reach-and-hold is established, move the target after a random dwell without
## resetting the robot. This creates reach -> hold -> reacquire transitions in the replay.
@export var relocate_target_during_training := true
@export var target_relocation_start_episode := 1500
@export_range(1, 1000, 1) var target_relocation_delay_steps_min := 45
@export_range(1, 1000, 1) var target_relocation_delay_steps_max := 120
@export var minimum_target_relocation_distance := 0.04
## Reach-and-HOLD curriculum breakpoints (absolute training episode). The hold requirement tightens
## in stages: a longer hold at a lower stillness threshold.
@export var hold_curriculum_stage1_until := 1000
@export var hold_curriculum_stage2_until := 2200
## Adaptive trainers send a normalized curriculum_level. This value maps level 1.0 to the
## episode-based curriculum below, preserving the existing behavior for terminal-only runs.
@export var adaptive_curriculum_full_episode := 6500
@export_category("Adaptive Curriculum Stages")
## Adaptive training uses factorized stages instead of mapping one level to every episode-based
## schedule at once. This prevents one promotion from simultaneously removing bootstrap support,
## adding many targets, tightening the pose gate, and extending the hold.
@export_range(0.0, 1.0, 0.05) var adaptive_bootstrap_full_level := 0.45
@export_range(0.0, 1.0, 0.05) var adaptive_reach_stage1_level := 0.40
@export_range(0.0, 1.0, 0.05) var adaptive_target_pool_start_level := 0.40
@export_range(0.0, 1.0, 0.05) var adaptive_target_pool_full_level := 0.70
@export_range(0.0, 1.0, 0.05) var adaptive_orientation_stage_level := 0.80
@export_range(0.0, 1.0, 0.05) var adaptive_reach_stage2_level := 0.90
@export_range(0.0, 1.0, 0.05) var adaptive_reach_stage3_level := 1.00
@export_range(0.0, 1.0, 0.01) var adaptive_orientation_initial_weight := 0.15
## Keep the first adaptive run focused on reach-and-stop. Longer holds and continuous tracking
## belong to a later fine-tuning run once regular reset success is reliable.
@export var adaptive_hold_physics_frames := 20
@export var adaptive_hold_max_joint_speed := 0.10
## Stage A (level < stage1) success gate. Defaults reproduce the previous hard-coded 8 cm / 45 deg;
## the stable reach-hold profile tightens the distance to 4 cm.
@export var adaptive_reach_stage_a_distance := 0.08
@export var adaptive_reach_stage_a_angle := 45.0
## When true the free curriculum_level interpolation is replaced by a DISCRETE 6-stage schedule
## (A..F at levels 0.0/0.2/0.4/0.6/0.8/1.0): per-stage region weights, per-region success gate and
## hold/speed come from a fixed table, and the reward orientation weight stays 0.10 in every stage.
@export var use_discrete_stage_curriculum := false
@export_category("OpenArm Collision Contract")
## Keep the scenario resilient to inherited-scene overrides saved by the editor.
@export var required_self_body_links := PackedStringArray(
	["openarm_body_link0"])
@export var required_self_check_links := PackedStringArray([
	"openarm_right_link3",
	"openarm_right_link4",
	"openarm_right_link5",
	"openarm_right_link6",
	"openarm_right_link7",
	"openarm_right_left_finger",
	"openarm_right_right_finger",
])

@onready var controller: ScenarioController = $ScenarioController
# Both robot backends implement the same contract without sharing a GDScript base class.
@onready var arm = $OpenarmAgent
@onready var target: Node3D = $Target
@onready var target_pose: Node3D = $Target/GraspPose
@onready var goal_event = $ScenarioController/ScenarioEventSystem/GoalReached
@onready var collision_event = $ScenarioController/ScenarioEventSystem/Collision
@onready var self_collision_event = $ScenarioController/ScenarioEventSystem/SelfCollision

var _training_episode := 0
var _curriculum_level_override := -1.0
## Index of the bootstrap demo pose chosen for the current bootstrap-reset episode (multi-pose mode).
var _active_bootstrap_index := 0
var _training_mode := false
var _evaluation_mode := false
var _eval_region_deck: Array = []
var _eval_region_deck_index := 0
# Demo-generation mode (M5 IK expert demos): force a single region + an explicit cell per episode
# so the recorder can balance coverage over every allowed cell. Python owns the schedule and sets
# demo_forced_cell via config before each reset; the scenario stays a passive, deterministic sampler.
var _demo_mode := false
var _demo_forced_region := "easy"
var _demo_forced_cell := -1
var _demo_plan_data: Array = []
var _current_reset_seed := 0
var _goal_terminal_reason := "target_reached"
var _episode_rng := RandomNumberGenerator.new()
var _target_pool_size := 1
var _target_relocation_step := -1
var _last_relocation_check_step := -1
var _last_target_sample_info: Dictionary = {}
var _target_height_ray: MeshInstance3D = null
var _target_height_mark: MeshInstance3D = null
var _target_axes: Node3D = null
var _target_axes_source: Node3D = null
var _tool_axes: Node3D = null
var _tool_axes_source: Node3D = null
var _region_schedule: Array[Area3D] = []
var _region_schedule_cursor := 0
var _region_schedule_signature := ""


func _ready() -> void:
	_apply_openarm_collision_contract()
	var configured_terminal_reason := str(goal_event.terminal_reason)
	if not configured_terminal_reason.is_empty():
		_goal_terminal_reason = configured_terminal_reason
	arm.target = target
	arm.target_pose = target_pose
	_spawn_target_axes_gizmo()
	# Let the manual J key sample fresh targets across the Easy region when capturing reference poses.
	if "manual_random_target_region" in arm:
		arm.manual_random_target_region = easy_target_region
	arm.target_reached.connect(_on_target_reached)
	arm.target_pose_relocated.connect(_on_target_pose_relocated)
	arm.obstacle_collision.connect(_on_obstacle_collision)
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)
	_apply_continuous_pose_tracking(continuous_pose_tracking)
	_validate_sampling_regions()


## Fail-closed guard: a target region whose allowlist is set but resolves to no cells (or whose
## box shape is missing) can never produce a valid target. Terminate the run loudly rather than
## silently training on degenerate/masked targets.
func _validate_sampling_regions() -> void:
	for region in [easy_target_region, medium_target_region, hard_target_region]:
		if region == null or not region.has_method("active_cells"):
			continue
		var has_allowlist: bool = ("allowed_cells" in region) and not region.allowed_cells.is_empty()
		if has_allowlist and region.active_cells().is_empty():
			push_error(
				"[openarm_scenario] region '%s' has allowed_cells set but no valid cell; "
				% region.name
				+ "invalid sampler configuration (fail-closed). Terminating.")
			get_tree().quit(1)
			return


## Axis triads on both frames of the orientation comparison: the demanded grasp pose and the tool
## pose it is measured against. Seeing them side by side turns orientation_error_deg into something
## readable while watching a run, and makes it obvious whether the azimuth sweep is applied at all.
func _spawn_target_axes_gizmo() -> void:
	if DisplayServer.get_name() == "headless":
		return
	_target_axes_source = target_pose
	_target_axes = _spawn_axes_gizmo(0.06)
	if arm != null:
		var tool_pose := (arm as Node3D).get_node_or_null("EndEffector/ToolPose") as Node3D
		if tool_pose == null:
			tool_pose = (arm as Node3D).get_node_or_null("EndEffector") as Node3D
		_tool_axes_source = tool_pose
		_tool_axes = _spawn_axes_gizmo(0.045)
	_spawn_target_height_ray()


## Plumb line from the target down to the floor, with a tick where it lands, because the target's
## height is the hardest thing to judge by eye. Parented to the scenario rather than to the target
## so the azimuth yaw never tilts it away from vertical.
func _spawn_target_height_ray() -> void:
	if target == null:
		return
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(1.0, 0.85, 0.2)
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED

	var ray_mesh := BoxMesh.new()
	ray_mesh.size = Vector3(0.004, 1.0, 0.004)
	ray_mesh.material = material
	_target_height_ray = MeshInstance3D.new()
	_target_height_ray.name = "TargetHeightRay"
	_target_height_ray.mesh = ray_mesh
	add_child(_target_height_ray)

	var mark_mesh := BoxMesh.new()
	mark_mesh.size = Vector3(0.04, 0.002, 0.04)
	mark_mesh.material = material
	_target_height_mark = MeshInstance3D.new()
	_target_height_mark.name = "TargetHeightMark"
	_target_height_mark.mesh = mark_mesh
	add_child(_target_height_mark)


func _process(_delta: float) -> void:
	_follow_frame(_target_axes, _target_axes_source)
	_follow_frame(_tool_axes, _tool_axes_source)
	if _target_height_ray == null or target == null:
		return
	var point := target.global_position
	var surface_y := _surface_below(point)
	var height := maxf(point.y - surface_y, 0.001)
	_target_height_ray.global_transform = Transform3D(
		Basis.IDENTITY.scaled(Vector3(1.0, height, 1.0)),
		Vector3(point.x, surface_y + height * 0.5, point.z))
	_target_height_mark.global_transform = Transform3D(
		Basis.IDENTITY, Vector3(point.x, surface_y + 0.002, point.z))


## Copy a frame's pose without its scale, so the rods stay square whatever the source link carries.
func _follow_frame(holder: Node3D, source: Node3D) -> void:
	if holder == null or source == null or not source.is_inside_tree():
		return
	holder.global_transform = Transform3D(
		source.global_basis.orthonormalized(), source.global_position)


## Height of the first support surface under a point, so the plumb line measures clearance over the
## table rather than over the floor. Only obstacles and the floor count: the arm passing underneath
## must not shorten the line.
func _surface_below(point: Vector3) -> float:
	if not is_inside_tree():
		return 0.0
	var query := PhysicsRayQueryParameters3D.create(
		point + Vector3.UP * 0.001, point + Vector3.DOWN * 5.0)
	query.collide_with_areas = false
	var hit := get_world_3d().direct_space_state.intersect_ray(query)
	var collider: Variant = hit.get("collider")
	if collider is Node and (
		(collider as Node).is_in_group("robot_obstacle")
		or (collider as Node).name == "Floor"
	):
		return (hit["position"] as Vector3).y
	return 0.0


## Three unshaded rods (X red, Y green, Z blue). The holder is a child of the scenario, not of the
## frame it depicts, and is driven from an orthonormalized copy of that frame in _process: a URDF
## link carrying a non-uniform scale would otherwise stretch the rods into something unreadable.
func _spawn_axes_gizmo(axis_length: float) -> Node3D:
	var holder := Node3D.new()
	holder.name = "AxesGizmo"
	add_child(holder)
	const AXIS_THICKNESS := 0.003
	var axis_colors := {
		Vector3.RIGHT: Color(1.0, 0.25, 0.25),
		Vector3.UP: Color(0.25, 1.0, 0.25),
		Vector3.BACK: Color(0.35, 0.55, 1.0),
	}
	for direction in axis_colors:
		var along: Vector3 = (direction as Vector3).abs()
		var mesh := BoxMesh.new()
		mesh.size = along * axis_length + (Vector3.ONE - along) * AXIS_THICKNESS
		var material := StandardMaterial3D.new()
		material.albedo_color = axis_colors[direction]
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mesh.material = material
		var rod := MeshInstance3D.new()
		rod.name = "Axis%s" % str(direction)
		rod.mesh = mesh
		rod.position = (direction as Vector3) * axis_length * 0.5
		holder.add_child(rod)
	return holder


func _apply_openarm_collision_contract() -> void:
	if not arm.self_body_collision_enabled:
		return
	arm.self_body_link_names = required_self_body_links.duplicate()
	arm.self_check_link_names = required_self_check_links.duplicate()
	if arm.has_method("_refresh_collision_geometry"):
		arm.call_deferred("_refresh_collision_geometry")


func _physics_process(_delta: float) -> void:
	if (
		not _training_mode
		or not relocate_target_during_training
		or _curriculum_episode() < target_relocation_start_episode
		or _target_relocation_step < 0
	):
		return
	var current_step := controller.step_count
	if current_step == _last_relocation_check_step:
		return
	_last_relocation_check_step = current_step
	if current_step >= _target_relocation_step:
		_relocate_target()


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))
	_curriculum_level_override = (
		clampf(float(config["curriculum_level"]), 0.0, 1.0)
		if config.has("curriculum_level")
		else -1.0)
	_training_mode = bool(config.get("training_mode", controller.training_mode))
	# Explicit frozen-evaluation flag (NOT derived from training_mode: run.py inference is also
	# non-training but must not apply the balanced eval deck). Reset the deck on every (re)config
	# so a stage change rebuilds it over the new active regions.
	_evaluation_mode = bool(config.get("evaluation_mode", false))
	_eval_region_deck.clear()
	_eval_region_deck_index = 0
	# Demo generation: force a single region + explicit cell (Python sends the full config each
	# episode). demo_forced_cell < 0 means "not set this episode" -> normal sampling in that region.
	_demo_mode = bool(config.get("demo_mode", false))
	_demo_forced_region = str(config.get("demo_forced_region", "easy")).strip_edges().to_lower()
	_demo_forced_cell = int(config.get("demo_forced_cell", -1))
	# Optional precomputed collision-free JOINT waypoint plan (home->goal) the arm tracks
	# closed-loop during demo recording. [] = follow ik_manual / policy instead.
	_demo_plan_data = config.get("demo_plan", [])
	if typeof(_demo_plan_data) != TYPE_ARRAY:
		_demo_plan_data = []
	var continue_after_success := bool(config.get(
		"continue_after_success", false))
	var scene_tracking_enabled := continuous_pose_tracking
	if _curriculum_level_override >= 0.0:
		scene_tracking_enabled = (
			scene_tracking_enabled
			and _curriculum_level_override
			>= continuous_tracking_curriculum_level)
	_apply_continuous_pose_tracking(
		continue_after_success or scene_tracking_enabled)


func _apply_continuous_pose_tracking(enabled: bool) -> void:
	controller.continue_after_success = enabled
	if arm.has_method("set_continue_after_success"):
		arm.set_continue_after_success(enabled)
	goal_event.terminal_reason = "" if enabled else _goal_terminal_reason


func _curriculum_episode() -> int:
	if _curriculum_level_override >= 0.0:
		return roundi(
			_curriculum_level_override
			* float(maxi(adaptive_curriculum_full_episode, 1)))
	return _training_episode


func _target_spawn_pool() -> Array[Marker3D]:
	# Prefer auto-collecting every Marker3D under target_spawns_root; fall back to the
	# manually-wired target_spawns array so existing scenes keep working.
	var result: Array[Marker3D] = []
	if target_spawns_root:
		for child in target_spawns_root.get_children():
			if child is Marker3D:
				result.append(child)
	if result.is_empty():
		for marker in target_spawns:
			if marker:
				result.append(marker)
	return result


func _on_episode_reset_started(_seed:int) -> void:
	var spawn_pool := _target_spawn_pool()
	if spawn_pool.is_empty():
		push_error("RobotArmReachingScenario requires at least one TargetSpawn")
		return
	_episode_rng.seed = _seed
	_current_reset_seed = _seed
	# Load the per-episode demo plan (if any) so the arm tracks it closed-loop this episode.
	if _demo_mode and arm.has_method("set_demo_plan"):
		arm.set_demo_plan(_demo_plan_data, int(controller.physics_frames_per_step))
	_target_relocation_step = -1
	_last_relocation_check_step = -1

	_target_pool_size = _curriculum_target_pool_size(spawn_pool.size())
	arm.orientation_progress_weight = _curriculum_orientation_weight()
	var joint_jitter_degrees := 0.0

	var curriculum_episode := _curriculum_episode()
	if _has_adaptive_curriculum():
		joint_jitter_degrees = _apply_adaptive_pose_curriculum()
	else:
		joint_jitter_degrees = _apply_episode_pose_curriculum(
			curriculum_episode)

	var regular_reset_fraction := _bootstrap_curriculum_fraction()
	var use_regular_reset := _sample_regular_reset(_episode_rng)
	if not use_regular_reset and not bootstrap_pose_joints.is_empty():
		_active_bootstrap_index = _episode_rng.randi_range(
			0, bootstrap_pose_joints.size() - 1)
	var sampled_target := _sample_target_transform(_episode_rng, spawn_pool)
	if per_region_gate_enabled and use_regular_reset:
		_apply_per_region_gate(str(_last_target_sample_info.get("target_region", "")))
	var sample_info := _last_target_sample_info.duplicate(true)
	if not use_regular_reset:
		sample_info = {
			"target_sampling": "bootstrap",
			"target_region": "bootstrap",
			"target_cell": -1,
		}
	var reset_info := {
		"task_reset_mode": "regular" if use_regular_reset else "bootstrap",
		"regular_reset": use_regular_reset,
		"regular_reset_fraction": regular_reset_fraction,
		"curriculum_level": _curriculum_level_override,
		"curriculum_episode": curriculum_episode,
	}
	reset_info.merge(sample_info, true)
	controller.merge_agent_reset_info(arm, reset_info)
	target.global_transform = _sample_reset_target_transform(
		sampled_target, _episode_rng, use_regular_reset)
	arm.set_reset_joint_offsets(_sample_reset_joint_offsets(
		_episode_rng, joint_jitter_degrees, use_regular_reset))


## Override the position gate per region (called after the target region is sampled). Easy/Medium
## tighten 4cm -> 3cm -> 2cm as the curriculum level rises (mastery); Hard stays at the loose 4cm.
## Unknown regions (marker/bootstrap) are left on whatever the level-based curriculum set.
func _apply_per_region_gate(region: String) -> void:
	if use_discrete_stage_curriculum:
		var stage := _discrete_stage()
		if stage.has(region):
			var gate: Array = stage[region]
			arm.success_distance = float(gate[0])
			arm.success_angle_degrees = float(gate[1])
		return
	var level := _adaptive_level() if _has_adaptive_curriculum() else 1.0
	if region == "easy":
		if level < easy_gate_stage1_level:
			arm.success_distance = per_region_gate_loose
		elif level < easy_gate_stage2_level:
			arm.success_distance = per_region_gate_mid
		else:
			arm.success_distance = easy_success_distance
	elif region == "medium":
		if level < medium_gate_stage1_level:
			arm.success_distance = per_region_gate_loose
		elif level < medium_gate_stage2_level:
			arm.success_distance = per_region_gate_mid
		else:
			arm.success_distance = medium_success_distance
	elif region == "hard":
		arm.success_distance = hard_success_distance


func _curriculum_target_pool_size(total_targets:int) -> int:
	var total := maxi(total_targets, 1)
	var easy := clampi(easy_target_count, 1, total)
	if _has_adaptive_curriculum():
		var start_level := clampf(
			adaptive_target_pool_start_level, 0.0, 1.0)
		var full_level := clampf(
			adaptive_target_pool_full_level, start_level, 1.0)
		if full_level <= start_level:
			return total if _adaptive_level() >= full_level else easy
		var adaptive_fraction := clampf(
			(_adaptive_level() - start_level)
			/ (full_level - start_level),
			0.0,
			1.0)
		return clampi(
			roundi(lerpf(float(easy), float(total), adaptive_fraction)),
			easy,
			total)
	if target_pool_curriculum_full_episode <= target_pool_curriculum_start_episode:
		return total
	var fraction := clampf(
		float(_curriculum_episode() - target_pool_curriculum_start_episode)
		/ float(
			target_pool_curriculum_full_episode
			- target_pool_curriculum_start_episode),
		0.0, 1.0)
	return clampi(roundi(lerpf(float(easy), float(total), fraction)), easy, total)


func _curriculum_orientation_weight() -> float:
	if use_discrete_stage_curriculum:
		return 0.10
	if _has_adaptive_curriculum():
		return (
			clampf(orientation_progress_final_weight, 0.0, 1.0)
			if _adaptive_level() >= adaptive_orientation_stage_level
			else clampf(adaptive_orientation_initial_weight, 0.0, 1.0)
		)
	if (
		orientation_progress_curriculum_full_episode
		<= orientation_progress_curriculum_start_episode
	):
		return clampf(orientation_progress_final_weight, 0.0, 1.0)
	var fraction := clampf(
		float(_curriculum_episode() - orientation_progress_curriculum_start_episode)
		/ float(
			orientation_progress_curriculum_full_episode
			- orientation_progress_curriculum_start_episode),
		0.0, 1.0)
	return lerpf(
		clampf(orientation_progress_start_weight, 0.0, 1.0),
		clampf(orientation_progress_final_weight, 0.0, 1.0),
		fraction)


func _has_bootstrap_pose() -> bool:
	if not bootstrap_pose_joints.is_empty():
		return true
	return bootstrap_target_pose != null and not bootstrap_joint_positions.is_empty()


func _bootstrap_curriculum_fraction() -> float:
	if not _has_bootstrap_pose():
		return 1.0
	if _has_adaptive_curriculum():
		var minimum := clampf(
			bootstrap_min_regular_reset_fraction, 0.0, 1.0)
		var full_level := clampf(
			adaptive_bootstrap_full_level, 0.0001, 1.0)
		var level_fraction := clampf(
			_adaptive_level() / full_level, 0.0, 1.0)
		# Also fade over EPISODES so a promotion level stuck on the untaught home-start reach does not
		# freeze the assist. Uses _training_episode (REAL global count), NOT _curriculum_episode() which
		# maps to the stuck level in adaptive mode. The regular (home-start) share grows with whichever
		# schedule is further along, so the arm keeps getting more real reach practice.
		var ep_start := bootstrap_curriculum_start_episode
		var ep_full := maxi(bootstrap_curriculum_full_episode, ep_start + 1)
		var ep_fraction := clampf(
			float(_training_episode - ep_start)
			/ float(ep_full - ep_start), 0.0, 1.0)
		return lerpf(minimum, 1.0, maxf(level_fraction, ep_fraction))
	if bootstrap_curriculum_full_episode <= bootstrap_curriculum_start_episode:
		return 1.0
	var fraction := clampf(
		float(_curriculum_episode() - bootstrap_curriculum_start_episode)
		/ float(
			bootstrap_curriculum_full_episode
			- bootstrap_curriculum_start_episode),
		0.0,
		1.0)
	# Floor the home-start (regular) fraction so the arm always practises some reaching, even at
	# level 0. Without this, level 0 is 100% bootstrap (pure holding) and the policy never learns to
	# travel home->target, so it collapses the moment the assist fades.
	return maxf(fraction, clampf(bootstrap_min_regular_reset_fraction, 0.0, 1.0))


func _has_adaptive_curriculum() -> bool:
	return _curriculum_level_override >= 0.0


func _adaptive_level() -> float:
	return clampf(_curriculum_level_override, 0.0, 1.0)


func _apply_adaptive_pose_curriculum() -> float:
	if use_discrete_stage_curriculum:
		var stage := _discrete_stage()
		arm.success_hold_physics_frames = maxi(int(stage["hold"]), 1)
		arm.success_max_joint_speed = maxf(float(stage["speed"]), 0.001)
		return 0.0
	var level := _adaptive_level()
	if level < adaptive_reach_stage1_level:
		arm.success_distance = adaptive_reach_stage_a_distance
		arm.success_angle_degrees = adaptive_reach_stage_a_angle
	elif level < adaptive_reach_stage2_level:
		arm.success_distance = 0.06
		arm.success_angle_degrees = 38.0
	elif level < adaptive_reach_stage3_level:
		arm.success_distance = 0.05
		arm.success_angle_degrees = 32.0
	else:
		arm.success_distance = 0.04
		arm.success_angle_degrees = 28.0

	# Hold/stillness tightens with the curriculum level so late-stage success requires a longer,
	# visibly still hold ON the target (not a ~0.33s touch at 0.30 joint speed). Early levels keep
	# the loose export values so learning stays easy; upper levels demand ~zero velocity + long hold.
	if level < adaptive_reach_stage1_level:
		arm.success_hold_physics_frames = maxi(adaptive_hold_physics_frames, 1)
		arm.success_max_joint_speed = maxf(adaptive_hold_max_joint_speed, 0.001)
	elif level < adaptive_reach_stage2_level:
		arm.success_hold_physics_frames = 40
		arm.success_max_joint_speed = 0.05
	elif level < adaptive_reach_stage3_level:
		arm.success_hold_physics_frames = 80
		arm.success_max_joint_speed = 0.04
	else:
		arm.success_hold_physics_frames = 120
		arm.success_max_joint_speed = 0.02
	return 0.0


func _apply_episode_pose_curriculum(curriculum_episode:int) -> float:
	var joint_jitter_degrees := 0.0
	if curriculum_episode < reach_curriculum_stage1_until:
		arm.success_distance = 0.08
		arm.success_angle_degrees = 35.0
	elif curriculum_episode < reach_curriculum_stage2_until:
		arm.success_distance = 0.06
		arm.success_angle_degrees = 25.0
	elif curriculum_episode < reach_curriculum_stage3_until:
		joint_jitter_degrees = intermediate_joint_jitter_degrees
		arm.success_distance = 0.04
		arm.success_angle_degrees = 20.0
	else:
		joint_jitter_degrees = late_joint_jitter_degrees
		arm.success_distance = 0.02
		arm.success_angle_degrees = 20.0

	if curriculum_episode < hold_curriculum_stage1_until:
		arm.success_hold_physics_frames = 20
		arm.success_max_joint_speed = 0.10
	elif curriculum_episode < hold_curriculum_stage2_until:
		arm.success_hold_physics_frames = 60
		arm.success_max_joint_speed = 0.05
	else:
		arm.success_hold_physics_frames = 120
		arm.success_max_joint_speed = 0.02
	return joint_jitter_degrees


func _sample_regular_reset(rng: RandomNumberGenerator) -> bool:
	var fraction := _bootstrap_curriculum_fraction()
	return fraction >= 1.0 or (fraction > 0.0 and rng.randf() < fraction)


func _sample_reset_target_transform(
		sampled_target: Transform3D,
		rng: RandomNumberGenerator,
		use_regular_reset: bool) -> Transform3D:
	if use_regular_reset:
		return sampled_target

	var result: Transform3D
	if not bootstrap_pose_targets.is_empty():
		var idx := clampi(
			_active_bootstrap_index, 0, bootstrap_pose_targets.size() - 1)
		result = Transform3D(Basis.IDENTITY, bootstrap_pose_targets[idx])
	else:
		result = bootstrap_target_pose.global_transform
	var jitter_radius := maxf(bootstrap_target_jitter_radius, 0.0)
	if jitter_radius > 0.0:
		result.origin += Vector3(
			rng.randf_range(-jitter_radius, jitter_radius),
			rng.randf_range(-jitter_radius, jitter_radius),
			rng.randf_range(-jitter_radius, jitter_radius))
	return result


func _sample_target_transform(
	rng: RandomNumberGenerator,
	spawn_pool: Array[Marker3D]) -> Transform3D:
	var target_index := rng.randi_range(
		0,
		maxi(mini(_target_pool_size, spawn_pool.size()) - 1, 0))
	var result := spawn_pool[target_index].global_transform
	if _should_sample_target_regions():
		var region := _select_target_region(rng)
		if region and region.has_method("sample_transform"):
			var forced_cell := _demo_forced_cell if _demo_mode else -1
			var sampled: Dictionary = region.call(
				"sample_transform", rng, result.basis, forced_cell)
			if sampled.get("valid", true) == false:
				push_error(
					"[openarm_scenario] region '%s' produced an invalid sample "
					% region.name
					+ "(broken allowlist/shape). Terminating (fail-closed).")
				get_tree().quit(1)
				return result
			var sampled_transform: Variant = sampled.get("transform")
			if sampled_transform is Transform3D:
				var sampled_region := str(sampled.get(
					"region", region.name)).strip_edges().to_lower()
				var sampled_cell := int(sampled.get("cell", -1))
				_last_target_sample_info = {
					"target_sampling": "stratified_region",
					"target_region": sampled_region,
					"target_cell": sampled_cell,
					# Deterministic identity of the exact pose (region+cell+reset seed). Demo train/val
					# split by this id so no trajectory is ever shared across the two sets.
					"target_pose_id": "%s:%d:%d" % [
						sampled_region, sampled_cell, _current_reset_seed],
					"target_coverage_cycle": int(sampled.get(
						"coverage_cycle", 0)),
					"target_coverage_fraction": float(sampled.get(
						"coverage_fraction", 0.0)),
				}
				var weights := _adaptive_region_weights()
				_last_target_sample_info["target_region_weights"] = [
					weights.x,
					weights.y,
					weights.z,
				]
				# Expected active regions this stage, so a frozen eval can fail closed (a required
				# region absent from the results -> selection_success_rate 0, not min-over-observed).
				var active_names: Array = []
				for region_node in _active_regions():
					active_names.append(str(region_node.region_name).to_lower())
				_last_target_sample_info["active_regions"] = active_names
				result = sampled_transform
				return _apply_target_orientation_randomization(rng, result)

	_last_target_sample_info = {
		"target_sampling": "marker",
		"target_region": "marker",
		"target_cell": target_index,
	}
	if (
		continuous_target_sampling
		and _curriculum_episode() >= continuous_target_sampling_start_episode
	):
		result.origin = _sample_target_position(rng, spawn_pool, result.origin)
	return _apply_target_orientation_randomization(rng, result)


func _apply_target_orientation_randomization(
	rng: RandomNumberGenerator,
	target_transform: Transform3D
) -> Transform3D:
	var result := target_transform
	if target_yaw_follows_azimuth:
		result.basis = (
			Basis(Vector3.UP, _target_azimuth_yaw(result.origin)) * result.basis)
	if (
		target_yaw_randomization_degrees > 0.0
		and _curriculum_episode() >= target_yaw_randomization_start_episode
	):
		var yaw := deg_to_rad(rng.randf_range(
			-target_yaw_randomization_degrees,
			target_yaw_randomization_degrees))
		result.basis = Basis(Vector3.UP, yaw) * result.basis
	# Right-multiplied so tilt and roll are about the target's own axes, i.e. relative to the yaw
	# already applied. Basis(BACK, -p) takes +X to (cos p, -sin p, 0): the approach axis dips.
	# Roll then spins about that dipped approach axis.
	var pitch := target_pitch_degrees
	if target_pitch_randomization_degrees > 0.0:
		pitch += rng.randf_range(
			-target_pitch_randomization_degrees, target_pitch_randomization_degrees)
	pitch = clampf(pitch, 0.0, 90.0)
	if not is_zero_approx(pitch):
		result.basis = result.basis * Basis(Vector3.BACK, deg_to_rad(-pitch))
	if target_roll_randomization_degrees > 0.0:
		var roll := rng.randf_range(
			-target_roll_randomization_degrees, target_roll_randomization_degrees)
		result.basis = result.basis * Basis(Vector3.RIGHT, deg_to_rad(roll))
	return result


## Yaw that sweeps the demanded pose with the target's azimuth around the arm base, measured from
## target_azimuth_reference_degrees. Basis(UP, t) * RIGHT == (cos t, 0, -sin t), hence the negated
## z. The gain sets how much of the sweep is applied and, when negative, its direction.
func _target_azimuth_yaw(target_position: Vector3) -> float:
	if arm == null:
		return 0.0
	var offset := target_position - (arm as Node3D).global_position
	if absf(offset.x) < 0.0001 and absf(offset.z) < 0.0001:
		return 0.0
	var azimuth := atan2(-offset.z, offset.x)
	var reference := deg_to_rad(target_azimuth_reference_degrees)
	return wrapf(azimuth - reference, -PI, PI) * target_azimuth_yaw_gain


func _should_sample_target_regions() -> bool:
	return (
		use_adaptive_target_regions
		and _has_adaptive_curriculum()
		and easy_target_region != null
		and easy_target_region.has_method("sample_transform"))


func _adaptive_region_weights() -> Vector3:
	if use_discrete_stage_curriculum:
		return _discrete_stage()["weights"]
	var level := _adaptive_level()
	var medium_start := clampf(
		adaptive_medium_region_start_level, 0.0, 1.0)
	var medium_full := clampf(
		adaptive_medium_region_full_level, medium_start, 1.0)
	var hard_start := clampf(
		adaptive_hard_region_start_level, medium_full, 1.0)
	var hard_full := clampf(
		adaptive_hard_region_full_level, hard_start, 1.0)

	if level < medium_start:
		return Vector3(1.0, 0.0, 0.0)
	if level < hard_start:
		var medium_fraction := lerpf(
			clampf(adaptive_medium_region_initial_weight, 0.0, 0.5),
			0.5,
			_range_fraction(level, medium_start, medium_full))
		return Vector3(1.0 - medium_fraction, medium_fraction, 0.0)

	var final_weights := _normalized_region_weights(
		adaptive_final_region_weights)
	var hard_fraction := _range_fraction(level, hard_start, hard_full)
	var hard_weight := lerpf(
		clampf(adaptive_hard_region_initial_weight, 0.0, 1.0),
		final_weights.z,
		hard_fraction)
	var remaining_weight := 1.0 - hard_weight
	var final_planar_weight := final_weights.x + final_weights.y
	var final_easy_ratio := (
		final_weights.x / final_planar_weight
		if final_planar_weight > 0.0
		else 0.5)
	var easy_ratio := lerpf(0.5, final_easy_ratio, hard_fraction)
	return _normalized_region_weights(Vector3(
		remaining_weight * easy_ratio,
		remaining_weight * (1.0 - easy_ratio),
		hard_weight))


func _range_fraction(value: float, start: float, end: float) -> float:
	if end <= start:
		return 1.0 if value >= end else 0.0
	return clampf((value - start) / (end - start), 0.0, 1.0)


## Discrete reach-hold curriculum: level -> stage index (0.0->A .. 1.0->F, step 0.2).
func _stage_index_for_level(level: float) -> int:
	return clampi(int(round(level / 0.2)), 0, 5)


## Fixed stage table: per-stage training region weights, global hold/speed, and per-region
## [success_distance_m, success_angle_deg]. Regions with zero weight in a stage are never sampled,
## so their gate values are placeholders. The reward contract (orientation weight 0.10) is unchanged.
func _discrete_stage() -> Dictionary:
	var stages := [
		{"name": "A", "weights": Vector3(1, 0, 0), "hold": 20, "speed": 0.30,
			"easy": [0.04, 45.0], "medium": [0.04, 45.0], "hard": [0.04, 45.0]},
		{"name": "B", "weights": Vector3(1, 0, 0), "hold": 30, "speed": 0.25,
			"easy": [0.03, 30.0], "medium": [0.03, 30.0], "hard": [0.04, 30.0]},
		{"name": "C", "weights": Vector3(1, 0, 0), "hold": 60, "speed": 0.15,
			"easy": [0.02, 20.0], "medium": [0.03, 30.0], "hard": [0.04, 30.0]},
		{"name": "D", "weights": Vector3(0.5, 0.5, 0), "hold": 40, "speed": 0.20,
			"easy": [0.02, 20.0], "medium": [0.04, 45.0], "hard": [0.04, 30.0]},
		{"name": "E", "weights": Vector3(0.5, 0.5, 0), "hold": 60, "speed": 0.15,
			"easy": [0.02, 20.0], "medium": [0.03, 30.0], "hard": [0.04, 30.0]},
		{"name": "F", "weights": Vector3(0.2, 0.4, 0.4), "hold": 60, "speed": 0.15,
			"easy": [0.02, 20.0], "medium": [0.03, 30.0], "hard": [0.04, 30.0]},
	]
	var idx := _stage_index_for_level(_adaptive_level())
	var stage: Dictionary = stages[idx]
	stage["index"] = idx
	return stage


func _normalized_region_weights(weights: Vector3) -> Vector3:
	var clean := Vector3(
		maxf(weights.x, 0.0) if easy_target_region else 0.0,
		maxf(weights.y, 0.0) if medium_target_region else 0.0,
		maxf(weights.z, 0.0) if hard_target_region else 0.0)
	var total := clean.x + clean.y + clean.z
	if total <= 0.0:
		return Vector3(1.0, 0.0, 0.0)
	return clean / total


## Active (training-weighted) regions for the current stage, in fixed easy/medium/hard order.
func _active_regions() -> Array:
	var w := _adaptive_region_weights()
	var out: Array = []
	if easy_target_region and w.x > 0.0 and easy_target_region.has_method("sample_transform"):
		out.append(easy_target_region)
	if medium_target_region and w.y > 0.0 and medium_target_region.has_method("sample_transform"):
		out.append(medium_target_region)
	if hard_target_region and w.z > 0.0 and hard_target_region.has_method("sample_transform"):
		out.append(hard_target_region)
	return out


## Frozen-evaluation region deck: every active region appears an EQUAL number of times (a
## seed-shuffled full cycle before any region repeats), independent of the training weights. Cell
## stratification is handled by TargetSamplingRegion3D. Deterministic given the eval seed.
func _select_eval_region(rng: RandomNumberGenerator) -> Area3D:
	var active := _active_regions()
	if active.is_empty():
		return null
	if _eval_region_deck_index >= _eval_region_deck.size():
		_eval_region_deck = active.duplicate()
		for i in range(_eval_region_deck.size() - 1, 0, -1):
			var j := rng.randi_range(0, i)
			var tmp = _eval_region_deck[i]
			_eval_region_deck[i] = _eval_region_deck[j]
			_eval_region_deck[j] = tmp
		_eval_region_deck_index = 0
	var region: Area3D = _eval_region_deck[_eval_region_deck_index]
	_eval_region_deck_index += 1
	return region


func _demo_region() -> Area3D:
	match _demo_forced_region:
		"medium":
			return medium_target_region
		"hard":
			return hard_target_region
		_:
			return easy_target_region


func _select_target_region(rng: RandomNumberGenerator) -> Area3D:
	if _demo_mode:
		return _demo_region()
	if _evaluation_mode:
		return _select_eval_region(rng)
	var weights := _normalized_region_weights(_adaptive_region_weights())
	var candidates: Array[Area3D] = []
	var candidate_weights: Array[float] = []
	for entry in [
		[easy_target_region, weights.x],
		[medium_target_region, weights.y],
		[hard_target_region, weights.z],
	]:
		var region := entry[0] as Area3D
		var weight := float(entry[1])
		if region and weight > 0.0 and region.has_method("sample_transform"):
			candidates.append(region)
			candidate_weights.append(weight)
	if candidates.is_empty():
		return null

	var signature_parts: Array[String] = []
	for index in range(candidates.size()):
		signature_parts.append(
			"%d:%.5f" % [
				candidates[index].get_instance_id(),
				candidate_weights[index],
			])
	var signature := "|".join(signature_parts)
	if (
		_region_schedule_signature != signature
		or _region_schedule_cursor >= _region_schedule.size()
	):
		_build_region_schedule(
			rng, candidates, candidate_weights, signature)
	var selected := _region_schedule[_region_schedule_cursor]
	_region_schedule_cursor += 1
	return selected


func _build_region_schedule(
	rng: RandomNumberGenerator,
	candidates: Array[Area3D],
	weights: Array[float],
	signature: String
) -> void:
	_region_schedule.clear()
	_region_schedule_cursor = 0
	_region_schedule_signature = signature
	var schedule_size := maxi(target_region_schedule_size, candidates.size())
	var total_weight := 0.0
	for weight in weights:
		total_weight += weight
	var counts: Array[int] = []
	var remainders: Array[float] = []
	var assigned := 0
	for weight in weights:
		var exact := (weight / total_weight) * float(schedule_size)
		var count := floori(exact)
		counts.append(count)
		remainders.append(exact - float(count))
		assigned += count
	while assigned < schedule_size:
		var best_index := 0
		for index in range(1, remainders.size()):
			if remainders[index] > remainders[best_index]:
				best_index = index
		counts[best_index] += 1
		remainders[best_index] = -1.0
		assigned += 1
	for index in range(candidates.size()):
		for _slot in range(counts[index]):
			_region_schedule.append(candidates[index])
	for index in range(_region_schedule.size() - 1, 0, -1):
		var swap_index := rng.randi_range(0, index)
		var temporary := _region_schedule[index]
		_region_schedule[index] = _region_schedule[swap_index]
		_region_schedule[swap_index] = temporary


func _relocate_target() -> void:
	var spawn_pool := _target_spawn_pool()
	if spawn_pool.is_empty():
		_target_relocation_step = -1
		return
	var candidate := target.global_transform
	for _attempt in range(8):
		candidate = _sample_target_transform(_episode_rng, spawn_pool)
		if (
			candidate.origin.distance_to(target.global_position)
			>= maxf(minimum_target_relocation_distance, 0.0)
		):
			break
	target.global_transform = candidate
	_target_relocation_step = -1
	arm.notify_target_pose_relocated()


func _sample_target_position(
		rng: RandomNumberGenerator,
		spawn_pool: Array[Marker3D],
		anchor: Vector3) -> Vector3:
	var lower := spawn_pool[0].global_position
	var upper := lower
	for marker in spawn_pool:
		lower = lower.min(marker.global_position)
		upper = upper.max(marker.global_position)

	var inset := Vector3(
		maxf(target_sampling_inset.x, 0.0),
		maxf(target_sampling_inset.y, 0.0),
		maxf(target_sampling_inset.z, 0.0))
	for axis in range(3):
		var half_extent := (upper[axis] - lower[axis]) * 0.5
		var axis_inset := minf(inset[axis], half_extent)
		lower[axis] += axis_inset
		upper[axis] -= axis_inset

	# Spatial curriculum: sample within a GROWING radius around the anchor marker instead of the
	# full volume at once. frac ramps 0->1 across [start, full]; the per-axis radius lerps from the
	# tight start radius (targets essentially on the learned markers) to the axis half-extent (full
	# volume). Clamped to the marker AABB so the target never leaves the workspace or drops below the
	# table (lower.y sits above it). full<=start disables the ramp (frac=1 -> full volume at once).
	var frac := 1.0
	if spatial_curriculum_full_episode > spatial_curriculum_start_episode:
		frac = clampf(
			float(_curriculum_episode() - spatial_curriculum_start_episode)
			/ float(spatial_curriculum_full_episode - spatial_curriculum_start_episode),
			0.0, 1.0)
	var out_position := Vector3()
	for axis in range(3):
		var half_extent := (upper[axis] - lower[axis]) * 0.5
		var radius := lerpf(spatial_curriculum_start_radius, half_extent, frac)
		var sampled := anchor[axis] + rng.randf_range(-radius, radius)
		out_position[axis] = clampf(sampled, lower[axis], upper[axis])
	return out_position


func _sample_joint_offsets(rng:RandomNumberGenerator, max_degrees:float) -> Array[float]:
	var offsets: Array[float] = []
	offsets.resize(arm.get_joint_count())
	for index in range(offsets.size()):
		offsets[index] = deg_to_rad(rng.randf_range(-max_degrees, max_degrees))
	return offsets


func _sample_reset_joint_offsets(
			rng: RandomNumberGenerator,
			regular_jitter_degrees: float,
			use_regular_reset: bool) -> Array[float]:
	if use_regular_reset:
		return _sample_joint_offsets(rng, regular_jitter_degrees)

	var offsets: Array[float] = []
	var joint_count: int = arm.get_joint_count()
	offsets.resize(joint_count)
	var home_positions: PackedFloat32Array = arm.joint_home_positions
	var jitter_degrees := maxf(bootstrap_joint_jitter_degrees, 0.0)
	var active_joints := bootstrap_joint_positions
	if not bootstrap_pose_joints.is_empty():
		active_joints = bootstrap_pose_joints[clampi(
			_active_bootstrap_index, 0, bootstrap_pose_joints.size() - 1)]
	var reverse_t := _sample_reverse_reset_fraction(rng)
	for index in range(joint_count):
		var home := (
			float(home_positions[index])
			if index < home_positions.size()
			else 0.0)
		var demo := (
			float(active_joints[index])
			if index < active_joints.size()
			else home)
		# Interpolate the start config between home and the demo pose (spatial reverse curriculum).
		var reference := home + reverse_t * (demo - home)
		var absolute_position := reference
		absolute_position += deg_to_rad(rng.randf_range(
			-jitter_degrees, jitter_degrees))
		offsets[index] = absolute_position - home
	return offsets


## Reverse-curriculum start distance: 1.0 = reset AT the demo pose (EE on the target, tiny reach),
## 0.0 = reset at HOME (full reach). At level 0 the window is [1,1] (all at the demo); it widens toward
## home as the level rises to reverse_curriculum_full_level, so the arm learns the collision-free
## approach corridor from the target backward.
func _sample_reverse_reset_fraction(rng: RandomNumberGenerator) -> float:
	# Demo recording must start from HOME (full reach), not the reverse-curriculum
	# near-target reset -- otherwise the demos teach short reaches the policy does
	# not need. reverse_t=0 => reference = home.
	if controller != null and controller.recording_mode:
		return 0.0
	if not reverse_curriculum_enabled:
		return 0.0  # home-start training (matches home-start demos)
	if reverse_curriculum_full_level <= 0.0:
		return 1.0
	# NOTE: this used to also bail out with `not _has_adaptive_curriculum()`, a leftover from when the
	# widening was level-paced. It is episode-paced now (see below), so it needs no curriculum level --
	# and the bail-out silently pinned reverse_t at 1.0 for every run launched with
	# --no-adaptive-curriculum, and for EVERY SB3 run since that backend never sends a level at all.
	# Pinned at 1.0 the assisted resets all start exactly ON the target, so the curriculum degenerates
	# into two difficulties (on-target or full home-start) with nothing in between -- the precise thing
	# a reverse curriculum exists to avoid.
	# EPISODE-paced (not level-paced) so a promotion level stuck on the untaught home-start reach does
	# not freeze the reverse curriculum. Use _training_episode (the REAL global episode count) - NOT
	# _curriculum_episode(), which in adaptive mode returns level*full_episode and so stays frozen with
	# the stuck level, defeating the whole point. t_low walks 1.0 -> 0.0 across [start, full] episodes.
	var start := reverse_curriculum_start_episode
	var full := maxi(reverse_curriculum_full_episode, start + 1)
	var frac := clampf(
		float(_training_episode - start) / float(full - start), 0.0, 1.0)
	var t_low := clampf(1.0 - frac, 0.0, 1.0)
	return rng.randf_range(t_low, 1.0)


func _on_target_reached() -> void:
	goal_event.trigger(str(arm.name))
	if (
		_training_mode
		and relocate_target_during_training
		and _curriculum_episode() >= target_relocation_start_episode
	):
		var minimum_delay := mini(
			target_relocation_delay_steps_min,
			target_relocation_delay_steps_max)
		var maximum_delay := maxi(
			target_relocation_delay_steps_min,
			target_relocation_delay_steps_max)
		_target_relocation_step = (
			controller.step_count
			+ _episode_rng.randi_range(minimum_delay, maximum_delay))


func _on_target_pose_relocated() -> void:
	controller.rebase_agent_tracking(arm)


func _on_obstacle_collision() -> void:
	# Route self-collisions (wrist/fingers into the arm's own body) to a distinct, harsher event
	# than environment collisions (table). Self-collision is a dumber mistake and should be
	# penalised more; both stay terminal via TerminalFailure's failure_terminal_reasons.
	var source := str(arm.get_last_collision_details().get("source", ""))
	if source == "self_body":
		self_collision_event.trigger(str(arm.name))
	else:
		collision_event.trigger(str(arm.name))
