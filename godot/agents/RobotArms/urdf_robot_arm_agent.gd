extends Node3D
class_name URDFRobotArmAgentBody

signal target_reached
signal obstacle_collision
signal target_pose_relocated

enum DistanceProgressMode {
	LINEAR_CLAMPED,
	INVERSE_DISTANCE,
}

enum PoseProgressMode {
	ADDITIVE,
	POSITION_GATED,
}

@export_category("URDF Robot")
@export var robot_path: NodePath = NodePath("urdf")
@export var controlled_joint_names := PackedStringArray()
## Uses a deterministic high-level joint-velocity servo: commands are integrated within URDF
## limits and links follow the resulting pose exactly. This is the default for reaching and
## trajectory policies, where the physical robot already owns the low-level motor loops.
## Disable only for a deliberately calibrated rigid-body/motor simulation; doing so changes the
## task dynamics and invalidates policies trained with kinematic control.
@export var use_kinematic_control := true
@export var joint_home_positions := PackedFloat32Array()
@export var default_joint_speed := 0.5
## Hard cap (rad/s) applied on top of each joint's URDF velocity limit. Zero disables it, keeping
## the raw URDF limit (the original behaviour). Set this when a URDF ships very high limits
## (e.g. 5-20 rad/s) but the reward/stillness thresholds were tuned for a ~1 rad/s regime: without
## the cap the policy controls a joint an order of magnitude faster than the still/hold gates expect.
@export var max_joint_speed_override := 0.0
## Tapers commands that push a bounded revolute joint toward its hard stop. The value is a
## fraction of that joint's complete URDF range; zero keeps only the hard stop at the limit.
@export_range(0.0, 0.25, 0.005) var joint_limit_slowdown_ratio := 0.05

@export_category("Tool Center Point")
@export var tcp_link_name := ""
@export var tcp_local_offset := Vector3.ZERO
@export var end_effector_path: NodePath = NodePath("EndEffector")
## Optional calibrated frame below EndEffector. Its transform maps the URDF hand axes to the
## canonical grasp axes used by the target pose.
@export var tool_pose_path: NodePath = NodePath("EndEffector/ToolPose")

@export_category("Task")
@export var target: Node3D
## Optional desired pose, normally a Marker3D below target. Falls back to target itself.
@export var target_pose: Node3D
@export var workspace_scale := 1.0
@export var success_distance := 0.05
@export_range(0.1, 180.0, 0.1) var success_angle_degrees := 12.0
@export var success_hold_physics_frames := 30
@export var terminate_on_success := true
@export_range(1.0, 10.0, 0.1) var success_rearm_distance_multiplier := 1.5
@export_range(1.0, 10.0, 0.1) var success_rearm_angle_multiplier := 1.5
## Reach-and-HOLD: success also requires the arm to be nearly still (maximum |joint speed| below
## the threshold), not just within distance. Without this, the arm can sweep THROUGH the target
## and "succeed" transiently, so it never learns to stop/hold (CAPS keeps a constant velocity
## smooth). Set success_require_still=false for a pure reach (no hold).
@export var success_require_still := true
@export var success_max_joint_speed := 0.15
## Reach-and-HOLD shaping: the "near target" zone (metres) where the hold reward/penalty terms
## activate. Away from the target they are zero so the fast reach stays unpenalised.
@export var near_target_distance := 0.20
## SEPARATE, much tighter distance that gates the stillness rewards (hold_stillness + near-target
## speed penalty). Sharing near_target_distance let the arm collect ~0.1/step for stopping at 19cm
## without ever reaching the 4cm success threshold -- a "barely enter the zone and camp still"
## exploit. Stillness must only pay right at the target; pose shaping (near_target_distance) may
## start much further out. Tie to success_distance-ish.
@export var hold_activation_distance := 0.06
## Orientation range over which pose tracking shaping fades to zero.
@export_range(1.0, 180.0, 0.1) var pose_reward_angle_degrees := 60.0
## Relative contribution of orientation to get_progress().
## Progress (drives the telescoping ProgressDelta approach reward) mixes position + orientation.
## Empirically THIS is what makes the arm learn orientation: it gives the orientation error a
## dense approach signal over the full range from far away. Setting it to 0 (position-only) killed
## orientation learning (arm nailed position, left orientation ~140deg). Position stays dominant.
@export_range(0.0, 1.0, 0.01) var orientation_progress_weight := 0.25
## Selects how distance becomes position progress. LINEAR_CLAMPED preserves the original
## 1-distance/scale behavior. INVERSE_DISTANCE remains informative beyond workspace_scale instead
## of becoming exactly zero when the tool moves farther away than expected.
@export var distance_progress_mode := DistanceProgressMode.LINEAR_CLAMPED
## ADDITIVE preserves the original position/orientation blend. POSITION_GATED makes position
## mandatory: orientation can refine progress, but can never replace approaching the target.
@export var pose_progress_mode := PoseProgressMode.ADDITIVE
## Joint speed (rad/s) at which the stillness reward decays to zero.
@export var hold_stillness_speed_reference := 0.5
## A target transform change larger than either threshold starts a new acquisition without
## requiring an episode reset. Small rigid-body jitter is ignored.
@export var target_relocation_position_epsilon := 0.001
@export_range(0.0, 30.0, 0.1) var target_relocation_angle_epsilon_degrees := 1.0

@export_category("Safety")
@export var safety_volumes: Array[Area3D] = []
@export var obstacle_group := "robot_obstacle"
## When true a detected collision terminates the episode and stops the joints. When false the collision
## is still recorded and penalised (via the obstacle_collision signal / reward) but the episode
## CONTINUES - so a from-scratch policy can graze an obstacle and recover, learning the collision-free
## approach corridor instead of dying on first contact. Manual control never terminates regardless.
@export var collision_terminates := true
@export var auto_detect_environment_collisions := true
@export_flags_3d_physics var environment_collision_mask := 0xFFFFFFFF
@export_range(1, 64, 1) var max_collision_results_per_shape := 8
## Self-body collision: flag (and terminate, like an obstacle hit) when a configured arm link
## overlaps the robot's OWN central body (e.g. the torso base link). Unlike environment collisions
## this targets specific self links so the arm learns not to fold into its own body. List only the
## DISTAL arm links in self_check_link_names — the shoulder links that always touch the body must be
## left out or they would false-trigger every frame. Reuses the same collision event/penalty path.
@export var self_body_collision_enabled := false
@export var self_body_link_names: PackedStringArray = []
@export var self_check_link_names: PackedStringArray = []
## Prints the first detected self-body overlap of each episode. Useful while calibrating a new URDF.
@export var self_collision_debug := false
## Prints the first collision of each episode, including source, querying link and collider.
@export var collision_debug := false

@export_category("Manual Control")
@export var manual_control := false
## Normalized velocity applied while using the selected-joint keyboard controls.
@export_range(0.01, 1.0, 0.01) var manual_command_scale := 0.20
## Multiplier applied while Shift is held for precise final alignment.
@export_range(0.01, 1.0, 0.01) var manual_fine_scale := 0.20
@export_range(0, 8, 1) var manual_selected_joint := 0
## Additional independent actuators available for calibration but excluded from the
## policy action and observation contract. Mimic followers must not be listed here.
@export var manual_extra_joint_names := PackedStringArray()
## Independent gripper actuator controlled directly with the manual close/open actions.
## Leave empty when the robot does not have a gripper.
@export var manual_gripper_joint_name: StringName = &""
@export var manual_debug_overlay := true
@export_dir var manual_capture_directory := "user://manual_arm_captures"
## Optional region (Area3D / Node3D with a BoxShape3D collision child) that the J key samples to place
## a fresh random target, for capturing reference poses spread across the region. The scenario wires
## this to the Easy area. If unset, J jitters the target within manual_random_target_extents of its
## current position.
@export var manual_random_target_region: Node3D
@export var manual_random_target_extents := Vector3(0.05, 0.05, 0.05)
## IK manual control: when enabled, apply_manual_action() drives the arm with a
## URDFIKController toward a target (in the RL action space) instead of joint-by-
## joint keys, so `metis record` captures smooth IK reach demos. Mode "auto"
## follows the scenario's reach target (full-coverage demos, no teleop); "teleop"
## follows ik_manual_target, which you move with W/A/S/D/Q/E + arrows/Z/X (T =
## toggle auto-orientation). ik_manual_max_speed MUST equal the per-joint max speed.
@export var ik_manual_enabled := false
@export_enum("auto", "teleop") var ik_manual_mode := "auto"
@export var ik_manual_target: Node3D
@export var ik_manual_track_orientation := true
@export_range(0.01, 5.0, 0.01) var ik_manual_max_speed := 0.5
var _ik_manual: URDFIKController = null
# Demo plan-follower (M5): a precomputed collision-free JOINT waypoint path (home->goal) that the
# arm tracks CLOSED-LOOP by emitting joint_velocity actions through the normal apply_action path.
# Empty = inactive (dormant during RL; takes precedence over ik_manual only while a plan is set).
var _demo_plan: Array = []
var _demo_wp_index := 0
var _demo_dt_step := 0.05
## Physics frames during which collision termination is suppressed after a manual recovery.
## This lets the operator move the arm out of an already-overlapping configuration.
@export_range(1, 300, 1) var manual_recovery_grace_physics_frames := 60

@export_category("Control")
@export var auto_configure_action_size := true

@onready var agent: Agent = $Agent

var _robot: GodotRobot
var end_effector: Node3D
var tool_pose: Node3D
var _joint_names := PackedStringArray()
var _manual_joint_names := PackedStringArray()
var _commands: Array[float] = []
var _previous_action: Array[float] = []
var _manual_extra_commands: Dictionary[String, float] = {}
var _pending_reset_offsets: Array[float] = []
var _robot_collision_shapes: Array[CollisionShape3D] = []
var _robot_collision_exclusions: Array[RID] = []
var _robot_collision_link_by_shape_id: Dictionary = {}
var _self_body_rids: Array[RID] = []
var _self_check_shapes: Array[CollisionShape3D] = []
var _self_check_exclusions: Array[RID] = []
var _self_check_link_by_shape_id: Dictionary = {}
var _last_self_collision_details: Dictionary = {}
var _last_collision_details: Dictionary = {}
var _terminal := false
var _succeeded := false
var _collided := false
var _success_frames := 0
var _training_active := true
var _last_target_pose_transform := Transform3D.IDENTITY
var _has_last_target_pose_transform := false
var _manual_overlay_layer: CanvasLayer
var _manual_overlay_label: Label
var _manual_overlay_elapsed := 0.0
var _manual_collision_grace_frames := 0


func _ready() -> void:
	_robot = get_node_or_null(robot_path) as GodotRobot
	end_effector = get_node_or_null(end_effector_path) as Node3D
	tool_pose = get_node_or_null(tool_pose_path) as Node3D
	if not tool_pose:
		tool_pose = end_effector
	if not _robot:
		push_error("URDFRobotArmAgentBody: GodotRobot not found at '%s'." % robot_path)
		set_physics_process(false)
		return

	_robot.control_mode = (
		GodotRobot.ControlMode.KINEMATIC
		if use_kinematic_control
		else GodotRobot.ControlMode.PHYSICS_MOTORS)
	# Update link poses before this adapter samples the TCP and terminal state.
	_robot.process_physics_priority = process_physics_priority - 1
	_joint_names = _resolve_controlled_joint_names()
	if _joint_names.is_empty():
		push_error("URDFRobotArmAgentBody: no actuated joints configured.")
		set_physics_process(false)
		return
	_manual_joint_names = _resolve_manual_joint_names()
	if ik_manual_enabled:
		_setup_ik_manual()

	_resize_state()
	_configure_action_size()
	_connect_safety_volumes()
	_cache_robot_collision_geometry()
	_cache_self_body_geometry()
	# Imported URDF link bodies can finish registering their physics RIDs after this
	# adapter's _ready(). Refresh once the complete scene tree has entered the world.
	call_deferred("_refresh_collision_geometry")
	_robot.reset_joint_positions(_build_home_positions())
	_update_end_effector()
	_capture_target_pose_transform()
	manual_selected_joint = clampi(
		manual_selected_joint, 0, maxi(get_manual_joint_count() - 1, 0))
	if manual_control:
		_enable_manual_control()


func _refresh_collision_geometry() -> void:
	if not is_inside_tree() or not _robot:
		return
	_cache_robot_collision_geometry()
	_cache_self_body_geometry()


func _physics_process(delta: float) -> void:
	if _manual_collision_grace_frames > 0:
		_manual_collision_grace_frames -= 1
	if manual_control:
		_manual_overlay_elapsed += delta
		if _manual_overlay_elapsed >= 0.10:
			_manual_overlay_elapsed = 0.0
			_update_manual_overlay()
	if not _training_active:
		return
	# In manual control the operator drives the joints directly, so motion must NEVER freeze: success
	# or collision only set informational flags here, they must not lock the arm (that would block
	# posing/capturing more references). Motion is gated on _terminal only during autonomous training.
	if manual_control:
		apply_manual_action()
	_update_end_effector()
	_detect_target_pose_relocation()
	_check_environment_collisions()
	_check_self_body_collisions()
	_update_success_state()


func apply_action(action: Variant) -> Variant:
	if (
		(typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME)
		and str(action) == "manual"
	):
		return apply_manual_action()

	if manual_control:
		_hide_manual_overlay()
	manual_control = false
	var values := agent.decode_continuous_action(action)
	for index in range(get_joint_count()):
		var requested_command := (
			clampf(float(values[index]), -1.0, 1.0)
			if index < values.size()
			else 0.0)
		var command := _limit_aware_command(index, requested_command)
		_commands[index] = command
		_previous_action[index] = command
		_robot.set_joint_target_velocity(
			_joint_names[index], command * _joint_max_speed(index))
	return _previous_action.duplicate()


func _setup_ik_manual() -> void:
	_ik_manual = URDFIKController.new()
	_ik_manual.name = "IKManual"
	_ik_manual.joint_names = _joint_names
	_ik_manual.tcp_link_name = tcp_link_name
	_ik_manual.tcp_local_offset = tcp_local_offset
	_ik_manual.max_joint_speed = ik_manual_max_speed
	# teleop starts position-only (free wrist); press T to toggle IK wrist control
	# (it then snaps to the current wrist so you fine-tune from there). auto uses
	# the configured flag.
	_ik_manual.track_orientation = false if ik_manual_mode == "teleop" else ik_manual_track_orientation
	# teleop: FULL orientation so the operator can set the exact wrist by rotating
	# the target; auto: pointing (roll-free) to avoid the 180-degree flip.
	_ik_manual.orientation_mode = "full" if ik_manual_mode == "teleop" else "pointing"
	_ik_manual.active = false  # driven manually from apply_manual_action
	add_child(_ik_manual)
	_ik_manual.set_robot(_robot)
	if ik_manual_mode == "teleop" and ik_manual_target == null:
		ik_manual_target = _create_teleop_marker()


func _create_teleop_marker() -> Node3D:
	var marker := Node3D.new()
	marker.name = "IKTeleopTarget"
	var mesh := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.02
	sphere.height = 0.04
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1.0, 0.9, 0.1)
	mat.emission_enabled = true
	mat.emission = Color(1.0, 0.9, 0.1)
	mesh.mesh = sphere
	mesh.material_override = mat
	# small axis gnomon so the operator sees the target orientation
	marker.add_child(mesh)
	add_child(marker)
	var tcp: Node3D = _robot.get_link_node(tcp_link_name)
	if tcp != null:
		# place the marker on the TCP TIP (link origin + tcp_local_offset), which is
		# the point the IK actually drives -- not the bare link origin.
		var tp := tcp.global_transform
		marker.global_transform = Transform3D(tp.basis, tp * tcp_local_offset)
	return marker


func _snap_ik_marker_to_target() -> void:
	# Jump the teleop marker onto the scenario reach target (position) so the arm
	# drives precisely there; keeps the wrist you already set.
	if not ik_manual_enabled or ik_manual_target == null:
		return
	var tgt: Node3D = _target_pose_node()
	if tgt != null:
		ik_manual_target.global_position = tgt.global_position
		print("[ik] marker snapped to target %.3v" % tgt.global_position)


func _toggle_ik_manual_orientation() -> void:
	if not ik_manual_enabled or _ik_manual == null:
		return
	_ik_manual.track_orientation = not _ik_manual.track_orientation
	if _ik_manual.track_orientation and ik_manual_target != null:
		# snap the target orientation to the current wrist so there is no jump; you
		# then fine-tune it from here with the arrows / Z / X.
		var tcp: Node3D = _robot.get_link_node(tcp_link_name)
		if tcp != null:
			ik_manual_target.global_transform.basis = tcp.global_transform.basis.orthonormalized()
	print("[ik] wrist control (auto-orientation) = ", _ik_manual.track_orientation)


func _apply_ik_manual_action() -> Array:
	var target: Node3D = (
		_target_pose_node() if ik_manual_mode == "auto" else ik_manual_target)
	if target == null or _ik_manual == null:
		return _previous_action.duplicate()
	if ik_manual_mode == "teleop":
		_teleop_ik_target(get_physics_process_delta_time())
	_ik_manual.set_target(target)
	_ik_manual.solve(false)  # compute only; apply_action drives + records previous_action
	var applied: Array = apply_action(Array(_ik_manual.get_normalized_action()))
	manual_control = true  # AFTER apply_action, which resets manual_control to false
	return applied


func _teleop_ik_target(delta: float) -> void:
	if ik_manual_target == null:
		return
	var s: float = 0.25 if Input.is_key_pressed(KEY_SHIFT) else 1.0
	var d: float = 0.25 * delta * s
	if Input.is_key_pressed(KEY_W): ik_manual_target.global_position.x += d
	if Input.is_key_pressed(KEY_S): ik_manual_target.global_position.x -= d
	if Input.is_key_pressed(KEY_A): ik_manual_target.global_position.y += d
	if Input.is_key_pressed(KEY_D): ik_manual_target.global_position.y -= d
	if Input.is_key_pressed(KEY_Q): ik_manual_target.global_position.z += d
	if Input.is_key_pressed(KEY_E): ik_manual_target.global_position.z -= d
	var r: float = 1.2 * delta * s
	if Input.is_key_pressed(KEY_UP): ik_manual_target.rotate_x(r)
	if Input.is_key_pressed(KEY_DOWN): ik_manual_target.rotate_x(-r)
	if Input.is_key_pressed(KEY_LEFT): ik_manual_target.rotate_y(r)
	if Input.is_key_pressed(KEY_RIGHT): ik_manual_target.rotate_y(-r)
	if Input.is_key_pressed(KEY_Z): ik_manual_target.rotate_z(r)
	if Input.is_key_pressed(KEY_X): ik_manual_target.rotate_z(-r)


## Load a demo plan (list of 7-joint waypoint configs, home->goal). While set it drives
## apply_manual_action(); reset per episode by re-calling with the new plan (or [] to clear).
func set_demo_plan(waypoints: Array, physics_frames_per_step: int = 3) -> void:
	_demo_plan = waypoints if waypoints != null else []
	_demo_wp_index = 0
	_demo_dt_step = maxf(float(physics_frames_per_step) / 60.0, 0.0001)


## Closed-loop tracking of the planned waypoints: read the CURRENT joint config, command the
## joint_velocity that reaches the active waypoint in one decision step (saturated), advance when
## close. The command goes through apply_action -> set_joint_target_velocity (the normal path);
## the returned applied command IS what is recorded, so recorded action == applied action.
func _apply_demo_plan_action() -> Array:
	var count := get_joint_count()
	# Advance PAST every already-reached waypoint (home == wp[0], or a waypoint reached on the
	# previous step) BEFORE computing, so the emitted action is always a REAL move toward the next
	# DISTINCT waypoint. Advancing after computing (the old order) made the very first action a zero
	# no-op toward the home waypoint, recording a contradictory (obs_home, action=0) first transition
	# whose next_obs equalled obs_home -- impossible for a feed-forward policy to fit.
	while _demo_wp_index < _demo_plan.size() - 1:
		var wp: Array = _demo_plan[_demo_wp_index]
		var reached := true
		for i in range(count):
			var qi: float = _robot.get_joint_position(_joint_names[i])
			if absf((float(wp[i]) - qi) if i < wp.size() else 0.0) >= 0.03:
				reached = false
				break
		if reached:
			_demo_wp_index += 1
		else:
			break
	var goal: Array = _demo_plan[_demo_wp_index]
	# Per-joint velocity that would reach the waypoint in one decision step (action units).
	var raw: Array = []
	raw.resize(count)
	var max_raw := 0.0
	for i in range(count):
		var q: float = _robot.get_joint_position(_joint_names[i])
		var gap: float = (float(goal[i]) - q) if i < goal.size() else 0.0
		var vmax: float = _joint_max_speed(i)
		var r: float = (gap / (vmax * _demo_dt_step)) if vmax > 0.0 else 0.0
		raw[i] = r
		max_raw = maxf(max_raw, absf(r))
	# Scale the WHOLE velocity vector uniformly so its largest component saturates at 1 -- this keeps
	# the command collinear with (waypoint - q), so closed-loop execution traces the SAME straight
	# joint-space segment the planner validated collision-free (per-joint clamping would bend the
	# path and clip obstacles). Near the waypoint (max_raw <= 1) it eases in proportionally.
	var scale: float = (1.0 / max_raw) if max_raw > 1.0 else 1.0
	var action: Array = []
	action.resize(count)
	for i in range(count):
		action[i] = clampf(float(raw[i]) * scale, -1.0, 1.0)
	# NOTE: unlike teleop ik_manual, the plan-follower does NOT set manual_control=true.
	# manual_control makes success informational-only (the arm never terminates), which would
	# make every demo run to truncation instead of a real target_reached hold-success. Leaving it
	# false lets the normal autonomous termination (hold -> target_reached) fire, which is exactly
	# what the recorder needs to accept a demo. The velocity set by apply_action persists across
	# the decision step's physics frames, so closed-loop tracking is unaffected.
	return apply_action(action)


func apply_manual_action() -> Array:
	if not _demo_plan.is_empty():
		return _apply_demo_plan_action()
	if ik_manual_enabled and _ik_manual != null:
		return _apply_ik_manual_action()
	var values: Array = []
	values.resize(get_joint_count())
	for index in range(get_joint_count()):
		var negative := StringName("joint_%d_negative" % index)
		var positive := StringName("joint_%d_positive" % index)
		var legacy_axis := 0.0
		if InputMap.has_action(negative):
			legacy_axis -= Input.get_action_strength(negative)
		if InputMap.has_action(positive):
			legacy_axis += Input.get_action_strength(positive)
		values[index] = legacy_axis

	var selected_axis := 0.0
	var selected_name := ""
	var selected_policy_index := -1
	if get_manual_joint_count() > 0:
		selected_name = _manual_joint_names[manual_selected_joint]
		selected_policy_index = _joint_names.find(selected_name)
		selected_axis = Input.get_axis(
			"robot_joint_negative", "robot_joint_positive")
		if not is_zero_approx(selected_axis) and selected_policy_index >= 0:
			values[selected_policy_index] = selected_axis
	var gripper_axis := 0.0
	if not manual_gripper_joint_name.is_empty():
		gripper_axis = Input.get_axis(
			"robot_gripper_close", "robot_gripper_open")
		var gripper_policy_index := _joint_names.find(
			String(manual_gripper_joint_name))
		if not is_zero_approx(gripper_axis) and gripper_policy_index >= 0:
			values[gripper_policy_index] = gripper_axis

	var _scale := manual_command_scale
	if Input.is_key_pressed(KEY_SHIFT):
		_scale *= manual_fine_scale
	for index in range(values.size()):
		values[index] = clampf(float(values[index]) * _scale, -1.0, 1.0)
	var applied: Array = apply_action(values)
	for joint_name in _manual_joint_names:
		if _joint_names.has(joint_name):
			continue
		var command := 0.0
		if (
			joint_name == String(manual_gripper_joint_name)
			and not is_zero_approx(gripper_axis)
		):
			command = clampf(gripper_axis * _scale, -1.0, 1.0)
			command = _limit_aware_named_command(joint_name, command)
		elif joint_name == selected_name:
			command = clampf(selected_axis * _scale, -1.0, 1.0)
			command = _limit_aware_named_command(joint_name, command)
		_manual_extra_commands[joint_name] = command
		_robot.set_joint_target_velocity(
			joint_name, command * _joint_max_speed_for_name(joint_name))
	manual_control = true
	return applied


func _unhandled_key_input(event: InputEvent) -> void:
	if not event is InputEventKey or not event.pressed or event.echo:
		return
	var key_event := event as InputEventKey
	var keycode := (
		key_event.physical_keycode
		if key_event.physical_keycode != KEY_NONE
		else key_event.keycode)

	if keycode == KEY_F2:
		set_manual_control_enabled(not manual_control)
		get_viewport().set_input_as_handled()
		return
	if not manual_control:
		return

	if keycode >= KEY_1 and keycode <= KEY_9:
		var joint_index := int(keycode) - int(KEY_1)
		if joint_index < get_manual_joint_count():
			_select_manual_joint(joint_index)
			get_viewport().set_input_as_handled()
		return

	match keycode:
		KEY_G:
			_select_manual_gripper()
		KEY_SPACE:
			_stop_manual_motion()
		KEY_HOME:
			_reset_manual_home()
		KEY_R:
			resume_manual_from_current_pose()
		KEY_P:
			print_manual_pose_diagnostics()
		KEY_F9:
			capture_manual_reference()
		KEY_J:
			randomize_manual_target()
		KEY_T:
			_toggle_ik_manual_orientation()
		KEY_F:
			_snap_ik_marker_to_target()
		KEY_H:
			_print_manual_help()
		_:
			return
	get_viewport().set_input_as_handled()


func set_manual_control_enabled(enabled: bool) -> void:
	if enabled:
		_enable_manual_control()
	else:
		_disable_manual_control()


func _enable_manual_control() -> void:
	var was_blocked := _terminal or _collided
	manual_control = true
	set_training_active(true)
	_terminal = false
	_collided = false
	_succeeded = false
	_success_frames = 0
	if was_blocked:
		_manual_collision_grace_frames = manual_recovery_grace_physics_frames
	_stop_manual_motion()
	_ensure_manual_overlay()
	_update_manual_overlay()
	_print_manual_help()
	_print_selected_manual_joint()


func _disable_manual_control() -> void:
	_stop_manual_motion()
	manual_control = false
	_hide_manual_overlay()
	print("[METIS ARM MANUAL] Disabled.")


func _select_manual_joint(index: int) -> void:
	if get_manual_joint_count() <= 0:
		return
	manual_selected_joint = clampi(index, 0, get_manual_joint_count() - 1)
	_stop_manual_motion()
	_update_manual_overlay()
	_print_selected_manual_joint()


func _select_manual_gripper() -> void:
	if manual_gripper_joint_name.is_empty():
		return
	var index := _manual_joint_names.find(String(manual_gripper_joint_name))
	if index >= 0:
		_select_manual_joint(index)


func _stop_manual_motion() -> void:
	if _robot:
		_robot.stop_all_joints()
	for index in range(_commands.size()):
		_commands[index] = 0.0
		_previous_action[index] = 0.0
	for joint_name in _manual_extra_commands:
		_manual_extra_commands[joint_name] = 0.0
		if _robot:
			_robot.set_joint_target_velocity(joint_name, 0.0)


func _reset_manual_home() -> void:
	_pending_reset_offsets.clear()
	reset_all(transform, false)
	manual_control = true
	_manual_collision_grace_frames = 0
	_update_manual_overlay()
	print("[METIS ARM MANUAL] Reset to the configured home joint positions.")


func resume_manual_from_current_pose() -> void:
	_terminal = false
	_collided = false
	_succeeded = false
	_success_frames = 0
	_training_active = true
	_manual_collision_grace_frames = manual_recovery_grace_physics_frames
	_stop_manual_motion()
	_update_end_effector()
	_capture_target_pose_transform()
	agent.reset_observation_sources()
	_update_manual_overlay()
	print(
		"[METIS ARM MANUAL] Resumed from the current pose with %d collision-grace "
		% manual_recovery_grace_physics_frames
		+ "physics frames. Hold Q/E immediately to move out of contact.")


func _print_manual_help() -> void:
	print(
		"[METIS ARM MANUAL] F2 toggle | 1-%d select joint | " %
		mini(get_manual_joint_count(), 9)
		+ "Q/E move -/+ | G select gripper | C/O close/open gripper | "
		+ "Shift fine | Space stop | R recover current pose | "
		+ "Home reset | J random target | "
		+ "P pose log | F9 screenshot + JSON")


func _print_selected_manual_joint() -> void:
	if not _robot or get_manual_joint_count() <= 0:
		return
	var joint_name := _manual_joint_names[manual_selected_joint]
	var joint := _robot.urdf.get_joint(joint_name) if _robot.urdf else null
	var limit_text := "unbounded"
	if joint and joint.type == "revolute" and joint.limit:
		limit_text = "[%.2f, %.2f] deg" % [
			rad_to_deg(minf(joint.limit.lower, joint.limit.upper)),
			rad_to_deg(maxf(joint.limit.lower, joint.limit.upper))]
	print(
		"[METIS ARM MANUAL] Selected joint %d/%d: %s position=%.3f rad (%.2f deg) "
		% [
			manual_selected_joint + 1,
			get_manual_joint_count(),
			joint_name,
			_robot.get_joint_position(joint_name),
			rad_to_deg(_robot.get_joint_position(joint_name))]
		+ "limits=%s" % limit_text)


func _ensure_manual_overlay() -> void:
	if not manual_debug_overlay:
		return
	if not _manual_overlay_layer:
		_manual_overlay_layer = CanvasLayer.new()
		_manual_overlay_layer.name = "ManualArmDiagnostics"
		_manual_overlay_layer.layer = 100
		add_child(_manual_overlay_layer)
	if not _manual_overlay_label:
		_manual_overlay_label = Label.new()
		_manual_overlay_label.position = Vector2(16.0, 16.0)
		_manual_overlay_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
		_manual_overlay_label.add_theme_font_size_override("font_size", 18)
		_manual_overlay_label.add_theme_color_override(
			"font_color", Color(0.92, 0.96, 1.0))
		_manual_overlay_label.add_theme_color_override(
			"font_outline_color", Color(0.02, 0.03, 0.04, 0.95))
		_manual_overlay_label.add_theme_constant_override("outline_size", 6)
		_manual_overlay_layer.add_child(_manual_overlay_label)
	_manual_overlay_layer.visible = true


func _hide_manual_overlay() -> void:
	if _manual_overlay_layer:
		_manual_overlay_layer.visible = false


func _update_manual_overlay() -> void:
	if not manual_control or not manual_debug_overlay:
		_hide_manual_overlay()
		return
	_ensure_manual_overlay()
	if not _manual_overlay_label or not _robot or get_manual_joint_count() <= 0:
		return

	var joint_name := _manual_joint_names[manual_selected_joint]
	var positions: Array[String] = []
	for _name in _manual_joint_names:
		positions.append("%.1f" % rad_to_deg(_robot.get_joint_position(_name)))
	_manual_overlay_label.text = (
		"MANUAL  J%d/%d  %s  command=%+.2f  state=%s\n"
		% [
			manual_selected_joint + 1,
			get_manual_joint_count(),
			joint_name,
			_manual_command_for_joint(joint_name),
			(
				"RECOVERY"
				if _manual_collision_grace_frames > 0
				else ("BLOCKED" if _terminal else "READY"))]
		+ "target error  %.4f m  %.2f deg  speed %.3f rad/s\n"
		% [_target_distance(), rad_to_deg(_target_angle_error()), _max_joint_speed()]
		+ "joints deg  [%s]" % ", ".join(positions))


func print_manual_pose_diagnostics() -> void:
	var diagnostics := get_manual_pose_diagnostics()
	print("[METIS_ARM_REFERENCE_BEGIN]")
	print(JSON.stringify(diagnostics, "\t", true, true))
	print("[METIS_ARM_REFERENCE_END]")


func capture_manual_reference() -> void:
	var diagnostics := get_manual_pose_diagnostics()
	var stamp := Time.get_datetime_string_from_system().replace(":", "-")
	var directory := manual_capture_directory.trim_suffix("/")
	var absolute_directory := ProjectSettings.globalize_path(directory)
	var directory_error := DirAccess.make_dir_recursive_absolute(absolute_directory)
	if directory_error != OK:
		push_error(
			"URDFRobotArmAgentBody: cannot create capture directory '%s' (error %d)." %
				[absolute_directory, directory_error])
		return

	var base_path := directory.path_join("arm_reference_%s" % stamp)
	var json_path := base_path + ".json"
	var json_file := FileAccess.open(json_path, FileAccess.WRITE)
	if not json_file:
		push_error(
			"URDFRobotArmAgentBody: cannot write '%s' (error %d)." %
				[json_path, FileAccess.get_open_error()])
		return
	json_file.store_string(JSON.stringify(diagnostics, "\t", true, true))
	json_file.close()

	# NOTE: no screenshot. Grabbing the viewport image from inside the input handler (with or without
	# await) was freezing/pausing the game after the first F9. The JSON is the authoritative capture
	# (joint config + target pose) - the PNG was only a visual aid and is not worth the freeze.
	print(
		"[METIS ARM MANUAL] Reference captured (JSON only): %s" %
			ProjectSettings.globalize_path(json_path))
	# Keep manual control usable so more references can be captured back-to-back: posing the arm on the
	# target makes it succeed/terminal ("BLOCKED"), which would freeze further motion after the capture.
	_terminal = false
	_collided = false
	_succeeded = false
	_success_frames = 0
	_manual_collision_grace_frames = manual_recovery_grace_physics_frames
	_update_manual_overlay()


## Move the target to a fresh random position (bound to the J key) so reference poses can be captured
## across the whole region. Samples inside manual_random_target_region's box when set, else jitters
## around the current target. Clears any terminal/success state so posing can continue immediately.
func randomize_manual_target() -> void:
	if target == null:
		push_warning("[METIS ARM MANUAL] No target node assigned to randomize.")
		return
	var point := _sample_manual_random_point()
	target.global_position = point
	_capture_target_pose_transform()
	_update_end_effector()
	_begin_new_target_acquisition()
	_terminal = false
	_collided = false
	_succeeded = false
	_success_frames = 0
	_manual_collision_grace_frames = manual_recovery_grace_physics_frames
	notify_target_pose_relocated()
	_update_manual_overlay()
	print("[METIS ARM MANUAL] New random target at (%.3f, %.3f, %.3f)." % [
		point.x, point.y, point.z])


func _sample_manual_random_point() -> Vector3:
	var shape_node := _find_box_collision_shape(manual_random_target_region) if manual_random_target_region else null
	if shape_node and shape_node.shape is BoxShape3D:
		var half := (shape_node.shape as BoxShape3D).size * 0.5
		var local := Vector3(
			randf_range(-half.x, half.x),
			randf_range(-half.y, half.y),
			randf_range(-half.z, half.z))
		return shape_node.global_transform * local
	var base := target.global_position
	return base + Vector3(
		randf_range(-manual_random_target_extents.x, manual_random_target_extents.x),
		randf_range(-manual_random_target_extents.y, manual_random_target_extents.y),
		randf_range(-manual_random_target_extents.z, manual_random_target_extents.z))


func _find_box_collision_shape(node: Node) -> CollisionShape3D:
	if node == null:
		return null
	if node is CollisionShape3D and (node as CollisionShape3D).shape is BoxShape3D:
		return node as CollisionShape3D
	for child in node.get_children():
		var found := _find_box_collision_shape(child)
		if found:
			return found
	return null


func get_manual_pose_diagnostics() -> Dictionary:
	var desired_pose := _target_pose_node()
	var current_pose := _tool_pose_node()
	var joints: Array[Dictionary] = []
	var mimic_joints: Array[Dictionary] = []
	if _robot:
		for index in range(get_manual_joint_count()):
			var joint_name := _manual_joint_names[index]
			var joint := _robot.urdf.get_joint(joint_name) if _robot.urdf else null
			var policy_index := _joint_names.find(joint_name)
			var joint_data := {
				"index": index,
				"policy_index": policy_index,
				"policy_controlled": policy_index >= 0,
				"name": joint_name,
				"position_rad": _robot.get_joint_position(joint_name),
				"position_deg": rad_to_deg(_robot.get_joint_position(joint_name)),
				"velocity_rad_s": _robot.get_joint_velocity(joint_name),
				"command_normalized": _manual_command_for_joint(joint_name),
				"max_velocity_rad_s": _joint_max_speed_for_name(joint_name),
			}
			if joint and joint.type in ["revolute", "prismatic"] and joint.limit:
				joint_data["lower_rad"] = minf(joint.limit.lower, joint.limit.upper)
				joint_data["upper_rad"] = maxf(joint.limit.lower, joint.limit.upper)
				if joint.type == "revolute":
					joint_data["lower_deg"] = rad_to_deg(
						minf(joint.limit.lower, joint.limit.upper))
					joint_data["upper_deg"] = rad_to_deg(
						maxf(joint.limit.lower, joint.limit.upper))
			joints.append(joint_data)
		for mimic in _robot.urdf.get_mimic_joints():
			mimic_joints.append({
				"name": mimic.name,
				"source": mimic.mimic_joint,
				"multiplier": mimic.mimic_multiplier,
				"offset": mimic.mimic_offset,
				"position_rad": _robot.get_joint_position(mimic.name),
				"position_deg": rad_to_deg(
					_robot.get_joint_position(mimic.name)),
				"velocity_rad_s": _robot.get_joint_velocity(mimic.name),
			})

	var position_error_world := Vector3.ZERO
	var position_error_base := Vector3.ZERO
	if desired_pose and current_pose:
		position_error_world = (
			desired_pose.global_position - current_pose.global_position)
		position_error_base = (
			global_transform.basis.inverse() * position_error_world)

	return {
		"timestamp": Time.get_datetime_string_from_system(),
		"agent": str(name),
		"manual_selected_joint": manual_selected_joint,
		"manual_selected_joint_name": (
			_manual_joint_names[manual_selected_joint]
			if get_manual_joint_count() > 0
			else ""),
		"policy_action_size": get_joint_count(),
		"manual_joint_names": Array(_manual_joint_names),
		"joints": joints,
		"mimic_joints": mimic_joints,
		"tool_pose_global": (
			_transform_diagnostics(current_pose.global_transform)
			if current_pose
			else {}),
		"target_pose_global": (
			_transform_diagnostics(desired_pose.global_transform)
			if desired_pose
			else {}),
		"position_error_world": _vector3_diagnostics(position_error_world),
		"position_error_base": _vector3_diagnostics(position_error_base),
		"position_error_m": _target_distance(),
		"orientation_error_axis_angle_normalized": _vector3_diagnostics(
			_target_orientation_error_vector()),
		"orientation_error_deg": rad_to_deg(_target_angle_error()),
		"max_joint_speed_rad_s": _max_joint_speed(),
		"hold_frames": _success_frames,
		"pose_held": _is_pose_held(),
		"target_acquired": _succeeded,
		"terminal": _terminal,
		"collided": _collided,
		"manual_collision_grace_frames": _manual_collision_grace_frames,
		"reward_terms_raw": {
			"pose_tracking": get_pose_tracking_reward(),
			"hold_stillness": get_hold_stillness_reward(),
			"hold_progress": get_hold_progress_reward(),
			"near_target_speed": get_near_target_speed_penalty(),
			"joint_motion": get_joint_motion_penalty(),
			"joint_limit": get_joint_limit_penalty(),
		},
		"success_thresholds": {
			"distance_m": success_distance,
			"orientation_deg": success_angle_degrees,
			"hold_physics_frames": success_hold_physics_frames,
			"max_joint_speed_rad_s": success_max_joint_speed,
		},
		"tool_pose_path": str(tool_pose_path),
		"target_pose_path": str(target_pose.get_path()) if target_pose else "",
	}


func _transform_diagnostics(value: Transform3D) -> Dictionary:
	var _basis := value.basis.orthonormalized()
	var _rotation := _basis.get_rotation_quaternion().normalized()
	return {
		"position": _vector3_diagnostics(value.origin),
		"quaternion_xyzw": [
			_rotation.x, _rotation.y, _rotation.z, _rotation.w],
		"axis_x": _vector3_diagnostics(_basis.x),
		"axis_y": _vector3_diagnostics(_basis.y),
		"axis_z": _vector3_diagnostics(_basis.z),
	}


func _vector3_diagnostics(value: Vector3) -> Array[float]:
	return [value.x, value.y, value.z]


func reset_all(original_transform: Variant, reset_rewards := true) -> void:
	set_training_active(true)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform

	_terminal = false
	_succeeded = false
	_collided = false
	_last_self_collision_details.clear()
	_last_collision_details.clear()
	_success_frames = 0
	_has_last_target_pose_transform = false
	for index in range(get_joint_count()):
		_commands[index] = 0.0
		_previous_action[index] = 0.0
	_robot.reset_joint_positions(_build_home_positions(true))
	_pending_reset_offsets.clear()
	_update_end_effector()
	_capture_target_pose_transform()

	# On reset the teleop marker snaps back to the (home) EE so the IK does not
	# chase the previous target from the new home; wrist control goes back to off.
	if ik_manual_enabled and ik_manual_target != null:
		var home_tcp: Node3D = _robot.get_link_node(tcp_link_name)
		if home_tcp != null:
			var tp := home_tcp.global_transform
			ik_manual_target.global_transform = Transform3D(tp.basis, tp * tcp_local_offset)
		if _ik_manual != null:
			_ik_manual.track_orientation = false

	if has_method("reset_physics_interpolation"):
		reset_physics_interpolation()
	agent.reset_observation_sources()
	if reset_rewards:
		agent.refresh_observation_sources()
		agent.reset_reward({"body": self})


func initialize_episode_from_current_state() -> void:
	_terminal = false
	_succeeded = false
	_collided = false
	_last_self_collision_details.clear()
	_last_collision_details.clear()
	_success_frames = 0
	_has_last_target_pose_transform = false
	set_training_active(true)
	for index in range(get_joint_count()):
		_commands[index] = 0.0
		_previous_action[index] = 0.0
	_robot.stop_all_joints()
	_update_end_effector()
	_capture_target_pose_transform()
	agent.reset_observation_sources()


func set_reset_joint_offsets(offsets_radians: Array) -> void:
	_pending_reset_offsets.clear()
	for offset in offsets_radians:
		_pending_reset_offsets.append(float(offset))


func set_training_active(enabled: bool) -> void:
	_training_active = enabled
	if not _robot:
		return
	if not enabled:
		for index in range(_commands.size()):
			_commands[index] = 0.0
		_robot.stop_all_joints()


func is_terminal() -> bool:
	return _terminal


func has_succeeded() -> bool:
	return _succeeded


func has_collided() -> bool:
	return _collided


func get_last_self_collision_details() -> Dictionary:
	return _last_self_collision_details.duplicate(true)


func get_last_collision_details() -> Dictionary:
	return _last_collision_details.duplicate(true)


func set_continue_after_success(enabled: bool) -> void:
	terminate_on_success = not enabled
	if enabled and _succeeded and not _collided:
		_terminal = false
		set_training_active(true)


func get_progress() -> float:
	var distance_ratio := _target_distance() / maxf(workspace_scale, 0.001)
	var position_progress := (
		1.0 / (1.0 + maxf(distance_ratio, 0.0))
		if distance_progress_mode == DistanceProgressMode.INVERSE_DISTANCE
		else 1.0 - clampf(distance_ratio, 0.0, 1.0)
	)
	var orientation_progress := 1.0 - clampf(_target_angle_error() / PI, 0.0, 1.0)
	var orientation_weight := clampf(orientation_progress_weight, 0.0, 1.0)
	if pose_progress_mode == PoseProgressMode.POSITION_GATED:
		return position_progress * lerpf(
			1.0, orientation_progress, orientation_weight)
	return lerpf(position_progress, orientation_progress, orientation_weight)


func get_target_distance() -> float:
	return _target_distance()


func get_joint_count() -> int:
	return _joint_names.size()


func get_controlled_joint_names() -> PackedStringArray:
	return _joint_names.duplicate()


func get_manual_control_joint_names() -> PackedStringArray:
	return _manual_joint_names.duplicate()


func get_manual_joint_count() -> int:
	return _manual_joint_names.size()


func get_joint_position_observation() -> Array:
	var result: Array = []
	for joint_name in _joint_names:
		var joint := _robot.urdf.get_joint(joint_name)
		var _position := _robot.get_joint_position(joint_name)
		if joint and joint.type in ["revolute", "prismatic"] and joint.limit:
			var lower := minf(joint.limit.lower, joint.limit.upper)
			var upper := maxf(joint.limit.lower, joint.limit.upper)
			if is_equal_approx(lower, upper):
				result.append(0.0)
			else:
				result.append(clampf(
					remap(_position, lower, upper, -1.0, 1.0), -1.0, 1.0))
		else:
			result.append(clampf(_position / PI, -1.0, 1.0))
	return result


func get_joint_velocity_observation() -> Array:
	var result: Array = []
	for index in range(get_joint_count()):
		var velocity := _robot.get_joint_velocity(_joint_names[index])
		result.append(clampf(
			velocity / maxf(_joint_max_speed(index), 0.000001), -1.0, 1.0))
	return result


func get_target_error_observation() -> Vector3:
	var desired_pose := _target_pose_node()
	var current_pose := _tool_pose_node()
	if not desired_pose or not current_pose:
		return Vector3.ZERO
	var error_world := desired_pose.global_position - current_pose.global_position
	var error_base := global_transform.basis.inverse() * error_world
	var _scale := maxf(workspace_scale, 0.001)
	return Vector3(
		clampf(error_base.x / _scale, -1.0, 1.0),
		clampf(error_base.y / _scale, -1.0, 1.0),
		clampf(error_base.z / _scale, -1.0, 1.0))


func get_target_orientation_error_observation() -> Vector3:
	# Shortest axis-angle error in the current tool frame, normalised so PI radians has length 1.
	# Zero therefore represents the desired orientation without quaternion sign ambiguity.
	return _target_orientation_error_vector()


func get_previous_action_observation() -> Array:
	return _previous_action.duplicate()


func get_joint_motion_penalty() -> float:
	var effort := 0.0
	for command in _commands:
		effort += absf(command)
	return -effort / maxf(float(_commands.size()), 1.0)


func get_joint_limit_penalty() -> float:
	var penalty := 0.0
	var margin := 0.10
	for joint_name in _joint_names:
		var joint := _robot.urdf.get_joint(joint_name)
		if (
			not joint
			or joint.type not in ["revolute", "prismatic"]
			or not joint.limit
		):
			continue
		var lower := minf(joint.limit.lower, joint.limit.upper)
		var upper := maxf(joint.limit.lower, joint.limit.upper)
		var ratio := clampf(inverse_lerp(
			lower, upper, _robot.get_joint_position(joint_name)), 0.0, 1.0)
		var edge_distance := minf(ratio, 1.0 - ratio)
		if edge_distance < margin:
			var error := (margin - edge_distance) / margin
			penalty -= error * error
	return penalty / maxf(float(get_joint_count()), 1.0)


func get_hold_stillness_reward() -> float:
	# Immobility reward, active ONLY within near_target_distance: rewards a low max joint speed so
	# the arm learns to SETTLE on the target instead of sweeping through it. Zero away from the
	# target, so the fast reach is never penalised.
	if not _is_near_target_pose():
		return 0.0
	# Zero (or less) makes the reward track the success gate instead of a fixed speed. A constant
	# reference silently drifts away from a tightening gate, and then the shaping keeps paying for a
	# speed the success criterion has already started rejecting.
	var reference := (
		hold_stillness_speed_reference
		if hold_stillness_speed_reference > 0.0
		else success_max_joint_speed)
	return clampf(1.0 - _max_joint_speed() / maxf(reference, 0.001), 0.0, 1.0)


func get_near_target_speed_penalty() -> float:
	# Penalty proportional to MAX joint speed, active ONLY near the target. Drives joint velocity
	# toward zero precisely where holding matters, without slowing the reach far away. Returns a
	# negative term (scaled by the component weight).
	if not _is_near_target_pose():
		return 0.0
	return -_max_joint_speed()


func get_pose_tracking_reward() -> float:
	# Orientation is GATED by proximity (multiplied by position_score), so orienting-while-far pays
	# NOTHING -- an earlier additive `0.4*orient` was paid every step even at 23cm, so the policy
	# learned the profitable INCOMPLETE strategy of orienting far and never approaching. The FAR
	# orientation-learning signal instead comes from orientation_progress_weight (improvement-based
	# via ProgressDelta -> no camp-and-collect). position_score is graded over near_target_distance
	# (now 0.20m) so it is DENSE well beyond 10cm and its monotone pull always favours getting closer
	# over camping. Full value needs BOTH position tight AND orientation aligned (joint).
	var position_score := 1.0 - clampf(
		_target_distance() / maxf(near_target_distance, 0.001), 0.0, 1.0)
	var orientation_full := 1.0 - clampf(_target_angle_error() / PI, 0.0, 1.0)
	return position_score * (0.3 + 0.7 * orientation_full)


func get_near_target_distance_penalty() -> float:
	# Per-step penalty proportional to distance, active ONLY within near_target_distance. Standing away
	# from the target keeps costing every step, so the arm cannot freeze at an overshoot (target behind
	# the tip): it must keep closing. Zero beyond near_target_distance so the fast reach far away is not
	# penalised. Returns a value in [-1, 0] (scaled by the component weight): -1 at the edge, 0 on target.
	var d := _target_distance()
	if d > near_target_distance:
		return 0.0
	return -clampf(d / maxf(near_target_distance, 0.001), 0.0, 1.0)


func get_hold_progress_reward() -> float:
	# Progressive reward for consecutive hold frames: grows as the arm SUSTAINS the hold, directly
	# rewarding staying put. Normalised by the required frames so the term stays in [0,1] across the
	# curriculum (10 -> 20 -> 30 frames).
	if success_hold_physics_frames <= 0:
		return 0.0
	return clampf(
		float(_success_frames) / float(success_hold_physics_frames), 0.0, 1.0)


func get_max_joint_speed() -> float:
	return _max_joint_speed()


func get_hold_frames() -> int:
	return _success_frames


func get_debug_metrics() -> Dictionary:
	# Generic per-step diagnostics surfaced to the trainer logs (merged into the step info by
	# ScenarioController when the body exposes this method).
	return {
		"position_error_m": _target_distance(),
		"orientation_error_deg": rad_to_deg(_target_angle_error()),
		"max_joint_speed": _max_joint_speed(),
		"hold_frames": _success_frames,
		"pose_held": _is_pose_held(),
		"target_acquired": _succeeded,
		"collided": _collided,
		"collision_source": str(_last_collision_details.get("source", "")),
		"collision_details": _last_collision_details.duplicate(true),
		"success_thresholds": {
			"distance_m": success_distance,
			"orientation_deg": success_angle_degrees,
			"hold_physics_frames": success_hold_physics_frames,
			"max_joint_speed_rad_s": success_max_joint_speed,
			"require_still": success_require_still,
		},
		# World-space frames behind orientation_error_deg. Without them the error is a single number
		# with no way to tell a mis-specified target pose from a policy that cannot reach it.
		"tool_pose_world": _pose_diagnostics(_tool_pose_node()),
		"target_pose_world": _pose_diagnostics(_target_pose_node()),
		"gripper_front": _gripper_front_direction(),
	}


## Direction from the TCP link towards the fingertips, i.e. the way the gripper actually faces,
## reported both in world space and in the TCP link's own frame. The latter is the ground truth for
## defining the tool pose: its +X should equal that vector, otherwise the orientation error is
## measured against an axis that has nothing to do with the grasp.
func _gripper_front_direction() -> Dictionary:
	if _robot == null or tcp_link_name.is_empty():
		return {}
	var hand := _robot.get_link_node(tcp_link_name) as Node3D
	if hand == null:
		return {}
	var centre := Vector3.ZERO
	var count := 0
	for link_name in self_check_link_names:
		if not str(link_name).contains("finger"):
			continue
		var node := _robot.get_link_node(link_name) as Node3D
		if node:
			centre += node.global_position
			count += 1
	if count == 0:
		return {}
	centre /= float(count)
	var world_direction := (centre - hand.global_position)
	if world_direction.length() < 0.0001:
		return {}
	world_direction = world_direction.normalized()
	var hand_basis := hand.global_basis.orthonormalized()
	var result := {
		"world": _vector3_diagnostics(world_direction),
		"hand_local": _vector3_diagnostics(hand_basis.inverse() * world_direction),
	}
	# Finger-to-finger axis: with the front direction it fixes the whole grasp frame, leaving no
	# free roll to guess at.
	var left := _robot.get_link_node("openarm_right_left_finger") as Node3D
	var right := _robot.get_link_node("openarm_right_right_finger") as Node3D
	if left != null and right != null:
		var span := right.global_position - left.global_position
		if span.length() > 0.0001:
			span = span.normalized()
			result["span_world"] = _vector3_diagnostics(span)
			result["span_hand_local"] = _vector3_diagnostics(hand_basis.inverse() * span)
	return result


func _pose_diagnostics(node: Node3D) -> Dictionary:
	if node == null:
		return {}
	var pose_basis := node.global_basis.orthonormalized()
	return {
		"position": _vector3_diagnostics(node.global_position),
		"axis_x": _vector3_diagnostics(pose_basis.x),
		"axis_y": _vector3_diagnostics(pose_basis.y),
		"axis_z": _vector3_diagnostics(pose_basis.z),
	}


func notify_target_pose_relocated() -> void:
	_begin_new_target_acquisition()
	_capture_target_pose_transform()
	target_pose_relocated.emit()


func get_control_input(input_name: String) -> float:
	if not input_name.begins_with("joint_velocity_"):
		return 0.0
	var index := int(input_name.trim_prefix("joint_velocity_"))
	return _commands[index] if index >= 0 and index < _commands.size() else 0.0


func report_obstacle_collision() -> void:
	_register_collision({"source": "reported_obstacle"})


func _resolve_controlled_joint_names() -> PackedStringArray:
	var available := _robot.get_actuated_joint_names()
	if controlled_joint_names.is_empty():
		return available
	var result := PackedStringArray()
	for joint_name in controlled_joint_names:
		if available.has(joint_name):
			result.append(joint_name)
		else:
			push_warning(
				"URDFRobotArmAgentBody: joint '%s' is missing or not actuated." %
				joint_name)
	return result


func _resolve_manual_joint_names() -> PackedStringArray:
	var result := _joint_names.duplicate()
	for joint_name in manual_extra_joint_names:
		if result.has(joint_name):
			continue
		var joint := _robot.urdf.get_joint(joint_name) if _robot.urdf else null
		if (
			joint
			and joint.type in ["revolute", "continuous", "prismatic"]
			and not joint.is_mimic()
		):
			result.append(joint_name)
			_manual_extra_commands[joint_name] = 0.0
		else:
			var reason := "missing"
			if joint:
				reason = "mimic" if joint.is_mimic() else "not independently actuated"
			push_warning(
				"URDFRobotArmAgentBody: manual joint '%s' is %s." %
				[joint_name, reason])
	return result


func _resize_state() -> void:
	_commands.resize(get_joint_count())
	_previous_action.resize(get_joint_count())
	for index in range(get_joint_count()):
		_commands[index] = 0.0
		_previous_action[index] = 0.0


func _configure_action_size() -> void:
	if not auto_configure_action_size:
		return
	var action_component := get_node_or_null("Agent/ActionSpace/JointVelocity")
	if action_component and action_component is ContinuousAction:
		action_component.size = get_joint_count()


func _build_home_positions(include_offsets := false) -> Dictionary:
	var result := {}
	for index in range(get_joint_count()):
		var _position := (
			float(joint_home_positions[index])
			if index < joint_home_positions.size()
			else 0.0)
		if include_offsets and index < _pending_reset_offsets.size():
			_position += float(_pending_reset_offsets[index])
		result[_joint_names[index]] = _position
	return result


func _joint_max_speed(index: int) -> float:
	if index < 0 or index >= get_joint_count():
		return maxf(default_joint_speed, 0.000001)
	return _joint_max_speed_for_name(_joint_names[index])


func _joint_max_speed_for_name(joint_name: String) -> float:
	var speed := default_joint_speed
	var joint := _robot.urdf.get_joint(joint_name)
	if joint and joint.limit and joint.limit.velocity > 0.0:
		speed = joint.limit.velocity
	if max_joint_speed_override > 0.0:
		speed = minf(speed, max_joint_speed_override)
	return maxf(speed, 0.000001)


func _limit_aware_command(index: int, command: float) -> float:
	if (
		index < 0
		or index >= get_joint_count()
	):
		return command
	return _limit_aware_named_command(_joint_names[index], command)


func _limit_aware_named_command(joint_name: String, command: float) -> float:
	if is_zero_approx(command) or not _robot or not _robot.urdf:
		return command
	var joint := _robot.urdf.get_joint(joint_name)
	if (
		not joint
		or joint.type not in ["revolute", "prismatic"]
		or not joint.limit
	):
		return command

	var lower := minf(joint.limit.lower, joint.limit.upper)
	var upper := maxf(joint.limit.lower, joint.limit.upper)
	var span := upper - lower
	if span <= 0.0:
		return 0.0

	var _position := clampf(_robot.get_joint_position(joint_name), lower, upper)
	var distance_to_limit := (
		_position - lower
		if command < 0.0
		else upper - _position)
	if distance_to_limit <= 0.000001:
		return 0.0

	var slowdown_margin := span * joint_limit_slowdown_ratio
	if slowdown_margin <= 0.000001:
		return command
	return command * clampf(distance_to_limit / slowdown_margin, 0.0, 1.0)


func _manual_command_for_joint(joint_name: String) -> float:
	var policy_index := _joint_names.find(joint_name)
	if policy_index >= 0:
		return _commands[policy_index]
	return float(_manual_extra_commands.get(joint_name, 0.0))


func _update_end_effector() -> void:
	if not end_effector or tcp_link_name.is_empty() or not _robot:
		return
	var tcp_link := _robot.get_link_node(tcp_link_name)
	if not tcp_link:
		return
	end_effector.global_transform = tcp_link.global_transform.translated_local(
		tcp_local_offset)


func _update_success_state() -> void:
	if _terminal or not _target_pose_node() or not _tool_pose_node():
		return
	var distance := _target_distance()
	var angle := _target_angle_error()
	var still := (not success_require_still) or (_max_joint_speed() <= success_max_joint_speed)
	var within_success := (
		distance <= success_distance
		and angle <= deg_to_rad(success_angle_degrees)
		and still
	)
	if _succeeded:
		var inside_rearm_zone := (
			distance <= success_distance * success_rearm_distance_multiplier
			and angle <= (
				deg_to_rad(success_angle_degrees)
				* success_rearm_angle_multiplier)
		)
		if terminate_on_success or inside_rearm_zone:
			_success_frames = (
				success_hold_physics_frames
				if within_success
				else 0)
			return
		_begin_new_target_acquisition()

	if within_success:
		_success_frames += 1
	else:
		_success_frames = 0
	if _success_frames < success_hold_physics_frames:
		return
	_succeeded = true
	# In manual control the operator keeps posing/capturing, so success must NOT terminate or stop the
	# joints (that froze the arm after posing on the target, blocking further F9/J). Only autonomous
	# training terminates on success.
	if terminate_on_success and not manual_control:
		_terminal = true
		_robot.stop_all_joints()
	if not manual_control:
		target_reached.emit()


func _detect_target_pose_relocation() -> void:
	var desired_pose := _target_pose_node()
	if not desired_pose:
		_has_last_target_pose_transform = false
		return
	if not _has_last_target_pose_transform:
		_capture_target_pose_transform()
		return

	var current_transform := desired_pose.global_transform
	var position_changed := (
		current_transform.origin.distance_to(_last_target_pose_transform.origin)
		> maxf(target_relocation_position_epsilon, 0.0)
	)
	var angle_changed := (
		_basis_angle_between(
			_last_target_pose_transform.basis,
			current_transform.basis)
		> deg_to_rad(maxf(target_relocation_angle_epsilon_degrees, 0.0))
	)
	if position_changed or angle_changed:
		notify_target_pose_relocated()


func _capture_target_pose_transform() -> void:
	var desired_pose := _target_pose_node()
	if not desired_pose:
		_has_last_target_pose_transform = false
		return
	_last_target_pose_transform = desired_pose.global_transform
	_has_last_target_pose_transform = true


func _begin_new_target_acquisition() -> void:
	_succeeded = false
	_success_frames = 0


func _connect_safety_volumes() -> void:
	for area in safety_volumes:
		if not area:
			continue
		area.monitoring = true
		if not area.body_entered.is_connected(_on_safety_body_entered):
			area.body_entered.connect(_on_safety_body_entered)
		if not area.area_entered.is_connected(_on_safety_area_entered):
			area.area_entered.connect(_on_safety_area_entered)


func _on_safety_body_entered(body: Node) -> void:
	if _node_or_parent_is_in_group(body, obstacle_group):
		_register_collision({
			"source": "safety_volume_body",
			"collider": str(body.name),
			"collider_path": str(body.get_path()),
		})


func _on_safety_area_entered(other: Area3D) -> void:
	if _node_or_parent_is_in_group(other, obstacle_group):
		_register_collision({
			"source": "safety_volume_area",
			"collider": str(other.name),
			"collider_path": str(other.get_path()),
		})


func _cache_robot_collision_geometry() -> void:
	_robot_collision_shapes.clear()
	_robot_collision_exclusions.clear()
	_robot_collision_link_by_shape_id.clear()
	if not _robot:
		return

	for link_name in _robot.links.keys():
		var link_node = _robot.links[link_name]
		if not link_node is Node:
			continue
		if link_node is CollisionObject3D:
			_robot_collision_exclusions.append(link_node.get_rid())
		var shapes_before := _robot_collision_shapes.size()
		_collect_collision_shapes(link_node)
		for index in range(shapes_before, _robot_collision_shapes.size()):
			var collision_shape := _robot_collision_shapes[index]
			_robot_collision_link_by_shape_id[
				collision_shape.get_instance_id()] = str(link_name)


func _collect_collision_shapes(node: Node) -> void:
	for child in node.get_children():
		if child is CollisionShape3D:
			_robot_collision_shapes.append(child)
		_collect_collision_shapes(child)


func _check_environment_collisions() -> void:
	if (
		_terminal
		or (manual_control and _manual_collision_grace_frames > 0)
		or not auto_detect_environment_collisions
		or _robot_collision_shapes.is_empty()
		or not is_inside_tree()
	):
		return

	var space_state := get_world_3d().direct_space_state
	for collision_shape in _robot_collision_shapes:
		if not collision_shape or collision_shape.disabled or not collision_shape.shape:
			continue
		var query := PhysicsShapeQueryParameters3D.new()
		query.shape = collision_shape.shape
		query.transform = collision_shape.global_transform
		query.collision_mask = environment_collision_mask
		query.collide_with_bodies = true
		query.collide_with_areas = true
		query.exclude = _robot_collision_exclusions
		for hit in space_state.intersect_shape(query, max_collision_results_per_shape):
			var collider: Variant = hit.get("collider")
			if collider is Node and _node_or_parent_is_in_group(collider, obstacle_group):
				_register_collision({
					"source": "environment",
					"checker_link": str(_robot_collision_link_by_shape_id.get(
						collision_shape.get_instance_id(), collision_shape.name)),
					"checker_shape": str(collision_shape.get_path()),
					"collider": str((collider as Node).name),
					"collider_path": str((collider as Node).get_path()),
				})
				return


func _cache_self_body_geometry() -> void:
	_self_body_rids.clear()
	_self_check_shapes.clear()
	_self_check_exclusions.clear()
	_self_check_link_by_shape_id.clear()
	if not self_body_collision_enabled or not _robot:
		return
	var body_names := {}
	for link_name in self_body_link_names:
		var node: Node = _robot.get_link_node(link_name)
		if node is CollisionObject3D:
			_self_body_rids.append((node as CollisionObject3D).get_rid())
			body_names[link_name] = true
	if _self_body_rids.is_empty():
		return
	# Exclude every robot link EXCEPT the central-body targets, so a checker query can only strike
	# the body. Adjacent shoulder links (which always touch the body) are excluded and never
	# false-trigger, and the checker's own link is excluded too.
	for link_name in _robot.links.keys():
		var link_node = _robot.links[link_name]
		if link_node is CollisionObject3D and not body_names.has(link_name):
			_self_check_exclusions.append((link_node as CollisionObject3D).get_rid())
	# Checker shapes: the configured distal arm links that can swing into the body.
	for link_name in self_check_link_names:
		var node: Node = _robot.get_link_node(link_name)
		if node:
			var link_shapes: Array[CollisionShape3D] = []
			_collect_shapes_into(node, link_shapes)
			for collision_shape in link_shapes:
				_self_check_shapes.append(collision_shape)
				_self_check_link_by_shape_id[collision_shape.get_instance_id()] = link_name


func _collect_shapes_into(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child is CollisionShape3D:
			out.append(child)
		_collect_shapes_into(child, out)


func _check_self_body_collisions() -> void:
	if (
		_terminal
		or (manual_control and _manual_collision_grace_frames > 0)
		or not self_body_collision_enabled
		or _self_check_shapes.is_empty()
		or _self_body_rids.is_empty()
		or not is_inside_tree()
	):
		return
	var space_state := get_world_3d().direct_space_state
	for collision_shape in _self_check_shapes:
		if not collision_shape or collision_shape.disabled or not collision_shape.shape:
			continue
		var query := PhysicsShapeQueryParameters3D.new()
		query.shape = collision_shape.shape
		query.transform = collision_shape.global_transform
		query.collision_mask = environment_collision_mask
		query.collide_with_bodies = true
		query.collide_with_areas = false
		query.exclude = _self_check_exclusions
		for hit in space_state.intersect_shape(query, max_collision_results_per_shape):
			if _self_body_rids.has(hit.get("rid")):
				var collider: Variant = hit.get("collider")
				_last_self_collision_details = {
					"checker_link": str(_self_check_link_by_shape_id.get(
						collision_shape.get_instance_id(), collision_shape.name)),
					"checker_shape": str(collision_shape.get_path()),
					"body_link": (
						str((collider as Node).name)
						if collider is Node
						else "<unknown>"),
					"body_path": (
						str((collider as Node).get_path())
						if collider is Node
						else ""),
				}
				var collision_details := _last_self_collision_details.duplicate(true)
				collision_details["source"] = "self_body"
				_register_collision(collision_details)
				return


func _node_or_parent_is_in_group(node: Node, group_name: String) -> bool:
	var current: Node = node
	while current:
		if current.is_in_group(group_name):
			return true
		current = current.get_parent()
	return false


func _register_collision(details: Dictionary = {}) -> void:
	if _terminal:
		return
	_last_collision_details = details.duplicate(true)
	if not _last_collision_details.has("source"):
		_last_collision_details["source"] = "unknown"
	_last_collision_details["physics_frame"] = Engine.get_physics_frames()
	_last_collision_details["succeeded_before_collision"] = _succeeded
	if collision_debug or (
		self_collision_debug
		and str(_last_collision_details.get("source", "")) == "self_body"
	):
		print(
			"[METIS ARM COLLISION] %s"
			% JSON.stringify(_last_collision_details))
	_collided = true
	# Manual control never terminates/stops (the operator keeps posing/capturing). In training the
	# collision is always penalised via the signal, but it only terminates + stops the joints when
	# collision_terminates is true; with it false the arm grazes and recovers (soft collision).
	if not manual_control:
		obstacle_collision.emit()
		if collision_terminates:
			_terminal = true
			_robot.stop_all_joints()


func _target_distance() -> float:
	var desired_pose := _target_pose_node()
	var current_pose := _tool_pose_node()
	if not desired_pose or not current_pose:
		return workspace_scale
	return current_pose.global_position.distance_to(desired_pose.global_position)


func _target_angle_error() -> float:
	return _target_orientation_error_vector().length() * PI


func _target_orientation_error_vector() -> Vector3:
	var desired_pose := _target_pose_node()
	var current_pose := _tool_pose_node()
	if not desired_pose or not current_pose:
		return Vector3.ZERO

	var current_basis := current_pose.global_basis.orthonormalized()
	var desired_basis := desired_pose.global_basis.orthonormalized()
	var relative_rotation := (
		current_basis.inverse() * desired_basis
	).get_rotation_quaternion().normalized()
	if relative_rotation.w < 0.0:
		relative_rotation = Quaternion(
			-relative_rotation.x,
			-relative_rotation.y,
			-relative_rotation.z,
			-relative_rotation.w)

	var vector_part := Vector3(
		relative_rotation.x,
		relative_rotation.y,
		relative_rotation.z)
	var sin_half_angle := vector_part.length()
	if sin_half_angle <= 0.000001:
		return Vector3.ZERO
	var angle := 2.0 * atan2(
		sin_half_angle,
		clampf(relative_rotation.w, -1.0, 1.0))
	return vector_part / sin_half_angle * (angle / PI)


func _basis_angle_between(first: Basis, second: Basis) -> float:
	var relative_rotation := (
		first.orthonormalized().inverse() * second.orthonormalized()
	).get_rotation_quaternion().normalized()
	return 2.0 * acos(clampf(absf(relative_rotation.w), 0.0, 1.0))


func _is_near_target_pose() -> bool:
	# Gates the STILLNESS rewards (hold_stillness + near-target speed penalty). Uses the tight
	# hold_activation_distance, NOT the wide pose-shaping near_target_distance, so the arm cannot get
	# paid for freezing far from the target -- stillness only matters right at the goal.
	return (
		_target_distance() <= hold_activation_distance
		and _target_angle_error() <= deg_to_rad(pose_reward_angle_degrees)
	)


func _is_pose_held() -> bool:
	return (
		_target_distance() <= success_distance
		and _target_angle_error() <= deg_to_rad(success_angle_degrees)
		and (
			not success_require_still
			or _max_joint_speed() <= success_max_joint_speed)
	)


func _tool_pose_node() -> Node3D:
	return tool_pose if tool_pose else end_effector


func _target_pose_node() -> Node3D:
	return target_pose if target_pose else target


func _max_joint_speed() -> float:
	# MAX |joint velocity| over the controlled joints; used by the reach-and-hold success gate to
	# require the arm to be nearly stationary. Max (not mean) so a single fast-sweeping joint (e.g.
	# the wide-range xarm2) cannot pass the gate while the average stays low.
	if not _robot:
		return 0.0
	var names := get_controlled_joint_names()
	if names.is_empty():
		return 0.0
	var fastest := 0.0
	for joint_name in names:
		fastest = maxf(fastest, absf(_robot.get_joint_velocity(joint_name)))
	return fastest
