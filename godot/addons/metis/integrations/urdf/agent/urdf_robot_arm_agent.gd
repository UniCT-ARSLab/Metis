extends MetisRobot3D
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
## Integrates joint-velocity commands within URDF limits.
@export var use_kinematic_control := true
@export var joint_home_positions := PackedFloat32Array()
@export var default_joint_speed := 0.5
## Hard cap (rad/s) applied on top of each joint's URDF velocity limit.
@export var max_joint_speed_override := 0.0
## Tapers commands that push a bounded revolute joint toward its hard stop.
@export_range(0.0, 0.25, 0.005) var joint_limit_slowdown_ratio := 0.05

@export_category("Tool Center Point")
@export var tcp_link_name := ""
@export var tcp_local_offset := Vector3.ZERO
@export var end_effector_path: NodePath = NodePath("EndEffector")
## Optional calibrated frame below EndEffector.
@export var tool_pose_path: NodePath = NodePath("EndEffector/ToolPose")

@export_category("Grasping")
## Joint that opens and closes the gripper.
@export var gripper_joint_name: StringName = &""
## The two finger links, used once at startup to measure where the pad SURFACES are.
@export var gripper_finger_link_names: PackedStringArray = []
## Make progress account for the grasp, not just the pose. Off leaves the reach behaviour untouched.
@export var grasp_gates_progress := false
## Share of progress attributable to reaching the object. The remainder is earned by holding it, so
## an empty hand cannot report more than this -- keep it BELOW the stall exemption threshold
## (ScenarioController's NoProgress ignore_stall_above_progress) or the gate does nothing.
@export_range(0.0, 1.0, 0.01) var grasp_progress_share := 0.85
## Stop the closing command once the pads have reached the object, so kinematic fingers cannot pass
## through it. See _gripper_command_blocked_by_object.
@export var gripper_stops_on_contact := true
## Separate speed cap for the gripper joint, in m/s. Zero leaves it on the arm's cap. See
## _joint_max_speed_for_name for why the arm's cap is the wrong one here.
@export var gripper_max_speed := 0.0
## Travel left unused at the CLOSED end of the gripper, in the joint's own units.
##
## A real gripper commanded shut against nothing runs into its mechanical stop and stays there under
## load; in simulation that is free, so a policy learns to hold the claws hard shut. This keeps a
## margin the closing command cannot cross. 0 leaves every existing arm exactly as it was.
@export var gripper_closed_margin := 0.0
## The grasp point ON THE OBJECT, which moves with it. Distinct from target_pose, the pose being
## DEMANDED, which stays where the object spawned so that carrying it off counts as leaving the
## target. Capture geometry is a question about where the object is, so it uses this one.
@export var grasp_object_pose: Node3D
## Width of the object to be grasped, in metres.
@export var grasp_object_width := 0.0
## Success means holding the object for success_hold_physics_frames, instead of matching the
## prescribed pose within success_distance / success_angle_degrees. The pose keeps guiding through
## the reward; it stops being the definition of the goal.
@export var success_requires_grasp := false
## While true, the pose gates cannot declare success however well they are met.
##
## Lets a scenario own a completion condition that cannot be inferred from the robot pose alone.
## The scenario switches this off once its external completion rule is satisfied.
@export var success_blocked := false
## Scale for the tool-frame position error observation.
@export var grasp_observation_scale := 0.05

@export_category("Task")
@export var target: Node3D
## Optional desired pose, normally a Marker3D below target.
@export var target_pose: Node3D
@export var workspace_scale := 1.0
@export var success_distance := 0.05
@export_range(0.1, 180.0, 0.1) var success_angle_degrees := 12.0
@export var success_hold_physics_frames := 30
@export var terminate_on_success := true
@export_range(1.0, 10.0, 0.1) var success_rearm_distance_multiplier := 1.5
@export_range(1.0, 10.0, 0.1) var success_rearm_angle_multiplier := 1.5
## Requires the pose gate and the joint-speed gate to hold together.
@export var success_require_still := true
@export var success_max_joint_speed := 0.15
## Reach-and-HOLD shaping: the "near target" zone (metres) where the hold reward/penalty terms activate.
@export var near_target_distance := 0.20
## SEPARATE, much tighter distance that gates the stillness rewards (hold_stillness + near-target speed penalty).
@export var hold_activation_distance := 0.06
## Orientation range over which pose tracking shaping fades to zero.
@export_range(1.0, 180.0, 0.1) var pose_reward_angle_degrees := 60.0
## Relative contribution of orientation to get_progress().
@export_range(0.0, 1.0, 0.01) var orientation_progress_weight := 0.25
## Selects how distance becomes position progress.
@export var distance_progress_mode := DistanceProgressMode.LINEAR_CLAMPED
## ADDITIVE preserves the original position/orientation blend.
@export var pose_progress_mode := PoseProgressMode.ADDITIVE
## Joint speed (rad/s) at which the stillness reward decays to zero.
@export var hold_stillness_speed_reference := 0.5
## A target transform change larger than either threshold starts a new acquisition without requiring an episode reset.
@export var target_relocation_position_epsilon := 0.001
@export_range(0.0, 30.0, 0.1) var target_relocation_angle_epsilon_degrees := 1.0

@export_category("Safety")
@export var safety_volumes: Array[Area3D] = []
@export var obstacle_group := "robot_obstacle"
## When true a detected collision terminates the episode and stops the joints.
@export var collision_terminates := true
@export var auto_detect_environment_collisions := true
@export_flags_3d_physics var environment_collision_mask := 0xFFFFFFFF
@export_range(1, 64, 1) var max_collision_results_per_shape := 8
## Links allowed to touch an obstacle without ending the episode.
##
## Needed when the object to pick up rests on a surface that is itself an obstacle: the claws have to
## reach down past the object's centre, so the same floor that must stop a forearm must not stop a
## fingertip. Empty by default, so an arm that does not declare exemptions behaves exactly as before.
@export var environment_collision_exempt_links: PackedStringArray = []
## Treats configured arm-to-body contacts as terminal self-collisions.
@export var self_body_collision_enabled := false
@export var self_body_link_names: PackedStringArray = []
@export var self_check_link_names: PackedStringArray = []
## Prints the first detected self-body overlap of each episode.
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
## Calibration actuators excluded from the policy contract.
@export var manual_extra_joint_names := PackedStringArray()
## Independent gripper actuator controlled directly with the manual close/open actions.
@export var manual_gripper_joint_name: StringName = &""
@export var manual_debug_overlay := true
@export_dir var manual_capture_directory := "user://manual_arm_captures"
## Region sampled by the J key while capturing manual reference poses.
@export var manual_random_target_region: Node3D
@export var manual_random_target_extents := Vector3(0.05, 0.05, 0.05)
## Uses IK rather than per-joint keys during manual recording.
@export var ik_manual_enabled := false
@export_enum("auto", "teleop") var ik_manual_mode := "auto"
@export var ik_manual_target: Node3D
@export var ik_manual_track_orientation := true
@export_range(0.01, 5.0, 0.01) var ik_manual_max_speed := 0.5
var _ik_manual: URDFIKController = null
# Optional collision-free joint path used while recording demonstrations.
var _demo_plan: Array = []
var _demo_wp_index := 0
var _demo_dt_step := 0.05
## Physics frames during which collision termination is suppressed after a manual recovery.
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
## Gripper geometry, measured once from the collision shapes (see _calibrate_gripper_geometry).
## True on any physics frame where a collision was detected. Distinct from _collided, which latches
## for the whole episode: a per-STEP cost needs to know whether the arm is still inside something,
## not whether it ever was.
var _colliding_now := false
## Measure the pad separation across the whole travel instead of assuming it grows as `2 * travel`.
##
## Off preserves the exact linear model for symmetric prismatic grippers. Enable this for linkages
## whose pad separation is nonlinear over the actuator range, such as revolute finger pairs.
@export var measure_pad_gap_curve := false
## Measured pad separation against gripper travel, ascending. See _sample_pad_gap_curve.
const GAP_CURVE_SAMPLES := 9
var _gap_curve_travel: Array[float] = []
var _gap_curve_gap: Array[float] = []
var _pad_gap_closed := 0.0
var _gripper_calibrated := false
var _grasp_attached := false
## Pad-face overlap in the tool frame. The component along the opening axis is filled in live from
## the current gap (see grasp_window), so it is left at zero here.
var _pad_window_min := Vector3.ZERO
var _pad_window_max := Vector3.ZERO
var _pad_window_axis := Vector3.ZERO


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
	_calibrate_gripper_geometry()
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


## Measures gripper geometry from the active collision shapes.
func _calibrate_gripper_geometry() -> void:
	_gripper_calibrated = false
	if gripper_joint_name.is_empty() or gripper_finger_link_names.size() < 2:
		return
	var axis := _gripper_opening_axis()
	if axis == Vector3.ZERO:
		return
	var inner := _measure_inner_faces(axis)
	if inner.size() < 2:
		return
	var measured_gap: float = absf(float(inner[0]) - float(inner[1]))
	if measure_pad_gap_curve:
		_sample_pad_gap_curve(axis)
	_pad_gap_closed = (
		_pad_gap_at_travel(0.0)
		if not _gap_curve_travel.is_empty()
		else measured_gap - 2.0 * _gripper_position())
	_gripper_calibrated = true
	_measure_grasp_window(axis, inner)
	print("[urdf_robot_arm_agent] gripper '%s': travel %.4f, pad gap %.4f m (closed %.4f m, %.4f m fully open)" % [
		String(gripper_joint_name),
		_gripper_position(),
		measured_gap,
		_pad_gap_closed,
		_pad_gap_at_travel(_gripper_travel_limit())])
	print("[urdf_robot_arm_agent] grasp window in the tool frame: min (%+.4f, %+.4f, %+.4f) max (%+.4f, %+.4f, %+.4f)" % [
		_pad_window_min.x, _pad_window_min.y, _pad_window_min.z,
		_pad_window_max.x, _pad_window_max.y, _pad_window_max.z])


## Where an object has to be, in the tool frame, for the pads to close around it rather than push it
## away. Taken from the pad FACES -- the vertices lying on each finger's innermost plane -- and
## intersected between the two fingers, because only the overlap of the two faces can actually grip.
##
## The two axes across the face are fixed by the finger geometry and measured here. The third, along
## the opening direction, is the gap and therefore changes with the travel, so grasp_window_min/max
## fill it in live rather than freezing it at calibration time.
## Signed distance of each finger's INNERMOST collision vertex from the tool axis. The inner face is
## the one closest to the opposite finger, so the extreme with the smaller magnitude.
func _measure_inner_faces(axis: Vector3) -> Array:
	var inner := []
	for link_name in gripper_finger_link_names:
		var link := _robot.get_link_node(String(link_name))
		if link == null:
			push_warning(
				"URDFRobotArmAgentBody: finger link '%s' not found; gripper geometry "
				% String(link_name)
				+ "is uncalibrated and the grasp observation will fall back to the raw travel.")
			return []
		# Which side of the tool frame this finger is on. Taking the globally closest vertex assumed
		# each finger sits entirely on its own side; where the tool frame is not between the fingers
		# their meshes straddle it, both report an offset near zero, and the measured gap collapses
		# to a couple of millimetres. Restricting each finger to its own side fixes that, and is a
		# no-op wherever the assumption already held.
		var side := signf(axis.dot(link.global_position - tool_pose.global_position))
		if side == 0.0:
			side = 1.0
		var closest := INF
		var found := false
		for point in _link_collision_points(link):
			var offset: float = axis.dot(link.global_transform * point - tool_pose.global_position)
			if offset * side < 0.0:
				continue
			if not found or absf(offset) < absf(closest):
				closest = offset
				found = true
		if not found:
			return []
		inner.append(closest)
	return inner


## Pad separation as a function of gripper travel, MEASURED across the whole range.
##
## The linear `closed + 2 * travel` model is exact for a symmetric prismatic pair but not for
## arbitrary linkages. Sampling the real geometry costs a handful of forward-kinematics evaluations
## once at calibration and avoids robot-specific formulas.
func _sample_pad_gap_curve(axis: Vector3) -> void:
	_gap_curve_travel.clear()
	_gap_curve_gap.clear()
	var limit := _gripper_travel_limit()
	if limit <= 0.0:
		return

	# Every joint has to be restored, not just the gripper: reset_joint_positions rewrites them all.
	var restore := {}
	for joint_name in _robot.get_actuated_joint_names():
		restore[joint_name] = _robot.get_joint_position(String(joint_name))

	for index in range(GAP_CURVE_SAMPLES):
		var travel := limit * float(index) / float(GAP_CURVE_SAMPLES - 1)
		var sample := restore.duplicate()
		# Back to a raw joint value: the curve is indexed by travel from the closed end.
		sample[String(gripper_joint_name)] = travel + _gripper_closed_limit()
		_robot.reset_joint_positions(sample)
		var inner := _measure_inner_faces(axis)
		if inner.size() < 2:
			continue
		_gap_curve_travel.append(travel)
		_gap_curve_gap.append(absf(float(inner[0]) - float(inner[1])))

	_robot.reset_joint_positions(restore)


func _measure_grasp_window(axis: Vector3, inner: Array) -> void:
	var lo := Vector3(-INF, -INF, -INF)
	var hi := Vector3(INF, INF, INF)
	var tool_inverse := tool_pose.global_transform.affine_inverse()
	var tool_axis := (tool_pose.global_basis.orthonormalized().inverse() * axis).normalized()
	for index in range(gripper_finger_link_names.size()):
		var link := _robot.get_link_node(String(gripper_finger_link_names[index]))
		if link == null:
			return
		var face_lo := Vector3(INF, INF, INF)
		var face_hi := Vector3(-INF, -INF, -INF)
		var face_points := 0
		for point in _link_collision_points(link):
			var world: Vector3 = link.global_transform * point
			if absf(axis.dot(world - tool_pose.global_position) - float(inner[index])) > 0.001:
				continue
			var local: Vector3 = tool_inverse * world
			face_lo = face_lo.min(local)
			face_hi = face_hi.max(local)
			face_points += 1
		if face_points == 0:
			return
		# Intersection: an object has to be inside BOTH faces to be gripped by both of them.
		lo = lo.max(face_lo)
		hi = hi.min(face_hi)
	# The opening axis is filled in live from the current gap, so blank it here rather than leaving
	# the calibration-time value to masquerade as a fixed bound.
	for component in range(3):
		if absf(tool_axis[component]) > 0.5:
			lo[component] = 0.0
			hi[component] = 0.0
	_pad_window_min = lo
	_pad_window_max = hi
	_pad_window_axis = tool_axis


## Where the CENTRE of the object may be, in the tool frame, for the claws to close around it.
##
## Across the pad faces the bounds are the measured overlap. Along the opening axis they are NOT
## half the gap: an object of finite width has to be centred well enough that both pads reach it, so
## the room to spare is (gap - object width) / 2. At a perfect fit that collapses to zero, which is
## correct -- there is exactly one place a snug object can be. Half the gap would instead call a
## 15 mm miss "inside the claws" whenever they happen to be wide open.
##
## Empty when the gripper is uncalibrated, so a caller cannot mistake a default for a measurement.
func grasp_window() -> Dictionary:
	if not _gripper_calibrated:
		return {}
	var half_room := maxf((pad_gap() - maxf(grasp_object_width, 0.0)) * 0.5, 0.0)
	var lo := _pad_window_min
	var hi := _pad_window_max
	for component in range(3):
		if absf(_pad_window_axis[component]) > 0.5:
			lo[component] = -half_room
			hi[component] = half_room
	return {"min": lo, "max": hi}


## Vector from the tool to the demanded grasp pose, in the tool frame, in METRES. The same quantity
## the tool-frame observation reports, before normalisation and clipping -- capture decisions need
## the real distance, not the scaled one.
func grasp_offset_in_tool() -> Vector3:
	var desired_pose := _target_pose_node()
	var current_pose := _tool_pose_node()
	if not desired_pose or not current_pose:
		return Vector3.ZERO
	return current_pose.global_basis.orthonormalized().inverse() * (
		desired_pose.global_position - current_pose.global_position)


## Vector from the tool to the grasp point ON THE OBJECT, in the tool frame, in metres.
func object_offset_in_tool() -> Vector3:
	var current_pose := _tool_pose_node()
	if grasp_object_pose == null or current_pose == null:
		return Vector3.ZERO
	return current_pose.global_basis.orthonormalized().inverse() * (
		grasp_object_pose.global_position - current_pose.global_position)


func is_grasp_attached() -> bool:
	return _grasp_attached


## Set by whoever owns the capture rule (the scenario), because only it knows about the object.
## Kept here because the observation and the success state read it.
func set_grasp_attached(value: bool) -> void:
	_grasp_attached = value


## Unit vector, in world space, along which the two fingers separate.
func _gripper_opening_axis() -> Vector3:
	if gripper_finger_link_names.size() < 2 or tool_pose == null:
		return Vector3.ZERO
	var first := _robot.get_link_node(String(gripper_finger_link_names[0]))
	var second := _robot.get_link_node(String(gripper_finger_link_names[1]))
	if first == null or second == null:
		return Vector3.ZERO
	var span := first.global_position - second.global_position
	if span.length() < 0.000001:
		return Vector3.ZERO
	return span.normalized()


## Travel measured FROM THE CLOSED END, so zero always means shut and larger always means wider.
##
## Joint ranges do not necessarily start at zero. Subtracting the lower limit keeps travel positive
## and reduces to the raw joint position when that lower limit is already zero.
func _gripper_position() -> float:
	if gripper_joint_name.is_empty() or _robot == null:
		return 0.0
	return (
		_robot.get_joint_position(String(gripper_joint_name))
		- _gripper_closed_limit())


func _gripper_closed_limit() -> float:
	if gripper_joint_name.is_empty() or _robot == null:
		return 0.0
	var joint_node := _robot.get_joint_node(String(gripper_joint_name))
	if joint_node == null or joint_node.joint == null or joint_node.joint.limit == null:
		return 0.0
	return float(joint_node.joint.limit.lower)


func _gripper_travel_limit() -> float:
	if gripper_joint_name.is_empty() or _robot == null:
		return 0.0
	var joint_node := _robot.get_joint_node(String(gripper_joint_name))
	if joint_node == null or joint_node.joint == null or joint_node.joint.limit == null:
		return 0.0
	# Span, not the raw upper bound: see _gripper_position.
	return float(joint_node.joint.limit.upper) - float(joint_node.joint.limit.lower)


## Every collision vertex under a link, in that link's own frame.
func _link_collision_points(link: Node3D) -> PackedVector3Array:
	var points := PackedVector3Array()
	_gather_link_collision_points(link, link, points)
	return points


func _gather_link_collision_points(
		node: Node, link: Node3D, points: PackedVector3Array) -> void:
	for child in node.get_children():
		var shape_node := child as CollisionShape3D
		if shape_node != null and shape_node.shape != null:
			var to_link := link.global_transform.affine_inverse() * shape_node.global_transform
			var shape := shape_node.shape
			if shape is ConvexPolygonShape3D:
				for p in (shape as ConvexPolygonShape3D).points:
					points.append(to_link * p)
			elif shape is BoxShape3D:
				var half := (shape as BoxShape3D).size * 0.5
				for sx in [-1.0, 1.0]:
					for sy in [-1.0, 1.0]:
						for sz in [-1.0, 1.0]:
							points.append(to_link * Vector3(
								half.x * sx, half.y * sy, half.z * sz))
			elif shape is ConcavePolygonShape3D:
				for p in (shape as ConcavePolygonShape3D).get_faces():
					points.append(to_link * p)
		_gather_link_collision_points(child, link, points)


func _physics_process(delta: float) -> void:
	if _manual_collision_grace_frames > 0:
		_manual_collision_grace_frames -= 1
	if manual_control:
		_manual_overlay_elapsed += delta
		if _manual_overlay_elapsed >= 0.10:
			_manual_overlay_elapsed = 0.0
			_update_manual_overlay()
	# Before the training gate: the end-effector marker is a VIEW of the robot's pose, not part of
	# the episode. Leaving it behind the gate froze it the moment an agent was marked done, so the
	# arm kept being drawn while the marker -- and the tool frame and debug gizmos hanging off it --
	# stayed at the pose the episode ended in. Cheap, and it costs nothing when nothing has moved.
	_update_end_effector()
	if not _training_active:
		return
	_colliding_now = false
	# Manual control keeps moving after terminal events so several references can be captured.
	if manual_control:
		apply_manual_action()
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
	# Teleoperation starts position-only; T enables wrist control from the current pose.
	_ik_manual.track_orientation = false if ik_manual_mode == "teleop" else ik_manual_track_orientation
	# Teleoperation controls full orientation; automatic mode leaves roll free.
	_ik_manual.orientation_mode = "full" if ik_manual_mode == "teleop" else "pointing"
	_ik_manual.active = false  # driven manually from apply_manual_action
	add_child(_ik_manual)
	_ik_manual.set_robot(_robot)
	# The teleoperation marker is created on FIRST MANUAL USE, not here.
	#
	# Spawning it at startup put a bright yellow ball on the tool tip and left it there: the operator
	# moves it with the keyboard, so under policy control it never moves again. On a rendered
	# training run it sits where the arm happened to start while the arm flies away from it, which
	# reads exactly like an end-effector gizmo that has come unstuck -- and it was reported as one.
	# The marker is therefore created only when teleoperation actually starts.
	if ik_manual_mode == "teleop" and manual_control and ik_manual_target == null:
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
	marker.add_child(mesh)
	add_child(marker)
	var tcp: Node3D = _robot.get_link_node(tcp_link_name)
	if tcp != null:
		# Place the marker on the TCP, not the link origin.
		var tp := tcp.global_transform
		marker.global_transform = Transform3D(tp.basis, tp * tcp_local_offset)
	return marker


func _snap_ik_marker_to_target() -> void:
	# Move to the scenario target without changing wrist orientation.
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
		# Start wrist control from the current orientation.
		var tcp: Node3D = _robot.get_link_node(tcp_link_name)
		if tcp != null:
			ik_manual_target.global_transform.basis = tcp.global_transform.basis.orthonormalized()
	print("[ik] wrist control (auto-orientation) = ", _ik_manual.track_orientation)


func _apply_ik_manual_action() -> Array:
	# Created here rather than at startup, so a policy run never shows a stray teleop marker.
	if ik_manual_mode == "teleop" and ik_manual_target == null:
		ik_manual_target = _create_teleop_marker()
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


## Load a demo plan (list of 7-joint waypoint configs, home->goal).
func set_demo_plan(waypoints: Array, physics_frames_per_step: int = 3) -> void:
	_demo_plan = waypoints if waypoints != null else []
	_demo_wp_index = 0
	_demo_dt_step = maxf(float(physics_frames_per_step) / 60.0, 0.0001)


## Tracks planned waypoints through bounded joint-velocity commands.
func _apply_demo_plan_action() -> Array:
	var count := get_joint_count()
	# Skip reached waypoints before computing, including the initial home waypoint.
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
	# Velocity that would reach the waypoint in one decision step.
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
	# Uniform scaling preserves the joint-space segment validated by the planner.
	var scale: float = (1.0 / max_raw) if max_raw > 1.0 else 1.0
	var action: Array = []
	action.resize(count)
	for i in range(count):
		action[i] = clampf(float(raw[i]) * scale, -1.0, 1.0)
	# Keep autonomous termination enabled so accepted demos end with target_reached.
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

	# JSON is authoritative; viewport grabs are unsafe inside this input callback.
	print(
		"[METIS ARM MANUAL] Reference captured (JSON only): %s" %
			ProjectSettings.globalize_path(json_path))
	# Clear terminal state so another reference can be captured immediately.
	_terminal = false
	_collided = false
	_succeeded = false
	_success_frames = 0
	_manual_collision_grace_frames = manual_recovery_grace_physics_frames
	_update_manual_overlay()


## Samples a new manual target with the J key.
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
	var pose_progress := get_pose_progress()
	if not grasp_gates_progress:
		return pose_progress
	# Grasping: arriving is most of the journey but it is NOT the destination, so arriving alone can
	# never report full progress.
	#
	# Measured before this existed: a run that reached the glass and then did nothing at all until
	# the 500-step truncation banked 71.0, against 75.0 for actually grasping it -- 95% of the reward
	# for none of the work, and truncation charges no terminal penalty. Two things made loitering
	# free. The near-target terms pay about 0.11 every step with no cap, and the stall terminator is
	# exempt above progress 0.9 -- an exemption meant for reach-and-HOLD, where standing on the
	# target IS the task, and which here protected exactly the pose the agent would camp in.
	#
	# Capping the approach below that threshold restores the stall guard for an empty hand while
	# leaving it lifted for a full one: hold the object and progress passes 0.9, so holding still is
	# properly exempt.
	var share := clampf(grasp_progress_share, 0.0, 1.0)
	if not _grasp_attached:
		return pose_progress * share
	var hold_fraction := (
		clampf(float(_success_frames) / float(maxi(success_hold_physics_frames, 1)), 0.0, 1.0)
		if success_hold_physics_frames > 0
		else 1.0)
	return share + (1.0 - share) * hold_fraction


## Progress towards the demanded POSE, ignoring whether anything has been grasped. Kept separate so
## the grasp gate above composes with it instead of duplicating it.
func get_pose_progress() -> float:
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


## Returns the target error in the tool frame at a near-target scale.
func get_target_error_tool_observation() -> Vector3:
	var desired_pose := _target_pose_node()
	var current_pose := _tool_pose_node()
	if not desired_pose or not current_pose:
		return Vector3.ZERO
	var error_world := desired_pose.global_position - current_pose.global_position
	var error_tool := current_pose.global_basis.orthonormalized().inverse() * error_world
	var _scale := maxf(grasp_observation_scale, 0.001)
	return Vector3(
		clampf(error_tool.x / _scale, -1.0, 1.0),
		clampf(error_tool.y / _scale, -1.0, 1.0),
		clampf(error_tool.z / _scale, -1.0, 1.0))


## Gripper state: whether an object is currently held, and how the opening compares with the object it has to enclose.
## Con false il flag "sto tenendo l'oggetto" esce dall'osservazione e resta la sola apertura delle
## chele, che e' un segnale che il robot vero ha davvero.
##
## Il flag e' verita' del simulatore: sul banco non c'e' niente che lo misuri, a meno di corrente o
## forza sulla pinza. Misurato sulle 121 dimostrazioni, non serve: la sola apertura normalizzata
## separa "sto tenendo" nel 99.5% delle transizioni (mediana -0.011 con il cubo in mano contro +0.939
## a mano vuota), perche' i pad si fermano sull'oggetto. E' anche ridondante due volte, visto che la
## chiusura comandata e' gia' dentro `previous_action`.
##
## Default true: ogni braccio che non lo tocca si comporta esattamente come prima.
@export var grasp_state_reports_attachment := true


func get_grasp_state_observation() -> Array:
	if not grasp_state_reports_attachment:
		return [grasp_clearance_normalised()]
	return [
		1.0 if _grasp_attached else 0.0,
		grasp_clearance_normalised(),
	]


## Returns signed clearance between the gripper pads and the object.
func grasp_clearance_normalised() -> float:
	var closed_gap := _pad_gap_at_travel(0.0)
	var open_gap := _pad_gap_at_travel(_gripper_travel_limit())
	var reference := grasp_object_width
	if reference <= 0.0:
		# No object published by the scenario: report the raw opening over its own travel instead,
		# so the value stays a bounded, monotone description of the gripper rather than a constant.
		reference = (closed_gap + open_gap) * 0.5
	var span := maxf(
		maxf(absf(open_gap - reference), absf(reference - closed_gap)), 0.001)
	return clampf((pad_gap() - reference) / span, -1.0, 1.0)


## Current distance between the finger pad SURFACES, in metres. Measured off the collision geometry
## rather than taken from the URDF: the frame origins are 12 mm apart at zero travel, but the pads
## reach past each other and are 5 mm OVERLAPPED there, and it is the pads that meet the object.
##
## Deliberately signed. A negative gap means the pads have crossed, which is a real state for a
## kinematic gripper -- nothing stops it closing -- and it is exactly what grasp_penetration has to
## measure. Clamping it at zero would hide the squeeze this task is supposed to penalise, and would
## also shrink the normalisation span enough to push grasp_clearance_normalised onto its clamp at
## full opening.
func pad_gap() -> float:
	return _pad_gap_at_travel(_gripper_position())


## Reads the measured curve. Falls back to the prismatic formula only if sampling never ran.
func _pad_gap_at_travel(travel: float) -> float:
	var count := _gap_curve_travel.size()
	if count == 0:
		return _pad_gap_closed + 2.0 * travel
	if count == 1 or travel <= _gap_curve_travel[0]:
		return _gap_curve_gap[0]
	if travel >= _gap_curve_travel[count - 1]:
		return _gap_curve_gap[count - 1]
	for index in range(1, count):
		if travel <= _gap_curve_travel[index]:
			var span: float = _gap_curve_travel[index] - _gap_curve_travel[index - 1]
			if span <= 0.0:
				return _gap_curve_gap[index]
			var t: float = (travel - _gap_curve_travel[index - 1]) / span
			return lerpf(_gap_curve_gap[index - 1], _gap_curve_gap[index], t)
	return _gap_curve_gap[count - 1]


## Gripper travel that brings the pads exactly onto an object of the given width.
func grasp_closure_for_width(width: float) -> float:
	if width <= 0.0:
		return 0.0
	var count := _gap_curve_travel.size()
	if count < 2:
		return clampf(
			(width - _pad_gap_closed) * 0.5, 0.0, _gripper_travel_limit())
	# Inverse of the measured curve. Monotonic by construction: opening the claws widens the gap.
	for index in range(1, count):
		var lower_gap: float = _gap_curve_gap[index - 1]
		var upper_gap: float = _gap_curve_gap[index]
		if width <= upper_gap:
			var span: float = upper_gap - lower_gap
			if span <= 0.0:
				return _gap_curve_travel[index]
			var t: float = clampf((width - lower_gap) / span, 0.0, 1.0)
			return lerpf(_gap_curve_travel[index - 1], _gap_curve_travel[index], t)
	return _gripper_travel_limit()


## How far the pads have closed past the object's surface, in metres, and zero unless the object is
## actually BETWEEN them. Kinematic fingers cannot be stopped by contact, so "do not squeeze" has to
## be modelled rather than simulated -- but the depth itself is exactly computable.
##
## The "between them" test is not decoration. Without it, closing the claws on empty air scores the
## same as crushing the glass: measured, an approach that missed by 15 mm and then shut its claws
## collected -86.9 for squeezing nothing at all, which teaches the arm never to close rather than
## never to squeeze.
func grasp_penetration() -> float:
	if grasp_object_width <= 0.0 or not _gripper_calibrated:
		return 0.0
	var depth := maxf(grasp_object_width - pad_gap(), 0.0) * 0.5
	if depth <= 0.0 or not is_object_between_pads():
		return 0.0
	return depth


## Whether the object's grasp point lies in the volume the pads sweep: inside the measured pad faces
## across their two fixed axes, and no further off the centreline than its own half width along the
## opening axis -- past that the pads slide down the side of it instead of onto it.
func is_object_between_pads() -> bool:
	if not _gripper_calibrated or grasp_object_pose == null:
		return false
	var offset := object_offset_in_tool()
	var half_width := maxf(grasp_object_width, 0.0) * 0.5
	for component in range(3):
		if absf(_pad_window_axis[component]) > 0.5:
			if absf(offset[component]) > half_width:
				return false
		elif (
			offset[component] < _pad_window_min[component]
			or offset[component] > _pad_window_max[component]
		):
			return false
	return true


## Whether the object is anywhere in the way of the closing pads, as opposed to nicely centred
## between them. Each non-opening axis of the measured pad window is widened by the object's half
## width, which is the condition for the object to OVERLAP the pad sweep at all.
##
## Stopping and capturing are different questions and were wrongly asked with the same test.
## Stopping is a physical constraint -- kinematic fingers are re-posed from forward kinematics and
## cannot be pushed back by contact, so nothing but this check prevents them driving through solid
## matter -- and it has to be permissive. Capturing hands out the terminal reward and has to be
## strict. Sharing the strict test meant an object 6.4-10.4 mm off centre was neither stopped nor
## charged, and the policy learned to clench and drive in: measured at episode 5200, six of ten home
## "grasps" had the pads at 17-21 mm on a 40 mm glass with the object never between them.
func is_object_in_pad_path() -> bool:
	if not _gripper_calibrated or grasp_object_pose == null:
		return false
	var offset := object_offset_in_tool()
	var half_width := maxf(grasp_object_width, 0.0) * 0.5
	for component in range(3):
		if absf(_pad_window_axis[component]) > 0.5:
			if absf(offset[component]) > half_width:
				return false
		elif (
			offset[component] < _pad_window_min[component] - half_width
			or offset[component] > _pad_window_max[component] + half_width
		):
			return false
	return true


## Object offset along the axis the pads open on, in metres. Zero means dead centre.
func _object_offset_across_pads() -> float:
	if not _gripper_calibrated or grasp_object_pose == null:
		return 0.0
	var offset := object_offset_in_tool()
	for component in range(3):
		if absf(_pad_window_axis[component]) > 0.5:
			return offset[component]
	return 0.0


func get_previous_action_observation() -> Array:
	return _previous_action.duplicate()


## Per-step cost, -1 while any part of the arm is inside something, 0 otherwise.
##
## This is what makes a NON-terminating collision safe. Under kinematic control the table cannot
## push the arm out of the way -- links are re-posed from forward kinematics regardless -- so if a
## contact merely fires a one-off penalty and the episode continues, the cheapest route to the glass
## is straight THROUGH the table. Charging every frame of overlap keeps the corridor enforced while
## still letting an arm that grazes an edge recover and go on to grasp.
func get_collision_contact_penalty() -> float:
	return -1.0 if _colliding_now else 0.0


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
	# Reward stillness only near the target, without slowing the approach.
	if not _is_near_target_pose():
		return 0.0
	# Non-positive values follow the current success-speed gate.
	var reference := (
		hold_stillness_speed_reference
		if hold_stillness_speed_reference > 0.0
		else success_max_joint_speed)
	return clampf(1.0 - _max_joint_speed() / maxf(reference, 0.001), 0.0, 1.0)


func get_near_target_speed_penalty() -> float:
	# Penalize the fastest joint only while the arm is near the target.
	if not _is_near_target_pose():
		return 0.0
	return -_max_joint_speed()


func get_pose_tracking_reward() -> float:
	# Gate orientation by proximity so it cannot replace approaching the target.
	var position_score := 1.0 - clampf(
		_target_distance() / maxf(near_target_distance, 0.001), 0.0, 1.0)
	# Skip orientation entirely when the task declares it free. An exact zero check preserves the
	# configured reward for every task that gives orientation a non-zero weight.
	if orientation_progress_weight <= 0.0:
		return position_score
	var orientation_full := 1.0 - clampf(_target_angle_error() / PI, 0.0, 1.0)
	return position_score * (0.3 + 0.7 * orientation_full)


func get_near_target_distance_penalty() -> float:
	# Near the target, keep a monotonic cost for the remaining distance.
	var d := _target_distance()
	if d > near_target_distance:
		return 0.0
	return -clampf(d / maxf(near_target_distance, 0.001), 0.0, 1.0)


func get_hold_progress_reward() -> float:
	# Normalize hold progress so curriculum changes keep this term in [0, 1].
	if success_hold_physics_frames <= 0:
		return 0.0
	return clampf(
		float(_success_frames) / float(success_hold_physics_frames), 0.0, 1.0)


func get_max_joint_speed() -> float:
	return _max_joint_speed()


func get_hold_frames() -> int:
	return _success_frames


func get_debug_metrics() -> Dictionary:
	# ScenarioController merges these values into step diagnostics.
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
		# Include both frames so orientation errors can be inspected directly.
		"tool_pose_world": _pose_diagnostics(_tool_pose_node()),
		"target_pose_world": _pose_diagnostics(_target_pose_node()),
		"gripper_front": _gripper_front_direction(),
		# Reported from a rendered run: the object sits between the claws and the arm does not close.
		# Three numbers separate the readings of that, which call for opposite fixes -- a policy that
		# never commands closing (an incentive problem) versus a closing command that contact keeps
		# refusing (a geometry problem). pad_gap is where the claws actually are, the command is what
		# the policy asked for last step, and the window flag says whether the object was even in a
		# position to be caught.
		"pad_gap_m": pad_gap(),
		"gripper_command": _last_gripper_command(),
		"object_between_pads": is_object_between_pads(),
		# Published so the analysis does not have to hardcode the object it is looking at.
		"grasp_object_width_m": grasp_object_width,
		"self_clearance_m": self_body_clearance(),
		# How far off the tool's centreline the object sits, along the axis the pads open on. Beyond
		# half the pad face, contact no longer stops the claws and the leading pad shoves the object
		# aside instead of enclosing it -- reported from a rendered run as the glass squirting out of
		# the gripper as it closes.
		"object_offset_across_pads_m": _object_offset_across_pads(),
	}


## The gripper term of the most recent action, or zero when the gripper is not a controlled joint.
func _last_gripper_command() -> float:
	if gripper_joint_name.is_empty():
		return 0.0
	var index := Array(get_controlled_joint_names()).find(String(gripper_joint_name))
	if index < 0 or index >= _previous_action.size():
		return 0.0
	return float(_previous_action[index])


## Direction from the TCP link towards the fingertips, i.e.
func _gripper_front_direction() -> Dictionary:
	if _robot == null or tcp_link_name.is_empty():
		return {}
	var hand := _robot.get_link_node(tcp_link_name) as Node3D
	if hand == null:
		return {}
	var centre := Vector3.ZERO
	var count := 0
	for link_name in gripper_finger_link_names:
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
	# The finger axis removes the remaining roll ambiguity.
	if gripper_finger_link_names.size() >= 2:
		var first := _robot.get_link_node(gripper_finger_link_names[0]) as Node3D
		var second := _robot.get_link_node(gripper_finger_link_names[1]) as Node3D
		if first == null or second == null:
			return result
		var span := second.global_position - first.global_position
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


## Lets a scenario declare an obstacle collision the shape query cannot see.
##
## The source is carried through to the collision details and the terminal reason, so a scenario
## rule reads apart from a geometric hit in the logs instead of hiding inside "reported_obstacle".
func report_obstacle_collision(source: String = "reported_obstacle") -> void:
	_register_collision({"source": source})


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
	# The gripper needs its own cap, because it is not a joint like the others: its whole travel is
	# 44 mm, and the pads close twice as fast as the joint moves. At the arm's 0.5 m/s a single
	# control step (three physics frames) sweeps 50 mm of gap -- the entire useful range -- so the
	# action collapses to open-or-shut and "close carefully" is not something the policy could
	# express, let alone learn. Measured: a clean grasp still ended 17.5 mm inside a 40 mm object.
	if (
		gripper_max_speed > 0.0
		and not gripper_joint_name.is_empty()
		and joint_name == String(gripper_joint_name)
	):
		speed = minf(speed, gripper_max_speed)
	return maxf(speed, 0.000001)


## Stop the gripper where a real one would stop: on the object.
##
## Under kinematic control a finger cannot be halted by contact -- the link pose is written from
## forward kinematics every frame -- so commanding "close" drives the pads straight THROUGH a rigid
## body. It looks wrong and it is wrong: the object ends up impaled instead of held.
##
## Treating it purely as a reward problem left the geometry broken and asked the policy to learn a
## restraint the hardware imposes for free. So the stop is enforced here, as a mechanical limit
## alongside the joint's own end stops, and only when the object is genuinely between the pads --
## closing on empty air stays free.
func _gripper_command_blocked_by_object(joint_name: String, command: float) -> bool:
	if (
		not gripper_stops_on_contact
		or gripper_joint_name.is_empty()
		or joint_name != String(gripper_joint_name)
		or grasp_object_width <= 0.0
		or not _gripper_calibrated
	):
		return false
	# Closing is negative travel; opening is always allowed, which is what lets a grasp be released.
	if command >= 0.0:
		return false
	if pad_gap() > grasp_object_width:
		return false
	# Permissive on purpose -- see is_object_in_pad_path.
	return is_object_in_pad_path()


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
	if _gripper_command_blocked_by_object(joint_name, command):
		return 0.0
	var joint := _robot.urdf.get_joint(joint_name)
	if (
		not joint
		or joint.type not in ["revolute", "prismatic"]
		or not joint.limit
	):
		return command

	var lower := minf(joint.limit.lower, joint.limit.upper)
	var upper := maxf(joint.limit.lower, joint.limit.upper)
	# Keep the gripper off its own hard stop. Which end is "closed" depends on the joint: the closing
	# direction is whichever bound the travel runs towards, so the margin is taken off the end the
	# gripper approaches while shutting.
	if (
		gripper_closed_margin > 0.0
		and not gripper_joint_name.is_empty()
		and joint_name == String(gripper_joint_name)
		and upper - lower > gripper_closed_margin
	):
		if joint.limit.lower < joint.limit.upper:
			lower += gripper_closed_margin
		else:
			upper -= gripper_closed_margin
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
	# Grasping succeeds by HOLDING THE OBJECT, not by matching a prescribed pose. The pose is still
	# shaped and still rewarded -- it is how the arm learns a workable approach -- but it stops being
	# the definition of the goal, so a different approach that grips better is not punished for
	# disagreeing with ours. The hold counter below is unchanged: capture still has to persist.
	# Grasping ADDS to the reach condition, it does not replace it: the arm must be holding the
	# object AND still be in the demanded pose AND be still, for the usual hold duration. Holding
	# alone was satisfied by grabbing the glass and flying off with it -- seen in a rendered run --
	# which is a swipe, not a grasp. After capture the object rides the hand while the demanded basis
	# stays fixed in world, so these gates keep measuring how WELL it was taken and whether it is
	# kept there.
	var pose_held := (
		distance <= success_distance
		and angle <= deg_to_rad(success_angle_degrees)
		and still
	)
	var within_success := (pose_held and _grasp_attached) if success_requires_grasp else pose_held
	if success_blocked:
		within_success = false
	if _succeeded:
		var pose_rearm := (
			distance <= success_distance * success_rearm_distance_multiplier
			and angle <= (
				deg_to_rad(success_angle_degrees)
				* success_rearm_angle_multiplier)
		)
		var inside_rearm_zone := (
			(pose_rearm and _grasp_attached) if success_requires_grasp else pose_rearm)
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
	# Manual sessions report success without ending control.
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
		var checker_link := str(_robot_collision_link_by_shape_id.get(
			collision_shape.get_instance_id(), collision_shape.name))
		if environment_collision_exempt_links.has(checker_link):
			continue
		for hit in space_state.intersect_shape(query, max_collision_results_per_shape):
			var collider: Variant = hit.get("collider")
			if collider is Node and _node_or_parent_is_in_group(collider, obstacle_group):
				_register_collision({
					"source": "environment",
					"checker_link": checker_link,
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
	# Restrict each query to configured central-body links.
	for link_name in _robot.links.keys():
		var link_node = _robot.links[link_name]
		if link_node is CollisionObject3D and not body_names.has(link_name):
			_self_check_exclusions.append((link_node as CollisionObject3D).get_rid())
	# Only configured distal links act as self-collision checkers.
	for link_name in self_check_link_names:
		var node: Node = _robot.get_link_node(link_name)
		if node:
			var link_shapes: Array[CollisionShape3D] = []
			_collect_shapes_into(node, link_shapes)
			for collision_shape in link_shapes:
				_self_check_shapes.append(collision_shape)
				_self_check_link_by_shape_id[collision_shape.get_instance_id()] = link_name


## Bracketed clearance between the arm and its own body, in metres.
##
## Terminal collisions say WHETHER the arm touched itself, never how close it was the rest of the
## time -- and that is the question that decides the remedy. If the arm normally skims its own body
## and only occasionally crosses, the constraint needs a gradient pushing it away; if it normally
## keeps well clear and the hits are rare excursions, the trajectory is the problem, not the margin.
##
## Godot exposes no separation distance, so the gap is bracketed: the same overlap query is repeated
## with the shape inflated by each margin in turn, and the smallest margin that produces a hit puts
## the true clearance below it. Off by default -- it costs a query per shape per margin, per step.
@export var measure_self_clearance := false
const SELF_CLEARANCE_MARGINS: Array[float] = [0.005, 0.01, 0.02, 0.04, 0.08]


## Cached per physics frame: the reward term and the telemetry both read it, and the query is the
## expensive part. Without the cache the cost would be paid twice per step for the same answer.
var _self_clearance_cached := -1.0
var _self_clearance_frame := -1


func self_body_clearance() -> float:
	var frame := Engine.get_physics_frames()
	if frame == _self_clearance_frame:
		return _self_clearance_cached
	_self_clearance_frame = frame
	_self_clearance_cached = _measure_self_body_clearance()
	return _self_clearance_cached


func _measure_self_body_clearance() -> float:
	if (
		not measure_self_clearance
		or not self_body_collision_enabled
		or _self_check_shapes.is_empty()
		or _self_body_rids.is_empty()
		or not is_inside_tree()
	):
		return -1.0
	var space_state := get_world_3d().direct_space_state
	for margin in SELF_CLEARANCE_MARGINS:
		for collision_shape in _self_check_shapes:
			if not collision_shape or collision_shape.disabled or not collision_shape.shape:
				continue
			var query := PhysicsShapeQueryParameters3D.new()
			query.shape = collision_shape.shape
			query.transform = collision_shape.global_transform
			query.margin = margin
			query.collision_mask = environment_collision_mask
			query.collide_with_bodies = true
			query.collide_with_areas = false
			query.exclude = _self_check_exclusions
			for hit in space_state.intersect_shape(query, max_collision_results_per_shape):
				if _self_body_rids.has(hit.get("rid")):
					return margin
	# Farther than the widest bracket; reported as that bracket so the value stays comparable.
	return SELF_CLEARANCE_MARGINS[SELF_CLEARANCE_MARGINS.size() - 1]


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
	_colliding_now = true
	# Training always reports collisions; termination remains configurable.
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
	# Stillness shaping uses a tighter gate than pose shaping.
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
	# Use the fastest joint so one moving axis cannot pass the stillness gate.
	if not _robot:
		return 0.0
	var names := get_controlled_joint_names()
	if names.is_empty():
		return 0.0
	var fastest := 0.0
	for joint_name in names:
		fastest = maxf(fastest, absf(_robot.get_joint_velocity(joint_name)))
	return fastest
