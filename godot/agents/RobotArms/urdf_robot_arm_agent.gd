extends Node3D
class_name URDFRobotArmAgentBody

signal target_reached
signal obstacle_collision
signal self_collision
signal object_grasped
signal object_dropped

enum TaskMode {
	REACHING,
	GRASPING
}

@export_category("URDF Robot")
@export var robot_path: NodePath = NodePath("urdf")
@export var controlled_joint_names := PackedStringArray()
@export var use_kinematic_control := true
@export var joint_home_positions := PackedFloat32Array()
@export var default_joint_speed := 0.5

@export_category("Tool Center Point")
@export var tcp_link_name := ""
@export var tcp_local_offset := Vector3.ZERO
@export var end_effector_path: NodePath = NodePath("EndEffector")

@export_category("Task")
@export var task_mode: TaskMode = TaskMode.REACHING
@export var target: Node3D
@export var workspace_scale := 1.0
@export var success_distance := 0.05
@export var success_hold_physics_frames := 10
@export var terminate_on_success := true
@export_range(1.0, 10.0, 0.1) var success_rearm_distance_multiplier := 1.5

@export_category("Grasping")
@export var grasp_target_body: RigidBody3D
@export var gripper_joint_name := "grip_left"
@export var gripper_open_position := 0.0
@export var gripper_closed_position := -1.2
@export_range(0.0, 1.0, 0.01) var grasp_close_threshold := 0.55
@export_range(0.0, 1.0, 0.01) var grasp_release_threshold := 0.20
@export var grasp_capture_distance := 0.035
@export var required_lift_height := 0.05
@export var grasp_hold_physics_frames := 30
@export var max_pregrasp_planar_displacement := 0.04
@export var target_drop_height_tolerance := 0.05
@export var target_linear_velocity_scale := 0.5
@export var target_angular_velocity_scale := 8.0
@export var max_grasp_target_speed := 0.15
@export var assisted_grasp := true
# Real-grasp gate: the glass must sit BETWEEN the two fingers and be touched by BOTH before
# the assisted capture triggers. Stops the "close in the air near the glass" fake grasp and
# the glued-object exploit; the disturbance penalty then stays active until a genuine grasp.
@export var require_finger_contact := true
@export var grasp_finger_link_names := PackedStringArray(["finger_left_link", "finger_right_link"])
# Max perpendicular distance from the glass to the line between the two fingers (the gripper
# "mouth" centerline) for the glass to count as enclosed. Slack added to the finger span.
@export var grasp_enclosure_tolerance := 0.03

@export_category("Safety")
@export var safety_volumes: Array[Area3D] = []
@export var obstacle_group := "robot_obstacle"
@export var auto_detect_environment_collisions := true
@export_flags_3d_physics var environment_collision_mask := 0xFFFFFFFF
@export_range(1, 64, 1) var max_collision_results_per_shape := 8
@export var support_surface_group := "robot_support_surface"
@export var support_contact_link_names := PackedStringArray()
@export var auto_detect_self_collisions := true
@export var ignore_adjacent_link_collisions := true
@export var self_collision_ignored_link_pairs := PackedStringArray()
@export var self_collision_ignored_link_sets := PackedStringArray()
@export_flags_3d_physics var self_collision_mask := 0xFFFFFFFF

@export_category("Control")
@export var manual_control := false
@export var auto_configure_action_size := true

@onready var agent: Agent = $Agent

var _robot: GodotRobot
var end_effector: Node3D
var _joint_names := PackedStringArray()
var _commands: Array[float] = []
var _previous_action: Array[float] = []
var _pending_reset_offsets: Array[float] = []
var _robot_collision_shapes: Array[CollisionShape3D] = []
var _robot_collision_exclusions: Array[RID] = []
var _collision_shape_owner: Dictionary = {}
var _robot_link_name_by_body_id: Dictionary = {}
var _robot_body_by_link_name: Dictionary = {}
var _ignored_self_collision_pairs: Dictionary = {}
var _terminal := false
var _succeeded := false
var _collided := false
var _self_collided := false
var _last_collision_info: Dictionary = {}
var _success_frames := 0
var _training_active := true
var _grasped := false
var _drop_registered := false
var _lift_hold_frames := 0
var _grasp_target_offset := Transform3D.IDENTITY
var _grasp_target_spawn_transform := Transform3D.IDENTITY
var _grasp_target_was_frozen := false
var _grasp_target_previous_freeze_mode := RigidBody3D.FREEZE_MODE_STATIC
var _grasp_finger_bodies: Array[Node3D] = []
var _grasp_finger_bodies_cached := false


func _ready() -> void:
	_robot = get_node_or_null(robot_path) as GodotRobot
	end_effector = get_node_or_null(end_effector_path) as Node3D
	if not _robot:
		push_error("URDFRobotArmAgentBody: GodotRobot not found at '%s'." % robot_path)
		set_physics_process(false)
		return

	_robot.set_control_mode(
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

	_resize_state()
	_configure_action_size()
	_connect_safety_volumes()
	_cache_robot_collision_geometry()
	_robot.reset_joint_positions(_build_home_positions())
	_update_end_effector()
	if grasp_target_body:
		set_grasp_target_spawn_transform(grasp_target_body.global_transform)


func _physics_process(_delta: float) -> void:
	if not _training_active:
		return
	if manual_control and not _terminal:
		apply_manual_action()
	_update_end_effector()
	_check_self_collisions()
	_check_environment_collisions()
	if task_mode == TaskMode.GRASPING:
		_update_grasping_state()
	else:
		_update_reaching_success_state()


func apply_action(action: Variant) -> Variant:
	if (
		(typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME)
		and str(action) == "manual"
	):
		return apply_manual_action()

	manual_control = false
	var values := agent.decode_continuous_action(action)
	for index in range(get_joint_count()):
		var command := (
			clampf(float(values[index]), -1.0, 1.0)
			if index < values.size()
			else 0.0)
		_commands[index] = command
		_previous_action[index] = command
		_robot.set_joint_target_velocity(
			_joint_names[index], command * _joint_max_speed(index))
	return _previous_action.duplicate()


func apply_manual_action() -> Array:
	var values: Array = []
	values.resize(get_joint_count())
	for index in range(get_joint_count()):
		var negative := StringName("joint_%d_negative" % index)
		var positive := StringName("joint_%d_positive" % index)
		values[index] = (
			Input.get_action_strength(positive)
			- Input.get_action_strength(negative))
	var applied: Array = apply_action(values)
	manual_control = true
	return applied


func reset_all(original_transform: Variant, reset_rewards := true) -> void:
	set_training_active(true)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform

	_terminal = false
	_succeeded = false
	_collided = false
	_self_collided = false
	_last_collision_info.clear()
	_success_frames = 0
	_reset_grasp_episode_state()
	for index in range(get_joint_count()):
		_commands[index] = 0.0
		_previous_action[index] = 0.0
	_robot.reset_joint_positions(_build_home_positions(true))
	_pending_reset_offsets.clear()
	_update_end_effector()

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
	_self_collided = false
	_last_collision_info.clear()
	_success_frames = 0
	_drop_registered = false
	_lift_hold_frames = 0
	set_training_active(true)
	for index in range(get_joint_count()):
		_commands[index] = 0.0
		_previous_action[index] = 0.0
	_robot.stop_all_joints()
	_update_end_effector()
	agent.reset_observation_sources()


func configure_grasp_target(body: RigidBody3D, grasp_point: Node3D = null) -> void:
	grasp_target_body = body
	target = grasp_point if grasp_point != null else body
	if body:
		set_grasp_target_spawn_transform(body.global_transform)


func prepare_grasp_target_reset() -> void:
	_detach_grasp_target(false)
	_grasped = false
	_drop_registered = false
	_lift_hold_frames = 0
	_succeeded = false
	_success_frames = 0


func set_grasp_target_spawn_transform(spawn_transform: Transform3D) -> void:
	_grasp_target_spawn_transform = spawn_transform


func is_object_grasped() -> bool:
	return _grasped


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


func has_self_collided() -> bool:
	return _self_collided


func get_last_collision_info() -> Dictionary:
	return _last_collision_info.duplicate(true)


func set_continue_after_success(enabled: bool) -> void:
	terminate_on_success = not enabled
	if enabled and _succeeded and not _collided:
		_terminal = false
		set_training_active(true)


func get_progress() -> float:
	if task_mode == TaskMode.GRASPING:
		var approach_progress := 1.0 - clampf(
			_target_distance() / maxf(workspace_scale, 0.001), 0.0, 1.0)
		if not _grasped:
			return approach_progress * 0.70
		var lift_progress := clampf(
			_target_lift_height() / maxf(required_lift_height, 0.001), 0.0, 1.0)
		return 0.75 + lift_progress * 0.25
	return 1.0 - clampf(
		_target_distance() / maxf(workspace_scale, 0.001), 0.0, 1.0)


func get_joint_count() -> int:
	return _joint_names.size()


func get_controlled_joint_names() -> PackedStringArray:
	return _joint_names.duplicate()


func get_joint_position_observation() -> Array:
	var result: Array = []
	for joint_name in _joint_names:
		var joint := _robot.urdf.get_joint(joint_name)
		var _position := _robot.get_joint_position(joint_name)
		if joint and joint.type == "revolute" and joint.limit:
			var lower := minf(joint.limit.lower, joint.limit.upper)
			var upper := maxf(joint.limit.lower, joint.limit.upper)
			result.append(remap(_position, lower, upper, -1.0, 1.0))
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
	if not target or not end_effector:
		return Vector3.ZERO
	var error_world := target.global_position - end_effector.global_position
	var error_base := global_transform.basis.inverse() * error_world
	var _scale := maxf(workspace_scale, 0.001)
	return Vector3(
		clampf(error_base.x / _scale, -1.0, 1.0),
		clampf(error_base.y / _scale, -1.0, 1.0),
		clampf(error_base.z / _scale, -1.0, 1.0))


func get_previous_action_observation() -> Array:
	return _previous_action.duplicate()


func get_target_motion_observation() -> Array:
	if not grasp_target_body:
		return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	var linear_local := global_transform.basis.inverse() * grasp_target_body.linear_velocity
	var angular_local := global_transform.basis.inverse() * grasp_target_body.angular_velocity
	var linear_scale := maxf(target_linear_velocity_scale, 0.001)
	var angular_scale := maxf(target_angular_velocity_scale, 0.001)
	return [
		clampf(linear_local.x / linear_scale, -1.0, 1.0),
		clampf(linear_local.y / linear_scale, -1.0, 1.0),
		clampf(linear_local.z / linear_scale, -1.0, 1.0),
		clampf(angular_local.x / angular_scale, -1.0, 1.0),
		clampf(angular_local.y / angular_scale, -1.0, 1.0),
		clampf(angular_local.z / angular_scale, -1.0, 1.0)
	]


func get_grasp_state_observation() -> Array:
	var lift_ratio := clampf(
		_target_lift_height() / maxf(required_lift_height, 0.001), 0.0, 1.0)
	return [
		_gripper_closed_fraction(),
		1.0 if _grasped else 0.0,
		lift_ratio
	]


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
		if not joint or joint.type != "revolute" or not joint.limit:
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


func get_target_disturbance_penalty() -> float:
	if task_mode != TaskMode.GRASPING or not grasp_target_body or _grasped:
		return 0.0
	var linear_ratio := grasp_target_body.linear_velocity.length() / maxf(
		target_linear_velocity_scale, 0.001)
	var angular_ratio := grasp_target_body.angular_velocity.length() / maxf(
		target_angular_velocity_scale, 0.001)
	return -clampf((linear_ratio + angular_ratio) * 0.5, 0.0, 1.0)


func get_premature_gripper_penalty() -> float:
	if task_mode != TaskMode.GRASPING or _grasped:
		return 0.0
	var safe_close_distance := maxf(grasp_capture_distance * 1.5, 0.001)
	if _target_distance() <= safe_close_distance:
		return 0.0
	return -_gripper_closed_fraction()


func get_empty_grasp_penalty() -> float:
	# Fingers closed onto each other with no glass between them = grasp attempted, target
	# missed. Kinematic fingers do not stop on contact, so this is detected geometrically.
	# Only penalise a genuine miss: overlapping AND not near the glass. Closing near the glass
	# is a legitimate attempt, rewarded by get_grasp_closing_reward instead of punished here.
	if task_mode != TaskMode.GRASPING or _grasped:
		return 0.0
	if _fingers_closed_on_nothing() and _target_distance() > grasp_capture_distance * 1.5:
		return -1.0
	return 0.0


func get_grasp_closing_reward() -> float:
	# Positive bootstrap: reward closing the gripper WHEN the glass is in the mouth (near and
	# between the fingers). Counters the pure-penalty trap where the policy learns never to
	# close. Capture triggers once conditions hold, so it cannot be farmed for long.
	if task_mode != TaskMode.GRASPING or _grasped:
		return 0.0
	if _target_distance() > grasp_capture_distance * 1.5:
		return 0.0
	if not _glass_between_fingers():
		return 0.0
	return _gripper_closed_fraction()


func _ensure_grasp_finger_bodies() -> void:
	# Lazy: resolving the fingers inside _ready shifted the first physics frame and tripped a
	# borderline self-collision in the reaching test. Do it on first grasp use instead, once.
	if _grasp_finger_bodies_cached:
		return
	_grasp_finger_bodies_cached = true
	_cache_grasp_finger_bodies()


func _cache_grasp_finger_bodies() -> void:
	# Resolve from the already-built collision map (populated by _cache_robot_collision_geometry,
	# which must run first). NOT _robot.get_link_node(): a cache miss there rebuilds the link
	# index and re-adds every joint, corrupting the self-collision RIDs cached just before.
	_grasp_finger_bodies.clear()
	for link_name in grasp_finger_link_names:
		var node := _robot_body_by_link_name.get(str(link_name)) as Node3D
		if node:
			_grasp_finger_bodies.append(node)
	if require_finger_contact and _grasp_finger_bodies.size() < 2:
		push_warning(
			("URDFRobotArmAgentBody: grasp finger links %s not resolved; "
			+ "falling back to distance-only capture.") % [grasp_finger_link_names])


func _glass_between_fingers() -> bool:
	# The glass must lie inside the gripper "mouth": between the two fingers along the closing
	# axis, and close to the line joining them. Unresolved fingers -> skip (old behavior).
	if not require_finger_contact:
		return true
	if _grasp_finger_bodies.size() < 2 or not grasp_target_body:
		return true
	var left := _grasp_finger_bodies[0].global_position
	var right := _grasp_finger_bodies[1].global_position
	var separation := right - left
	var separation_length := separation.length()
	if separation_length < 0.0001:
		return false
	var axis := separation / separation_length
	var center := (left + right) * 0.5
	var offset := grasp_target_body.global_position - center
	var lateral := offset.dot(axis)
	var perpendicular := (offset - axis * lateral).length()
	# The glass must be roughly CENTERED between the fingers, not merely somewhere on the line
	# joining them. At one finger perpendicular is ~0 and |lateral| ~= half the span; the old
	# span check accepted that, so a one-sided touch could capture. Require |lateral| small
	# (central portion of the gripper mouth), auto-scaled to the current opening.
	return (
		absf(lateral) <= separation_length * 0.30
		and perpendicular <= grasp_enclosure_tolerance)


func _both_fingers_contact_glass() -> bool:
	if not require_finger_contact:
		return true
	# Unresolved fingers -> skip (warned at startup); otherwise both must touch the glass.
	if _grasp_finger_bodies.size() < 2:
		return true
	if not grasp_target_body or not grasp_target_body.contact_monitor:
		return false
	var touching := grasp_target_body.get_colliding_bodies()
	return _grasp_finger_bodies[0] in touching and _grasp_finger_bodies[1] in touching


func _fingers_closed_on_nothing() -> bool:
	# The finger collision hulls never touch each other (they stay ~4.8 cm apart even fully
	# closed), so finger-finger collision is undetectable -- the visible overlap is only the
	# meshes. The functional failure is the gripper closed while NEITHER finger is on the
	# glass. Finger-glass contact IS detectable: the glass is dynamic, so kinematic-vs-dynamic
	# generates a reported contact (unlike kinematic-vs-kinematic).
	if _gripper_closed_fraction() < grasp_close_threshold:
		return false
	return not _both_fingers_contact_glass()


func get_control_input(input_name: String) -> float:
	if not input_name.begins_with("joint_velocity_"):
		return 0.0
	var index := int(input_name.trim_prefix("joint_velocity_"))
	return _commands[index] if index >= 0 and index < _commands.size() else 0.0


func report_obstacle_collision() -> void:
	_register_collision(false)


func report_self_collision() -> void:
	_register_collision(true)


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
	var joint := _robot.urdf.get_joint(_joint_names[index])
	if joint and joint.limit and joint.limit.velocity > 0.0:
		return joint.limit.velocity
	return maxf(default_joint_speed, 0.000001)


func _update_end_effector() -> void:
	if not end_effector or tcp_link_name.is_empty() or not _robot:
		return
	var tcp_link := _robot.get_link_node(tcp_link_name)
	if not tcp_link:
		return
	end_effector.global_transform = tcp_link.global_transform.translated_local(
		tcp_local_offset)


func _update_reaching_success_state() -> void:
	if _terminal or not target or not end_effector:
		return
	var distance := _target_distance()
	if _succeeded:
		if (
			not terminate_on_success
			and distance > success_distance * success_rearm_distance_multiplier
		):
			_succeeded = false
			_success_frames = 0
		return
	if distance <= success_distance:
		_success_frames += 1
	else:
		_success_frames = 0
	if _success_frames < success_hold_physics_frames:
		return
	_complete_task()


func _update_grasping_state() -> void:
	if _terminal or not grasp_target_body or not target or not end_effector:
		return
	_ensure_grasp_finger_bodies()

	if _grasped:
		_sync_grasp_target()
		if _gripper_closed_fraction() < grasp_release_threshold:
			_detach_grasp_target(true)
			_register_drop()
			return
		if _target_lift_height() >= required_lift_height:
			_lift_hold_frames += 1
		else:
			_lift_hold_frames = 0
		if _lift_hold_frames >= max(1, grasp_hold_physics_frames):
			_complete_task()
		return

	if _target_was_disturbed_before_grasp():
		_register_drop()
		return

	var can_capture := (
		_target_distance() <= grasp_capture_distance
		and _glass_between_fingers()
		and _gripper_closed_fraction() >= grasp_close_threshold
		and _both_fingers_contact_glass()
		and grasp_target_body.linear_velocity.length() <= max_grasp_target_speed
	)
	if can_capture:
		_attach_grasp_target()


func _complete_task() -> void:
	if _succeeded:
		return
	_succeeded = true
	if terminate_on_success:
		_terminal = true
		_robot.stop_all_joints()
	target_reached.emit()


func _attach_grasp_target() -> void:
	if _grasped or not grasp_target_body or not end_effector:
		return
	_grasped = true
	_lift_hold_frames = 0
	_grasp_target_offset = (
		end_effector.global_transform.affine_inverse()
		* grasp_target_body.global_transform)
	_grasp_target_was_frozen = grasp_target_body.freeze
	_grasp_target_previous_freeze_mode = grasp_target_body.freeze_mode
	if assisted_grasp:
		grasp_target_body.freeze_mode = RigidBody3D.FREEZE_MODE_KINEMATIC
		grasp_target_body.freeze = true
		grasp_target_body.linear_velocity = Vector3.ZERO
		grasp_target_body.angular_velocity = Vector3.ZERO
	object_grasped.emit()


func _sync_grasp_target() -> void:
	if not assisted_grasp or not _grasped or not grasp_target_body or not end_effector:
		return
	grasp_target_body.global_transform = (
		end_effector.global_transform * _grasp_target_offset)
	grasp_target_body.linear_velocity = Vector3.ZERO
	grasp_target_body.angular_velocity = Vector3.ZERO


func _detach_grasp_target(mark_as_release: bool) -> void:
	if not grasp_target_body:
		_grasped = false
		return
	if assisted_grasp and _grasped:
		grasp_target_body.freeze_mode = _grasp_target_previous_freeze_mode
		grasp_target_body.freeze = _grasp_target_was_frozen
		if not grasp_target_body.freeze:
			grasp_target_body.sleeping = false
	_grasped = false
	_lift_hold_frames = 0
	if mark_as_release:
		grasp_target_body.linear_velocity = Vector3.ZERO
		grasp_target_body.angular_velocity = Vector3.ZERO


func _reset_grasp_episode_state() -> void:
	_detach_grasp_target(false)
	_grasped = false
	_drop_registered = false
	_lift_hold_frames = 0


func _target_was_disturbed_before_grasp() -> bool:
	if not grasp_target_body:
		return false
	var offset := grasp_target_body.global_position - _grasp_target_spawn_transform.origin
	var planar_displacement := Vector2(offset.x, offset.z).length()
	var fell := offset.y < -absf(target_drop_height_tolerance)
	return planar_displacement > max_pregrasp_planar_displacement or fell


func _register_drop() -> void:
	if _drop_registered or _terminal:
		return
	_drop_registered = true
	_terminal = true
	_robot.stop_all_joints()
	object_dropped.emit()


func _gripper_closed_fraction() -> float:
	if not _robot or gripper_joint_name.is_empty():
		return 0.0
	var denominator := gripper_closed_position - gripper_open_position
	if is_zero_approx(denominator):
		return 0.0
	return clampf(
		(_robot.get_joint_position(gripper_joint_name) - gripper_open_position)
		/ denominator,
		0.0,
		1.0)


func _target_lift_height() -> float:
	if not grasp_target_body:
		return 0.0
	return maxf(
		grasp_target_body.global_position.y - _grasp_target_spawn_transform.origin.y,
		0.0)


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
		_register_collision(false)


func _on_safety_area_entered(other: Area3D) -> void:
	if _node_or_parent_is_in_group(other, obstacle_group):
		_register_collision(false)


func _cache_robot_collision_geometry() -> void:
	_robot_collision_shapes.clear()
	_robot_collision_exclusions.clear()
	_collision_shape_owner.clear()
	_robot_link_name_by_body_id.clear()
	_robot_body_by_link_name.clear()
	_ignored_self_collision_pairs.clear()
	if not _robot:
		return

	for link_name in _robot.links.keys():
		var link_node: Node3D = _robot.links[link_name]
		if not link_node is Node:
			continue
		if link_node is CollisionObject3D:
			_robot_collision_exclusions.append(link_node.get_rid())
			_robot_link_name_by_body_id[link_node.get_instance_id()] = str(link_name)
			_robot_body_by_link_name[str(link_name)] = link_node
			_collect_collision_shapes(link_node, link_node)

	if ignore_adjacent_link_collisions and _robot.urdf:
		for joint in _robot.urdf.joints:
			_ignore_self_collision_pair(joint.parent, joint.child)
	for pair_spec in self_collision_ignored_link_pairs:
		var names := str(pair_spec).split(":", false, 1)
		if names.size() != 2:
			push_warning(
				"URDFRobotArmAgentBody: invalid ignored self-collision pair '%s'; " +
				"use 'link_a:link_b'." % pair_spec)
			continue
		_ignore_self_collision_pair(names[0].strip_edges(), names[1].strip_edges())
	for set_spec in self_collision_ignored_link_sets:
		var names := str(set_spec).split(",", false)
		for first_index in range(names.size()):
			for second_index in range(first_index + 1, names.size()):
				_ignore_self_collision_pair(
					names[first_index].strip_edges(),
					names[second_index].strip_edges())


func _collect_collision_shapes(node: Node, owner: CollisionObject3D) -> void:
	for child in node.get_children():
		if child is CollisionShape3D:
			_robot_collision_shapes.append(child)
			_collision_shape_owner[child.get_instance_id()] = owner
		_collect_collision_shapes(child, owner)


func _check_environment_collisions() -> void:
	if (
		_terminal
		or not auto_detect_environment_collisions
		or _robot_collision_shapes.is_empty()
		or not is_inside_tree()
	):
		return

	var space_state := get_world_3d().direct_space_state
	for collision_shape in _robot_collision_shapes:
		if not collision_shape or collision_shape.disabled or not collision_shape.shape:
			continue
		var owner := _collision_owner(collision_shape)
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
				if _is_allowed_support_contact(owner, collider):
					continue
				_register_collision(
					false,
					_robot_link_name(owner),
					str(collider.name))
				return


func _check_self_collisions() -> void:
	if (
		_terminal
		or not auto_detect_self_collisions
		or _robot_collision_shapes.is_empty()
		or not is_inside_tree()
	):
		return

	var space_state := get_world_3d().direct_space_state
	for collision_shape in _robot_collision_shapes:
		if not collision_shape or collision_shape.disabled or not collision_shape.shape:
			continue
		var owner := _collision_owner(collision_shape)
		if not owner:
			continue
		var owner_name := _robot_link_name(owner)
		var query := PhysicsShapeQueryParameters3D.new()
		query.shape = collision_shape.shape
		query.transform = collision_shape.global_transform
		query.collision_mask = self_collision_mask
		query.collide_with_bodies = true
		query.collide_with_areas = false
		query.exclude = _self_collision_exclusions(owner_name)
		for hit in space_state.intersect_shape(query, max_collision_results_per_shape):
			var collider: Variant = hit.get("collider")
			if not collider is CollisionObject3D:
				continue
			var collider_name := _robot_link_name(collider)
			if collider_name.is_empty():
				continue
			if not _is_ignored_self_collision_pair(owner_name, collider_name):
				_register_collision(true, owner_name, collider_name)
				return


func _collision_owner(collision_shape: CollisionShape3D) -> CollisionObject3D:
	return _collision_shape_owner.get(collision_shape.get_instance_id()) as CollisionObject3D


func _robot_link_name(body: CollisionObject3D) -> String:
	return str(_robot_link_name_by_body_id.get(body.get_instance_id(), ""))


func _is_allowed_support_contact(owner: CollisionObject3D, collider: Node) -> bool:
	if not owner or support_surface_group.is_empty():
		return false
	if not _node_or_parent_is_in_group(collider, support_surface_group):
		return false
	return support_contact_link_names.has(_robot_link_name(owner))


func _self_collision_exclusions(owner_name: String) -> Array[RID]:
	var exclusions: Array[RID] = []
	for link_name in _robot_body_by_link_name.keys():
		if _is_ignored_self_collision_pair(owner_name, str(link_name)):
			var body := _robot_body_by_link_name[link_name] as CollisionObject3D
			if body:
				exclusions.append(body.get_rid())
	return exclusions


func _ignore_self_collision_pair(first: String, second: String) -> void:
	if first.is_empty() or second.is_empty():
		return
	_ignored_self_collision_pairs[_self_collision_pair_key(first, second)] = true


func _is_ignored_self_collision_pair(first: String, second: String) -> bool:
	if first == second:
		return true
	return _ignored_self_collision_pairs.has(
		_self_collision_pair_key(first, second))


func _self_collision_pair_key(first: String, second: String) -> String:
	var names := PackedStringArray([first, second])
	names.sort()
	return "%s|%s" % [names[0], names[1]]


func _node_or_parent_is_in_group(node: Node, group_name: String) -> bool:
	var current: Node = node
	while current:
		if current.is_in_group(group_name):
			return true
		current = current.get_parent()
	return false


func _register_collision(
		is_self_collision := false,
		first_body := "",
		second_body := "") -> void:
	if _terminal:
		return
	_collided = true
	_self_collided = is_self_collision
	_last_collision_info = {
		"type": "self" if is_self_collision else "environment",
		"first_body": first_body,
		"second_body": second_body,
	}
	_terminal = true
	_robot.stop_all_joints()
	if is_self_collision:
		self_collision.emit()
	else:
		obstacle_collision.emit()


func _target_distance() -> float:
	if not target or not end_effector:
		return workspace_scale
	return end_effector.global_position.distance_to(target.global_position)
