extends Node3D
class_name URDFRobotArmAgentBody

signal target_reached
signal obstacle_collision
signal target_pose_relocated

@export_category("URDF Robot")
@export var robot_path: NodePath = NodePath("urdf")
@export var controlled_joint_names := PackedStringArray()
@export var use_kinematic_control := true
@export var joint_home_positions := PackedFloat32Array()
@export var default_joint_speed := 0.5
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
@export var near_target_distance := 0.10
## Orientation range over which pose tracking shaping fades to zero.
@export_range(1.0, 180.0, 0.1) var pose_reward_angle_degrees := 60.0
## Relative contribution of orientation to get_progress().
@export_range(0.0, 1.0, 0.01) var orientation_progress_weight := 0.25
## Joint speed (rad/s) at which the stillness reward decays to zero.
@export var hold_stillness_speed_reference := 0.5
## A target transform change larger than either threshold starts a new acquisition without
## requiring an episode reset. Small rigid-body jitter is ignored.
@export var target_relocation_position_epsilon := 0.001
@export_range(0.0, 30.0, 0.1) var target_relocation_angle_epsilon_degrees := 1.0

@export_category("Safety")
@export var safety_volumes: Array[Area3D] = []
@export var obstacle_group := "robot_obstacle"
@export var auto_detect_environment_collisions := true
@export_flags_3d_physics var environment_collision_mask := 0xFFFFFFFF
@export_range(1, 64, 1) var max_collision_results_per_shape := 8

@export_category("Control")
@export var manual_control := false
@export var auto_configure_action_size := true

@onready var agent: Agent = $Agent

var _robot: GodotRobot
var end_effector: Node3D
var tool_pose: Node3D
var _joint_names := PackedStringArray()
var _commands: Array[float] = []
var _previous_action: Array[float] = []
var _pending_reset_offsets: Array[float] = []
var _robot_collision_shapes: Array[CollisionShape3D] = []
var _robot_collision_exclusions: Array[RID] = []
var _terminal := false
var _succeeded := false
var _collided := false
var _success_frames := 0
var _training_active := true
var _last_target_pose_transform := Transform3D.IDENTITY
var _has_last_target_pose_transform := false


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

	_resize_state()
	_configure_action_size()
	_connect_safety_volumes()
	_cache_robot_collision_geometry()
	_robot.reset_joint_positions(_build_home_positions())
	_update_end_effector()
	_capture_target_pose_transform()


func _physics_process(_delta: float) -> void:
	if not _training_active:
		return
	if manual_control and not _terminal:
		apply_manual_action()
	_update_end_effector()
	_detect_target_pose_relocation()
	_check_environment_collisions()
	_update_success_state()


func apply_action(action: Variant) -> Variant:
	if (
		(typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME)
		and str(action) == "manual"
	):
		return apply_manual_action()

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
	_success_frames = 0
	_has_last_target_pose_transform = false
	for index in range(get_joint_count()):
		_commands[index] = 0.0
		_previous_action[index] = 0.0
	_robot.reset_joint_positions(_build_home_positions(true))
	_pending_reset_offsets.clear()
	_update_end_effector()
	_capture_target_pose_transform()

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


func set_continue_after_success(enabled: bool) -> void:
	terminate_on_success = not enabled
	if enabled and _succeeded and not _collided:
		_terminal = false
		set_training_active(true)


func get_progress() -> float:
	var position_progress := 1.0 - clampf(
		_target_distance() / maxf(workspace_scale, 0.001), 0.0, 1.0)
	var orientation_progress := 1.0 - clampf(_target_angle_error() / PI, 0.0, 1.0)
	var orientation_weight := clampf(orientation_progress_weight, 0.0, 1.0)
	return lerpf(position_progress, orientation_progress, orientation_weight)


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


func get_hold_stillness_reward() -> float:
	# Immobility reward, active ONLY within near_target_distance: rewards a low max joint speed so
	# the arm learns to SETTLE on the target instead of sweeping through it. Zero away from the
	# target, so the fast reach is never penalised.
	if not _is_near_target_pose():
		return 0.0
	return clampf(
		1.0 - _max_joint_speed() / maxf(hold_stillness_speed_reference, 0.001), 0.0, 1.0)


func get_near_target_speed_penalty() -> float:
	# Penalty proportional to MAX joint speed, active ONLY near the target. Drives joint velocity
	# toward zero precisely where holding matters, without slowing the reach far away. Returns a
	# negative term (scaled by the component weight).
	if not _is_near_target_pose():
		return 0.0
	return -_max_joint_speed()


func get_pose_tracking_reward() -> float:
	# Dense reward for matching position AND orientation. A pure product gives ZERO orientation
	# gradient while the arm is still far (position_score ~0), so orientation is only ever shaped at
	# the very end -- exactly where it is kinematically hardest to fix without disturbing position.
	# The 6-DOF xArm reliably solves position (min 0.011m) but leaves orientation stuck (~35deg even
	# when close). So reward each half INDEPENDENTLY (additive) to teach the policy to pre-orient
	# during the approach, and keep a smaller product term to still reward solving BOTH at once.
	var position_score := 1.0 - clampf(
		_target_distance() / maxf(near_target_distance, 0.001), 0.0, 1.0)
	var orientation_score := 1.0 - clampf(
		_target_angle_error() / maxf(deg_to_rad(pose_reward_angle_degrees), 0.001),
		0.0,
		1.0)
	return 0.4 * position_score + 0.4 * orientation_score + 0.2 * position_score * orientation_score


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
		"target_acquired": _succeeded
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
	_register_collision()


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


func _limit_aware_command(index: int, command: float) -> float:
	if (
		is_zero_approx(command)
		or index < 0
		or index >= get_joint_count()
		or not _robot
		or not _robot.urdf
	):
		return command
	var joint := _robot.urdf.get_joint(_joint_names[index])
	if not joint or joint.type != "revolute" or not joint.limit:
		return command

	var lower := minf(joint.limit.lower, joint.limit.upper)
	var upper := maxf(joint.limit.lower, joint.limit.upper)
	var span := upper - lower
	if span <= 0.0:
		return 0.0

	var position := clampf(_robot.get_joint_position(_joint_names[index]), lower, upper)
	var distance_to_limit := (
		position - lower
		if command < 0.0
		else upper - position)
	if distance_to_limit <= 0.000001:
		return 0.0

	var slowdown_margin := span * joint_limit_slowdown_ratio
	if slowdown_margin <= 0.000001:
		return command
	return command * clampf(distance_to_limit / slowdown_margin, 0.0, 1.0)


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
	if terminate_on_success:
		_terminal = true
		_robot.stop_all_joints()
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
		_register_collision()


func _on_safety_area_entered(other: Area3D) -> void:
	if _node_or_parent_is_in_group(other, obstacle_group):
		_register_collision()


func _cache_robot_collision_geometry() -> void:
	_robot_collision_shapes.clear()
	_robot_collision_exclusions.clear()
	if not _robot:
		return

	for link_node in _robot.links.values():
		if not link_node is Node:
			continue
		if link_node is CollisionObject3D:
			_robot_collision_exclusions.append(link_node.get_rid())
		_collect_collision_shapes(link_node)


func _collect_collision_shapes(node: Node) -> void:
	for child in node.get_children():
		if child is CollisionShape3D:
			_robot_collision_shapes.append(child)
		_collect_collision_shapes(child)


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
				_register_collision()
				return


func _node_or_parent_is_in_group(node: Node, group_name: String) -> bool:
	var current: Node = node
	while current:
		if current.is_in_group(group_name):
			return true
		current = current.get_parent()
	return false


func _register_collision() -> void:
	if _terminal:
		return
	_collided = true
	_terminal = true
	_robot.stop_all_joints()
	obstacle_collision.emit()


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
	return (
		_target_distance() <= near_target_distance
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
