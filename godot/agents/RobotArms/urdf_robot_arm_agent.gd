extends Node3D
class_name URDFRobotArmAgentBody

signal target_reached
signal obstacle_collision

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
@export var target: Node3D
@export var workspace_scale := 1.0
@export var success_distance := 0.05
@export var success_hold_physics_frames := 10
@export var terminate_on_success := true
@export_range(1.0, 10.0, 0.1) var success_rearm_distance_multiplier := 1.5

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


func _ready() -> void:
	_robot = get_node_or_null(robot_path) as GodotRobot
	end_effector = get_node_or_null(end_effector_path) as Node3D
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


func _physics_process(_delta: float) -> void:
	if not _training_active:
		return
	if manual_control and not _terminal:
		apply_manual_action()
	_update_end_effector()
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
	_success_frames = 0
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


func _update_end_effector() -> void:
	if not end_effector or tcp_link_name.is_empty() or not _robot:
		return
	var tcp_link := _robot.get_link_node(tcp_link_name)
	if not tcp_link:
		return
	end_effector.global_transform = tcp_link.global_transform.translated_local(
		tcp_local_offset)


func _update_success_state() -> void:
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
	_succeeded = true
	if terminate_on_success:
		_terminal = true
		_robot.stop_all_joints()
	target_reached.emit()


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
	if not target or not end_effector:
		return workspace_scale
	return end_effector.global_position.distance_to(target.global_position)
