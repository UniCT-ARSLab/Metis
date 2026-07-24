@tool
class_name GodotRobot
extends Node3D

enum ControlMode {
	PHYSICS_MOTORS,
	KINEMATIC,
}

var _transform_cache: Dictionary[String, Transform3D] = {}
var _joint_defs: Dictionary[String, Dictionary] = {}
var _joint_nodes: Dictionary[String, URDF6DOFJoint3D] = {}
var _joint_by_child_link: Dictionary[String, URDFJoint] = {}
var _mimic_order: Array[URDFJoint] = []
var _joint_positions: Dictionary[String, float] = {}
var _joint_velocities: Dictionary[String, float] = {}
var _joint_velocity_targets: Dictionary[String, float] = {}

var links: Dictionary[String, Node3D] = {}

@export var urdf: URDFRobot
@export var control_mode: ControlMode = ControlMode.PHYSICS_MOTORS
@export var synchronize_mimic_joints: bool = true


func _ready() -> void:
	_rebuild_runtime_joint_index()
	_rebuild_runtime_link_index()
	_initialize_joint_state()
	_validate_mimic_joints()
	if control_mode == ControlMode.KINEMATIC:
		_apply_kinematic_pose()


func _physics_process(delta: float) -> void:
	if Engine.is_editor_hint():
		return
	if control_mode == ControlMode.KINEMATIC:
		_step_kinematic_control(delta)
	elif synchronize_mimic_joints:
		_sync_mimic_targets()


func get_joint_node(joint_name: String) -> URDF6DOFJoint3D:
	if not _joint_nodes.has(joint_name):
		_rebuild_runtime_joint_index()
	return _joint_nodes.get(joint_name)


func get_actuated_joint_names() -> PackedStringArray:
	return urdf.get_actuated_joint_names() if urdf else PackedStringArray()


func get_joint_position(joint_name: String) -> float:
	return float(_joint_positions.get(joint_name, 0.0))


func get_joint_velocity(joint_name: String) -> float:
	return float(_joint_velocities.get(joint_name, 0.0))


func get_joint_positions(joint_names: PackedStringArray = PackedStringArray()) -> Array:
	var names := joint_names if not joint_names.is_empty() else get_actuated_joint_names()
	var result: Array = []
	for joint_name in names:
		result.append(get_joint_position(joint_name))
	return result


func get_joint_velocities(joint_names: PackedStringArray = PackedStringArray()) -> Array:
	var names := joint_names if not joint_names.is_empty() else get_actuated_joint_names()
	var result: Array = []
	for joint_name in names:
		result.append(get_joint_velocity(joint_name))
	return result


func get_link_node(link_name: String) -> Node3D:
	if not links.has(link_name):
		_rebuild_runtime_link_index()
	return links.get(link_name)


func set_joint_target_velocity(joint_name: String, target_velocity: float) -> bool:
	var joint_data := urdf.get_joint(joint_name) if urdf else null
	if not joint_data:
		push_warning("URDF joint '%s' was not found." % joint_name)
		return false
	if joint_data.is_mimic():
		push_warning("URDF mimic joint '%s' cannot be commanded directly." % joint_name)
		return false
	var velocity := _clamp_joint_velocity(joint_data, target_velocity)
	_joint_velocity_targets[joint_name] = velocity
	if control_mode == ControlMode.KINEMATIC:
		return true

	var joint_node := get_joint_node(joint_name)
	if not joint_node:
		push_warning("URDF joint node '%s' was not found." % joint_name)
		return false
	joint_node.set_motor_target_velocity(velocity)
	_sync_mimic_targets()
	return true


func set_joint_target_position(
		joint_name: String,
		target_position: float,
		stiffness: float = -1.0,
		damping: float = -1.0) -> bool:
	var joint_data := urdf.get_joint(joint_name) if urdf else null
	if not joint_data:
		push_warning("URDF joint '%s' was not found." % joint_name)
		return false
	if joint_data.is_mimic():
		push_warning("URDF mimic joint '%s' cannot be commanded directly." % joint_name)
		return false
	var position := _clamp_joint_position(joint_data, target_position)
	_joint_positions[joint_name] = position
	_joint_velocities[joint_name] = 0.0
	_joint_velocity_targets[joint_name] = 0.0
	if control_mode == ControlMode.KINEMATIC:
		_apply_kinematic_pose()
		return true

	var joint_node := get_joint_node(joint_name)
	if not joint_node:
		push_warning("URDF joint node '%s' was not found." % joint_name)
		return false
	joint_node.set_spring_target_position(
		position, stiffness, damping)
	_sync_mimic_targets()
	return true


func reset_joint_positions(positions: Dictionary = {}) -> void:
	_initialize_joint_state()
	for joint_name in get_actuated_joint_names():
		var joint_data := urdf.get_joint(joint_name)
		var position := float(positions.get(joint_name, 0.0))
		_joint_positions[joint_name] = _clamp_joint_position(joint_data, position)
		_joint_velocities[joint_name] = 0.0
		_joint_velocity_targets[joint_name] = 0.0
		if control_mode == ControlMode.PHYSICS_MOTORS:
			var joint_node := get_joint_node(joint_name)
			if joint_node:
				joint_node.set_motor_target_velocity(0.0)
	_apply_kinematic_pose() if control_mode == ControlMode.KINEMATIC else _sync_mimic_targets()


func stop_all_joints() -> void:
	for joint_name in get_actuated_joint_names():
		set_joint_target_velocity(joint_name, 0.0)


func _rebuild_runtime_joint_index() -> void:
	_joint_nodes.clear()
	for child in get_children():
		if child is URDF6DOFJoint3D and child.joint:
			_joint_nodes[child.joint.name] = child
	_rebuild_mimic_order()


func _rebuild_runtime_link_index() -> void:
	links.clear()
	for child in get_children():
		var link_data: URDFLink = null
		if child is URDFVisualNode or child is URDFRigidBody3D or child is URDFStaticBody3D:
			link_data = child.link
		if link_data:
			links[link_data.name] = child

	_joint_by_child_link.clear()
	_joint_defs.clear()
	_transform_cache.clear()
	if not urdf:
		return
	for joint in urdf.joints:
		_joint_by_child_link[joint.child] = joint
		add_joint(
			joint,
			URDFUtils.xyz_rpy_to_transform3d(
				joint.origin_xyz, joint.origin_rpy))


func _initialize_joint_state() -> void:
	if not urdf:
		return
	for joint in urdf.joints:
		if not _joint_positions.has(joint.name):
			_joint_positions[joint.name] = 0.0
		_joint_velocities[joint.name] = 0.0
		_joint_velocity_targets[joint.name] = 0.0


func _rebuild_mimic_order() -> void:
	_mimic_order.clear()
	if not urdf:
		return
	var pending := urdf.get_mimic_joints()
	var added: Dictionary[String, bool] = {}
	while not pending.is_empty():
		var made_progress := false
		for index in range(pending.size() - 1, -1, -1):
			var joint: URDFJoint = pending[index]
			var source := urdf.get_joint(joint.mimic_joint)
			if source and (not source.is_mimic() or added.has(source.name)):
				_mimic_order.append(joint)
				added[joint.name] = true
				pending.remove_at(index)
				made_progress = true
		if not made_progress:
			break


func _validate_mimic_joints() -> void:
	if not urdf:
		return
	for message in urdf.validate_mimic_joints():
		push_error("URDF mimic: " + message)


func _sync_mimic_targets() -> void:
	if not urdf or _mimic_order.is_empty():
		return
	for mimic in _mimic_order:
		var source := get_joint_node(mimic.mimic_joint)
		var follower := get_joint_node(mimic.name)
		if not source or not follower:
			continue

		var source_velocity := source.get_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY)
		follower.set_motor_target_velocity(
			source_velocity * mimic.mimic_multiplier)

		var spring_enabled := source.get_flag_z(
			Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_SPRING)
		follower.set_flag_z(
			Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_SPRING,
			spring_enabled)
		if spring_enabled:
			var source_target := source.get_param_z(
				Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT)
			follower.set_spring_target_position(
				source_target * mimic.mimic_multiplier + mimic.mimic_offset,
				source.get_param_z(
					Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_STIFFNESS),
					source.get_param_z(
						Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_DAMPING))


func _step_kinematic_control(delta: float) -> void:
	for joint_name in get_actuated_joint_names():
		var joint_data := urdf.get_joint(joint_name)
		var old_position := get_joint_position(joint_name)
		var velocity := float(_joint_velocity_targets.get(joint_name, 0.0))
		var new_position := _clamp_joint_position(
			joint_data, old_position + velocity * delta)
		_joint_positions[joint_name] = new_position
		_joint_velocities[joint_name] = (
			(new_position - old_position) / maxf(delta, 0.000001))
	_apply_kinematic_pose()


func _apply_kinematic_pose() -> void:
	if not urdf:
		return
	for mimic in _mimic_order:
		var source_position := get_joint_position(mimic.mimic_joint)
		var source_velocity := get_joint_velocity(mimic.mimic_joint)
		_joint_positions[mimic.name] = _clamp_joint_position(
			mimic,
			source_position * mimic.mimic_multiplier + mimic.mimic_offset)
		_joint_velocities[mimic.name] = source_velocity * mimic.mimic_multiplier

	var pose_cache: Dictionary[String, Transform3D] = {}
	for link in urdf.links:
		var link_node := get_link_node(link.name)
		if not link_node:
			continue
		if link_node is RigidBody3D:
			link_node.freeze = true
			link_node.linear_velocity = Vector3.ZERO
			link_node.angular_velocity = Vector3.ZERO
		link_node.transform = _get_kinematic_link_transform(link.name, pose_cache)


func _get_kinematic_link_transform(
		link_name: String,
		cache: Dictionary[String, Transform3D]) -> Transform3D:
	if cache.has(link_name):
		return cache[link_name]
	if not _joint_by_child_link.has(link_name):
		cache[link_name] = Transform3D.IDENTITY
		return Transform3D.IDENTITY

	var joint: URDFJoint = _joint_by_child_link[link_name]
	var parent_transform := _get_kinematic_link_transform(joint.parent, cache)
	var origin := URDFUtils.xyz_rpy_to_transform3d(
		joint.origin_xyz, joint.origin_rpy)
	var motion := Transform3D.IDENTITY
	if joint.type in ["revolute", "continuous"] and not joint.axis.is_zero_approx():
		motion.basis = Basis(Quaternion(
			joint.axis.normalized(), get_joint_position(joint.name)))
	var result := parent_transform * origin * motion
	cache[link_name] = result
	return result


func _clamp_joint_position(joint: URDFJoint, value: float) -> float:
	if joint and joint.type == "revolute" and joint.limit:
		return clampf(
			value,
			minf(joint.limit.lower, joint.limit.upper),
			maxf(joint.limit.lower, joint.limit.upper))
	return value


func _clamp_joint_velocity(joint: URDFJoint, value: float) -> float:
	if joint and joint.limit and joint.limit.velocity > 0.0:
		return clampf(value, -joint.limit.velocity, joint.limit.velocity)
	return value

func add_joint(joint: URDFJoint, local_transform: Transform3D):
	_joint_defs[joint.child] = {
		"parent": joint.parent,
		"transform": local_transform
	}


func get_rel_transform(link_name: String) -> Transform3D:
	if _transform_cache.has(link_name):
		return _transform_cache[link_name]
	
	if not _joint_defs.has(link_name):
		_transform_cache[link_name] = Transform3D.IDENTITY
		return Transform3D.IDENTITY
	
	var joint = _joint_defs[link_name]
	var parent_transform = get_rel_transform(joint.parent)
	var transform = parent_transform * joint.transform
	
	_transform_cache[link_name] = transform
	return transform


func init_data(
		robot: URDFRobot,
		parent: Node3D,
		owner: Node3D,
		options: Dictionary,
		source_path: String) -> void:
	self.urdf = robot
	self.name = robot.name
	if parent:
		parent.add_child(self)
	if owner:
		self.owner = owner
	else:
		owner = self
	
	for link in robot.links:
		var link_node = null
		if link.colliders.size() > 0:
			var physics: bool = options.get("create_physics", true)
			var physics_body = URDFRigidBody3D.new() \
				 if physics else URDFStaticBody3D.new()
			link_node = physics_body
		elif link.visuals.size() > 0:
			link_node = URDFVisualNode.new()

		if not link_node:
			# neither visual elements nor collider, skip
			continue

		link_node.update_link(
			link, self, owner, options, source_path)
		links[link.name] = link_node

	for joint in robot.joints:
		var child_node: Node3D = links.get(joint.child)

		# Set center position and store in cache so we can
		# calculate the global positions later.
		var local_transform: Transform3D = URDFUtils.xyz_rpy_to_transform3d(
				joint.origin_xyz, joint.origin_rpy)

		if child_node:
			child_node.name = joint.name
		self.add_joint(joint, local_transform)
		self.create_godot_joint(joint, owner)

	for link_name in links.keys():
		var global_rel_transform = self.get_rel_transform(link_name)
		var link_node = links[link_name]
		link_node.transform = global_rel_transform
	_rebuild_runtime_joint_index()
	_rebuild_runtime_link_index()
	_initialize_joint_state()

func create_godot_joint(joint: URDFJoint, owner: Node3D):
	var collision_node_a = links.get(joint.parent)
	var collision_node_b = links.get(joint.child)
	if !collision_node_a:
		# print(
		# 	"Can not find parent joint " + joint.parent + " for " + joint.name)
		return
	if !collision_node_b:
		# print(
		# 	"Can not find child joint " + joint.child + " for " + joint.name)
		return

	# print(
	# 	"Create joint:" + joint.name + ": " + 
	# 	joint.parent + " -> " + joint.child)
	var godot_joint: URDF6DOFJoint3D = URDF6DOFJoint3D.new()
	godot_joint.update_joint(self, owner, joint)

	godot_joint.node_a = godot_joint.get_path_to(collision_node_a)
	godot_joint.node_b = godot_joint.get_path_to(collision_node_b)
