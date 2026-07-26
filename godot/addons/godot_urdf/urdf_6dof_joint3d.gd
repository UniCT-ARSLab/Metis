class_name URDF6DOFJoint3D
extends Generic6DOFJoint3D

@export var joint: URDFJoint


func clamp_target_position(target_position: float) -> float:
	if joint and joint.type == "revolute" and joint.limit:
		var lower := minf(joint.limit.lower, joint.limit.upper)
		var upper := maxf(joint.limit.lower, joint.limit.upper)
		return clampf(target_position, lower, upper)
	return target_position


func clamp_target_velocity(target_velocity: float) -> float:
	if joint and joint.limit and joint.limit.velocity > 0.0:
		return clampf(target_velocity, -joint.limit.velocity, joint.limit.velocity)
	return target_velocity


func set_motor_target_velocity(target_velocity: float) -> void:
	set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, true)
	set_param_z(
		Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,
		clamp_target_velocity(target_velocity))


func set_spring_target_position(
		target_position: float,
		stiffness: float = -1.0,
		damping: float = -1.0) -> void:
	set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_SPRING, true)
	set_param_z(
		Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT,
		clamp_target_position(target_position))
	if stiffness >= 0.0:
		set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_STIFFNESS,
			stiffness)
	if damping >= 0.0:
		set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_DAMPING,
			damping)

func update_joint(
		godot_robot: GodotRobot,
		owner: Node3D,
		joint: URDFJoint) -> void:
	godot_robot.add_child(self)
	self.owner = owner

	self.joint = joint
	self.name = "joint_" + joint.name
	self.exclude_nodes_from_collision = false

	# lock all movement and rotation (fixed = default)
	for axis_func in ["set_param_x", "set_param_y", "set_param_z"]:
		self.call(
			axis_func, Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0.0)
		self.call(
			axis_func, Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0.0)
	for axis_func in ["set_param_x", "set_param_y", "set_param_z"]:
		self.call(
			axis_func, Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, 0.0)
		self.call(
			axis_func, Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, 0.0)

	if joint.type == "revolute" and joint.limit:
		# Limited hinge
		var lower := minf(joint.limit.lower, joint.limit.upper)
		var upper := maxf(joint.limit.lower, joint.limit.upper)
		self.set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, lower)
		self.set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, upper)

	elif joint.type == "continuous":
		# Unlimited hinge: Lower > Upper disables the limit
		self.set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, 1.0)
		self.set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, 0.0)

	var is_movable := joint.type in ["revolute", "continuous"]
	self.set_flag_z(
		Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, is_movable)
	self.set_param_z(
		Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0.0)

	if joint.dynamics:
		if joint.dynamics.friction > 0:
			self.set_param_z(
				Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT,
				joint.dynamics.friction)
		if joint.dynamics.damping > 0:
			# Damping is not supported by Jolt Physics, use Springinstead
			# if joint.dynamics.damping > 0:
			# 	godot_joint.set_param_z(
			# 		Generic6DOFJoint3D.PARAM_ANGULAR_DAMPING,
			# 		joint.dynamics.damping)
			self.set_flag_z(
				Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_SPRING, true)
			self.set_param_z(
				Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_DAMPING,
				joint.dynamics.damping)

	if joint.limit and joint.limit.effort > 0:
		# If friction is already using the motor, we ensure the effort is
		# at least as high as friction
		var max_force = max(
			joint.limit.effort,
			joint.dynamics.friction if joint.dynamics else 0.0)
		self.set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, max_force)

	# apply transform to position and rotate hinge
	var child_transform = godot_robot.get_rel_transform(joint.child)
	self.position = child_transform.origin
	var base_basis = child_transform.basis
	var urdf_axis = joint.axis.normalized()
	var axis_rotation = Quaternion(Vector3.FORWARD, urdf_axis)
	self.transform.basis = base_basis * Basis(axis_rotation)
