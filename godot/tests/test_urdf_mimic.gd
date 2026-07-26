extends SceneTree


func _initialize() -> void:
	var parser := URDFXMLParser.new()
	var parsed := parser.parse("res://assets/urdf/xarm/xarm_fixed.urdf")
	if not _validate_parsed_xarm(parsed):
		quit(1)
		return

	var runtime_ok := await _validate_runtime_propagation()
	if not runtime_ok:
		quit(1)
		return

	print("URDF mimic test passed")
	quit(0)


func _validate_parsed_xarm(robot: URDFRobot) -> bool:
	if not robot:
		push_error("URDF mimic test: xArm URDF was not parsed.")
		return false
	var actuated := robot.get_actuated_joint_names()
	var mimics := robot.get_mimic_joints()
	var grip_right := robot.get_joint("grip_right")
	var wrist_mount := robot.get_joint("wrist_roll")
	var passed := actuated.size() == 6 and mimics.size() == 5
	passed = passed and not actuated.has("wrist_roll")
	passed = passed and wrist_mount != null
	passed = passed and wrist_mount.type == "fixed"
	passed = passed and wrist_mount.origin_rpy.is_equal_approx(
		Vector3(0.0, PI / 2.0, 0.0))
	passed = passed and actuated.has("grip_left")
	passed = passed and not actuated.has("grip_right")
	passed = passed and grip_right != null
	passed = passed and grip_right.mimic_joint == "grip_left"
	passed = passed and is_equal_approx(grip_right.mimic_multiplier, -1.0)
	passed = passed and is_equal_approx(grip_right.mimic_offset, 0.0)
	passed = passed and robot.validate_mimic_joints().is_empty()
	if not passed:
		push_error(
			"URDF mimic parse failed: actuated=%s mimic_count=%d" %
			[actuated, mimics.size()])
	return passed


func _validate_runtime_propagation() -> bool:
	var source_data := URDFJoint.new()
	source_data.name = "source"
	source_data.type = "revolute"
	source_data.limit = URDFLimit.new()
	source_data.limit.lower = -1.0
	source_data.limit.upper = 1.0
	source_data.limit.velocity = 2.0

	var follower_data := URDFJoint.new()
	follower_data.name = "follower"
	follower_data.type = "revolute"
	follower_data.mimic_joint = "source"
	follower_data.mimic_multiplier = -0.5
	follower_data.mimic_offset = 0.25
	follower_data.limit = URDFLimit.new()
	follower_data.limit.lower = -1.0
	follower_data.limit.upper = 1.0
	follower_data.limit.velocity = 2.0

	var robot_data := URDFRobot.new()
	robot_data.joints.assign([source_data, follower_data])
	var robot := GodotRobot.new()
	robot.urdf = robot_data
	var source := URDF6DOFJoint3D.new()
	source.joint = source_data
	var follower := URDF6DOFJoint3D.new()
	follower.joint = follower_data
	robot.add_child(source)
	robot.add_child(follower)
	root.add_child(robot)
	await process_frame

	var velocity_ok := robot.set_joint_target_velocity("source", 0.8)
	var follower_velocity := follower.get_param_z(
		Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY)
	velocity_ok = velocity_ok and is_equal_approx(follower_velocity, -0.4)

	var position_ok := robot.set_joint_target_position("source", 0.5, 20.0, 2.0)
	var follower_position := follower.get_param_z(
		Generic6DOFJoint3D.PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT)
	position_ok = position_ok and is_equal_approx(follower_position, 0.0)
	position_ok = position_ok and not robot.set_joint_target_velocity("follower", 1.0)

	robot.free()
	if not velocity_ok or not position_ok:
		push_error(
			"URDF mimic runtime failed: velocity=%f position=%f" %
			[follower_velocity, follower_position])
	return velocity_ok and position_ok
