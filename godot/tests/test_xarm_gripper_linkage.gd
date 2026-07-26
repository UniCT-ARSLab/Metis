extends SceneTree


func _initialize() -> void:
	var packed := load("res://agents/RobotArms/xarm/x_arm_agent.tscn") as PackedScene
	if not packed:
		push_error("XArm gripper linkage test: agent scene could not be loaded.")
		quit(1)
		return

	var body := packed.instantiate() as URDFRobotArmAgentBody
	root.add_child(body)
	await process_frame
	await physics_frame

	var robot: GodotRobot = body.get_node("xarm")
	var hand := robot.get_link_node("hand_link")
	var finger_left := robot.get_link_node("finger_left_link")
	var finger_right := robot.get_link_node("finger_right_link")
	var hand_basis := hand.global_basis
	var finger_left_basis := finger_left.global_basis
	var finger_right_basis := finger_right.global_basis
	var finger_left_position := finger_left.global_position
	var finger_right_position := finger_right.global_position

	var commanded := robot.set_joint_target_position("grip_left", -0.8)
	await physics_frame

	var passed := commanded
	passed = passed and is_equal_approx(
		robot.get_joint_position("grip_left"), -0.8)
	passed = passed and is_equal_approx(
		robot.get_joint_position("grip_right"), 0.8)
	passed = passed and is_equal_approx(
		robot.get_joint_position("tendon_left"), -0.8)
	passed = passed and is_equal_approx(
		robot.get_joint_position("tendon_right"), 0.8)
	passed = passed and is_equal_approx(
		robot.get_joint_position("finger_left"), 0.8)
	passed = passed and is_equal_approx(
		robot.get_joint_position("finger_right"), -0.8)
	passed = passed and hand.global_basis.is_equal_approx(hand_basis)
	passed = passed and finger_left.global_basis.is_equal_approx(
		finger_left_basis)
	passed = passed and finger_right.global_basis.is_equal_approx(
		finger_right_basis)
	passed = passed and finger_left.global_position.distance_to(
		finger_left_position) > 0.001
	passed = passed and finger_right.global_position.distance_to(
		finger_right_position) > 0.001

	if not passed:
		push_error(
			"XArm gripper linkage failed: positions=%s" % {
				"grip_left": robot.get_joint_position("grip_left"),
				"grip_right": robot.get_joint_position("grip_right"),
				"tendon_left": robot.get_joint_position("tendon_left"),
				"tendon_right": robot.get_joint_position("tendon_right"),
				"finger_left": robot.get_joint_position("finger_left"),
				"finger_right": robot.get_joint_position("finger_right"),
			})
		body.free()
		quit(1)
		return

	print("XArm gripper linkage test passed")
	body.free()
	quit(0)
