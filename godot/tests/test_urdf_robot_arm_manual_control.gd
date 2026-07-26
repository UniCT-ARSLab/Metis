extends SceneTree


func _initialize() -> void:
	var packed := load("res://agents/RobotArms/xarm/x_arm_agent.tscn") as PackedScene
	if not packed:
		push_error("URDF manual control test: agent scene could not be loaded.")
		quit(1)
		return

	var body := packed.instantiate() as URDFRobotArmAgentBody
	root.add_child(body)
	await process_frame
	await physics_frame

	var robot: GodotRobot = body.get_node("xarm")
	var joint_names := body.get_controlled_joint_names()
	var passed := joint_names.size() == 5
	var wrist_mount := robot.get_joint_node("wrist_roll")
	var wrist_link := robot.get_link_node("xarm_2_link")
	var hand_link := robot.get_link_node("hand_link")
	passed = passed and wrist_mount != null
	passed = passed and wrist_link != null and hand_link != null
	passed = passed and not robot.get_actuated_joint_names().has("wrist_roll")
	passed = passed and is_zero_approx(wrist_mount.get_param_z(
		Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT))
	passed = passed and is_zero_approx(wrist_mount.get_param_z(
		Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT))
	passed = passed and not wrist_mount.get_flag_z(
		Generic6DOFJoint3D.FLAG_ENABLE_MOTOR)
	passed = passed and not robot.set_joint_target_velocity("wrist_roll", 1.0)
	var hand_mount_transform := (
		wrist_link.global_transform.affine_inverse()
		* hand_link.global_transform)
	var expected_mount := URDFUtils.xyz_rpy_to_transform3d(
		Vector3.ZERO,
		Vector3(0.0, PI / 2.0, 0.0))
	passed = passed and hand_mount_transform.basis.is_equal_approx(
		expected_mount.basis)

	body.manual_selected_joint = 2
	body.set_manual_control_enabled(true)
	Input.action_press("robot_joint_positive")
	var applied: Array = body.apply_manual_action()
	Input.action_release("robot_joint_positive")
	passed = passed and body.manual_control
	passed = passed and applied.size() == joint_names.size()
	for index in range(applied.size()):
		var expected := body.manual_command_scale if index == 2 else 0.0
		passed = passed and is_equal_approx(float(applied[index]), expected)

	await physics_frame
	var diagnostics := body.get_manual_pose_diagnostics()
	passed = passed and diagnostics.get("joints", []).size() == joint_names.size()
	passed = passed and diagnostics.has("tool_pose_global")
	passed = passed and diagnostics.has("target_pose_global")
	passed = passed and diagnostics.has("reward_terms_raw")
	passed = passed and str(
		diagnostics.get("manual_selected_joint_name", "")) == joint_names[2]

	body.report_obstacle_collision()
	passed = passed and body.is_terminal() and body.has_collided()
	body.resume_manual_from_current_pose()
	passed = passed and not body.is_terminal() and not body.has_collided()
	passed = passed and int(
		body.get_manual_pose_diagnostics().get(
			"manual_collision_grace_frames", 0)) > 0

	body.set_manual_control_enabled(false)
	await physics_frame
	passed = passed and not body.manual_control
	for command in body.get_previous_action_observation():
		passed = passed and is_zero_approx(float(command))
	for joint_name in joint_names:
		passed = passed and is_zero_approx(robot.get_joint_velocity(joint_name))

	if not passed:
		push_error(
			"URDF manual control test failed: selected=%s applied=%s diagnostics=%s" %
				[joint_names[2], applied, diagnostics])
		body.free()
		quit(1)
		return

	print("URDF robot arm manual control test passed")
	body.free()
	quit(0)
