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
	var manual_joint_names := body.get_manual_control_joint_names()
	var passed := joint_names.size() == 5
	passed = passed and joint_names[4] == "xarm_2_joint"
	passed = passed and manual_joint_names.size() == 6
	passed = passed and manual_joint_names[5] == "grip_left"
	passed = passed and body.agent.get_action_size() == 5
	passed = passed and body.agent.get_observation_size() == 21
	var wrist_mount := robot.get_joint_node("wrist_roll")
	var wrist_link := robot.get_link_node("xarm_2_link")
	var hand_link := robot.get_link_node("hand_link")
	passed = passed and wrist_mount != null
	passed = passed and wrist_link != null and hand_link != null
	passed = passed and not robot.get_actuated_joint_names().has("wrist_roll")
	passed = passed and robot.urdf.get_joint("wrist_roll").type == "fixed"
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
	passed = passed and diagnostics.get("joints", []).size() == manual_joint_names.size()
	passed = passed and diagnostics.has("tool_pose_global")
	passed = passed and diagnostics.has("target_pose_global")
	passed = passed and diagnostics.has("reward_terms_raw")
	passed = passed and str(
		diagnostics.get("manual_selected_joint_name", "")) == joint_names[2]

	# Keep the last arm joint selected: direct gripper controls must never rotate the hand.
	body.manual_selected_joint = 4
	var wrist_position_before_grip := robot.get_joint_position("xarm_2_joint")
	var hand_basis_before_grip := hand_link.global_basis
	var left_grip_basis_before := robot.get_link_node("grip_left_link").global_basis
	var right_grip_basis_before := robot.get_link_node("grip_right_link").global_basis
	var left_finger_position_before := (
		robot.get_link_node("finger_left_link").global_position)
	var right_finger_position_before := (
		robot.get_link_node("finger_right_link").global_position)
	Input.action_press("robot_gripper_close")
	var gripper_policy_action: Array = body.apply_manual_action()
	await physics_frame
	Input.action_release("robot_gripper_close")
	body.apply_manual_action()
	passed = passed and gripper_policy_action.size() == joint_names.size()
	for command in gripper_policy_action:
		passed = passed and is_zero_approx(float(command))
	passed = passed and is_equal_approx(
		robot.get_joint_position("xarm_2_joint"), wrist_position_before_grip)
	passed = passed and hand_link.global_basis.is_equal_approx(
		hand_basis_before_grip)
	passed = passed and not robot.get_link_node(
		"grip_left_link").global_basis.is_equal_approx(left_grip_basis_before)
	passed = passed and not robot.get_link_node(
		"grip_right_link").global_basis.is_equal_approx(right_grip_basis_before)
	passed = passed and robot.get_link_node(
		"finger_left_link").global_position.distance_to(
			left_finger_position_before) > 0.000001
	passed = passed and robot.get_link_node(
		"finger_right_link").global_position.distance_to(
			right_finger_position_before) > 0.000001
	passed = passed and robot.get_joint_position("grip_left") < 0.0
	passed = passed and robot.get_joint_position("grip_right") > 0.0
	passed = passed and robot.get_joint_position("tendon_left") < 0.0
	passed = passed and robot.get_joint_position("tendon_right") > 0.0
	passed = passed and robot.get_joint_position("finger_left") > 0.0
	passed = passed and robot.get_joint_position("finger_right") < 0.0
	diagnostics = body.get_manual_pose_diagnostics()
	passed = passed and diagnostics.get("joints", []).size() == manual_joint_names.size()
	passed = passed and diagnostics.get("mimic_joints", []).size() == 5
	passed = passed and int(diagnostics.get("policy_action_size", 0)) == 5
	passed = passed and str(
		diagnostics.get("manual_selected_joint_name", "")) == "xarm_2_joint"

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
	for joint_name in manual_joint_names:
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
