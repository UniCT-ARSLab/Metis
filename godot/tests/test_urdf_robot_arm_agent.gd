extends SceneTree


func _initialize() -> void:
	var packed := load("res://agents/RobotArms/xarm/x_arm_agent.tscn") as PackedScene
	if not packed:
		push_error("URDF robot arm test: agent scene could not be loaded.")
		quit(1)
		return

	var body := packed.instantiate() as Node3D
	root.add_child(body)
	var controller := ScenarioController.new()
	controller.controlled_agents.append(body)
	root.add_child(controller)
	await process_frame
	await physics_frame

	var agent: Agent = body.get_node("Agent")
	var robot: GodotRobot = body.get_node("xarm")
	var tcp_link := robot.get_link_node("hand_link")
	var end_effector := body.get_node("EndEffector") as Node3D
	var tool_pose := body.get_node("EndEffector/ToolPose") as Node3D
	var initial_tcp := end_effector.global_position
	var initial_link_position := tcp_link.global_position if tcp_link else Vector3.ZERO
	var passed := agent.get_action_type() == "continuous"
	passed = passed and agent.get_action_size() == 5
	passed = passed and agent.get_observation_size() == 21
	passed = passed and body.get_joint_count() == 5
	passed = passed and robot.get_actuated_joint_names().size() == 6
	passed = passed and not body.has_collided()
	var tool_basis_is_valid := is_equal_approx(
		tool_pose.global_basis.determinant(), 1.0)
	passed = passed and tool_basis_is_valid
	var links_are_kinematic := true
	for link_node in robot.links.values():
		if link_node is RigidBody3D:
			links_are_kinematic = (
				links_are_kinematic
				and link_node.freeze
				and link_node.freeze_mode == RigidBody3D.FREEZE_MODE_KINEMATIC)

	body.apply_action([0.0, 1.0, 0.0, 0.0, 0.0])
	for _index in range(4):
		await physics_frame
	var moved_position := robot.get_joint_position("xarm_5_joint")
	var moved_tcp := end_effector.global_position
	var moved_link_position := tcp_link.global_position if tcp_link else Vector3.ZERO
	passed = passed and moved_position > 0.0
	passed = passed and moved_tcp.distance_to(initial_tcp) > 0.000001
	robot.set_joint_target_position("grip_left", -0.4)
	var mimic_right_ok := is_equal_approx(
		robot.get_joint_position("grip_right"), 0.4)
	var mimic_tendon_ok := is_equal_approx(
		robot.get_joint_position("tendon_left"), -0.4)

	var reset_offsets := [0.05, 0.0, 0.0, 0.0, 0.0]
	body.set_reset_joint_offsets(reset_offsets)
	body.reset_all(body.transform)
	await physics_frame
	var reset_position := robot.get_joint_position("xarm_6_joint")
	passed = passed and is_equal_approx(reset_position, 0.05)

	agent.reset_reward({"body": body})
	body.apply_action([0.0, 0.0, 0.0, 0.0, 0.0])
	agent.get_reward({"body": body})
	body.apply_action([1.0, -1.0, 1.0, -1.0, 1.0])
	agent.get_reward({"body": body})
	var reward_terms := agent.get_reward_terms()
	var reward_terms_ok := (
		reward_terms.has("joint_motion")
		and reward_terms.has("action_smoothness"))
	var reward_caller_ok: bool = (
		body.get_node("Agent/RewardSystem/JointLimit").node_caller == body)
	passed = passed and reward_terms_ok and reward_caller_ok

	body.reset_all(body.transform)
	await physics_frame
	var home_reset_ok := is_zero_approx(
		robot.get_joint_position("xarm_5_joint"))
	passed = passed and home_reset_ok
	passed = passed and agent.get_observation_size() == 21

	var moving_target := Node3D.new()
	root.add_child(moving_target)
	body.target = moving_target
	body.target_pose = moving_target
	body.success_hold_physics_frames = 1
	body.set_continue_after_success(true)
	moving_target.global_transform = tool_pose.global_transform
	await physics_frame
	var acquired_without_terminal: bool = body.has_succeeded() and not body.is_terminal()
	var aligned_orientation_error: float = (
		body.get_target_orientation_error_observation().length())
	passed = passed and acquired_without_terminal
	passed = passed and aligned_orientation_error < 0.0001
	moving_target.global_position += Vector3(body.success_distance * 3.0, 0.0, 0.0)
	await physics_frame
	var rearmed_after_relocation: bool = (
		not body.has_succeeded() and not body.is_terminal())
	passed = passed and rearmed_after_relocation
	moving_target.free()
	body.target = null
	body.target_pose = null
	body.set_continue_after_success(false)
	body.initialize_episode_from_current_state()

	var robot_shape := body.get_node(
		"xarm/xarm_6_joint/xarm_6_link_collision") as CollisionShape3D
	var obstacle := StaticBody3D.new()
	obstacle.add_to_group("robot_obstacle")
	root.add_child(obstacle)
	var obstacle_shape := CollisionShape3D.new()
	obstacle_shape.shape = robot_shape.shape
	obstacle.add_child(obstacle_shape)
	obstacle.global_transform = robot_shape.global_transform
	await physics_frame
	await physics_frame
	var collision_detected: bool = body.has_collided()
	passed = passed and collision_detected
	obstacle.free()
	var preserved_joint_position := robot.get_joint_position("xarm_5_joint")
	var reset_started_count := [0]
	controller.episode_reset_started.connect(
		func(_seed:int) -> void: reset_started_count[0] += 1)
	var preserved_reset: Dictionary = await controller.reset_episode_with_request({
		"seed": 123,
		"preserve_state": true
	})
	var preserve_cleared_terminal: bool = (
		not body.has_collided() and not body.is_terminal())
	var preserve_joint_ok := is_equal_approx(
		robot.get_joint_position("xarm_5_joint"), preserved_joint_position)
	var preserve_signal_ok: bool = reset_started_count[0] == 0
	var preserve_flag_ok := bool(
		preserved_reset.get("info", {}).get("preserve_state", false))
	var preserve_mode_ok := str(
		preserved_reset.get("info", {}).get("reset", {}).get("mode", "")) == "current_state"
	passed = (
		passed
		and preserve_cleared_terminal
		and preserve_joint_ok
		and preserve_signal_ok
		and preserve_flag_ok
		and preserve_mode_ok)

	if not passed:
		push_error("URDF robot arm test failed: %s" % [{
			"action_size": agent.get_action_size(),
			"observation_size": agent.get_observation_size(),
				"tool_basis_is_valid": tool_basis_is_valid,
			"links_are_kinematic": links_are_kinematic,
			"mimic_right_ok": mimic_right_ok,
			"mimic_tendon_ok": mimic_tendon_ok,
			"reset_position": reset_position,
			"reward_terms_ok": reward_terms_ok,
			"reward_caller_ok": reward_caller_ok,
			"home_reset_ok": home_reset_ok,
			"acquired_without_terminal": acquired_without_terminal,
			"aligned_orientation_error": aligned_orientation_error,
			"rearmed_after_relocation": rearmed_after_relocation,
			"collision_detected": collision_detected,
			"preserve_cleared_terminal": preserve_cleared_terminal,
			"preserve_joint_ok": preserve_joint_ok,
			"preserve_signal_ok": preserve_signal_ok,
			"preserve_flag_ok": preserve_flag_ok,
			"preserve_mode_ok": preserve_mode_ok
		}])
		body.free()
		quit(1)
		return
	controller.free()
	body.free()
	print("URDF robot arm agent test passed")
	quit(0)
