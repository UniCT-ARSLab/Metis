extends SceneTree


func _initialize() -> void:
	var packed := load("res://agents/RobotArms/xarm/x_arm_agent.tscn") as PackedScene
	if not packed:
		push_error("URDF robot arm test: agent scene could not be loaded.")
		quit(1)
		return

	var body := packed.instantiate() as Node3D
	root.add_child(body)
	await process_frame
	await physics_frame

	var agent: Agent = body.get_node("Agent")
	var robot: GodotRobot = body.get_node("xarm")
	var tcp_link := robot.get_link_node("hand_link")
	var end_effector := body.get_node("EndEffector") as Node3D
	var initial_tcp := end_effector.global_position
	var initial_link_position := tcp_link.global_position if tcp_link else Vector3.ZERO
	var passed := agent.get_action_type() == "continuous"
	passed = passed and agent.get_action_size() == 6
	passed = passed and agent.get_observation_size() == 21
	passed = passed and body.get_joint_count() == 6
	passed = passed and robot.get_actuated_joint_names().size() == 7

	body.apply_action([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
	for _index in range(4):
		await physics_frame
	var moved_position := robot.get_joint_position("xarm_5_joint")
	var moved_tcp := end_effector.global_position
	var moved_link_position := tcp_link.global_position if tcp_link else Vector3.ZERO
	passed = passed and moved_position > 0.0
	passed = passed and moved_tcp.distance_to(initial_tcp) > 0.000001
	passed = passed and robot.set_joint_target_position("grip_left", -0.4)
	passed = passed and is_equal_approx(
		robot.get_joint_position("grip_right"), 0.4)
	passed = passed and is_equal_approx(
		robot.get_joint_position("tendon_left"), -0.4)

	var reset_offsets := [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
	body.set_reset_joint_offsets(reset_offsets)
	body.reset_all(body.transform)
	await physics_frame
	var reset_position := robot.get_joint_position("xarm_6_joint")
	passed = passed and is_equal_approx(reset_position, 0.05)

	agent.reset_reward({"body": body})
	body.apply_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	agent.get_reward({"body": body})
	body.apply_action([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
	agent.get_reward({"body": body})
	var reward_terms := agent.get_reward_terms()
	passed = passed and reward_terms.has("joint_motion")
	passed = passed and reward_terms.has("action_smoothness")
	passed = passed and body.get_node("Agent/RewardSystem/JointLimit").node_caller == body

	body.reset_all(body.transform)
	await physics_frame
	passed = passed and is_zero_approx(
		robot.get_joint_position("xarm_5_joint"))
	passed = passed and agent.get_observation_size() == 21

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
	passed = passed and body.has_collided()
	obstacle.free()

	if not passed:
		push_error(
				"URDF robot arm test failed: action_size=%d obs_size=%d joint=%f reset=%f tcp_delta=%f link_delta=%f terms=%s collided=%s configured_tcp=%s physics=%s links=%s" %
				[
					agent.get_action_size(),
					agent.get_observation_size(),
					moved_position,
					reset_position,
					moved_tcp.distance_to(initial_tcp),
					moved_link_position.distance_to(initial_link_position),
					reward_terms,
					body.has_collided(),
					body.get("end_effector"),
				body.is_physics_processing(),
				robot.links.keys(),
			])
		body.free()
		quit(1)
		return
	body.free()
	print("URDF robot arm agent test passed")
	quit(0)
