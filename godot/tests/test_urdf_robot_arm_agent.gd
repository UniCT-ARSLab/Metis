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
	var initial_tcp := end_effector.global_position
	var initial_link_position := tcp_link.global_position if tcp_link else Vector3.ZERO
	var initial_collision_info: Dictionary = body.get_last_collision_info()
	var passed := agent.get_action_type() == "continuous"
	passed = passed and agent.get_action_size() == 7
	passed = passed and agent.get_observation_size() == 33
	passed = passed and body.get_joint_count() == 7
	passed = passed and robot.get_actuated_joint_names().size() == 7
	passed = passed and not body.has_collided()

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
	passed = passed and agent.get_observation_size() == 33

	var moving_target := Node3D.new()
	root.add_child(moving_target)
	body.target = moving_target
	body.success_hold_physics_frames = 1
	body.set_continue_after_success(true)
	moving_target.global_position = end_effector.global_position
	await physics_frame
	passed = passed and body.has_succeeded() and not body.is_terminal()
	moving_target.global_position += Vector3(body.success_distance * 3.0, 0.0, 0.0)
	await physics_frame
	passed = passed and not body.has_succeeded() and not body.is_terminal()
	moving_target.free()
	body.target = null
	body.set_continue_after_success(false)

	var grasp_target := RigidBody3D.new()
	grasp_target.freeze = true
	root.add_child(grasp_target)
	var grasp_point := Marker3D.new()
	grasp_target.add_child(grasp_point)
	grasp_target.global_position = end_effector.global_position
	body.task_mode = URDFRobotArmAgentBody.TaskMode.GRASPING
	body.configure_grasp_target(grasp_target, grasp_point)
	body.grasp_capture_distance = 0.05
	body.required_lift_height = 0.02
	body.grasp_hold_physics_frames = 1
	body.set_grasp_target_spawn_transform(
		Transform3D(grasp_target.global_basis, grasp_target.global_position - Vector3(0.0, 0.03, 0.0)))
	robot.set_joint_target_position("grip_left", -1.0)
	await physics_frame
	passed = passed and body.is_object_grasped()
	passed = passed and float(body.get_grasp_state_observation()[0]) > 0.5
	await physics_frame
	passed = passed and body.has_succeeded() and body.is_terminal()
	body.prepare_grasp_target_reset()
	grasp_target.free()
	body.task_mode = URDFRobotArmAgentBody.TaskMode.REACHING
	body.target = null
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
	passed = passed and body.has_collided()
	obstacle.free()
	var preserved_joint_position := robot.get_joint_position("xarm_5_joint")
	var reset_started_count := [0]
	controller.episode_reset_started.connect(
		func(_seed:int) -> void: reset_started_count[0] += 1)
	var preserved_reset: Dictionary = await controller.reset_episode_with_request({
		"seed": 123,
		"preserve_state": true
	})
	passed = passed and not body.has_collided() and not body.is_terminal()
	passed = passed and is_equal_approx(
		robot.get_joint_position("xarm_5_joint"), preserved_joint_position)
	passed = passed and reset_started_count[0] == 0
	passed = passed and bool(preserved_reset.get("info", {}).get("preserve_state", false))
	passed = passed and str(
		preserved_reset.get("info", {}).get("reset", {}).get("mode", "")) == "current_state"

	var non_adjacent_shapes := robot.get_link_node("xarm_4_link").find_children(
		"*", "CollisionShape3D", true, false)
	if not non_adjacent_shapes.is_empty():
		var non_adjacent_shape := non_adjacent_shapes[0] as CollisionShape3D
		var original_shape_transform := non_adjacent_shape.global_transform
		non_adjacent_shape.global_transform = robot_shape.global_transform
		body.call("_check_self_collisions")
		passed = passed and body.has_self_collided()
		passed = passed and str(
			body.get_last_collision_info().get("type", "")) == "self"
		non_adjacent_shape.global_transform = original_shape_transform
	else:
		passed = false

	if not passed:
		push_error(
				"URDF robot arm test failed: action_size=%d obs_size=%d joint=%f reset=%f tcp_delta=%f link_delta=%f terms=%s collided=%s initial_collision=%s collision_info=%s configured_tcp=%s physics=%s links=%s" %
				[
					agent.get_action_size(),
					agent.get_observation_size(),
					moved_position,
					reset_position,
					moved_tcp.distance_to(initial_tcp),
					moved_link_position.distance_to(initial_link_position),
					reward_terms,
					body.has_collided(),
					initial_collision_info,
					body.get_last_collision_info(),
					body.get("end_effector"),
				body.is_physics_processing(),
				robot.links.keys(),
			])
		body.free()
		quit(1)
		return
	controller.free()
	body.free()
	print("URDF robot arm agent test passed")
	quit(0)
