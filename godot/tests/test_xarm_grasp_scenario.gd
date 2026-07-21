extends SceneTree


func _initialize() -> void:
	var packed := load("res://scenarios/robotarms/XarmScenario.tscn") as PackedScene
	if not packed:
		push_error("XArm grasp scenario test: scene could not be loaded.")
		quit(1)
		return

	var scenario := packed.instantiate() as Node3D
	var bridge := scenario.get_node("BridgeServer")
	scenario.remove_child(bridge)
	bridge.free()
	root.add_child(scenario)
	await process_frame
	await physics_frame

	var arm: URDFRobotArmAgentBody = scenario.get_node("RobotArm")
	var robot: GodotRobot = arm.get_node("xarm")
	var target: RigidBody3D = scenario.get_node("Target")
	var grasp_point: Marker3D = scenario.get_node("Target/GraspPoint")
	var floor: StaticBody3D = scenario.get_node("Environment/Floor")
	var event_system: ScenarioEventSystem = scenario.get_node(
		"ScenarioController/ScenarioEventSystem")
	var self_collision_penalty: EventScenarioReward = scenario.get_node(
		"ScenarioController/ScenarioRewardSystem/SelfCollisionPenalty")
	var ground_collision_info: Dictionary = {}
	var initial_collision_info: Dictionary = arm.get_last_collision_info()
	var self_reward_value := 0.0
	var self_terminal_reason := ""
	var spawn_pool: Array[Marker3D] = scenario.call("_target_spawn_pool")
	var passed := spawn_pool.size() == 29
	passed = passed and arm.task_mode == URDFRobotArmAgentBody.TaskMode.GRASPING
	passed = passed and arm.grasp_target_body == target
	passed = passed and arm.target == grasp_point
	passed = passed and arm.agent.get_action_size() == 7
	passed = passed and arm.agent.get_observation_size() == 33
	passed = passed and not arm.has_collided()
	passed = passed and floor.is_in_group("robot_obstacle")
	passed = passed and floor.is_in_group("robot_support_surface")
	passed = passed and arm.support_contact_link_names.has("xarm_6_link")
	passed = passed and self_collision_penalty.event_name == "self_collision"
	passed = passed and is_equal_approx(self_collision_penalty.reward, -30.0)
	for marker in spawn_pool:
		passed = passed and marker.global_position.x >= 0.182
		passed = passed and marker.global_position.x <= 0.264
		passed = passed and marker.global_position.z >= -0.096
		passed = passed and marker.global_position.z <= 0.096

	scenario.call("_on_episode_reset_started", 123)
	var matches_spawn := false
	for marker in spawn_pool:
		var planar_distance := Vector2(
			target.global_position.x - marker.global_position.x,
			target.global_position.z - marker.global_position.z).length()
		if planar_distance < 0.000001:
			matches_spawn = true
			break
	passed = passed and matches_spawn
	passed = passed and target.linear_velocity.is_zero_approx()
	passed = passed and target.angular_velocity.is_zero_approx()

	var hand_shapes := robot.get_link_node("hand_link").find_children(
		"*", "CollisionShape3D", true, false)
	if not hand_shapes.is_empty():
		var hand_shape := hand_shapes[0] as CollisionShape3D
		var original_shape_transform := hand_shape.global_transform
		hand_shape.global_transform = Transform3D(
			hand_shape.global_basis,
			floor.global_position - Vector3(0.0, 0.001, 0.0))
		arm.call("_check_environment_collisions")
		ground_collision_info = arm.get_last_collision_info()
		passed = passed and arm.has_collided() and not arm.has_self_collided()
		passed = passed and str(
			arm.get_last_collision_info().get("second_body", "")) == "Floor"
		hand_shape.global_transform = original_shape_transform
		arm.initialize_episode_from_current_state()
		event_system.reset_events()
		event_system.reset_agent(arm, {"agent_id": str(arm.name)})
	else:
		passed = false

	arm.report_self_collision()
	event_system.update_agent(arm, {"agent_id": str(arm.name)})
	var event_context := event_system.get_agent_context(str(arm.name))
	self_terminal_reason = event_system.get_terminal_reason(str(arm.name))
	self_reward_value = self_collision_penalty.compute_reward(
		arm,
		{"agent_id": str(arm.name), "self_collision": 1.0})
	passed = passed and arm.has_collided() and arm.has_self_collided()
	passed = passed and float(event_context.get("self_collision", 0.0)) == 1.0
	passed = passed and self_terminal_reason == "self_collision"
	passed = passed and is_equal_approx(self_reward_value, -30.0)

	if not passed:
		push_error(
			"XArm grasp scenario test failed: spawns=%d action=%d obs=%d target=%s point=%s collision=%s initial=%s ground=%s self=%s context=%s terminal=%s reward=%f floor=%s/%s support=%s" % [
				spawn_pool.size(),
				arm.agent.get_action_size(),
				arm.agent.get_observation_size(),
				arm.grasp_target_body,
				arm.target,
				arm.has_collided(),
				initial_collision_info,
				ground_collision_info,
				arm.get_last_collision_info(),
				event_context,
				self_terminal_reason,
				self_reward_value,
				floor.is_in_group("robot_obstacle"),
				floor.is_in_group("robot_support_surface"),
				arm.support_contact_link_names
			])
		scenario.free()
		quit(1)
		return

	scenario.free()
	print("XArm grasp scenario test passed")
	quit(0)
