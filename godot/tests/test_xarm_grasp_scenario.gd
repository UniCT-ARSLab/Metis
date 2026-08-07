extends SceneTree


func _initialize() -> void:
	var packed := load("res://scenarios/robotarms/XarmScenario.tscn") as PackedScene
	if not packed:
		push_error("XArm pose scenario test: scene could not be loaded.")
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
	var target: Node3D = scenario.get_node("Target")
	var target_pose: Marker3D = scenario.get_node("Target/GraspPose")
	var table: StaticBody3D = scenario.get_node("Workcell/Table")
	var controller: ScenarioController = scenario.get_node("ScenarioController")
	var goal_event: ManualScenarioEventSource = scenario.get_node(
		"ScenarioController/ScenarioEventSystem/GoalReached")
	var collision_event: ManualScenarioEventSource = scenario.get_node(
		"ScenarioController/ScenarioEventSystem/Collision")
	var goal_reward: EventScenarioReward = scenario.get_node(
		"ScenarioController/ScenarioRewardSystem/GoalReward")
	var collision_penalty: EventScenarioReward = scenario.get_node(
		"ScenarioController/ScenarioRewardSystem/CollisionPenalty")
	var spawn_pool: Array[Marker3D] = scenario.call("_target_spawn_pool")

	var passed := spawn_pool.size() == 25
	passed = passed and arm.target == target
	passed = passed and arm.target_pose == target_pose
	passed = passed and arm.agent.get_action_size() == 5
	passed = passed and arm.agent.get_observation_size() == 21
	passed = passed and arm.get_controlled_joint_names().size() == 5
	passed = passed and arm.get_manual_control_joint_names().has("grip_left")
	passed = passed and not arm.has_collided()
	passed = passed and table.is_in_group("robot_obstacle")
	passed = passed and controller.physics_frames_per_step == 3
	passed = passed and controller.continue_after_success
	passed = passed and not arm.terminate_on_success
	passed = passed and goal_event.event_name == "target_reached"
	passed = passed and goal_event.terminal_reason.is_empty()
	passed = passed and collision_event.terminal_reason == "collision"
	passed = passed and is_equal_approx(goal_reward.reward, 30.0)
	passed = passed and is_equal_approx(collision_penalty.reward, -20.0)

	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
	})
	await controller.reset_episode(456)
	passed = passed and not arm.is_terminal()
	passed = passed and not arm.has_collided()
	passed = passed and is_equal_approx(arm.success_distance, 0.08)
	passed = passed and is_equal_approx(arm.success_angle_degrees, 35.0)
	passed = passed and arm.success_hold_physics_frames == 20
	passed = passed and is_equal_approx(arm.success_max_joint_speed, 0.30)
	passed = passed and _matches_spawn_position(target.global_position, spawn_pool)
	passed = passed and _all_finite(arm.agent.get_observation_vector())

	scenario.call("_on_scenario_configured", {
		"training_episode": 10000,
		"training_mode": true,
	})
	await controller.reset_episode(789)
	passed = passed and not arm.is_terminal()
	passed = passed and is_equal_approx(arm.success_distance, 0.02)
	passed = passed and is_equal_approx(arm.success_angle_degrees, 20.0)
	passed = passed and arm.success_hold_physics_frames == 120
	passed = passed and is_equal_approx(arm.success_max_joint_speed, 0.12)

	if not passed:
		push_error(
			"XArm pose scenario test failed: spawns=%d action=%d obs=%d joints=%s manual=%s target=%s pose=%s progress=%f thresholds=%s" % [
				spawn_pool.size(),
				arm.agent.get_action_size(),
				arm.agent.get_observation_size(),
				arm.get_controlled_joint_names(),
				arm.get_manual_control_joint_names(),
				arm.target,
				arm.target_pose,
				arm.get_progress(),
				arm.get_debug_metrics().get("success_thresholds", {}),
			])
		scenario.free()
		quit(1)
		return

	scenario.free()
	print("XArm pose scenario test passed")
	quit(0)


func _matches_spawn_position(position: Vector3, spawns: Array[Marker3D]) -> bool:
	for marker in spawns:
		if position.is_equal_approx(marker.global_position):
			return true
	return false


func _all_finite(values: Array) -> bool:
	for value in values:
		if is_nan(value) or is_inf(value):
			return false
	return true
