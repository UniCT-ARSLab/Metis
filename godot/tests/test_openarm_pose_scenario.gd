extends SceneTree


func _initialize() -> void:
	var packed := load(
		"res://scenarios/robotarms/openarm_scenario.tscn") as PackedScene
	if not packed:
		push_error("OpenArm pose scenario test: scene could not be loaded.")
		quit(1)
		return

	var scenario := packed.instantiate() as Node3D
	var bridge := scenario.get_node_or_null("BridgeServer")
	if bridge:
		scenario.remove_child(bridge)
		bridge.free()
	root.add_child(scenario)
	await process_frame
	await physics_frame

	var arm: URDFRobotArmAgentBody = scenario.get_node("OpenarmAgent")
	var controller: ScenarioController = scenario.get_node("ScenarioController")
	var robot: GodotRobot = arm.get_node("openarm")
	var easy_region: Area3D = scenario.get_node("EasyArea")
	var medium_region: Area3D = scenario.get_node("MediumArea")
	var hard_region: Area3D = scenario.get_node("HardArea")
	var expected_home := [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	var failures: Array[String] = []
	if arm.agent.get_action_size() != 7:
		failures.append("action_size")
	if arm.agent.get_observation_size() != 27:
		failures.append("observation_size")
	if arm.distance_progress_mode != arm.DistanceProgressMode.INVERSE_DISTANCE:
		failures.append("distance_progress_mode")
	if arm.pose_progress_mode != arm.PoseProgressMode.POSITION_GATED:
		failures.append("pose_progress_mode")
	if not is_equal_approx(arm.workspace_scale, 1.0):
		failures.append("workspace_scale")
	if not is_equal_approx(arm.max_joint_speed_override, 0.5):
		failures.append("joint_speed_limit")
	if not arm.self_body_collision_enabled:
		failures.append("self_collision_disabled")
	if arm._self_body_rids.is_empty():
		failures.append("body_link")
	if not arm.get_manual_control_joint_names().has(
			"openarm_right_finger_joint1"):
		failures.append("manual_gripper")
	if not _positions_match(arm, expected_home):
		failures.append("initial_home")
	if arm.is_terminal() or arm.has_collided():
		failures.append("initial_terminal")
	for region in [easy_region, medium_region, hard_region]:
		if (
			not region.has_method("sample_transform")
			or region.collision_layer != 0
			or region.collision_mask != 0
		):
			failures.append("target_region_configuration_%s" % region.name)

	var spawn_pool: Array = scenario.call("_target_spawn_pool")
	scenario.set("_training_episode", 0)
	if int(scenario.call("_curriculum_target_pool_size", spawn_pool.size())) != 4:
		failures.append("target_pool_initial")
	scenario.set("_training_episode", 2000)
	var middle_pool_size := int(scenario.call(
		"_curriculum_target_pool_size", spawn_pool.size()))
	if middle_pool_size <= 4 or middle_pool_size >= spawn_pool.size():
		failures.append("target_pool_middle")
	scenario.set("_training_episode", 3200)
	if (
		int(scenario.call("_curriculum_target_pool_size", spawn_pool.size()))
		!= spawn_pool.size()
	):
		failures.append("target_pool_full")
	scenario.set("_training_episode", 0)
	if not is_equal_approx(float(scenario.call(
		"_bootstrap_curriculum_fraction")),
		float(scenario.bootstrap_min_regular_reset_fraction)):
		failures.append("bootstrap_fraction_initial")
	var bootstrap_middle_episode := roundi(
		(float(scenario.bootstrap_curriculum_start_episode)
		+ float(scenario.bootstrap_curriculum_full_episode)) * 0.5)
	scenario.set("_training_episode", bootstrap_middle_episode)
	if not is_equal_approx(
		float(scenario.call("_bootstrap_curriculum_fraction")), 0.5):
		failures.append("bootstrap_fraction_middle")
	scenario.set("_training_episode", scenario.bootstrap_curriculum_full_episode)
	if not is_equal_approx(float(scenario.call(
		"_bootstrap_curriculum_fraction")), 1.0):
		failures.append("bootstrap_fraction_full")
	scenario.call("_on_scenario_configured", {
		"training_episode": 5000,
		"training_mode": true,
		"curriculum_level": 0.0,
	})
	if int(scenario.call("_curriculum_episode")) != 0:
		failures.append("adaptive_curriculum_initial")
	if controller.continue_after_success:
		failures.append("adaptive_curriculum_initial_tracking")
	var adaptive_episode_fraction := clampf(
			(5000.0 - float(scenario.bootstrap_curriculum_start_episode))
			/ float(
				scenario.bootstrap_curriculum_full_episode
				- scenario.bootstrap_curriculum_start_episode),
			0.0,
			1.0)
	var expected_adaptive_bootstrap := lerpf(
		float(scenario.bootstrap_min_regular_reset_fraction),
		1.0,
		adaptive_episode_fraction)
	var actual_adaptive_bootstrap := float(scenario.call(
		"_bootstrap_curriculum_fraction"))
	if not is_equal_approx(actual_adaptive_bootstrap, expected_adaptive_bootstrap):
		failures.append(
			"adaptive_bootstrap_initial_%.4f_expected_%.4f"
			% [actual_adaptive_bootstrap, expected_adaptive_bootstrap])
	if int(scenario.call(
		"_curriculum_target_pool_size", spawn_pool.size())) != 4:
		failures.append("adaptive_target_pool_initial")
	scenario.call("_apply_adaptive_pose_curriculum")
	if (
		not is_equal_approx(float(arm.success_distance), 0.08)
		or not is_equal_approx(float(arm.success_angle_degrees), 45.0)
	):
		failures.append("adaptive_reach_initial")
	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
		"curriculum_level": 0.4,
	})
	if int(scenario.call(
		"_curriculum_target_pool_size", spawn_pool.size())) != 4:
		failures.append("adaptive_target_pool_changed_with_precision")
	scenario.call("_apply_adaptive_pose_curriculum")
	if (
		not is_equal_approx(float(arm.success_distance), 0.06)
		or not is_equal_approx(float(arm.success_angle_degrees), 38.0)
	):
		failures.append("adaptive_reach_stage1")
	var expected_level_bootstrap := lerpf(
		float(scenario.bootstrap_min_regular_reset_fraction),
		1.0,
		clampf(
			0.4 / float(scenario.adaptive_bootstrap_full_level), 0.0, 1.0))
	var actual_level_bootstrap := float(scenario.call(
		"_bootstrap_curriculum_fraction"))
	if not is_equal_approx(actual_level_bootstrap, expected_level_bootstrap):
		failures.append(
			"adaptive_bootstrap_level_fraction_%.4f_expected_%.4f"
			% [actual_level_bootstrap, expected_level_bootstrap])
	if not is_equal_approx(float(scenario.call(
		"_curriculum_orientation_weight")), 0.15):
		failures.append("adaptive_orientation_initial")
	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
		"curriculum_level": 0.7,
	})
	if (
		int(scenario.call("_curriculum_target_pool_size", spawn_pool.size()))
		!= spawn_pool.size()
	):
		failures.append("adaptive_target_pool_full")
	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
		"curriculum_level": 0.8,
	})
	if not is_equal_approx(float(scenario.call(
		"_curriculum_orientation_weight")), 0.25):
		failures.append("adaptive_orientation_final")

	for level_and_expected in [
		[0.4, Vector3(1.0, 0.0, 0.0)],
		[0.55, Vector3(0.5, 0.5, 0.0)],
		[0.7, Vector3(0.4, 0.4, 0.2)],
		[0.75, Vector3(0.2916667, 0.4083333, 0.30)],
	]:
		scenario.set("_curriculum_level_override", level_and_expected[0])
		var weights: Vector3 = scenario.call("_adaptive_region_weights")
		if not weights.is_equal_approx(level_and_expected[1]):
			failures.append(
				"adaptive_region_weights_%.2f_%s"
				% [level_and_expected[0], weights])

	var region_rng := RandomNumberGenerator.new()
	region_rng.seed = 91234
	scenario.set("_curriculum_level_override", 0.45)
	var balanced_region_counts := {
		"EasyArea": 0,
		"MediumArea": 0,
	}
	for _draw in range(20):
		var selected_region: Area3D = scenario.call(
			"_select_target_region", region_rng)
		balanced_region_counts[selected_region.name] += 1
	if (
		balanced_region_counts["EasyArea"] != 10
		or balanced_region_counts["MediumArea"] != 10
	):
		failures.append(
			"adaptive_balanced_region_schedule_%s"
			% balanced_region_counts)

	scenario.set("_curriculum_level_override", 0.75)
	var region_counts := {
		"EasyArea": 0,
		"MediumArea": 0,
		"HardArea": 0,
	}
	for _draw in range(20):
		var selected_region: Area3D = scenario.call(
			"_select_target_region", region_rng)
		region_counts[selected_region.name] += 1
	if (
		region_counts["EasyArea"] != 6
		or region_counts["MediumArea"] != 8
		or region_counts["HardArea"] != 6
	):
		failures.append("adaptive_region_schedule_%s" % region_counts)

	easy_region.call("reset_sampling_sequence")
	var sampled_cells := {}
	for _draw in range(20):
		var sampled: Dictionary = easy_region.call(
			"sample_transform", region_rng, Basis.IDENTITY)
		var sampled_transform: Transform3D = sampled["transform"]
		sampled_cells[int(sampled["cell"])] = true
		if not bool(easy_region.call(
			"contains_global_position", sampled_transform.origin)):
			failures.append("target_region_sample_outside")
	if sampled_cells.size() != 20:
		failures.append("target_region_coverage_%d" % sampled_cells.size())

	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
		"curriculum_level": 1.0,
	})
	if int(scenario.call("_curriculum_episode")) != scenario.adaptive_curriculum_full_episode:
		failures.append("adaptive_curriculum_full")
	if controller.continue_after_success:
		failures.append("adaptive_curriculum_unexpected_tracking")
	if not is_equal_approx(float(scenario.call(
		"_bootstrap_curriculum_fraction")), 1.0):
		failures.append("adaptive_bootstrap_full")
	scenario.call("_apply_adaptive_pose_curriculum")
	if (
		not is_equal_approx(float(arm.success_distance), 0.04)
		or not is_equal_approx(float(arm.success_angle_degrees), 28.0)
		or arm.success_hold_physics_frames != 120
		or not is_equal_approx(float(arm.success_max_joint_speed), 0.10)
	):
		failures.append("adaptive_reach_final")
	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
	})
	scenario.set("_training_episode", 0)
	# The next block specifically validates the assisted reset pose. Production
	# training keeps at least half of the resets regular even at curriculum level 0.
	scenario.bootstrap_min_regular_reset_fraction = 0.0
	scenario.bootstrap_target_jitter_radius = 0.0
	scenario.bootstrap_joint_jitter_degrees = 0.0

	var target: Node3D = scenario.get_node("Target")
	var bootstrap_target: Marker3D = scenario.get_node("BootstrapTargetPose")
	var table: StaticBody3D = scenario.get_node("Workcell/Table")
	var table_shape: CollisionShape3D = table.get_node("CollisionShape3D")
	var table_box := table_shape.shape as BoxShape3D
	var table_top_y := table.global_position.y + table_box.size.y * 0.5
	if bootstrap_target.global_position.y - table_top_y < 0.055:
		failures.append("bootstrap_target_table_clearance")
	for spawn in spawn_pool:
		if spawn.global_position.y - table_top_y < 0.08:
			failures.append("spawn_target_table_clearance")
			break
	var tool_pose: Node3D = arm.get_node("EndEffector/ToolPose")
	var saved_target_transform := target.global_transform
	target.global_position = tool_pose.global_position + Vector3(2.0, 0.0, 0.0)
	var far_progress := arm.get_progress()
	target.global_position = tool_pose.global_position + Vector3(1.0, 0.0, 0.0)
	var nearer_progress := arm.get_progress()
	if far_progress <= 0.0:
		failures.append("far_progress_saturated")
	if nearer_progress <= far_progress:
		failures.append("distance_progress_direction")
	target.global_transform = saved_target_transform
	arm.notify_target_pose_relocated()

	var distance_reward = scenario.get_node_or_null(
		"ScenarioController/ScenarioRewardSystem/DistanceApproach")
	if not distance_reward:
		failures.append("distance_reward_missing")
	else:
		if not is_equal_approx(float(distance_reward.approach_reward_scale), 5.0):
			failures.append("distance_approach_scale")
		if not is_equal_approx(float(distance_reward.retreat_penalty_scale), 3.0):
			failures.append("distance_retreat_scale")
	var goal_reward = scenario.get_node_or_null(
		"ScenarioController/ScenarioRewardSystem/GoalReward")
	if (
		not goal_reward
		or not is_equal_approx(float(goal_reward.reward), 50.0)
	):
		failures.append("goal_reward")
	var joint_action = arm.get_node_or_null("Agent/ActionSpace/JointVelocity")
	if (
		not joint_action
		or not is_equal_approx(float(joint_action.get("exploration_low")), -0.35)
		or not is_equal_approx(float(joint_action.get("exploration_high")), 0.35)
	):
		failures.append("joint_exploration_bounds")
	if not is_equal_approx(float(arm.near_target_distance), 0.2):
		failures.append("near_target_distance")
	var hold_stillness := arm.get_node("Agent/RewardSystem/HoldStillness")
	if not is_equal_approx(float(hold_stillness.weight), 0.05):
		failures.append("hold_stillness_weight")
	var pose_tracking := arm.get_node("Agent/RewardSystem/PoseTracking")
	if not is_equal_approx(float(pose_tracking.weight), 0.1):
		failures.append("pose_tracking_weight")
	if bool(scenario.continuous_target_sampling):
		failures.append("continuous_sampling_enabled")
	var stall_reward = scenario.get_node(
		"ScenarioController/ScenarioRewardSystem/NoProgress")
	if not is_zero_approx(float(stall_reward.stalled_progress_penalty)):
		failures.append("duplicated_stall_terminal_penalty")
	if float(stall_reward.no_progress_step_penalty) >= 0.0:
		failures.append("stall_step_penalty")
	if not is_equal_approx(float(stall_reward.ignore_stall_above_progress), 0.95):
		failures.append("stall_near_target_threshold")
	var terminal_failure = scenario.get_node_or_null(
		"ScenarioController/ScenarioRewardSystem/TerminalFailure")
	if not terminal_failure:
		failures.append("terminal_failure_missing")
	elif (
		not is_equal_approx(float(terminal_failure.base_failure_penalty), -10.0)
		or not is_equal_approx(
			float(terminal_failure.remaining_progress_penalty), -20.0)
		or not bool(terminal_failure.penalize_truncation)
	):
		failures.append("terminal_failure_scale")

	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
	})
	for seed in range(12):
		var reset_result: Dictionary = await controller.reset_episode(7000 + seed)
		for _frame in range(5):
			await physics_frame
		var reset_channel := _first_agent_channel(reset_result)
		var reset_info: Dictionary = reset_channel.get("info", {}).get("reset", {})
		if str(reset_info.get("task_reset_mode", "")) != "bootstrap":
			failures.append("bootstrap_reset_mode_%d" % seed)
		if arm.is_terminal() or arm.has_collided():
			failures.append("terminal_seed_%d" % seed)
		if not arm.get_last_self_collision_details().is_empty():
			failures.append("collision_details_seed_%d" % seed)
		if not arm.get_last_collision_details().is_empty():
			failures.append("generic_collision_details_seed_%d" % seed)
		var bootstrap_index := int(scenario.get("_active_bootstrap_index"))
		var expected_bootstrap: Array = Array(
			scenario.bootstrap_pose_joints[bootstrap_index])
		if not _positions_match(arm, expected_bootstrap):
			failures.append("bootstrap_seed_%d" % seed)
		var expected_target: Vector3 = scenario.bootstrap_pose_targets[
			bootstrap_index]
		if target.global_position.distance_to(expected_target) > 0.0001:
			failures.append("bootstrap_target_seed_%d" % seed)
		if seed == 0:
			for _frame in range(30):
				await physics_frame
			if not arm.has_succeeded():
				failures.append("bootstrap_pose_not_held")
			if arm.has_collided():
				failures.append("bootstrap_pose_collision")

	# A bootstrap pose must also tolerate the first small policy command. The previous
	# reference was collision-free while static but put link4 on the base boundary.
	for joint_index in range(arm.get_joint_count()):
		for direction in [-1.0, 1.0]:
			await controller.reset_episode(7600 + joint_index * 2 + int(direction > 0.0))
			var action: Array[float] = []
			action.resize(arm.get_joint_count())
			action.fill(0.0)
			action[joint_index] = 0.1 * direction
			arm.apply_action(action)
			for _frame in range(3):
				await physics_frame
			arm.apply_action(Array())
			if arm.has_collided():
				var details := arm.get_last_collision_details()
				failures.append(
					"bootstrap_action_collision_%d_%s_%s"
					% [joint_index, str(direction), str(details)])

	# Intermediate adaptive levels must select between complete safe reset configurations.
	# Interpolating their joint angles used to place the hand inside the table at level 0.1.
	for level in [0.1, 0.25, 0.5, 0.75]:
		scenario.call("_on_scenario_configured", {
			"training_episode": 0,
			"training_mode": true,
			"curriculum_level": level,
		})
		for seed in range(12):
			await controller.reset_episode(
				7700 + roundi(level * 100.0) * 20 + seed)
			for _frame in range(5):
				await physics_frame
			if arm.has_collided():
				failures.append(
					"adaptive_reset_collision_%.2f_%d details=%s joints=%s"
					% [
						level,
						seed,
						str(arm.get_last_collision_details()),
						str(arm.get_manual_pose_diagnostics().get("joints", [])),
					])

	# The regular home reset must admit the first bounded policy command too.
	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
		"curriculum_level": 1.0,
	})
	for joint_index in range(arm.get_joint_count()):
		for direction in [-1.0, 1.0]:
			await controller.reset_episode(
				9000 + joint_index * 2 + int(direction > 0.0))
			var action: Array[float] = []
			action.resize(arm.get_joint_count())
			action.fill(0.0)
			action[joint_index] = 0.1 * direction
			arm.apply_action(action)
			for _frame in range(3):
				await physics_frame
			arm.apply_action(Array())
			if arm.has_collided():
				failures.append(
					"home_action_collision_%d_%s_%s"
					% [
						joint_index,
						str(direction),
						str(arm.get_last_collision_details()),
					])

	scenario.call("_on_scenario_configured", {
		"training_episode": 0,
		"training_mode": true,
	})
	scenario.set("_training_episode", scenario.bootstrap_curriculum_full_episode)
	await controller.reset_episode(7800)
	for _frame in range(3):
		await physics_frame
	if not _positions_match(arm, expected_home):
		failures.append("bootstrap_did_not_fade")
	scenario.set("_training_episode", 0)

	# These configurations were captured from visually-correct manual reaches.
	# They guard against evaluating the imported URDF wrist frame as the grasp frame.
	var manual_references := [
		{
			"name": "near_aligned",
			"joints": [
				0.4183333333333322,
				-0.0340295555293905,
				-0.02333333333333334,
				0.7050000000000027,
				-0.0949999999999999,
				0.32166666666666655,
				-0.018333333333333337,
			],
			"position_error": 0.01108228,
			"orientation_limit": 1.0,
		},
		{
			"name": "raised_wrist",
			"joints": [
				0.4149999999999989,
				-0.11092151511395537,
				0.0,
				0.7816666666666712,
				-0.1183333333333332,
				0.0,
				0.0,
			],
			"position_error": 0.05363322,
			"orientation_limit": 20.0,
		},
		{
			"name": "offset_elbow",
			"joints": [
				0.438333333333332,
				0.07683290081155485,
				-0.21833333333333377,
				0.7183333333333364,
				0.0066666666666666706,
				0.3366666666666664,
				0.0,
			],
			"position_error": 0.01877926,
			"orientation_limit": 10.0,
		},
	]
	var joint_names := arm.get_controlled_joint_names()
	for reference_index in range(manual_references.size()):
		var reference: Dictionary = manual_references[reference_index]
		await controller.reset_episode(7999 + reference_index)
		target.global_transform = Transform3D(
			Basis.IDENTITY,
			Vector3(0.43611306, 0.46049988, 0.06939696))
		var positions: Array = reference["joints"]
		var joint_positions := {}
		for joint_index in range(joint_names.size()):
			joint_positions[joint_names[joint_index]] = positions[joint_index]
		robot.reset_joint_positions(joint_positions)
		for _frame in range(3):
			await physics_frame
		var reference_metrics := arm.get_debug_metrics()
		var position_error := float(
			reference_metrics.get("position_error_m", 1.0))
		var orientation_error := float(
			reference_metrics.get("orientation_error_deg", 180.0))
		if absf(position_error - float(reference["position_error"])) > 0.002:
			failures.append(
				"manual_%s_position_%.4f"
				% [reference["name"], position_error])
		if orientation_error > float(reference["orientation_limit"]):
			failures.append(
				"manual_%s_orientation_%.2f"
				% [reference["name"], orientation_error])
		if arm.has_collided():
			failures.append("manual_%s_collision" % reference["name"])

	# Exercise the complete collision contract independently from the exact URDF contact pair:
	# agent flag -> scenario event -> terminal reason -> one progress-aware failure penalty.
	await controller.reset_episode(8000)
	arm.report_obstacle_collision()
	var step_result: Dictionary = await controller.step({
		str(arm.name): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	})
	var step_channel := _first_agent_channel(step_result)
	var step_info: Dictionary = step_channel.get("info", {})
	var scenario_terms: Dictionary = step_info.get("scenario_terms", {})
	var component_values: Dictionary = scenario_terms.get(
		"component_values", {})
	if not bool(step_channel.get("terminated", false)):
		failures.append("collision_not_terminal")
	if str(step_info.get("terminal_reason", "")) != "collision":
		failures.append("collision_terminal_reason")
	if str(step_info.get("collision_source", "")) != "reported_obstacle":
		failures.append("collision_source_missing")
	var collision_progress := clampf(
		float(step_info.get("track_progress", 0.0)), 0.0, 1.0)
	var expected_collision_penalty := -10.0 - 20.0 * (
		1.0 - collision_progress)
	if not is_equal_approx(
		float(component_values.get("terminal_failure", 0.0)),
		expected_collision_penalty):
		failures.append("collision_reward")
	var cached_result: Dictionary = await controller.step({
		str(arm.name): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	})
	var cached_channel := _first_agent_channel(cached_result)
	if not is_zero_approx(float(cached_channel.get("reward", 0.0))):
		failures.append("collision_reward_repeated")

	await controller.reset_episode(8001)
	if not robot.set_joint_target_position("openarm_right_finger_joint1", 0.02):
		failures.append("gripper_command")
	await physics_frame
	if not is_equal_approx(
		robot.get_joint_position("openarm_right_finger_joint1"), 0.02):
		failures.append("gripper_source_position")
	if not is_equal_approx(
		robot.get_joint_position("openarm_right_finger_joint2"), 0.02):
		failures.append("gripper_mimic_position")

	if not failures.is_empty():
		push_error(
			(
				"OpenArm pose scenario test failed: action=%d obs=%d joints=%s "
				+ "terminal=%s collided=%s self_collision=%s failures=%s "
				+ "body_names=%s body_node=%s positions=%s collision_step=%s"
			) % [
				arm.agent.get_action_size(),
				arm.agent.get_observation_size(),
				arm.get_controlled_joint_names(),
				arm.is_terminal(),
				arm.has_collided(),
				arm.get_last_self_collision_details(),
				failures,
				arm.self_body_link_names,
				arm._robot.get_link_node("openarm_body_link0"),
				arm.get_manual_pose_diagnostics().get("joints", []),
				step_result,
			])
		scenario.free()
		quit(1)
		return

	scenario.free()
	print("OpenArm pose scenario test passed")
	quit(0)


func _positions_match(
		arm: URDFRobotArmAgentBody,
		expected: Array,
		tolerance := 0.0001) -> bool:
	var diagnostics := arm.get_manual_pose_diagnostics()
	var joints: Array = diagnostics.get("joints", [])
	if joints.size() < expected.size():
		return false
	for index in range(expected.size()):
		if (
			absf(float(joints[index].get("position_rad", 0.0)) - expected[index])
			> tolerance
		):
			return false
	return true


func _first_agent_channel(result: Dictionary) -> Dictionary:
	if result.has("reward"):
		return result
	var agents: Array = result.get("agents", [])
	if agents.is_empty() or typeof(agents[0]) != TYPE_DICTIONARY:
		return {}
	return agents[0]
