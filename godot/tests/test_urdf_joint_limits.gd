extends SceneTree


func _initialize() -> void:
	var packed := load("res://agents/RobotArms/xarm/x_arm_agent.tscn") as PackedScene
	if not packed:
		push_error("URDF joint limit test: agent scene could not be loaded.")
		quit(1)
		return

	var body := packed.instantiate() as URDFRobotArmAgentBody
	root.add_child(body)
	await process_frame
	await physics_frame

	var robot: GodotRobot = body.get_node("xarm")
	var joint_names := body.get_controlled_joint_names()
	var passed := not joint_names.is_empty()
	var details: Array[String] = []

	for index in range(joint_names.size()):
		var joint_name := joint_names[index]
		var joint := robot.urdf.get_joint(joint_name)
		if not joint or joint.type != "revolute" or not joint.limit:
			passed = false
			details.append(
				"%s has no bounded revolute definition (type=%s limit=%s)" % [
					joint_name,
					str(joint.type) if joint else "<missing>",
					str(joint.limit) if joint else "<missing>",
				])
			continue

		var lower := minf(joint.limit.lower, joint.limit.upper)
		var upper := maxf(joint.limit.lower, joint.limit.upper)
		robot.set_joint_target_position(joint_name, upper + 10.0)
		var upper_position := robot.get_joint_position(joint_name)
		var positive_action: Array[float] = []
		positive_action.resize(joint_names.size())
		positive_action.fill(0.0)
		positive_action[index] = 1.0
		var upper_applied: Array = body.apply_action(positive_action)
		await physics_frame

		robot.set_joint_target_position(joint_name, lower - 10.0)
		var lower_position := robot.get_joint_position(joint_name)
		var negative_action: Array[float] = []
		negative_action.resize(joint_names.size())
		negative_action.fill(0.0)
		negative_action[index] = -1.0
		var lower_applied: Array = body.apply_action(negative_action)
		await physics_frame

		var joint_passed := (
			upper_position <= upper + 0.000001
			and lower_position >= lower - 0.000001
			and is_zero_approx(float(upper_applied[index]))
			and is_zero_approx(float(lower_applied[index]))
		)
		passed = passed and joint_passed
		details.append(
			"%s=[%.4f,%.4f] measured=[%.4f,%.4f] guarded=[%.4f,%.4f]" % [
				joint_name,
				lower,
				upper,
				lower_position,
				upper_position,
				float(lower_applied[index]),
				float(upper_applied[index]),
			])

	for value in body.get_joint_position_observation():
		passed = passed and float(value) >= -1.0 and float(value) <= 1.0

	if not passed:
		push_error("URDF joint limit test failed: %s" % "; ".join(details))
		body.free()
		quit(1)
		return

	print("URDF joint limit test passed: %s" % "; ".join(details))
	body.free()
	quit(0)
