extends SceneTree


func _initialize() -> void:
	var packed := load("res://agents/RobotArms/xarm/x_arm_agent.tscn") as PackedScene
	if not packed:
		push_error("URDF kinematic contact test: agent scene could not be loaded.")
		quit(1)
		return

	var body := packed.instantiate() as URDFRobotArmAgentBody
	root.add_child(body)
	await process_frame
	await physics_frame

	body.set_physics_process(false)
	var robot: GodotRobot = body.get_node("xarm")
	robot.set_physics_process(false)
	var moving_link := robot.get_link_node("hand_link") as RigidBody3D
	if not moving_link:
		push_error("URDF kinematic contact test: hand_link was not found.")
		body.free()
		quit(1)
		return

	for link_node in robot.links.values():
		if not link_node is RigidBody3D:
			continue
		for collision_shape in link_node.find_children(
				"*", "CollisionShape3D", true, false):
			(collision_shape as CollisionShape3D).disabled = true

	var moving_shape := CollisionShape3D.new()
	var moving_box := BoxShape3D.new()
	moving_box.size = Vector3(0.08, 0.08, 0.08)
	moving_shape.shape = moving_box
	moving_link.add_child(moving_shape)
	moving_link.global_transform = Transform3D(
		Basis.IDENTITY, Vector3(0.0, 1.0, 0.0))

	var target := RigidBody3D.new()
	target.mass = 0.02
	target.gravity_scale = 0.0
	target.linear_damp = 0.0
	target.angular_damp = 0.0
	target.continuous_cd = true
	var target_shape := CollisionShape3D.new()
	var target_sphere := SphereShape3D.new()
	target_sphere.radius = 0.025
	target_shape.shape = target_sphere
	target.add_child(target_shape)
	root.add_child(target)
	target.global_position = Vector3(0.085, 1.0, 0.0)

	await physics_frame
	await physics_frame
	var initial_target_x := target.global_position.x
	for _index in range(12):
		moving_link.global_position += Vector3(0.01, 0.0, 0.0)
		await physics_frame

	var passed := (
		moving_link.freeze
		and moving_link.freeze_mode == RigidBody3D.FREEZE_MODE_KINEMATIC
		and target.global_position.x > initial_target_x + 0.01)
	if not passed:
		push_error(
			"URDF kinematic contact test failed: mode=%d start_x=%f end_x=%f" % [
				moving_link.freeze_mode,
				initial_target_x,
				target.global_position.x,
			])
		target.free()
		body.free()
		quit(1)
		return

	target.free()
	body.free()
	print("URDF kinematic contact test passed")
	quit(0)
