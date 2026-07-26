extends SceneTree


const REFERENCE_JOINTS := {
	"xarm_6_joint": 1.5142961185835626,
	"xarm_5_joint": -0.11333333333333329,
	"xarm_4_joint": 1.201451257384437,
	"xarm_3_joint": 0.9933333333333295,
	"xarm_2_joint": 0.0,
	"grip_left": -1.0166666666666766,
}


func _initialize() -> void:
	var packed := load(
		"res://agents/RobotArms/xarm/x_arm_agent.tscn") as PackedScene
	if not packed:
		push_error("XArm reference pose test: agent could not be loaded.")
		quit(1)
		return

	var target_pose := Node3D.new()
	target_pose.name = "ReferenceTargetPose"
	target_pose.position = Vector3(
		0.2123040407896042,
		0.13049985468387604,
		0.016337677836418152)
	root.add_child(target_pose)
	var arm := packed.instantiate() as URDFRobotArmAgentBody
	arm.position.y = 0.0006389916
	arm.target = target_pose
	arm.target_pose = target_pose
	root.add_child(arm)
	await process_frame
	await physics_frame

	var robot: GodotRobot = arm.get_node("xarm")
	robot.reset_joint_positions(REFERENCE_JOINTS)
	await physics_frame
	await physics_frame

	var metrics := arm.get_debug_metrics()
	var position_error := float(metrics.get("position_error_m", INF))
	var orientation_error := float(metrics.get("orientation_error_deg", INF))
	var calibrated_local_transform := (
		arm.end_effector.global_transform.affine_inverse()
		* target_pose.global_transform)
	var passed := arm.agent.get_action_size() == 5
	passed = passed and arm.agent.get_observation_size() == 21
	passed = passed and not robot.get_actuated_joint_names().has("wrist_roll")
	passed = passed and is_equal_approx(
		robot.get_joint_position("grip_left"), -1.0166666666666766)
	passed = passed and position_error < 0.0001
	passed = passed and orientation_error < 0.01

	if not passed:
		push_error(
			"XArm reference pose failed: action=%d observation=%d position=%f "
			% [
				arm.agent.get_action_size(),
				arm.agent.get_observation_size(),
				position_error]
			+ "orientation=%f metrics=%s" % [orientation_error, metrics])
		print(
			"XArm recommended ToolPose local transform: %s"
			% calibrated_local_transform)
		arm.free()
		target_pose.free()
		quit(1)
		return

	print(
		"XArm reference pose test passed: position=%.6f m orientation=%.6f deg"
		% [position_error, orientation_error])
	arm.free()
	target_pose.free()
	quit(0)
