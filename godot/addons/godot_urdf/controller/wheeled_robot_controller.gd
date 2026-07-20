@tool
extends Node3D
class_name WheeledRobotController

@export var robot: GodotRobot

@export var max_speed: float = 20.0

@export_group("Robot Type")
enum MobilityType {
	TWO_WHEEL_DIFF_DRIVE,
	FOUR_WHEEL_DIFF_DRIVE,
	# ACKERMANN
	# add hexapods, quadrupeds and humanoids?!
}

@export_group("Control")
@export var mobile_type: MobilityType = MobilityType.FOUR_WHEEL_DIFF_DRIVE:
	set(value):
		if mobile_type != value:
			mobile_type = value
			notify_property_list_changed()

# TODO: have names as Generic6DOFJoint3D directly?
var front_left_wheel: String = ""
var front_right_wheel: String = ""
var rear_left_wheel: String = ""
var rear_right_wheel: String = ""

var front_left_joint: Generic6DOFJoint3D
var front_right_joint: Generic6DOFJoint3D
var rear_left_joint: Generic6DOFJoint3D
var rear_right_joint: Generic6DOFJoint3D

func _physics_process(delta):
	if Engine.is_editor_hint():
		return
	var throttle = Input.get_axis("decelerate", "accelerate")
	var steering = Input.get_axis("right", "left")
	var left_target = throttle - steering
	var right_target = throttle + steering
	var left_velocity = left_target * max_speed
	var right_velocity = right_target * max_speed


	match mobile_type:
		MobilityType.TWO_WHEEL_DIFF_DRIVE:
			apply_motor_velocity(front_left_joint, left_velocity)
			apply_motor_velocity(front_right_joint, right_velocity)
		MobilityType.FOUR_WHEEL_DIFF_DRIVE:
			apply_motor_velocity(front_left_joint, left_velocity)
			apply_motor_velocity(front_right_joint, right_velocity)
			apply_motor_velocity(rear_left_joint, left_velocity)
			apply_motor_velocity(rear_right_joint, right_velocity)


func apply_motor_velocity(joint: Generic6DOFJoint3D, target_velocity: float):
	if joint:
		joint.set_param_z(
			Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY,
			target_velocity)
		joint.set_flag_z(
			Generic6DOFJoint3D.FLAG_ENABLE_MOTOR,
			true)

func _get_property_list():
	var list = []
	list.append({name = "front_left_wheel", type = TYPE_STRING})
	list.append({name = "front_right_wheel", type = TYPE_STRING})
	match mobile_type:
		MobilityType.FOUR_WHEEL_DIFF_DRIVE: #, MobilityType.ACKERMANN:
			list.append({name = "rear_left_wheel", type = TYPE_STRING})
			list.append({name = "rear_right_wheel", type = TYPE_STRING})
	return list

func _ready():
	if not Engine.is_editor_hint():
		_initialize_robot()

func _initialize_robot():
	if not robot:
		push_error("Robot Controller: No Robot Node assigned!")
		return
	front_left_joint = robot.find_child(front_left_wheel, true, false)
	front_right_joint = robot.find_child(front_right_wheel, true, false)
	match mobile_type:
		MobilityType.FOUR_WHEEL_DIFF_DRIVE: #, MobilityType.ACKERMANN:
			rear_left_joint = robot.find_child(rear_left_wheel, true, false)
			rear_right_joint = robot.find_child(rear_right_wheel, true, false)
