extends "res://scripts/agent/reward_components/RewardComponent.gd"
class_name LowSpeedPenaltyReward

@export var min_abs_speed := 0.5
@export var grace_steps := 15
@export var penalty := -0.02

var _low_speed_steps := 0


func reset_reward(_context:Dictionary = {}) -> void:
	_low_speed_steps = 0


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if body == null:
		return 0.0
	if body.has_method("is_crashed") and bool(body.is_crashed()):
		return 0.0

	var speed := absf(_get_signed_forward_speed(body))
	if speed >= min_abs_speed:
		_low_speed_steps = 0
		return 0.0

	_low_speed_steps += 1
	if _low_speed_steps <= grace_steps:
		return 0.0

	return penalty * weight


func _get_signed_forward_speed(body) -> float:
	if body.has_method("get_signed_forward_speed"):
		return float(body.get_signed_forward_speed())

	if not (body is Node3D):
		return 0.0

	var velocity := Vector3.ZERO
	if body is CharacterBody3D:
		velocity = body.velocity
	elif body is RigidBody3D:
		velocity = body.linear_velocity

	var forward: Vector3 = body.global_transform.basis.z
	forward.y = 0.0
	if forward.length() <= 0.000001:
		return 0.0

	velocity.y = 0.0
	return velocity.dot(forward.normalized())
