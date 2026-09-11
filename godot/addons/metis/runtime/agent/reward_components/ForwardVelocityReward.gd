extends RewardComponent
class_name ForwardVelocityReward

@export var reward := 0.01
@export var reference_speed := 10.0
@export var clamp_value := true


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if body == null:
		return 0.0

	var signed_speed := _get_signed_forward_speed(body)
	var normalized_speed := signed_speed / maxf(reference_speed, 0.000001)
	if clamp_value:
		normalized_speed = clampf(normalized_speed, -1.0, 1.0)

	return reward * normalized_speed * weight


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
	forward = forward.normalized()

	velocity.y = 0.0
	return velocity.dot(forward)
