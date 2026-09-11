extends ObservationSource
class_name BodyKinematicsObservationSource

@export var include_position := true
@export var include_forward := true
@export var include_local_velocity := true
@export var position_name := "position"
@export var forward_name := "forward"
@export var local_velocity_name := "local_velocity"
@export var position_scale := 10.0
@export var speed_scale := 20.0

var _body: Node


func register_observations(agent:Agent, body:Node) -> void:
	_body = body
	if include_position:
		agent.add_observation(position_name, Callable(self, "_get_position"))
	if include_forward:
		agent.add_observation(forward_name, Callable(self, "_get_forward"))
	if include_local_velocity:
		agent.add_observation(local_velocity_name, Callable(self, "_get_local_velocity"))


func _get_position() -> Vector3:
	if not (_body is Node3D):
		return Vector3.ZERO
	return _body.global_position / maxf(position_scale, 0.001)


func _get_forward() -> Vector3:
	if not (_body is Node3D):
		return Vector3.ZERO
	return _body.global_transform.basis.z.normalized()


func _get_local_velocity() -> Vector3:
	if not (_body is Node3D):
		return Vector3.ZERO

	var velocity := Vector3.ZERO
	if _body is CharacterBody3D:
		velocity = _body.velocity
	elif _body is RigidBody3D:
		velocity = _body.linear_velocity

	var local_velocity: Vector3 = _body.global_transform.basis.inverse() * velocity
	return local_velocity / maxf(speed_scale, 0.001)
