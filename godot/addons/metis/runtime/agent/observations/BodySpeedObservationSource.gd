extends "res://addons/metis/runtime/agent/observations/ObservationSource.gd"
class_name BodySpeedObservationSource

@export var include_forward_speed := true
@export var include_absolute_speed := true
@export var forward_speed_name := "forward_speed"
@export var absolute_speed_name := "speed"
@export var speed_scale := 20.0

var _body: Node


func register_observations(agent:Agent, body:Node) -> void:
	_body = body
	if include_forward_speed:
		agent.add_observation(forward_speed_name, Callable(self, "_normalized_forward_speed"))
	if include_absolute_speed:
		agent.add_observation(absolute_speed_name, Callable(self, "_normalized_absolute_speed"))


func _normalized_forward_speed() -> float:
	return clampf(_signed_forward_speed() / maxf(speed_scale, 0.000001), -1.0, 1.0)


func _normalized_absolute_speed() -> float:
	return clampf(_planar_velocity().length() / maxf(speed_scale, 0.000001), 0.0, 1.0)


func _signed_forward_speed() -> float:
	var velocity := _planar_velocity()
	if _body is Node3D:
		var forward: Vector3 = _body.global_transform.basis.z
		forward.y = 0.0
		return velocity.dot(forward.normalized()) if forward.length() > 0.000001 else 0.0
	if _body is Node2D:
		var forward_2d: Vector2 = _body.global_transform.x.normalized()
		return Vector2(velocity.x, velocity.y).dot(forward_2d)
	return 0.0


func _planar_velocity() -> Vector3:
	if _body is CharacterBody3D:
		return Vector3(_body.velocity.x, 0.0, _body.velocity.z)
	if _body is RigidBody3D:
		return Vector3(_body.linear_velocity.x, 0.0, _body.linear_velocity.z)
	if _body is CharacterBody2D:
		return Vector3(_body.velocity.x, _body.velocity.y, 0.0)
	if _body is RigidBody2D:
		return Vector3(_body.linear_velocity.x, _body.linear_velocity.y, 0.0)
	return Vector3.ZERO
