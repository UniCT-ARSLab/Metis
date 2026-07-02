extends "res://scripts/agent/reward_components/RewardComponent.gd"
class_name MovementReward

@export var reward := 0.002
@export var reference_delta := 0.01

var _previous_position := Vector3.ZERO
var _has_previous_position := false


func reset_reward(context:Dictionary = {}) -> void:
	var body = context.get("body")
	if body is Node3D:
		_previous_position = body.global_position
		_has_previous_position = true
	else:
		_previous_position = Vector3.ZERO
		_has_previous_position = false


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if not (body is Node3D):
		return 0.0

	var current_position: Vector3 = body.global_position
	if not _has_previous_position:
		_previous_position = current_position
		_has_previous_position = true
		return 0.0

	var movement_delta: float = current_position.distance_to(_previous_position)
	_previous_position = current_position

	if movement_delta <= 0.0:
		return 0.0

	var normalized_delta: float = clampf(movement_delta / maxf(reference_delta, 0.000001), 0.0, 1.0)
	return reward * normalized_delta * weight
