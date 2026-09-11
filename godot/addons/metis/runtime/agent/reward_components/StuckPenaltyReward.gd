extends RewardComponent
class_name StuckPenaltyReward

@export var input_observation := "move_input"
@export var penalty := -0.01
@export var input_threshold := 0.2
@export var delta_threshold := 0.001

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

	var observations: Dictionary = context.get("observations", {})
	var input_value := absf(float(observations.get(input_observation, 0.0)))

	var current_position: Vector3 = body.global_position
	if not _has_previous_position:
		_previous_position = current_position
		_has_previous_position = true
		return 0.0

	var movement_delta: float = current_position.distance_to(_previous_position)
	_previous_position = current_position

	if input_value < input_threshold:
		return 0.0
	if movement_delta >= delta_threshold:
		return 0.0

	return penalty * weight
