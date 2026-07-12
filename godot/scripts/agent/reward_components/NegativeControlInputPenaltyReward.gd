extends "res://scripts/agent/reward_components/RewardComponent.gd"
class_name NegativeControlInputPenaltyReward

@export var input_name := "move_input"
@export var deadzone := 0.05
@export var penalty_scale := -0.08


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if body == null:
		return 0.0
	if body.has_method("is_crashed") and bool(body.is_crashed()):
		return 0.0

	var value := _read_input_value(body)
	if value >= -deadzone:
		return 0.0

	return absf(value) * penalty_scale * weight


func _read_input_value(body) -> float:
	if body.has_method("get_control_input"):
		return float(body.get_control_input(input_name))

	var observations: Dictionary = {}
	if body.has_method("get_observations"):
		observations = body.get_observations()

	return float(observations.get(input_name, 0.0))
