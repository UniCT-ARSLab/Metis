extends "res://scripts/agent/reward_components/RewardComponent.gd"
class_name ActionSmoothnessPenaltyReward

@export var input_names: PackedStringArray = ["throttle_input", "rotation_input"]
@export var free_delta := 0.08
@export var penalty_scale := -0.04
@export var use_absolute_delta := true
@export var normalize_by_input_count := false

var _previous_values := {}
var _has_previous := false


func reset_reward(context:Dictionary = {}) -> void:
	_previous_values.clear()
	_has_previous = false

	var body = context.get("body")
	if body == null:
		return

	for input_name in input_names:
		_previous_values[str(input_name)] = _read_input_value(body, str(input_name))


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if body == null:
		return 0.0
	if body.has_method("is_crashed") and bool(body.is_crashed()):
		return 0.0

	var penalty := 0.0
	for input_name_variant in input_names:
		var input_name := str(input_name_variant)
		var current_value := _read_input_value(body, input_name)
		var previous_value := float(_previous_values.get(input_name, current_value))
		var delta := current_value - previous_value
		if use_absolute_delta:
			delta = absf(delta)
		else:
			delta = maxf(delta, 0.0)

		if _has_previous:
			penalty += maxf(delta - free_delta, 0.0) * penalty_scale

		_previous_values[input_name] = current_value

	_has_previous = true
	if normalize_by_input_count and not input_names.is_empty():
		penalty /= float(input_names.size())
	return penalty * weight


func _read_input_value(body, input_name:String) -> float:
	if body.has_method("get_control_input"):
		return float(body.get_control_input(input_name))

	var observations: Dictionary = {}
	if body.has_method("get_observations"):
		observations = body.get_observations()

	return float(observations.get(input_name, 0.0))
