extends "res://addons/metis/runtime/agent/reward_components/RewardComponent.gd"
class_name ClearPathSteeringPenaltyReward

@export var input_name := "rotation_input"
@export var clear_threshold := 0.72
@export var steering_deadzone := 0.18
@export var penalty_scale := -0.12
@export var speed_reference := 16.0
@export_range(0.0, 1.0, 0.01) var min_speed_factor := 0.15


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if body == null:
		return 0.0
	if body.has_method("is_crashed") and bool(body.is_crashed()):
		return 0.0

	var clearance := _read_forward_clearance(context, body)
	if clearance < clear_threshold:
		return 0.0

	var steering := absf(_read_input_value(context, body))
	if steering <= steering_deadzone:
		return 0.0

	var speed_factor := maxf(_read_forward_speed_norm(context, body), min_speed_factor)
	var excess := (steering - steering_deadzone) / maxf(1.0 - steering_deadzone, 0.000001)
	var clear_factor := (clearance - clear_threshold) / maxf(1.0 - clear_threshold, 0.000001)
	return excess * clear_factor * speed_factor * penalty_scale * weight


func _read_forward_clearance(context:Dictionary, body) -> float:
	var observations: Dictionary = context.get("observations", {})
	if observations.has("forward_clearance"):
		return clampf(float(observations["forward_clearance"]), 0.0, 1.0)
	if body.has_method("get_forward_clearance"):
		return clampf(float(body.get_forward_clearance()), 0.0, 1.0)
	return 1.0


func _read_input_value(context:Dictionary, body) -> float:
	var observations: Dictionary = context.get("observations", {})
	if observations.has(input_name):
		return clampf(float(observations[input_name]), -1.0, 1.0)
	if body.has_method("get_control_input"):
		return clampf(float(body.get_control_input(input_name)), -1.0, 1.0)
	if body.has_method("get_observations"):
		observations = body.get_observations()

	return clampf(float(observations.get(input_name, 0.0)), -1.0, 1.0)


func _read_forward_speed_norm(context:Dictionary, body) -> float:
	if body.has_method("get_signed_forward_speed"):
		return clampf(maxf(float(body.get_signed_forward_speed()), 0.0) / maxf(speed_reference, 0.000001), 0.0, 1.0)
	var observations: Dictionary = context.get("observations", {})
	if observations.has("forward_speed"):
		return clampf(maxf(float(observations["forward_speed"]), 0.0), 0.0, 1.0)
	if body.has_method("get_normalized_forward_speed"):
		return clampf(maxf(float(body.get_normalized_forward_speed()), 0.0), 0.0, 1.0)
	return 0.0
