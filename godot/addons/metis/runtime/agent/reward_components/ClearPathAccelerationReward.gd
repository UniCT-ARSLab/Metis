extends RewardComponent
class_name ClearPathAccelerationReward

@export var input_name := "move_input"
@export var clear_threshold := 0.65
@export var blocked_threshold := 0.35
@export var target_input := 0.8
@export var reward_scale := 0.12
@export var below_target_penalty := -0.08
@export var reverse_penalty_scale := -0.25
@export var speed_reference := 18.0
@export var clear_speed_reward_scale := 0.08
@export var min_clear_speed := 0.45
@export var low_speed_clear_penalty := -0.04
@export var blocked_throttle_penalty_scale := -0.12


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if body == null:
		return 0.0
	if body.has_method("is_crashed") and bool(body.is_crashed()):
		return 0.0

	var clearance := _read_forward_clearance(context, body)
	var input_value := _read_input_value(context, body)
	var forward_speed_norm := _read_forward_speed_norm(context, body)

	if input_value < 0.0:
		return absf(input_value) * reverse_penalty_scale * weight

	if clearance <= blocked_threshold:
		if input_value <= target_input:
			return 0.0
		return (input_value - target_input) / maxf(1.0 - target_input, 0.000001) * blocked_throttle_penalty_scale * weight

	if clearance < clear_threshold:
		return 0.0

	var reward_value := input_value * reward_scale
	reward_value += forward_speed_norm * clear_speed_reward_scale

	if input_value < target_input:
		reward_value += below_target_penalty * (target_input - input_value) / maxf(target_input, 0.000001)

	if forward_speed_norm < min_clear_speed:
		reward_value += low_speed_clear_penalty * (min_clear_speed - forward_speed_norm) / maxf(min_clear_speed, 0.000001)

	return reward_value * weight


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
