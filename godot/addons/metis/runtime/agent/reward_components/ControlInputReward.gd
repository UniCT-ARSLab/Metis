extends RewardComponent
class_name ControlInputReward

@export var input_name := "throttle_input"
@export var target_min_value := 0.35
@export var below_target_penalty := -0.01
@export var value_reward_scale := 0.002
@export var grace_steps := 5

var _below_target_steps := 0


func reset_reward(_context:Dictionary = {}) -> void:
	_below_target_steps = 0


func compute_reward(context:Dictionary) -> float:
	var body = context.get("body")
	if body == null:
		return 0.0
	if body.has_method("is_crashed") and bool(body.is_crashed()):
		return 0.0

	var value := _read_input_value(context, body)
	var reward_value := value * value_reward_scale

	if value >= target_min_value:
		_below_target_steps = 0
		return reward_value * weight

	_below_target_steps += 1
	if _below_target_steps <= grace_steps:
		return reward_value * weight

	return (reward_value + below_target_penalty) * weight


func _read_input_value(context:Dictionary, body) -> float:
	var observations: Dictionary = context.get("observations", {})
	if observations.has(input_name):
		return clampf(float(observations[input_name]), 0.0, 1.0)
	if body.has_method("get_control_input"):
		return clampf(float(body.get_control_input(input_name)), 0.0, 1.0)
	if body.has_method("get_observations"):
		observations = body.get_observations()

	return clampf(float(observations.get(input_name, 0.0)), 0.0, 1.0)
