extends RewardComponent
class_name ObservationMarginReward

## Continuous centering cost: zero at the target, -scale at either band edge.
@export var observation_name := ""
@export var target_value := 0.0
@export_range(0.000001, 1000000.0, 0.001, "or_greater") var half_width := 1.0
@export var scale := 1.0
@export_enum("Linear", "Quadratic") var shape := 1
@export var gate_observation := ""
@export var gate_min_value := -1000000.0
@export var gate_max_value := 1000000.0


func compute_reward(context: Dictionary) -> float:
	var observations: Dictionary = context.get("observations", {})
	if observation_name.is_empty() or not observations.has(observation_name):
		return 0.0
	if not gate_observation.is_empty():
		if not observations.has(gate_observation):
			return 0.0
		var gate_value := float(observations[gate_observation])
		if not is_finite(gate_value) or gate_value < gate_min_value or gate_value > gate_max_value:
			return 0.0
	var measured := float(observations[observation_name])
	if not is_finite(measured) or not is_finite(target_value) or not is_finite(half_width) or half_width <= 0.0:
		return 0.0
	var error := absf(measured - target_value) / half_width
	var cost := error if shape == 0 else error * error
	var reward := -scale * cost * weight
	return reward if is_finite(reward) else 0.0
