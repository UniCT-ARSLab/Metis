extends RewardComponent
class_name ObservationThresholdReward

enum ThresholdMode {
	BELOW,
	ABOVE
}

@export var observation_prefix := ""
@export var value := -0.02
@export var threshold := 0.35
@export var mode: ThresholdMode = ThresholdMode.BELOW
@export var gate_observation := ""
@export var gate_min_value := -1000000.0
@export var gate_max_value := 1000000.0


func compute_reward(context:Dictionary) -> float:
	var observations: Dictionary = context.get("observations", {})

	if not _gate_is_open(observations):
		return 0.0

	var worst_severity := 0.0
	for key in observations.keys():
		var observation_name := str(key)
		if not observation_prefix.is_empty() and not observation_name.begins_with(observation_prefix):
			continue

		var observation_value := float(observations[key])
		var severity := _compute_severity(observation_value)
		worst_severity = maxf(worst_severity, severity)

	return value * worst_severity * weight


func _gate_is_open(observations:Dictionary) -> bool:
	if gate_observation.is_empty():
		return true
	if not observations.has(gate_observation):
		return false

	var gate_value := float(observations[gate_observation])
	return gate_value >= gate_min_value and gate_value <= gate_max_value


func _compute_severity(observation_value:float) -> float:
	if mode == ThresholdMode.BELOW:
		if observation_value >= threshold:
			return 0.0
		return clampf((threshold - observation_value) / maxf(absf(threshold), 0.000001), 0.0, 1.0)

	if observation_value <= threshold:
		return 0.0
	return clampf((observation_value - threshold) / maxf(absf(threshold), 0.000001), 0.0, 1.0)
