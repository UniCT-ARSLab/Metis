extends "res://addons/metis/runtime/agent/reward_components/RewardComponent.gd"
class_name EventReward

@export var event_name := ""
@export var scale := 1.0


func compute_reward(context:Dictionary) -> float:
	if event_name.is_empty():
		return 0.0

	var events: Dictionary = context.get("events", {})
	return float(events.get(event_name, 0.0)) * scale * weight
