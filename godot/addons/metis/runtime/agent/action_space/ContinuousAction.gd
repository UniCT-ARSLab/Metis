extends Metis
class_name ContinuousAction

@export var action_name := ""
@export var size := 1
@export var low := -1.0
@export var high := 1.0
@export var exploration_low := -1.0
@export var exploration_high := 1.0
@export var use_custom_exploration_bounds := false


func get_action_spec() -> Dictionary:
	var spec := {
		"name": action_name if not action_name.is_empty() else str(name),
		"size": max(1, size),
		"action_type": "continuous",
		"low": low,
		"high": high
	}
	if use_custom_exploration_bounds:
		spec["exploration_low"] = exploration_low
		spec["exploration_high"] = exploration_high
	return spec
