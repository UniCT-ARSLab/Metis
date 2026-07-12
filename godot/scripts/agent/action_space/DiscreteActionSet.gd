extends Node
class_name DiscreteActionSet

@export var action_name := "action"
@export var names: PackedStringArray = []


func get_action_spec() -> Dictionary:
	return {
		"name": action_name,
		"size": names.size(),
		"action_type": "discrete",
		"names": Array(names)
	}
