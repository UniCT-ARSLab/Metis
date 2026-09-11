extends Metis
class_name DiscreteActionSet

@export var action_name := "action"
@export_group("Legacy")
@export var names: PackedStringArray = []


func get_action_spec() -> Dictionary:
	var action_names := get_action_names()
	return {
		"name": action_name,
		"size": action_names.size(),
		"action_type": "discrete",
		"names": action_names
	}


func get_action_names() -> Array:
	var bindings := _action_bindings()
	if bindings.is_empty():
		return Array(names)

	var result := []
	for binding in bindings:
		result.append(str(binding.get_action_name()))
	return result


func execute_action(action:Variant, default_target:Node) -> int:
	var bindings := _action_bindings()
	if bindings.is_empty():
		return ERR_UNAVAILABLE

	var action_names := get_action_names()
	var action_id := -1
	if typeof(action) == TYPE_INT or typeof(action) == TYPE_FLOAT:
		action_id = int(action)
	elif typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME:
		action_id = action_names.find(str(action))

	if action_id < 0 or action_id >= bindings.size():
		return ERR_INVALID_PARAMETER
	return int(bindings[action_id].invoke(default_target))


func _action_bindings() -> Array:
	var result := []
	for child in get_children():
		if child.has_method("get_action_name") and child.has_method("invoke"):
			result.append(child)
	return result
