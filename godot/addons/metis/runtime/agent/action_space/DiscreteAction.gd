extends Metis
class_name DiscreteAction

@export var action_name := ""
@export var target_path: NodePath
@export var method_name: StringName
@export var arguments: Array = []


func get_action_name() -> String:
	return action_name if not action_name.is_empty() else str(name)


func invoke(default_target:Node) -> int:
	var target := default_target
	if not target_path.is_empty():
		target = default_target.get_node_or_null(target_path) if default_target != null else null
	if target == null:
		push_warning("Discrete action '%s' target not found: %s" % [get_action_name(), target_path])
		return ERR_DOES_NOT_EXIST
	if method_name.is_empty() or not target.has_method(method_name):
		push_warning("Discrete action '%s' method not found: %s" % [get_action_name(), method_name])
		return ERR_METHOD_NOT_FOUND

	target.callv(method_name, arguments)
	return OK
