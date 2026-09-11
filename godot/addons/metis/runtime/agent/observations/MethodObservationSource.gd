@tool
extends ObservationSource
class_name MethodObservationSource

@export var observation_name := ""
@export var source_path: NodePath
@export var method_name: StringName
@export var bind_string_arg := ""


func register_observations(agent:Agent, body:Node) -> void:
	if observation_name.is_empty() or method_name.is_empty():
		return
	var source := _resolve_source(body)
	if source == null or not source.has_method(method_name):
		push_warning("Observation method not found: %s" % method_name)
		return

	var callable := Callable(source, method_name)
	if not bind_string_arg.is_empty():
		callable = callable.bind(bind_string_arg)
	agent.add_observation(observation_name, callable)


func get_editor_source() -> Node:
	return _resolve_source(_editor_body())


func _resolve_source(body:Node) -> Node:
	if body == null or source_path.is_empty():
		return body
	return body.get_node_or_null(source_path)


func _editor_body() -> Node:
	var observation_system := get_parent()
	var agent := observation_system.get_parent() if observation_system != null else null
	return agent.get_parent() if agent != null else null
