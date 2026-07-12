extends "res://scripts/agent/observations/ObservationSource.gd"
class_name MethodObservationSource

@export var observation_name := ""
@export var method_name := ""
@export var bind_string_arg := ""


func register_observations(agent:Agent, body:Node) -> void:
	if observation_name.is_empty() or method_name.is_empty():
		return
	if body == null or not body.has_method(method_name):
		push_warning("Observation method not found: %s" % method_name)
		return

	var callable := Callable(body, method_name)
	if not bind_string_arg.is_empty():
		callable = callable.bind(bind_string_arg)
	agent.add_observation(observation_name, callable)
