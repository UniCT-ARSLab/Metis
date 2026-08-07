extends "res://addons/metis/runtime/agent/progress/ProgressProvider.gd"
class_name MethodProgressProvider

@export var source_path:NodePath
@export var method_name:StringName = &"get_progress"
@export var pass_agent_to_source := true


func measure_progress(agent:Node, _context:Dictionary = {}) -> float:
	var source := agent if source_path.is_empty() else get_node_or_null(source_path)
	if source == null or not source.has_method(method_name):
		return 0.0
	if source == agent or not pass_agent_to_source:
		return float(source.call(method_name))
	return float(source.call(method_name, agent))
