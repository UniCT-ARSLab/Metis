extends "res://addons/metis/runtime/agent/observations/ObservationSource.gd"
class_name RaycastClearanceObservationSource

@export var observation_name := "forward_clearance"
@export var raycast_source_path: NodePath = NodePath("../Raycasts")
@export_range(0.0, 1.0, 0.01) var forward_dot_threshold := 0.86

var _body: Node
var _raycast_source: Node


func register_observations(agent:Agent, body:Node) -> void:
	_body = body
	_raycast_source = get_node_or_null(raycast_source_path)
	if _raycast_source == null or not _raycast_source.has_method("get_forward_clearance"):
		push_warning("Raycast clearance source not found at %s" % raycast_source_path)
		return
	agent.add_observation(observation_name, Callable(self, "_get_clearance"))


func _get_clearance() -> float:
	if _raycast_source == null:
		return 1.0
	return float(_raycast_source.get_forward_clearance(_body, forward_dot_threshold))
