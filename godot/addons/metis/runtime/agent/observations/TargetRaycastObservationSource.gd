extends "res://addons/metis/runtime/agent/observations/ObservationSource.gd"
class_name TargetRaycastObservationSource

@export var target_group := "target"
@export var visible_observation_name := "target_visible"
@export var signal_observations: Dictionary = {}

var _raycasts := {}


func register_observations(agent:Agent, body:Node) -> void:
	_raycasts.clear()
	if not visible_observation_name.is_empty():
		agent.add_observation(visible_observation_name, Callable(self, "_get_target_visible"))

	for observation_name in signal_observations.keys():
		var path := NodePath(str(signal_observations[observation_name]))
		var raycast := get_node_or_null(path)
		if raycast == null and body != null:
			raycast = body.get_node_or_null(path)
		if raycast is RayCast3D:
			_raycasts[str(observation_name)] = raycast
			agent.add_observation(str(observation_name), Callable(self, "_get_target_signal").bind(raycast))


func refresh_source() -> void:
	for raycast in _raycasts.values():
		if is_instance_valid(raycast):
			raycast.force_raycast_update()


func _get_target_visible() -> float:
	for raycast in _raycasts.values():
		if _raycast_hits_target(raycast):
			return 1.0
	return 0.0


func _get_target_signal(raycast:RayCast3D) -> float:
	if not _raycast_hits_target(raycast):
		return 0.0

	var max_distance := raycast.target_position.length()
	if max_distance <= 0.001:
		return 1.0
	var hit_distance := raycast.global_position.distance_to(raycast.get_collision_point())
	var normalized_distance := clampf(hit_distance / max_distance, 0.0, 1.0)
	return 1.0 - normalized_distance


func _raycast_hits_target(raycast:RayCast3D) -> bool:
	if not is_instance_valid(raycast):
		return false
	raycast.force_raycast_update()
	if not raycast.is_colliding():
		return false
	var collider := raycast.get_collider()
	if collider == null:
		return false
	return _node_or_parent_is_in_group(collider, target_group)


func _node_or_parent_is_in_group(node:Node, group_name:String) -> bool:
	var current := node
	while current != null:
		if current.is_in_group(group_name):
			return true
		current = current.get_parent()
	return false
