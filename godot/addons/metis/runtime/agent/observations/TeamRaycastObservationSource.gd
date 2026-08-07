extends "res://addons/metis/runtime/agent/observations/ObservationSource.gd"
class_name TeamRaycastObservationSource

@export var team_method_name: StringName = &"get_team_id"
@export var enemy_visible_observation_name := "enemy_visible"
@export var enemy_signal_observations: Dictionary = {}
@export var ally_signal_observations: Dictionary = {}

var _body: Node
var _enemy_raycasts := {}
var _ally_raycasts := {}


func register_observations(agent:Agent, body:Node) -> void:
	_body = body
	_enemy_raycasts = _register_signal_observations(agent, body, enemy_signal_observations, true)
	_ally_raycasts = _register_signal_observations(agent, body, ally_signal_observations, false)
	if not enemy_visible_observation_name.is_empty():
		agent.add_observation(enemy_visible_observation_name, Callable(self, "_get_enemy_visible"))


func refresh_source() -> void:
	for raycast in _all_raycasts():
		if is_instance_valid(raycast):
			raycast.force_raycast_update()


func _register_signal_observations(agent:Agent, body:Node, definitions:Dictionary, wants_enemy:bool) -> Dictionary:
	var result := {}
	for observation_name in definitions.keys():
		var path := NodePath(str(definitions[observation_name]))
		var raycast := get_node_or_null(path)
		if raycast == null and body != null:
			raycast = body.get_node_or_null(path)
		if raycast is RayCast3D:
			result[str(observation_name)] = raycast
			agent.add_observation(
				str(observation_name),
				Callable(self, "_get_team_signal").bind(raycast, wants_enemy)
			)
	return result


func _get_enemy_visible() -> float:
	for raycast in _enemy_raycasts.values():
		if _raycast_matches_team(raycast, true):
			return 1.0
	return 0.0


func _get_team_signal(raycast:RayCast3D, wants_enemy:bool) -> float:
	if not _raycast_matches_team(raycast, wants_enemy):
		return 0.0
	var max_distance := raycast.target_position.length()
	if max_distance <= 0.001:
		return 1.0
	var hit_distance := raycast.global_position.distance_to(raycast.get_collision_point())
	return 1.0 - clampf(hit_distance / max_distance, 0.0, 1.0)


func _raycast_matches_team(raycast:RayCast3D, wants_enemy:bool) -> bool:
	if not is_instance_valid(raycast):
		return false
	raycast.force_raycast_update()
	if not raycast.is_colliding():
		return false
	var target := _find_team_owner(raycast.get_collider())
	if target == null or _body == null or not _body.has_method(team_method_name):
		return false
	var same_team := int(target.call(team_method_name)) == int(_body.call(team_method_name))
	return not same_team if wants_enemy else same_team


func _find_team_owner(node:Node) -> Node:
	var current := node
	while current != null:
		if current.has_method(team_method_name):
			return current
		current = current.get_parent()
	return null


func _all_raycasts() -> Array:
	var result := []
	for raycast in _enemy_raycasts.values():
		if not result.has(raycast):
			result.append(raycast)
	for raycast in _ally_raycasts.values():
		if not result.has(raycast):
			result.append(raycast)
	return result
