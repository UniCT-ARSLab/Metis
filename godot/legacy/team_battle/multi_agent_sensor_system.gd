extends Node3D

@export var controller_path: NodePath
@export var ray_length := 7.0

var controller
var ray_dirs := [
	Vector3(0, 0, -1),
	Vector3(-0.7, 0, -0.7).normalized(),
	Vector3(0.7, 0, -0.7).normalized(),
	Vector3(-1, 0, 0),
	Vector3(1, 0, 0)
]

func _ready() -> void:
	controller = get_node(controller_path)

func get_team_channels(team_id: int) -> Array:
	var team_units = controller.get_team_units(team_id)
	var channels: Array = []
	var idx := 0
	for unit in team_units:
		channels.append({
			"id": unit.name,
			"obs": _agent_observation(unit, team_id, idx)
		})
		idx += 1
	return channels

func get_teams_channels(team_ids: Array) -> Array:
	var channels: Array = []
	for team_id in team_ids:
		channels.append_array(get_team_channels(int(team_id)))
	return channels

func _agent_observation(agent, team_id: int, agent_index: int) -> Array:
	var obs: Array = []
	var allies = controller.get_living_units(team_id)
	var enemies = controller.get_living_units(1 - team_id)

	var nearest_enemy = _nearest_other(agent, enemies)
	var nearest_ally = _nearest_other(agent, allies, true)

	obs.append(float(agent_index))
	obs.append(1.0 if agent.alive else 0.0)
	obs.append(agent.hp_norm())
	obs.append(agent.reload_norm())
	obs.append(clampf(agent.get_forward_speed() / 10.0, -1.0, 1.0))
	obs.append(clampf(agent.global_transform.origin.x / agent.arena_half_extent, -1.0, 1.0))
	obs.append(clampf(agent.global_transform.origin.z / agent.arena_half_extent, -1.0, 1.0))

	if nearest_enemy != null:
		var enemy_vec = nearest_enemy.global_transform.origin - agent.global_transform.origin
		var enemy_local = agent.global_transform.basis.inverse() * enemy_vec
		obs.append(clampf(enemy_local.x / 20.0, -1.0, 1.0))
		obs.append(clampf(enemy_local.z / 20.0, -1.0, 1.0))
		obs.append(clampf(enemy_vec.length() / 20.0, 0.0, 1.0))
		obs.append(1.0 if controller.has_line_of_sight(agent, nearest_enemy) else 0.0)
	else:
		obs.append(0.0)
		obs.append(0.0)
		obs.append(1.0)
		obs.append(0.0)

	if nearest_ally != null:
		var ally_vec = nearest_ally.global_transform.origin - agent.global_transform.origin
		var ally_local = agent.global_transform.basis.inverse() * ally_vec
		obs.append(clampf(ally_local.x / 20.0, -1.0, 1.0))
		obs.append(clampf(ally_local.z / 20.0, -1.0, 1.0))
	else:
		obs.append(0.0)
		obs.append(0.0)

	for d in ray_dirs:
		obs.append(_ray_distance_normalized(agent, d))

	return obs

func _nearest_other(agent, units: Array, exclude_self: bool = false):
	var best = null
	var best_d := INF
	for unit in units:
		if unit == null:
			continue
		if exclude_self and unit == agent:
			continue
		if not unit.alive:
			continue
		var d = agent.global_transform.origin.distance_to(unit.global_transform.origin)
		if d < best_d:
			best_d = d
			best = unit
	return best

func _ray_distance_normalized(agent, local_dir: Vector3) -> float:
	var space_state = get_world_3d().direct_space_state
	var from = agent.global_transform.origin + Vector3.UP * 0.6
	var world_dir = agent.global_transform.basis * local_dir.normalized()
	var to = from + world_dir * ray_length
	var query := PhysicsRayQueryParameters3D.create(from, to)
	query.exclude = [agent]
	var hit := space_state.intersect_ray(query)
	if hit.is_empty():
		return 1.0
	return clamp(from.distance_to(hit.position) / ray_length, 0.0, 1.0)
