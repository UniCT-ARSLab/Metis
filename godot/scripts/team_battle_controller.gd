extends Node

@export var team_a_root_path: NodePath
@export var team_b_root_path: NodePath
@export var sensors_path: NodePath
@export var reward_system_path: NodePath
@export var max_steps := 400
@export var shot_range := 8.5
@export var shot_fov_cos := 0.92
@export var shot_damage := 25.0

var team_a_root
var team_b_root
var sensors
var reward_system
var step_count := 0

var spawn_a := [
	Transform3D(Basis.IDENTITY, Vector3(-7, 0.5, -5)),
	Transform3D(Basis.IDENTITY, Vector3(-7, 0.5, 5))
]
var spawn_b := [
	Transform3D(Basis.from_euler(Vector3(0, PI, 0)), Vector3(7, 0.5, -5)),
	Transform3D(Basis.from_euler(Vector3(0, PI, 0)), Vector3(7, 0.5, 5))
]

func _ready() -> void:
	team_a_root = get_node(team_a_root_path)
	team_b_root = get_node(team_b_root_path)
	sensors = get_node(sensors_path)
	reward_system = get_node(reward_system_path)
	

func get_team_units(team_id: int) -> Array:
	var root = team_a_root if team_id == 0 else team_b_root
	var result: Array = []
	for child in root.get_children():
		result.append(child)
	return result

func get_all_units() -> Array:
	var arr: Array = []
	arr.append_array(get_team_units(0))
	arr.append_array(get_team_units(1))
	return arr

func get_living_units(team_id: int) -> Array:
	var result: Array = []
	for unit in get_team_units(team_id):
		if unit.alive:
			result.append(unit)
	return result

func reset_episode(seed: int, observed_teams: Array = [0]) -> Dictionary:
	step_count = 0
	var i := 0
	for unit in get_team_units(0):
		unit.reset_unit(spawn_a[i], 0)
		i += 1
	i = 0
	for unit in get_team_units(1):
		unit.reset_unit(spawn_b[i], 1)
		i += 1
	reward_system.reset_reward()
	return {
		"agents": sensors.get_teams_channels(observed_teams),
		"info": {
			"episode_step": step_count,
			"team_a_alive": get_living_units(0).size(),
			"team_b_alive": get_living_units(1).size(),
			"winner": -1
		}
	}

func configure(config: Dictionary) -> Dictionary:
	if config.has("max_steps"):
		max_steps = int(config.max_steps)
	if config.has("shot_damage"):
		shot_damage = float(config.shot_damage)
	return {
		"ok": true,
		"max_steps": max_steps,
		"shot_damage": shot_damage
	}

func step_episode(action_map: Dictionary, controlled_teams: Array = [0], observed_teams: Array = [0]) -> Dictionary:
	step_count += 1

	for unit in get_team_units(0):
		if unit.alive and _team_is_controlled(0, controlled_teams):
			var act := int(action_map.get(unit.name, 0))
			unit.apply_action(act)
		elif unit.alive:
			unit.apply_action(0)
		else:
			unit.apply_action(0)

	for enemy in get_team_units(1):
		if enemy.alive and _team_is_controlled(1, controlled_teams):
			var act := int(action_map.get(enemy.name, 0))
			enemy.apply_action(act)
		elif enemy.alive:
			enemy.apply_action(_scripted_enemy_action(enemy))
		else:
			enemy.apply_action(0)

	var boundary_hits := {}
	for _i in range(4):
		for unit in get_all_units():
			unit.sim_step(1.0 / 30.0)
			if unit.hit_boundary:
				boundary_hits[unit.name] = int(boundary_hits.get(unit.name, 0)) + 1

	var step_events := _resolve_all_shots()
	step_events["boundary_hits"] = boundary_hits
	var winner := _winner_team_id()
	var truncated := step_count >= max_steps
	var per_agent_rewards = reward_system.compute_per_agent_rewards(step_events, winner)

	var channels: Array = []
	var channels_obs = sensors.get_teams_channels(observed_teams)
	for ch in channels_obs:
		var agent_id = ch.id
		var agent_unit = _find_unit(agent_id)
		channels.append({
			"id": agent_id,
			"obs": ch.obs,
			"reward": float(per_agent_rewards.get(agent_id, 0.0)),
			"done": winner != -1 or truncated or not agent_unit.alive,
			"alive": agent_unit.alive
		})

	return {
		"agents": channels,
		"terminated": winner != -1,
		"truncated": truncated,
		"info": {
			"episode_step": step_count,
			"winner": winner,
			"team_a_alive": get_living_units(0).size(),
			"team_b_alive": get_living_units(1).size(),
			"per_agent_rewards": per_agent_rewards
		}
	}

func _team_is_controlled(team_id: int, controlled_teams: Array) -> bool:
	for item in controlled_teams:
		if int(item) == team_id:
			return true
	return false

func _find_unit(name: String):
	for unit in get_all_units():
		if unit.name == name:
			return unit
	return null

func _winner_team_id() -> int:
	var alive_a := get_living_units(0).size()
	var alive_b := get_living_units(1).size()
	if alive_b <= 0:
		return 0
	if alive_a <= 0:
		return 1
	return -1

func _scripted_enemy_action(unit) -> int:
	if _near_arena_edge(unit):
		return _turn_towards_center_action(unit)

	var targets := get_living_units(0)
	if targets.is_empty():
		return 0
	var target = _nearest_enemy(unit, targets)
	if target == null:
		return 0
	var delta = target.global_transform.origin - unit.global_transform.origin
	var local = unit.global_transform.basis.inverse() * delta
	var angle := atan2(local.x, -local.z)
	var dist = delta.length()

	if abs(angle) > 0.18:
		return 3 if angle > 0.0 else 2
	if dist < shot_range and has_line_of_sight(unit, target) and unit.can_fire():
		return 6
	if _front_blocked(unit):
		return 3 if local.x > 0.0 else 2
	return 7 if unit.can_fire() else 1

func _near_arena_edge(unit) -> bool:
	var margin := 1.0
	var pos = unit.global_transform.origin
	return abs(pos.x) > unit.arena_half_extent - margin or abs(pos.z) > unit.arena_half_extent - margin

func _turn_towards_center_action(unit) -> int:
	var to_center = -unit.global_transform.origin
	var local = unit.global_transform.basis.inverse() * to_center
	var angle := atan2(local.x, -local.z)
	if abs(angle) > 0.25:
		return 3 if angle > 0.0 else 2
	return 1

func _front_blocked(unit) -> bool:
	var space_state = get_viewport().world_3d.direct_space_state
	var from_pos = unit.global_transform.origin + Vector3.UP * 0.6
	var forward = (-unit.global_transform.basis.z).normalized()
	var to_pos = from_pos + forward * 1.8
	var query := PhysicsRayQueryParameters3D.create(from_pos, to_pos)
	query.exclude = [unit]
	var hit := space_state.intersect_ray(query)
	if hit.is_empty():
		return false
	if hit.get("collider") == null:
		return false
	return true

func _nearest_enemy(unit, candidates: Array):
	var best = null
	var best_d := INF
	for other in candidates:
		if not other.alive:
			continue
		var d = unit.global_transform.origin.distance_to(other.global_transform.origin)
		if d < best_d:
			best_d = d
			best = other
	return best

func _resolve_all_shots() -> Dictionary:
	var events := {
		"damage_dealt": {},
		"kills": {}
	}
	_resolve_team_shots(get_team_units(0), get_team_units(1), events)
	_resolve_team_shots(get_team_units(1), get_team_units(0), events)
	return events

func _resolve_team_shots(shooters: Array, targets: Array, events: Dictionary) -> void:
	for shooter in shooters:
		if not shooter.alive:
			continue
		if not shooter.request_fire:
			continue
		if not shooter.can_fire():
			continue
		var target = _best_target_in_front(shooter, targets)
		if target != null:
			shooter.consume_fire()
			var died = target.take_damage(shot_damage)
			if not events.damage_dealt.has(shooter.name):
				events.damage_dealt[shooter.name] = 0.0
			events.damage_dealt[shooter.name] += shot_damage
			if died:
				if not events.kills.has(shooter.name):
					events.kills[shooter.name] = 0
				events.kills[shooter.name] += 1

func _best_target_in_front(shooter, targets: Array):
	var best = null
	var best_score := -INF
	var forward = (-shooter.global_transform.basis.z).normalized()
	for target in targets:
		if not target.alive:
			continue
		var vec = target.global_transform.origin - shooter.global_transform.origin
		var dist = vec.length()
		if dist > shot_range or dist <= 0.001:
			continue
		var dir = vec.normalized()
		var align = forward.dot(dir)
		if align < shot_fov_cos:
			continue
		if not has_line_of_sight(shooter, target):
			continue
		var score = align - dist * 0.01
		if score > best_score:
			best_score = score
			best = target
	return best

func has_line_of_sight(from_unit, to_unit) -> bool:
	if from_unit == null or to_unit == null:
		return false
	var space_state = get_viewport().world_3d.direct_space_state
	var from_pos = from_unit.global_transform.origin + Vector3.UP * 0.6
	var to_pos = to_unit.global_transform.origin + Vector3.UP * 0.6
	var query := PhysicsRayQueryParameters3D.create(from_pos, to_pos)
	query.exclude = [from_unit]
	var hit := space_state.intersect_ray(query)
	if hit.is_empty():
		return true
	return hit.collider == to_unit
