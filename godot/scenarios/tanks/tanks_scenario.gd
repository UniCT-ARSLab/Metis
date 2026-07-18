extends Node3D

@export var tanks: Array[BattleTank] = []
@export var layout_scenes: Array[PackedScene] = []
@export var layout_container: Node3D

@onready var controller: ScenarioController = $ScenarioController
@onready var enemy_damage: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/EnemyDamage
@onready var enemy_kill: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/EnemyKill
@onready var team_kill_assist: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamKillAssist
@onready var damage_taken: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/DamageTaken
@onready var destroyed_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/Destroyed
@onready var team_won: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamWon
@onready var team_lost: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/TeamLost
@onready var match_draw: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/MatchDraw

var _training_episode := 0
var _match_finished := false
var _current_layout: Node


func _ready() -> void:
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)
	for tank in tanks:
		tank.damage_received.connect(_on_damage_received)
		tank.destroyed.connect(_on_tank_destroyed)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))


func _on_episode_reset_started(episode_seed:int) -> void:
	_match_finished = false
	_clear_projectiles()
	_select_layout(episode_seed)


func _on_damage_received(victim:BattleTank, attacker:BattleTank, amount:float) -> void:
	if _match_finished or attacker == null:
		return
	if attacker.get_team_id() == victim.get_team_id():
		return
	var normalized_damage := amount / maxf(victim.max_health, 0.001)
	enemy_damage.trigger(str(attacker.name), normalized_damage)
	damage_taken.trigger(str(victim.name), normalized_damage)


func _on_tank_destroyed(victim:BattleTank, killer:BattleTank) -> void:
	if _match_finished:
		return
	destroyed_event.trigger(str(victim.name))
	if killer != null and killer.get_team_id() != victim.get_team_id():
		enemy_kill.trigger(str(killer.name))
		for teammate in tanks:
			if teammate.get_team_id() == killer.get_team_id():
				team_kill_assist.trigger(str(teammate.name))
	_check_match_finished()


func _check_match_finished() -> void:
	var red_alive := _alive_count(0)
	var blue_alive := _alive_count(1)
	if red_alive > 0 and blue_alive > 0:
		return

	_match_finished = true
	if red_alive == 0 and blue_alive == 0:
		for tank in tanks:
			match_draw.trigger(str(tank.name))
		return

	var winner_team := 0 if red_alive > 0 else 1
	for tank in tanks:
		if tank.get_team_id() == winner_team:
			team_won.trigger(str(tank.name))
		else:
			team_lost.trigger(str(tank.name))


func _alive_count(team:int) -> int:
	var count := 0
	for tank in tanks:
		if tank.get_team_id() == team and tank.is_alive():
			count += 1
	return count


func _clear_projectiles() -> void:
	for projectile in get_tree().get_nodes_in_group("projectile"):
		var parent := projectile.get_parent()
		if parent != null:
			parent.remove_child(projectile)
		projectile.queue_free()


func _select_layout(episode_seed:int) -> void:
	if layout_scenes.is_empty() or layout_container == null:
		return
	if is_instance_valid(_current_layout):
		layout_container.remove_child(_current_layout)
		_current_layout.queue_free()

	var available := 1
	if _training_episode >= 500:
		available = mini(2, layout_scenes.size())
	if _training_episode >= 1500:
		available = layout_scenes.size()

	var rng := RandomNumberGenerator.new()
	rng.seed = episode_seed
	var selected := rng.randi_range(0, maxi(available - 1, 0))
	_current_layout = layout_scenes[selected].instantiate()
	layout_container.add_child(_current_layout)
