extends Node3D

@export var target_spawns: Array[Marker3D] = []
## Optional: assign a node whose Marker3D children are auto-collected as spawn points, so
## adding a marker in the editor is enough (no need to also wire it into target_spawns).
@export var target_spawns_root: Node3D
@export var easy_target_count := 4

@onready var controller: ScenarioController = $ScenarioController
# Entrambi i backend implementano lo stesso contratto, ma non ereditano dalla
# stessa classe GDScript.
@onready var arm = $RobotArm
@onready var target: Node3D = $Target
@onready var goal_event = $ScenarioController/ScenarioEventSystem/GoalReached
@onready var collision_event = $ScenarioController/ScenarioEventSystem/Collision

var _training_episode := 0
var _goal_terminal_reason := "target_reached"


func _ready() -> void:
	_goal_terminal_reason = str(goal_event.terminal_reason)
	arm.target = target
	arm.target_reached.connect(_on_target_reached)
	arm.obstacle_collision.connect(_on_obstacle_collision)
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))
	var continue_after_success := bool(config.get(
		"continue_after_success", controller.continue_after_success))
	if arm.has_method("set_continue_after_success"):
		arm.set_continue_after_success(continue_after_success)
	goal_event.terminal_reason = "" if continue_after_success else _goal_terminal_reason


func _target_spawn_pool() -> Array[Marker3D]:
	# Prefer auto-collecting every Marker3D under target_spawns_root; fall back to the
	# manually-wired target_spawns array so existing scenes keep working.
	var result: Array[Marker3D] = []
	if target_spawns_root:
		for child in target_spawns_root.get_children():
			if child is Marker3D:
				result.append(child)
	if result.is_empty():
		for marker in target_spawns:
			if marker:
				result.append(marker)
	return result


func _on_episode_reset_started(_seed:int) -> void:
	var spawn_pool := _target_spawn_pool()
	if spawn_pool.is_empty():
		push_error("RobotArmReachingScenario requires at least one TargetSpawn")
		return
	var rng := RandomNumberGenerator.new()
	rng.seed = _seed

	var target_pool_size := spawn_pool.size()
	var joint_jitter_degrees := 0.0

	# Curriculum on the success radius. workspace_scale=0.5, so the progress reward saturates
	# ~7cm from target; making the final radius 3cm meant the +30 goal almost never fired and
	# the arm never learned to close the last stretch (~4% success). End at 5cm so the goal is
	# reachable and teaches closing in; tighten later once success is high.
	if _training_episode < 300:
		target_pool_size = maxi(1, mini(easy_target_count, spawn_pool.size()))
		arm.success_distance = 0.08
	elif _training_episode < 800:
		arm.success_distance = 0.065
	elif _training_episode < 1500:
		joint_jitter_degrees = 2.0
		arm.success_distance = 0.055
	else:
		joint_jitter_degrees = 5.0
		arm.success_distance = 0.05

	var target_index := rng.randi_range(0, maxi(target_pool_size - 1, 0))
	target.global_transform = spawn_pool[target_index].global_transform
	arm.set_reset_joint_offsets(_sample_joint_offsets(rng, joint_jitter_degrees))


func _sample_joint_offsets(rng:RandomNumberGenerator, max_degrees:float) -> Array[float]:
	var offsets: Array[float] = []
	offsets.resize(arm.get_joint_count())
	for index in range(offsets.size()):
		offsets[index] = deg_to_rad(rng.randf_range(-max_degrees, max_degrees))
	return offsets


func _on_target_reached() -> void:
	goal_event.trigger(str(arm.name))


func _on_obstacle_collision() -> void:
	collision_event.trigger(str(arm.name))
