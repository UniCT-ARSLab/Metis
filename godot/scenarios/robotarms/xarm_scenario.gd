extends Node3D

@export var target_spawns: Array[Marker3D] = []
@export var easy_target_count := 4

@onready var controller: ScenarioController = $ScenarioController
# Entrambi i backend implementano lo stesso contratto, ma non ereditano dalla
# stessa classe GDScript.
@onready var arm = $RobotArm
@onready var target: Node3D = $Target
@onready var goal_event = $ScenarioController/ScenarioEventSystem/GoalReached
@onready var collision_event = $ScenarioController/ScenarioEventSystem/Collision

var _training_episode := 0


func _ready() -> void:
	arm.target = target
	arm.target_reached.connect(_on_target_reached)
	arm.obstacle_collision.connect(_on_obstacle_collision)
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))


func _on_episode_reset_started(_seed:int) -> void:
	if target_spawns.is_empty():
		push_error("RobotArmReachingScenario requires at least one TargetSpawn")
		return
	var rng := RandomNumberGenerator.new()
	rng.seed = _seed

	var target_pool_size := target_spawns.size()
	var joint_jitter_degrees := 0.0

	if _training_episode < 300:
		target_pool_size = maxi(1, mini(easy_target_count, target_spawns.size()))
		arm.success_distance = 0.08
	elif _training_episode < 800:
		arm.success_distance = 0.05
	elif _training_episode < 1500:
		joint_jitter_degrees = 2.0
		arm.success_distance = 0.04
	else:
		joint_jitter_degrees = 5.0
		arm.success_distance = 0.03

	var target_index := rng.randi_range(0, maxi(target_pool_size - 1, 0))
	target.global_transform = target_spawns[target_index].global_transform
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
