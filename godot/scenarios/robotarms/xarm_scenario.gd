extends Node3D

@export var target_spawns: Array[Marker3D] = []
@export var target_spawns_root: Node3D

@export_category("Curriculum")
@export var easy_target_count := 4
@export var easy_stage_end_episode := 3000
@export var intermediate_stage_end_episode := 6000
@export var advanced_stage_end_episode := 9000

@onready var controller: ScenarioController = $ScenarioController
# Robot body adapters share the same contract without requiring a common GDScript base.
@onready var arm = $RobotArm
@onready var target: RigidBody3D = $Target
@onready var grasp_point: Marker3D = $Target/GraspPoint
@onready var goal_event = $ScenarioController/ScenarioEventSystem/GoalReached
@onready var collision_event = $ScenarioController/ScenarioEventSystem/Collision
@onready var self_collision_event = $ScenarioController/ScenarioEventSystem/SelfCollision
@onready var grasp_event = $ScenarioController/ScenarioEventSystem/ObjectGrasped
@onready var drop_event = $ScenarioController/ScenarioEventSystem/ObjectDropped

var _training_episode := 0
var _goal_terminal_reason := "target_reached"
var _target_waiting_for_reset_completion := false


func _ready() -> void:
	_goal_terminal_reason = str(goal_event.terminal_reason)
	arm.configure_grasp_target(target, grasp_point)
	arm.target_reached.connect(_on_target_reached)
	arm.obstacle_collision.connect(_on_obstacle_collision)
	arm.self_collision.connect(_on_self_collision)
	arm.object_grasped.connect(_on_object_grasped)
	arm.object_dropped.connect(_on_object_dropped)
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)
	controller.episode_reset_completed.connect(_on_episode_reset_completed)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))
	var continue_after_success := bool(config.get(
		"continue_after_success", controller.continue_after_success))
	if arm.has_method("set_continue_after_success"):
		arm.set_continue_after_success(continue_after_success)
	goal_event.terminal_reason = "" if continue_after_success else _goal_terminal_reason


func _on_episode_reset_started(_seed:int) -> void:
	var spawn_pool := _target_spawn_pool()
	if spawn_pool.is_empty():
		push_error("RobotArmReachingScenario requires at least one TargetSpawn")
		return
	var rng := RandomNumberGenerator.new()
	rng.seed = _seed

	var target_pool_size := spawn_pool.size()
	var joint_jitter_degrees := 0.0

	if _training_episode < easy_stage_end_episode:
		target_pool_size = maxi(1, mini(easy_target_count, spawn_pool.size()))
		arm.grasp_capture_distance = 0.060
		arm.required_lift_height = 0.020
		arm.grasp_hold_physics_frames = 10
		arm.max_grasp_target_speed = 0.25
		arm.max_pregrasp_planar_displacement = 0.060
	elif _training_episode < intermediate_stage_end_episode:
		target_pool_size = maxi(
			1, mini(easy_target_count * 2, spawn_pool.size()))
		arm.grasp_capture_distance = 0.050
		arm.required_lift_height = 0.030
		arm.grasp_hold_physics_frames = 15
		arm.max_grasp_target_speed = 0.20
		arm.max_pregrasp_planar_displacement = 0.055
	elif _training_episode < advanced_stage_end_episode:
		joint_jitter_degrees = 2.0
		arm.grasp_capture_distance = 0.040
		arm.required_lift_height = 0.040
		arm.grasp_hold_physics_frames = 20
		arm.max_grasp_target_speed = 0.18
		arm.max_pregrasp_planar_displacement = 0.045
	else:
		joint_jitter_degrees = 5.0
		arm.grasp_capture_distance = 0.032
		arm.required_lift_height = 0.050
		arm.grasp_hold_physics_frames = 30
		arm.max_grasp_target_speed = 0.15
		arm.max_pregrasp_planar_displacement = 0.040

	var target_index := rng.randi_range(0, maxi(target_pool_size - 1, 0))
	_reset_target_body(spawn_pool[target_index].global_transform)
	arm.set_reset_joint_offsets(_sample_joint_offsets(rng, joint_jitter_degrees))


func _on_episode_reset_completed(_seed:int) -> void:
	if not _target_waiting_for_reset_completion:
		return
	target.linear_velocity = Vector3.ZERO
	target.angular_velocity = Vector3.ZERO
	# Keep the target immovable while only REACHING: a solid kinematic arm would otherwise shove
	# the dynamic glass (and its GraspPoint) out of reach, so the reach target could never be hit.
	# Grasping needs it dynamic (to lift), so only freeze it for reaching.
	target.freeze = arm.task_mode == URDFRobotArmAgentBody.TaskMode.REACHING
	target.sleeping = false
	_target_waiting_for_reset_completion = false


func _target_spawn_pool() -> Array[Marker3D]:
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


func _reset_target_body(spawn_transform: Transform3D) -> void:
	arm.prepare_grasp_target_reset()
	target.freeze_mode = RigidBody3D.FREEZE_MODE_STATIC
	target.freeze = true
	target.linear_velocity = Vector3.ZERO
	target.angular_velocity = Vector3.ZERO
	target.global_transform = spawn_transform
	if target.has_method("reset_physics_interpolation"):
		target.reset_physics_interpolation()
	arm.set_grasp_target_spawn_transform(spawn_transform)
	_target_waiting_for_reset_completion = true


func _sample_joint_offsets(rng:RandomNumberGenerator, max_degrees:float) -> Array[float]:
	var offsets: Array[float] = []
	offsets.resize(arm.get_joint_count())
	var joint_names: PackedStringArray = arm.get_controlled_joint_names()
	for index in range(offsets.size()):
		if index < joint_names.size() and joint_names[index] == arm.gripper_joint_name:
			offsets[index] = 0.0
		else:
			offsets[index] = deg_to_rad(rng.randf_range(-max_degrees, max_degrees))
	return offsets


func _on_target_reached() -> void:
	goal_event.trigger(str(arm.name))


func _on_obstacle_collision() -> void:
	collision_event.trigger(str(arm.name))


func _on_self_collision() -> void:
	self_collision_event.trigger(str(arm.name))


func _on_object_grasped() -> void:
	grasp_event.trigger(str(arm.name))


func _on_object_dropped() -> void:
	drop_event.trigger(str(arm.name))
