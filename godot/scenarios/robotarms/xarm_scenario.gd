extends Node3D

@export var target_spawns: Array[Marker3D] = []
## Optional: assign a node whose Marker3D children are auto-collected as spawn points, so
## adding a marker in the editor is enough (no need to also wire it into target_spawns).
@export var target_spawns_root: Node3D
@export var easy_target_count := 4
## Once the basic reach is learned, sample the complete axis-aligned volume delimited by the
## Marker3D nodes instead of memorizing a finite set of target coordinates.
@export var continuous_target_sampling := true
@export var continuous_target_sampling_start_episode := 800
## Shrinks the marker-defined volume on each axis. Useful when the outer markers sit too close to
## a wall or to the physical edge of the robot workspace.
@export var target_sampling_inset := Vector3.ZERO
@export var late_joint_jitter_degrees := 10.0
## Reach-and-HOLD curriculum breakpoints (absolute training episode). The hold requirement tightens
## in stages: a longer hold at a lower stillness threshold. Defaults assume resuming a reach policy
## around episode 5400. Stages: <stage1 -> 10 frames / 0.30, <stage2 -> 20 / 0.20, else 30 / 0.15.
@export var hold_curriculum_stage1_until := 5900
@export var hold_curriculum_stage2_until := 6600

@onready var controller: ScenarioController = $ScenarioController
# Both robot backends implement the same contract without sharing a GDScript base class.
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

	# Curriculum on the success radius. Coarse first (easy to hit + earn the +30 goal), then
	# progressively tighter so the arm learns to stop ON the target, not just within 5cm. With
	# terminate_on_success the radius is where the arm stops, so tightening it moves the stop point
	# onto the target; the dense ProximityScenarioReward pulls it through the final approach.
	# workspace_scale=0.5.
	if _training_episode < 300:
		target_pool_size = maxi(1, mini(easy_target_count, spawn_pool.size()))
		arm.success_distance = 0.08
	elif _training_episode < 800:
		arm.success_distance = 0.06
	elif _training_episode < 1500:
		joint_jitter_degrees = 2.0
		arm.success_distance = 0.05
	else:
		joint_jitter_degrees = late_joint_jitter_degrees
		arm.success_distance = 0.04

	# Reach-and-HOLD curriculum: require a progressively LONGER hold at a TIGHTER stillness threshold.
	# The reach (success_distance) is already at its tightest by this episode range; this teaches the
	# arm to STOP and stay, not just touch. 10 frames / 0.30 rad/s -> 20 / 0.20 -> 30 / 0.15.
	if _training_episode < hold_curriculum_stage1_until:
		arm.success_hold_physics_frames = 10
		arm.success_max_joint_speed = 0.30
	elif _training_episode < hold_curriculum_stage2_until:
		arm.success_hold_physics_frames = 20
		arm.success_max_joint_speed = 0.20
	else:
		arm.success_hold_physics_frames = 30
		arm.success_max_joint_speed = 0.15

	var target_index := rng.randi_range(0, maxi(target_pool_size - 1, 0))
	target.global_transform = spawn_pool[target_index].global_transform
	if (
		continuous_target_sampling
		and _training_episode >= continuous_target_sampling_start_episode
	):
		target.global_position = _sample_target_position(rng, spawn_pool)
	arm.set_reset_joint_offsets(_sample_joint_offsets(rng, joint_jitter_degrees))


func _sample_target_position(
		rng: RandomNumberGenerator,
		spawn_pool: Array[Marker3D]) -> Vector3:
	var lower := spawn_pool[0].global_position
	var upper := lower
	for marker in spawn_pool:
		lower = lower.min(marker.global_position)
		upper = upper.max(marker.global_position)

	var inset := Vector3(
		maxf(target_sampling_inset.x, 0.0),
		maxf(target_sampling_inset.y, 0.0),
		maxf(target_sampling_inset.z, 0.0))
	for axis in range(3):
		var half_extent := (upper[axis] - lower[axis]) * 0.5
		var axis_inset := minf(inset[axis], half_extent)
		lower[axis] += axis_inset
		upper[axis] -= axis_inset

	return Vector3(
		rng.randf_range(lower.x, upper.x),
		rng.randf_range(lower.y, upper.y),
		rng.randf_range(lower.z, upper.z))


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
