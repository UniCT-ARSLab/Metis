extends Node3D

@export var target_spawns: Array[Marker3D] = []
## Collects Marker3D children automatically when assigned.
@export var target_spawns_root: Node3D
@export var easy_target_count := 4
## Samples the volume bounded by the markers after the initial reach phase.
@export var continuous_target_sampling := true
@export var continuous_target_sampling_start_episode := 800
## Expands sampling from each marker to the full marker volume.
@export var spatial_curriculum_start_episode := 0
@export var spatial_curriculum_full_episode := 0
@export var spatial_curriculum_start_radius := 0.01
## Shrinks the marker-defined volume on each axis.
@export var target_sampling_inset := Vector3.ZERO
@export var late_joint_jitter_degrees := 10.0
@export_category("Continuous Pose Tracking")
@export var continuous_pose_tracking := true
## Random yaw teaches the policy that the target is a pose, not only a point.
@export var target_yaw_randomization_start_episode := 800
@export_range(0.0, 180.0, 1.0) var target_yaw_randomization_degrees := 30.0
## Once static reach-and-hold is established, move the target after a random dwell without resetting the robot.
@export var relocate_target_during_training := true
@export var target_relocation_start_episode := 1500
@export_range(1, 1000, 1) var target_relocation_delay_steps_min := 45
@export_range(1, 1000, 1) var target_relocation_delay_steps_max := 120
@export var minimum_target_relocation_distance := 0.04
## Absolute episode boundaries for the hold curriculum.
@export var hold_curriculum_stage1_until := 1000
@export var hold_curriculum_stage2_until := 2200

@onready var controller: ScenarioController = $ScenarioController
@onready var arm = $RobotArm
@onready var target: Node3D = $Target
@onready var target_pose: Node3D = $Target/GraspPose
@onready var goal_event = $ScenarioController/ScenarioEventSystem/GoalReached
@onready var collision_event = $ScenarioController/ScenarioEventSystem/Collision

var _training_episode := 0
var _training_mode := false
var _goal_terminal_reason := "target_reached"
var _episode_rng := RandomNumberGenerator.new()
var _target_pool_size := 1
var _target_relocation_step := -1
var _last_relocation_check_step := -1


func _ready() -> void:
	var configured_terminal_reason := str(goal_event.terminal_reason)
	if not configured_terminal_reason.is_empty():
		_goal_terminal_reason = configured_terminal_reason
	arm.target = target
	arm.target_pose = target_pose
	arm.target_reached.connect(_on_target_reached)
	arm.target_pose_relocated.connect(_on_target_pose_relocated)
	arm.obstacle_collision.connect(_on_obstacle_collision)
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_on_episode_reset_started)
	_apply_continuous_pose_tracking(continuous_pose_tracking)


func _physics_process(_delta: float) -> void:
	if (
		not _training_mode
		or not relocate_target_during_training
		or _training_episode < target_relocation_start_episode
		or _target_relocation_step < 0
	):
		return
	var current_step := controller.step_count
	if current_step == _last_relocation_check_step:
		return
	_last_relocation_check_step = current_step
	if current_step >= _target_relocation_step:
		_relocate_target()


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))
	_training_mode = bool(config.get("training_mode", controller.training_mode))
	var continue_after_success := bool(config.get(
		"continue_after_success", controller.continue_after_success))
	_apply_continuous_pose_tracking(
		continue_after_success or continuous_pose_tracking)


func _apply_continuous_pose_tracking(enabled: bool) -> void:
	controller.continue_after_success = enabled
	if arm.has_method("set_continue_after_success"):
		arm.set_continue_after_success(enabled)
	goal_event.terminal_reason = "" if enabled else _goal_terminal_reason


func _target_spawn_pool() -> Array[Marker3D]:
	# Explicit markers remain the fallback for older scenes.
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
	_episode_rng.seed = _seed
	_target_relocation_step = -1
	_last_relocation_check_step = -1

	_target_pool_size = spawn_pool.size()
	var joint_jitter_degrees := 0.0

	# Tighten the pose gate only after the arm can reach reliably.
	if _training_episode < 300:
		_target_pool_size = maxi(1, mini(easy_target_count, spawn_pool.size()))
		arm.success_distance = 0.08
		arm.success_angle_degrees = 35.0
	elif _training_episode < 800:
		arm.success_distance = 0.06
		arm.success_angle_degrees = 25.0
	elif _training_episode < 1500:
		joint_jitter_degrees = 2.0
		arm.success_distance = 0.05
		arm.success_angle_degrees = 18.0
	else:
		joint_jitter_degrees = late_joint_jitter_degrees
		arm.success_distance = 0.02
		# With the fixed wrist roll, 20 degrees is reachable across the workspace.
		arm.success_angle_degrees = 20.0

	# Later stages require a longer, steadier hold. At 60 Hz, 120 frames are two seconds.
	if _training_episode < hold_curriculum_stage1_until:
		arm.success_hold_physics_frames = 20
		arm.success_max_joint_speed = 0.30
	elif _training_episode < hold_curriculum_stage2_until:
		arm.success_hold_physics_frames = 60
		arm.success_max_joint_speed = 0.20
	else:
		arm.success_hold_physics_frames = 120
		arm.success_max_joint_speed = 0.12

	target.global_transform = _sample_target_transform(_episode_rng, spawn_pool)
	arm.set_reset_joint_offsets(
		_sample_joint_offsets(_episode_rng, joint_jitter_degrees))


func _sample_target_transform(
		rng: RandomNumberGenerator,
		spawn_pool: Array[Marker3D]) -> Transform3D:
	var target_index := rng.randi_range(
		0,
		maxi(mini(_target_pool_size, spawn_pool.size()) - 1, 0))
	var result := spawn_pool[target_index].global_transform
	if (
		continuous_target_sampling
		and _training_episode >= continuous_target_sampling_start_episode
	):
		result.origin = _sample_target_position(rng, spawn_pool, result.origin)
	if (
		target_yaw_randomization_degrees > 0.0
		and _training_episode >= target_yaw_randomization_start_episode
	):
		var yaw := deg_to_rad(rng.randf_range(
			-target_yaw_randomization_degrees,
			target_yaw_randomization_degrees))
		result.basis = Basis(Vector3.UP, yaw) * result.basis
	return result


func _relocate_target() -> void:
	var spawn_pool := _target_spawn_pool()
	if spawn_pool.is_empty():
		_target_relocation_step = -1
		return
	var candidate := target.global_transform
	for _attempt in range(8):
		candidate = _sample_target_transform(_episode_rng, spawn_pool)
		if (
			candidate.origin.distance_to(target.global_position)
			>= maxf(minimum_target_relocation_distance, 0.0)
		):
			break
	target.global_transform = candidate
	_target_relocation_step = -1
	arm.notify_target_pose_relocated()


func _sample_target_position(
		rng: RandomNumberGenerator,
		spawn_pool: Array[Marker3D],
		anchor: Vector3) -> Vector3:
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

	# Grow each axis around the anchor, while staying inside the marker bounds.
	var frac := 1.0
	if spatial_curriculum_full_episode > spatial_curriculum_start_episode:
		frac = clampf(
			float(_training_episode - spatial_curriculum_start_episode)
			/ float(spatial_curriculum_full_episode - spatial_curriculum_start_episode),
			0.0, 1.0)
	var out_position := Vector3()
	for axis in range(3):
		var half_extent := (upper[axis] - lower[axis]) * 0.5
		var radius := lerpf(spatial_curriculum_start_radius, half_extent, frac)
		var sampled := anchor[axis] + rng.randf_range(-radius, radius)
		out_position[axis] = clampf(sampled, lower[axis], upper[axis])
	return out_position


func _sample_joint_offsets(rng:RandomNumberGenerator, max_degrees:float) -> Array[float]:
	var offsets: Array[float] = []
	offsets.resize(arm.get_joint_count())
	for index in range(offsets.size()):
		offsets[index] = deg_to_rad(rng.randf_range(-max_degrees, max_degrees))
	return offsets


func _on_target_reached() -> void:
	goal_event.trigger(str(arm.name))
	if (
		_training_mode
		and relocate_target_during_training
		and _training_episode >= target_relocation_start_episode
	):
		var minimum_delay := mini(
			target_relocation_delay_steps_min,
			target_relocation_delay_steps_max)
		var maximum_delay := maxi(
			target_relocation_delay_steps_min,
			target_relocation_delay_steps_max)
		_target_relocation_step = (
			controller.step_count
			+ _episode_rng.randi_range(minimum_delay, maximum_delay))


func _on_target_pose_relocated() -> void:
	controller.rebase_agent_tracking(arm)


func _on_obstacle_collision() -> void:
	collision_event.trigger(str(arm.name))
