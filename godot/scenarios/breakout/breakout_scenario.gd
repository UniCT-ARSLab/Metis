extends Node2D

@export var ball_speed := 420.0
@export var paddle_half_width := 64.0
@export_range(0.05, 0.50) var min_ball_vertical_ratio := 0.20
@export_category("Reset Randomization")
@export_range(0.0, 600.0, 1.0) var ball_position_jitter_x := 120.0
@export_range(0.0, 1.0, 0.01) var launch_horizontal_range := 0.35
@export_range(0.0, 0.25, 0.01) var ball_speed_jitter_ratio := 0.05
@export_category("Reset Curriculum")
@export var reset_curriculum_enabled := true
@export_range(0.0, 600.0, 1.0) var curriculum_final_ball_jitter_x := 420.0
@export_range(1, 10000, 1) var curriculum_ramp_episodes := 800
@export_category("Manual Test")
@export var auto_start_manual := true
@export var manual_seed := 1

@onready var controller := $ScenarioController
@onready var paddle: BreakoutPaddle = $BreakoutPaddle
@onready var paddle_agent: Agent = $BreakoutPaddle/Agent
@onready var ball: BreakoutBall = $Ball
@onready var loss_zone: Area2D = $LossZone
@onready var brick_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/BrickDestroyed
@onready var lost_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/LifeLost
@onready var cleared_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/LevelCleared
@onready var ball_hit_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/BallHitEvent


var _ball_start_transform: Transform2D
var _bricks: Array[Node2D] = []
var _remaining_bricks := 0
var _episode_ball_speed := 0.0
var _training_episode := 0


func _ready() -> void:
	_ball_start_transform = ball.global_transform
	_episode_ball_speed = ball_speed
	for child in $Bricks.get_children():
		if child is Node2D:
			_bricks.append(child)
	controller.episode_reset_started.connect(_reset_game)
	controller.scenario_configured.connect(_on_scenario_configured)
	ball.body_entered.connect(_on_ball_body_entered)
	loss_zone.body_entered.connect(_on_loss_zone_body_entered)
	if auto_start_manual and paddle.manual_control:
		call_deferred("_reset_game", manual_seed)

func _reset_game(seed:int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	paddle.set_episode_terminal(false)
	_remaining_bricks = _bricks.size()

	for brick in _bricks:
		brick.visible = true
		brick.process_mode = Node.PROCESS_MODE_INHERIT
		if brick is CollisionObject2D:
			brick.collision_layer = 1

	var reset_transform := _ball_start_transform
	var position_jitter := _current_ball_position_jitter_x()
	reset_transform.origin.x += rng.randf_range(-position_jitter, position_jitter)
	var launch_direction := Vector2(
		rng.randf_range(-launch_horizontal_range, launch_horizontal_range),
		1.0
	).normalized()
	var speed_scale := rng.randf_range(
		1.0 - ball_speed_jitter_ratio,
		1.0 + ball_speed_jitter_ratio
	)
	_episode_ball_speed = ball_speed * speed_scale
	ball.reset_ball(reset_transform, launch_direction * _episode_ball_speed)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = maxi(0, int(config.get("training_episode", _training_episode)))


func _current_ball_position_jitter_x() -> float:
	if not reset_curriculum_enabled:
		return ball_position_jitter_x
	var ratio := clampf(float(_training_episode) / float(maxi(curriculum_ramp_episodes, 1)), 0.0, 1.0)
	return lerpf(ball_position_jitter_x, curriculum_final_ball_jitter_x, ratio)


func get_brick_progress() -> float:
	if _bricks.is_empty():
		return 0.0
	return clampf(
		float(_bricks.size() - _remaining_bricks) / float(_bricks.size()),
		0.0,
		1.0
	)
	
func _on_ball_body_entered(body:Node) -> void:
	if body == paddle:
		# The contact offset lets the agent influence the outgoing bounce.
		var horizontal_offset := clampf(
			(ball.global_position.x - paddle.global_position.x) / maxf(paddle_half_width, 0.001),
			-1.0,
			1.0
		)
		call_deferred("_apply_paddle_bounce", horizontal_offset)
		return

	# Let Godot resolve the contact, then restore the intended arcade speed.
	call_deferred("_stabilize_ball_velocity")
	var brick := body as Node2D
	if brick == null or not brick.is_in_group("brick") or not brick.visible:
		return
	brick.visible = false
	brick.set_deferred("process_mode", Node.PROCESS_MODE_DISABLED)
	if brick is CollisionObject2D:
		brick.set_deferred("collision_layer", 0)
	_remaining_bricks -= 1
	brick_event.trigger(str(paddle.name))
	if _remaining_bricks <= 0:
		ball.stop_ball()
		cleared_event.trigger(str(paddle.name))
		paddle.set_episode_terminal(true)


func _on_loss_zone_body_entered(body:Node) -> void:
	if body != ball:
		return
	ball.stop_ball()
	lost_event.trigger(str(paddle.name))
	paddle.set_episode_terminal(true)


func _stabilize_ball_velocity() -> void:
	if ball.freeze or ball.linear_velocity.is_zero_approx():
		return
	_set_ball_direction(ball.linear_velocity.normalized())


func _apply_paddle_bounce(horizontal_offset:float) -> void:
	_set_ball_direction(Vector2(horizontal_offset, -1.0))
	ball_hit_event.trigger(str(paddle.name))


func _set_ball_direction(direction:Vector2) -> void:
	if direction.is_zero_approx():
		direction = Vector2(0.0, -1.0) 
	var vertical_sign := signf(direction.y)
	if is_zero_approx(vertical_sign):
		vertical_sign = -1.0
	if absf(direction.y) < min_ball_vertical_ratio:
		direction.y = vertical_sign * min_ball_vertical_ratio
	ball.linear_velocity = direction.normalized() * _episode_ball_speed
	ball.angular_velocity = 0.0
