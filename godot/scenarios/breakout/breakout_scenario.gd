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

@export_category("Calibration")
## Derives movement and observation limits from the scene geometry.
@export var auto_calibrate_from_geometry := true
## Keep the ball's spawn within reach of the paddle for the ball's flight time.
@export_range(0.0, 1.0, 0.05) var reachable_spawn_margin := 0.9

@export_category("Serve")
## Serve the ball off the paddle, upward, instead of dropping it from a fixed point in mid air.
@export var serve_from_paddle := true
## Half-width of the launch cone, as |dx| against a vertical of 1.
@export_range(0.0, 0.9, 0.05) var serve_horizontal_range := 0.6
## Clearance between the paddle's top edge and the ball, so the serve does not start in contact.
@export_range(0.0, 40.0, 1.0) var serve_gap := 6.0

@onready var controller := $ScenarioController
@onready var paddle: BreakoutPaddle = $BreakoutPaddle
@onready var paddle_agent: Agent = $BreakoutPaddle/Agent
@onready var ball: BreakoutBall = $Ball
@onready var loss_zone: Area2D = $LossZone
@onready var brick_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/BrickDestroyed
@onready var lost_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/LifeLost
@onready var cleared_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/LevelCleared
@onready var ball_hit_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/BallHitEvent
@onready var time_label: Label = get_node_or_null("UI/VBoxContainer/Time2")
@onready var reward_label: Label = get_node_or_null("UI/VBoxContainer/Reward2")


var _ball_start_transform: Transform2D
var _paddle_start := Vector2.ZERO
var _bricks: Array[Node2D] = []
var _remaining_bricks := 0
var _episode_ball_speed := 0.0
var _training_episode := 0
# Playfield interior, measured from the walls (see _calibrate_from_geometry).
var _field := Rect2()
var _ball_radius := 0.0
# Physics steps remain meaningful when training runs faster than real time.
var _episode_steps := 0
var _episode_reward := 0.0


func _ready() -> void:
	_ball_start_transform = ball.global_transform
	_paddle_start = paddle.global_position
	_episode_ball_speed = ball_speed
	for child in $Bricks.get_children():
		if child is Node2D:
			_bricks.append(child)
	_measure_field()
	if auto_calibrate_from_geometry:
		_calibrate_from_geometry()
	controller.episode_reset_started.connect(_reset_game)
	controller.episode_step_completed.connect(_on_step_completed)
	controller.scenario_configured.connect(_on_scenario_configured)
	ball.body_entered.connect(_on_ball_body_entered)
	loss_zone.body_entered.connect(_on_loss_zone_body_entered)
	_refresh_hud()
	if auto_start_manual and paddle.manual_control:
		call_deferred("_reset_game", manual_seed)


# geometry

func _shape_rect(path:String) -> Rect2:
	## World-space rect of a CollisionShape2D holding a RectangleShape2D, or an empty rect.
	var node := get_node_or_null(path) as CollisionShape2D
	if node == null or not (node.shape is RectangleShape2D):
		return Rect2()
	var half := (node.shape as RectangleShape2D).size * 0.5
	return Rect2(node.global_position - half, half * 2.0)


func _measure_field() -> void:
	## The interior the ball can occupy: inside the walls, above the loss line.
	var left := _shape_rect("Walls/LeftWall/CollisionShape2D")
	var right := _shape_rect("Walls/RightWall/CollisionShape2D")
	var ceiling := _shape_rect("Walls/Ceiling/CollisionShape2D")
	var loss := _shape_rect("LossZone/CollisionShape2D")
	if left.size == Vector2.ZERO or right.size == Vector2.ZERO \
			or ceiling.size == Vector2.ZERO or loss.size == Vector2.ZERO:
		push_warning("Breakout: could not measure the playfield; keeping the exported values.")
		return
	var shape := (ball.get_node_or_null("CollisionShape2D") as CollisionShape2D)
	if shape != null and shape.shape is CircleShape2D:
		_ball_radius = (shape.shape as CircleShape2D).radius
	_field = Rect2(
		Vector2(left.end.x, ceiling.end.y),
		Vector2(right.position.x - left.end.x, loss.position.y - ceiling.end.y))


func _paddle_half_extent() -> float:
	## Half the paddle's collision width.
	var shape := paddle.get_node_or_null("CollisionShape2D") as CollisionShape2D
	if shape == null:
		return paddle_half_width
	if shape.shape is CapsuleShape2D:
		var capsule := shape.shape as CapsuleShape2D
		var along := capsule.height * 0.5
		var across := capsule.radius
		var rotated := absf(sin(shape.global_rotation)) > 0.5
		return along if rotated else across
	if shape.shape is RectangleShape2D:
		return (shape.shape as RectangleShape2D).size.x * 0.5
	return paddle_half_width


func _calibrate_from_geometry() -> void:
	if _field.size == Vector2.ZERO:
		return
	var half := _paddle_half_extent()
	paddle_half_width = half
	paddle.horizontal_center = _field.position.x + _field.size.x * 0.5
	# Account for the paddle width so the observation has no unreachable edge values.
	paddle.horizontal_limit = maxf(1.0, _field.size.x * 0.5 - half)
	# Field dimensions map the relative position across the full [-1, 1] range.
	paddle.observation_position_scale = _field.size
	print("Breakout field: x=[%.0f, %.0f] y=[%.0f, %.0f] paddle_half=%.0f limit=%.0f" % [
		_field.position.x, _field.end.x, _field.position.y, _field.end.y,
		half, paddle.horizontal_limit], "\n")

func _reset_game(seed:int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	paddle.set_episode_terminal(false)
	_remaining_bricks = _bricks.size()

	# Park the ball before restoring bricks to avoid reset-time contacts.
	ball.park_outside(_field.position - Vector2(1000.0, 1000.0))

	for brick in _bricks:
		brick.visible = true
		brick.process_mode = Node.PROCESS_MODE_INHERIT
		if brick is CollisionObject2D:
			brick.collision_layer = 1

	_episode_steps = 0
	_episode_reward = 0.0
	_refresh_hud()

	var speed_scale := rng.randf_range(
		1.0 - ball_speed_jitter_ratio,
		1.0 + ball_speed_jitter_ratio
	)
	_episode_ball_speed = ball_speed * speed_scale

	var reset_transform := _ball_start_transform
	var launch_direction: Vector2
	if serve_from_paddle:
		# Vary the launch angle while keeping every opening serve reachable.
		launch_direction = Vector2(
			rng.randf_range(-serve_horizontal_range, serve_horizontal_range),
			-1.0
		).normalized()
		reset_transform.origin = Vector2(
			_paddle_start.x,
			_paddle_start.y - _paddle_half_height() - _ball_radius - serve_gap)
	else:
		# The reachable offset depends on the sampled direction and flight time.
		launch_direction = Vector2(
			rng.randf_range(-launch_horizontal_range, launch_horizontal_range),
			1.0
		).normalized()
		var position_jitter := _current_ball_position_jitter_x()
		position_jitter = minf(position_jitter, _spawn_jitter_limit(launch_direction))
		if _field.size != Vector2.ZERO:
			# Clamp the range before sampling to avoid bias at the field edges.
			var origin := _ball_start_transform.origin.x
			position_jitter = minf(position_jitter, minf(
				origin - (_field.position.x + _ball_radius),
				(_field.end.x - _ball_radius) - origin))
		position_jitter = maxf(position_jitter, 0.0)
		reset_transform.origin.x += rng.randf_range(-position_jitter, position_jitter)
	ball.reset_ball(reset_transform, launch_direction * _episode_ball_speed)


func _paddle_half_height() -> float:
	## Half the paddle's collision height -- the SHORT axis of the rotated capsule, the mirror of _paddle_half_extent().
	var shape := paddle.get_node_or_null("CollisionShape2D") as CollisionShape2D
	if shape == null:
		return 13.0
	if shape.shape is CapsuleShape2D:
		var capsule := shape.shape as CapsuleShape2D
		var rotated := absf(sin(shape.global_rotation)) > 0.5
		return capsule.radius if rotated else capsule.height * 0.5
	if shape.shape is RectangleShape2D:
		return (shape.shape as RectangleShape2D).size.y * 0.5
	return 13.0


func _spawn_jitter_limit(launch_direction:Vector2) -> float:
	## Largest sideways spawn offset the paddle can still answer, for this ball.
	var vertical_speed := _episode_ball_speed * absf(launch_direction.y)
	if vertical_speed <= 0.0 or paddle == null or paddle.speed <= 0.0:
		return 1e9
	var fall_distance := _paddle_start.y - _ball_start_transform.origin.y
	if fall_distance <= 0.0:
		# Served from below or level with the paddle: reachability says nothing useful here.
		return 1e9
	var flight := fall_distance / vertical_speed
	var reach := paddle.speed * flight * reachable_spawn_margin
	var drift := absf(_episode_ball_speed * launch_direction.x) * flight
	# Include both the initial offset and reset jitter in the reachability budget.
	var offset := absf(_ball_start_transform.origin.x - _paddle_start.x)
	if controller != null and controller.randomize_reset:
		offset += absf(controller.reset_position_jitter.x)
	return maxf(0.0, reach - drift - offset)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = maxi(0, int(config.get("training_episode", _training_episode)))


# HUD

func _on_step_completed(step:int) -> void:
	## Accumulates the same per-step reward sent to Python.
	_episode_steps = step
	_episode_reward += controller.last_reward
	_refresh_hud()


func _refresh_hud() -> void:
	if time_label != null:
		time_label.text = "%d steps" % _episode_steps
	if reward_label != null:
		reward_label.text = "%.2f" % _episode_reward


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
