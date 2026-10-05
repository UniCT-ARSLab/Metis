extends Node2D

@onready var controller := $ScenarioController
@onready var left_paddle: PongPaddle = $LeftPaddle
@onready var right_paddle: PongPaddle = $RightPaddle
@onready var ball: PongBall = $Ball
@onready var left_goal: Area2D = $LeftGoal
@onready var right_goal: Area2D = $RightGoal
@onready var rally_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/RallyHit
@onready var win_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/PointWon
@onready var loss_event: ManualScenarioEventSource = $ScenarioController/ScenarioEventSystem/PointLost

@export_category("Calibration")
# Derives paddle travel, observation scales and bounce height from the walls and goals.
@export var auto_calibrate_from_geometry := true

@export_category("Serve")
@export_range(0.0, 300.0, 1.0) var serve_position_y_range := 180.0
@export_range(0.0, 1.0, 0.01) var minimum_serve_vertical_component := 0.08
@export_range(0.0, 1.5, 0.01) var maximum_serve_vertical_component := 0.8

@export_category("Rally speed")
# Every point starts slow and speeds up at each paddle hit, up to a cap that a perfect player can still return.
@export var ball_start_speed := 300.0
@export_range(1.0, 1.5, 0.005) var hit_speed_gain := 1.07
# Measured with scripted players at constant speed: a paddle parked at one end of its travel loses 19 points in 20 at 1200 px/s, one that returns to the middle loses none up to 1400.
@export var max_ball_speed := 1200.0

var _ball_start_transform: Transform2D
# Midpoint between the paddles' spawn positions; read at _ready because the controller restores the paddles only after the serve.
var _serve_x := 0.0
var _point_finished := false
# Playfield interior between the walls and the goal lines; empty when the scene could not be measured.
var _field := Rect2()


func _ready() -> void:
	_ball_start_transform = ball.transform
	_serve_x = (left_paddle.position.x + right_paddle.position.x) * 0.5
	_measure_field()
	_apply_geometry()
	controller.episode_reset_started.connect(_reset_point)
	ball.paddle_hit.connect(_on_paddle_hit)
	ball.goal_line_crossed.connect(_on_goal_line_crossed)
	left_goal.body_entered.connect(_on_left_goal_entered)
	right_goal.body_entered.connect(_on_right_goal_entered)


func _shape_rect(path:String) -> Rect2:
	## World-space rect of a CollisionShape2D holding a RectangleShape2D, or an empty rect.
	var node := get_node_or_null(path) as CollisionShape2D
	if node == null or not (node.shape is RectangleShape2D):
		return Rect2()
	var half := (node.shape as RectangleShape2D).size * 0.5 * node.global_transform.get_scale().abs()
	return Rect2(node.global_position - half, half * 2.0)


func _measure_field() -> void:
	var top := _shape_rect("TopWall/CollisionShape2D")
	var bottom := _shape_rect("BottomWall/CollisionShape2D")
	var left := _shape_rect("LeftGoal/CollisionShape2D")
	var right := _shape_rect("RightGoal/CollisionShape2D")
	if top.size == Vector2.ZERO or bottom.size == Vector2.ZERO \
			or left.size == Vector2.ZERO or right.size == Vector2.ZERO:
		push_warning("Pong: could not measure the playfield; keeping the exported values.")
		return
	_field = Rect2(
		Vector2(left.end.x, top.end.y),
		Vector2(right.position.x - left.end.x, bottom.position.y - top.end.y))


func _apply_geometry() -> void:
	ball.hit_speed_gain = hit_speed_gain
	ball.max_speed = max_ball_speed
	if _field.size == Vector2.ZERO:
		return
	ball.left_goal_line = _field.position.x
	ball.right_goal_line = _field.end.x
	var radius := ball.radius
	for paddle in [left_paddle, right_paddle]:
		paddle.set_field(_field.position.y + radius, _field.end.y - radius, _field.size.y, radius)
	if not auto_calibrate_from_geometry:
		return
	var half := left_paddle.get_half_height()
	# The steepest bounce comes from a contact at the tip of the capsule, where the ball centre sits half a paddle plus one radius away.
	ball.paddle_half_height = half + radius
	for paddle in [left_paddle, right_paddle]:
		paddle.vertical_center = _field.get_center().y
		paddle.vertical_limit = maxf(1.0, _field.size.y * 0.5 - half - 1.0)
		paddle.observation_scale = _field.size
		paddle.ball_speed_scale = max_ball_speed
	print("Pong field: x=[%.0f, %.0f] y=[%.0f, %.0f] paddle_half=%.1f limit=%.1f ball_radius=%.1f" % [
		_field.position.x, _field.end.x, _field.position.y, _field.end.y,
		half, left_paddle.vertical_limit, radius])


func _reset_point(_seed:int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = _seed
	_point_finished = false
	left_paddle.set_episode_terminal(false)
	right_paddle.set_episode_terminal(false)

	ball.transform = _ball_start_transform
	if _field.size != Vector2.ZERO:
		# Serve from the midpoint between the paddles so neither side starts closer to the ball.
		ball.position = Vector2(_serve_x, _field.get_center().y)
	ball.position.y += rng.randf_range(-serve_position_y_range, serve_position_y_range)
	ball.speed = _serve_speed()
	var serve_direction := -1.0 if rng.randi() % 2 == 0 else 1.0
	var vertical_direction := -1.0 if rng.randi() % 2 == 0 else 1.0
	var vertical_component := rng.randf_range(
		minimum_serve_vertical_component,
		maxf(minimum_serve_vertical_component, _serve_vertical_limit())
	)
	ball.launch(serve_direction, vertical_direction * vertical_component)
	if ball.has_method("reset_physics_interpolation"):
		ball.reset_physics_interpolation()


func _serve_speed() -> float:
	return ball_start_speed


func _serve_vertical_limit() -> float:
	return maximum_serve_vertical_component


func _on_paddle_hit(paddle:Node) -> void:
	if not _point_finished:
		rally_event.trigger(str(paddle.name))


func _on_goal_line_crossed(side:int) -> void:
	if side < 0:
		_finish_point(right_paddle, left_paddle)
	else:
		_finish_point(left_paddle, right_paddle)


func _on_left_goal_entered(body:Node) -> void:
	# Fallback for an unmeasured field; a stale signal from before a reset finds the ball back in the middle and is ignored.
	if body != ball:
		return
	if is_inf(ball.left_goal_line) or ball.global_position.x - ball.radius <= ball.left_goal_line + 1.0:
		_finish_point(right_paddle, left_paddle)


func _on_right_goal_entered(body:Node) -> void:
	if body != ball:
		return
	if is_inf(ball.right_goal_line) or ball.global_position.x + ball.radius >= ball.right_goal_line - 1.0:
		_finish_point(left_paddle, right_paddle)


func _finish_point(winner:PongPaddle, loser:PongPaddle) -> void:
	if _point_finished:
		return
	_point_finished = true
	ball.velocity = Vector2.ZERO
	win_event.trigger(str(winner.name))
	loss_event.trigger(str(loser.name))
	winner.set_episode_terminal(true)
	loser.set_episode_terminal(true)
