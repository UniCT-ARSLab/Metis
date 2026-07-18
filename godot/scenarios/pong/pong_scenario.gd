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

var _ball_start_transform: Transform2D
var _training_episode := 0
var _point_finished := false

@export_range(0.0, 300.0, 1.0) var serve_position_y_range := 180.0
@export_range(0.0, 1.0, 0.01) var minimum_serve_vertical_component := 0.08


func _ready() -> void:
	_ball_start_transform = ball.transform
	controller.scenario_configured.connect(_on_scenario_configured)
	controller.episode_reset_started.connect(_reset_point)
	ball.paddle_hit.connect(_on_paddle_hit)
	left_goal.body_entered.connect(_on_left_goal_entered)
	right_goal.body_entered.connect(_on_right_goal_entered)


func _on_scenario_configured(config:Dictionary) -> void:
	_training_episode = int(config.get("training_episode", _training_episode))


func _reset_point(_seed:int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = _seed
	_point_finished = false
	left_paddle.set_episode_terminal(false)
	right_paddle.set_episode_terminal(false)
	ball.transform = _ball_start_transform
	ball.position.y += rng.randf_range(-serve_position_y_range, serve_position_y_range)
	ball.speed = _curriculum_ball_speed()
	var serve_direction := -1.0 if rng.randi() % 2 == 0 else 1.0
	var vertical_direction := -1.0 if rng.randi() % 2 == 0 else 1.0
	var vertical_component := rng.randf_range(
		minimum_serve_vertical_component,
		maxf(minimum_serve_vertical_component, _curriculum_angle())
	)
	ball.launch(serve_direction, vertical_direction * vertical_component)
	if ball.has_method("reset_physics_interpolation"):
		ball.reset_physics_interpolation()


func _on_paddle_hit(paddle:Node) -> void:
	if not _point_finished:
		rally_event.trigger(str(paddle.name))


func _on_left_goal_entered(body:Node) -> void:
	if body == ball:
		_finish_point(right_paddle, left_paddle)


func _on_right_goal_entered(body:Node) -> void:
	if body == ball:
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


func _curriculum_ball_speed() -> float:
	if _training_episode < 300:
		return 300.0
	if _training_episode < 800:
		return 380.0
	return 460.0


func _curriculum_angle() -> float:
	if _training_episode < 300:
		return 0.25
	if _training_episode < 800:
		return 0.50
	return 0.80
