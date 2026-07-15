extends CharacterBody2D
class_name BreakoutPaddle

@export var speed := 420.0
@export var horizontal_center := 0.0
@export var horizontal_limit := 540.0
@export var lock_vertical_position := true
@export var ball: RigidBody2D
@export var observation_position_scale := Vector2(540.0, 360.0)
@export var observation_ball_speed_scale := 500.0
@export var manual_control := false

@onready var agent: Agent = $Agent

var _move_input := 0.0
var _terminal := false
var _training_active := true
var _locked_y := 0.0


func _ready() -> void:
	_locked_y = global_position.y

func _physics_process(_delta:float) -> void:
	if not _training_active:
		return
	if manual_control:
		_move_input = Input.get_axis("move_left", "move_right")
	velocity = Vector2(_move_input * speed, 0.0)
	move_and_slide()
	if lock_vertical_position:
		global_position.y = _locked_y
		velocity.y = 0.0
	global_position.x = clampf(
		global_position.x,
		horizontal_center - horizontal_limit,
		horizontal_center + horizontal_limit
	)
	
func apply_action(action:Variant) -> Variant:
	if str(action) == "manual":
		return apply_manual_action()
	manual_control = false
	stop()
	var action_id := int(action)
	if agent.act(action_id) != OK:
		return 0
	return action_id

func apply_manual_action() -> int:
	var axis := Input.get_axis("move_left", "move_right")
	var action_id := 0
	if axis < -0.2:
		action_id = 1
	elif axis > 0.2:
		action_id = 2
	return apply_action(action_id)

func stop() -> void:
	_move_input = 0.0

func move_left() -> void:
	_move_input = -1.0

func move_right() -> void:
	_move_input = 1.0

func get_paddle_x_observation() -> float:
	return clampf(
		(global_position.x - horizontal_center) / maxf(horizontal_limit, 0.001),
		-1.0,
		1.0
	)

func get_ball_relative_position_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var relative := ball.global_position - global_position
	return Vector2(
		clampf(relative.x / maxf(observation_position_scale.x, 0.001), -1.0, 1.0),
		clampf(relative.y / maxf(observation_position_scale.y, 0.001), -1.0, 1.0)
	)

func get_ball_velocity_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	return (ball.linear_velocity / maxf(observation_ball_speed_scale, 0.001)).clamp(
		Vector2(-1.0, -1.0),
		Vector2(1.0, 1.0)
	)

func set_episode_terminal(value:bool) -> void:
	_terminal = value

func is_terminal() -> bool:
	return _terminal

func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	_terminal = false
	stop()
	velocity = Vector2.ZERO
	if typeof(original_transform) == TYPE_TRANSFORM2D:
		transform = original_transform
	_locked_y = global_position.y
	if has_method("reset_physics_interpolation"):
		reset_physics_interpolation()
	agent.reset_observation_sources()
	if reset_rewards:
		agent.refresh_observation_sources()
		agent.reset_reward({"body": self})

func set_training_active(enabled:bool) -> void:
	_training_active = enabled
	set_physics_process(enabled)
	if not enabled:
		stop()
		velocity = Vector2.ZERO
