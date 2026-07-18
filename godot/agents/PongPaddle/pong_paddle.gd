extends CharacterBody2D
class_name PongPaddle

@export var speed := 420.0
# Vertical frame of the play field. The scene is laid out in positive coordinates, so the
# clamp and the y observation must be centered on the field center, not on y=0. Same
# pattern as BreakoutPaddle.horizontal_center. Default 0 keeps a y=0-centered field working.
@export var vertical_center := 0.0
@export var vertical_limit := 300.0
@export var ball_speed_scale := 520.0
@export var view_direction := 1.0
@export var team_id := 0
@export var ball: CharacterBody2D
@export var opponent: PongPaddle
@export var manual_control := false

@onready var agent: Agent = $Agent

var _move_input := 0.0
var _terminal := false
var _training_active := true
# Previous normalized |ball_rel_y| for the incoming-ball tracking reward.
# A negative value means that the next sample must establish a new baseline.
var _prev_abs_rel_y := -1.0


func get_team_id() -> int:
	return team_id


func _physics_process(_delta:float) -> void:
	if not _training_active:
		return
	if manual_control:
		_move_input = Input.get_axis("move_up", "move_down")
	velocity = Vector2(0.0, _move_input * speed)
	move_and_slide()
	global_position.y = clampf(
		global_position.y,
		vertical_center - vertical_limit,
		vertical_center + vertical_limit
	)


func apply_action(action:Variant) -> Variant:
	if str(action) == "manual":
		return apply_manual_action()
	stop()
	var action_id := int(action)
	if agent.act(action_id) != OK:
		return 0
	return action_id


func apply_manual_action() -> int:
	var axis := Input.get_axis("move_up", "move_down")
	var action_id := 0
	if axis < -0.2:
		action_id = 1
	elif axis > 0.2:
		action_id = 2
	return apply_action(action_id)


func stop() -> void:
	_move_input = 0.0


func move_up() -> void:
	_move_input = -1.0


func move_down() -> void:
	_move_input = 1.0


func get_paddle_y_observation() -> float:
	return clampf((global_position.y - vertical_center) / maxf(vertical_limit, 0.001), -1.0, 1.0)


func get_ball_relative_position_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var relative := ball.global_position - global_position
	relative.x *= view_direction
	return Vector2(
		clampf(relative.x / 640.0, -1.0, 1.0),
		clampf(relative.y / maxf(vertical_limit, 0.001), -1.0, 1.0)
	)


func get_ball_velocity_observation() -> Vector2:
	if ball == null:
		return Vector2.ZERO
	var perceived_velocity := ball.velocity
	perceived_velocity.x *= view_direction
	return (perceived_velocity / maxf(ball_speed_scale, 0.001)).clamp(
		Vector2(-1.0, -1.0),
		Vector2(1.0, 1.0)
	)


func get_opponent_delta_y_observation() -> float:
	if opponent == null:
		return 0.0
	return clampf(
		(opponent.global_position.y - global_position.y) / maxf(vertical_limit * 2.0, 0.001),
		-1.0,
		1.0
	)


func get_vertical_alignment_reward() -> float:
	if ball == null:
		return 0.0

	# In each Paddle's mirrored frame, negative X velocity means that the ball is
	# approaching. Tracking it while it moves away wastes motion and can conflict with
	# preparing for the next return, so a new shaping phase starts only on approach.
	var perceived_velocity_x := ball.velocity.x * view_direction
	if perceived_velocity_x >= 0.0:
		_prev_abs_rel_y = -1.0
		return 0.0

	var abs_rel_y := clampf(
		absf(ball.global_position.y - global_position.y) / maxf(vertical_limit, 0.001),
		0.0,
		1.0
	)
	if _prev_abs_rel_y < 0.0:
		_prev_abs_rel_y = abs_rel_y
		return 0.0
	var delta := _prev_abs_rel_y - abs_rel_y
	_prev_abs_rel_y = abs_rel_y
	return delta


func set_episode_terminal(value:bool) -> void:
	_terminal = value


func is_terminal() -> bool:
	return _terminal


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	_terminal = false
	_prev_abs_rel_y = -1.0  # new episode: no baseline until the first step
	stop()
	velocity = Vector2.ZERO
	if typeof(original_transform) == TYPE_TRANSFORM2D:
		transform = original_transform
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
