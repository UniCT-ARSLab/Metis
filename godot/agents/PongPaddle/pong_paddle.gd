extends CharacterBody2D
class_name PongPaddle

enum AlignmentTarget { INTERCEPT, BALL_Y }

@export var speed := 420.0
# Vertical frame of the play field. The scene is laid out in positive coordinates, so the clamp and the y observation are centred on the field centre, not on y=0.
# PongScenario overwrites these from the wall geometry when auto_calibrate_from_geometry is on.
@export var vertical_center := 0.0
@export var vertical_limit := 300.0
# Divisors of the ball position relative to this paddle; the field size keeps every value inside [-1, 1].
@export var observation_scale := Vector2(640.0, 258.0)
@export var ball_speed_scale := 520.0
@export var view_direction := 1.0
@export var team_id := 0
@export var ball: CharacterBody2D
@export var opponent: PongPaddle
@export var manual_control := false

@export_category("Alignment shaping")
# INTERCEPT pays for closing the gap to where the ball will cross this paddle's line; BALL_Y chases the ball's current height.
@export var alignment_target := AlignmentTarget.INTERCEPT
# Gap that counts as aligned, as a fraction of the paddle half height, so the shaping does not push every hit to the centre.
@export_range(0.0, 1.0, 0.05) var alignment_deadzone_ratio := 0.5

@onready var agent: Agent = $Agent

var _move_input := 0.0
var _terminal := false
var _training_active := true
# Previous alignment gap for the incoming-ball shaping; negative means that the next sample sets a new baseline.
var _prev_alignment_gap := -1.0
# Ball-centre range and field height, set by PongScenario; zero height means that the geometry is unknown.
var _ball_y_min := 0.0
var _ball_y_max := 0.0
var _field_height := 0.0
var _ball_radius := 0.0
var _half_height := 0.0
# Sliding along the round ball can push the body sideways, so x is pinned to where the paddle spawned.
var _home_x := 0.0


func _ready() -> void:
	_home_x = global_position.x
	_half_height = _measure_half_height()
	_ensure_manual_input_actions()


func get_team_id() -> int:
	return team_id


func get_half_height() -> float:
	return _half_height


func set_field(ball_y_min:float, ball_y_max:float, field_height:float, ball_radius:float) -> void:
	_ball_y_min = ball_y_min
	_ball_y_max = ball_y_max
	_field_height = field_height
	_ball_radius = ball_radius


func _physics_process(_delta:float) -> void:
	if not _training_active:
		return
	if manual_control:
		_move_input = Input.get_axis("move_up", "move_down")
	velocity = Vector2(0.0, _move_input * speed)
	move_and_slide()
	global_position = Vector2(
		_home_x,
		clampf(global_position.y, vertical_center - vertical_limit, vertical_center + vertical_limit)
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
		clampf(relative.x / maxf(observation_scale.x, 0.001), -1.0, 1.0),
		clampf(relative.y / maxf(observation_scale.y, 0.001), -1.0, 1.0)
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


func is_ball_approaching() -> bool:
	# In each paddle's mirrored frame a negative x velocity means that the ball is coming towards it.
	return ball != null and ball.velocity.x * view_direction < 0.0


func predict_intercept_y() -> float:
	## Height at which the ball centre will reach this paddle's contact line, folding the path on the walls.
	var contact_x := global_position.x + view_direction * (_paddle_radius() + _ball_radius)
	var vx := ball.velocity.x
	if is_zero_approx(vx):
		return ball.global_position.y
	var t := maxf((contact_x - ball.global_position.x) / vx, 0.0)
	var y := ball.global_position.y + ball.velocity.y * t
	if _field_height <= 0.0 or _ball_y_max <= _ball_y_min:
		return y
	var span := _ball_y_max - _ball_y_min
	var folded := fposmod(y - _ball_y_min, 2.0 * span)
	if folded > span:
		folded = 2.0 * span - folded
	return _ball_y_min + folded


func get_vertical_alignment_reward() -> float:
	if ball == null:
		return 0.0

	# Tracking the ball while it moves away wastes motion and can conflict with preparing for the next return, so a new shaping phase starts only on approach.
	if not is_ball_approaching():
		_prev_alignment_gap = -1.0
		return 0.0

	var gap := _alignment_gap()
	if _prev_alignment_gap < 0.0:
		_prev_alignment_gap = gap
		return 0.0
	var delta := _prev_alignment_gap - gap
	_prev_alignment_gap = gap
	return delta


func _alignment_gap() -> float:
	if alignment_target == AlignmentTarget.BALL_Y or _field_height <= 0.0:
		return clampf(
			absf(ball.global_position.y - global_position.y) / maxf(vertical_limit, 0.001),
			0.0,
			1.0
		)
	var deadzone := alignment_deadzone_ratio * _half_height
	var miss := maxf(absf(predict_intercept_y() - global_position.y) - deadzone, 0.0)
	return clampf(miss / _field_height, 0.0, 1.0)


func _paddle_radius() -> float:
	var shape_node := get_node_or_null("CollisionShape2D") as CollisionShape2D
	if shape_node != null and shape_node.shape is CapsuleShape2D:
		return (shape_node.shape as CapsuleShape2D).radius * absf(shape_node.global_transform.get_scale().x)
	return 20.0


func _measure_half_height() -> float:
	var shape_node := get_node_or_null("CollisionShape2D") as CollisionShape2D
	if shape_node != null and shape_node.shape is CapsuleShape2D:
		return (shape_node.shape as CapsuleShape2D).height * 0.5 * absf(shape_node.global_transform.get_scale().y)
	return 80.0


func _ensure_manual_input_actions() -> void:
	# The project input map does not ship these actions; without them manual_control reads nothing.
	var bindings := {"move_up": [KEY_W, KEY_UP], "move_down": [KEY_S, KEY_DOWN]}
	for action_name in bindings:
		if InputMap.has_action(action_name):
			continue
		InputMap.add_action(action_name)
		for keycode in bindings[action_name]:
			var event := InputEventKey.new()
			event.physical_keycode = keycode
			InputMap.action_add_event(action_name, event)


func set_episode_terminal(value:bool) -> void:
	_terminal = value


func is_terminal() -> bool:
	return _terminal


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	_terminal = false
	_prev_alignment_gap = -1.0
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
