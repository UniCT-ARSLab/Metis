extends CharacterBody2D
class_name PongBall

# Emitted when a paddle returns the ball, not on every contact: a ball squeezed against a paddle touches it every tick.
signal paddle_hit(paddle:Node)
# Emitted in the same physics tick in which the ball reaches a goal line: -1 for the left goal, +1 for the right one.
signal goal_line_crossed(side:int)

@export var speed := 420.0
@export_range(0.0, 45.0, 1.0) var minimum_bounce_angle_degrees := 8.0
@export_range(15.0, 80.0, 1.0) var maximum_bounce_angle_degrees := 60.0
# Vertical offset from the paddle centre that maps to the steepest bounce.
@export_range(1.0, 300.0, 1.0) var paddle_half_height := 103.0
@export_category("Rally speed")
# Speed multiplier applied at every paddle hit; 1.0 keeps the serve speed for the whole point.
@export_range(1.0, 1.5, 0.005) var hit_speed_gain := 1.0
@export var max_speed := 460.0
@export_category("Goal lines")
# x of the goal lines, set by the scenario; the ball edge touching one ends the point.
@export var left_goal_line := -INF
@export var right_goal_line := INF

var radius := 0.0
var _goal_reported := false


func _ready() -> void:
	var shape_node := get_node_or_null("CollisionShape2D") as CollisionShape2D
	if shape_node != null and shape_node.shape is CircleShape2D:
		radius = (shape_node.shape as CircleShape2D).radius * absf(shape_node.global_transform.get_scale().x)


func _physics_process(delta:float) -> void:
	var motion := velocity * delta
	# Dropping the motion left after a contact freezes a ball wedged between a paddle and a wall, so it is spent on the new heading.
	for _contact in range(4):
		var collision := move_and_collide(motion)
		if collision == null:
			break
		_bounce(collision)
		if velocity.is_zero_approx():
			break
		motion = velocity.normalized() * collision.get_remainder().length()
	_check_goal_lines()


func _bounce(collision:KinematicCollision2D) -> void:
	var collider := collision.get_collider()
	if collider is Node2D and collider.is_in_group("paddle"):
		var paddle := collider as Node2D
		var returned := velocity.x * (paddle.global_position.x - global_position.x) > 0.0
		if returned:
			speed = minf(speed * hit_speed_gain, maxf(max_speed, speed))
		_apply_arcade_paddle_bounce(paddle)
		if returned:
			paddle_hit.emit(paddle)
	else:
		velocity = velocity.bounce(collision.get_normal()).normalized() * speed
		_limit_vertical_trajectory()


func launch(direction:float, vertical_component:float) -> void:
	_goal_reported = false
	velocity = Vector2(direction, vertical_component).normalized() * speed
	_limit_vertical_trajectory()


func _check_goal_lines() -> void:
	# The Area2D goals report one tick late, so the point would end one decision after the ball went in.
	if _goal_reported or velocity.is_zero_approx():
		return
	if global_position.x - radius <= left_goal_line:
		_goal_reported = true
		goal_line_crossed.emit(-1)
	elif global_position.x + radius >= right_goal_line:
		_goal_reported = true
		goal_line_crossed.emit(1)


func _apply_arcade_paddle_bounce(paddle:Node2D) -> void:
	var hit_offset := clampf(
		(global_position.y - paddle.global_position.y) / maxf(paddle_half_height, 1.0),
		-1.0,
		1.0
	)
	var vertical_direction := signf(hit_offset)
	if is_zero_approx(vertical_direction):
		vertical_direction = signf(velocity.y)
	if is_zero_approx(vertical_direction):
		vertical_direction = 1.0

	var minimum_angle := deg_to_rad(minimum_bounce_angle_degrees)
	var maximum_angle := deg_to_rad(maximum_bounce_angle_degrees)
	var bounce_angle := lerpf(minimum_angle, maximum_angle, absf(hit_offset))
	var horizontal_direction := signf(global_position.x - paddle.global_position.x)
	if is_zero_approx(horizontal_direction):
		horizontal_direction = signf(velocity.x) * -1.0
	if is_zero_approx(horizontal_direction):
		horizontal_direction = 1.0

	velocity = Vector2(
		horizontal_direction * cos(bounce_angle),
		vertical_direction * sin(bounce_angle)
	) * speed


func _limit_vertical_trajectory() -> void:
	if velocity.is_zero_approx():
		return
	var maximum_angle := deg_to_rad(maximum_bounce_angle_degrees)
	var horizontal_direction := signf(velocity.x)
	if is_zero_approx(horizontal_direction):
		horizontal_direction = 1.0
	var vertical_direction := signf(velocity.y)
	if is_zero_approx(vertical_direction):
		vertical_direction = 1.0
	var current_angle := atan2(absf(velocity.y), absf(velocity.x))
	if current_angle <= maximum_angle:
		velocity = velocity.normalized() * speed
		return
	velocity = Vector2(
		horizontal_direction * cos(maximum_angle),
		vertical_direction * sin(maximum_angle)
	) * speed
