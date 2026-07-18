extends CharacterBody2D
class_name PongBall

signal paddle_hit(paddle:Node)

@export var speed := 420.0
@export_range(0.0, 45.0, 1.0) var minimum_bounce_angle_degrees := 8.0
@export_range(15.0, 80.0, 1.0) var maximum_bounce_angle_degrees := 60.0
@export_range(1.0, 300.0, 1.0) var paddle_half_height := 103.0


func _physics_process(delta:float) -> void:
	var collision := move_and_collide(velocity * delta)
	if collision == null:
		return
	var collider := collision.get_collider()
	if collider is Node and collider.is_in_group("paddle"):
		_apply_arcade_paddle_bounce(collider)
		paddle_hit.emit(collider)
	else:
		velocity = velocity.bounce(collision.get_normal()).normalized() * speed
		_limit_vertical_trajectory()


func launch(direction:float, vertical_component:float) -> void:
	velocity = Vector2(direction, vertical_component).normalized() * speed
	_limit_vertical_trajectory()


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
