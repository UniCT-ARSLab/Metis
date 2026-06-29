extends CharacterBody3D

@export var move_speed := 20.0
@export var turn_speed := 2.0
@export var gravity := 20.0
@export var manualControl:bool = true

var _move_input = 0.0
var _turn_input = 0.0

func _physics_process(delta):
	if not is_on_floor():
		velocity.y -= gravity * delta

	var y_velocity := velocity.y
	manual_control()

	velocity = transform.basis.z * _move_input * move_speed*delta
	velocity.y = y_velocity

	rotate_y(_turn_input * turn_speed * delta)
	move_and_slide()

func manual_control():
	if manualControl:
		_move_input = Input.get_axis("move_back", "move_forward")
		_turn_input = Input.get_axis("turn_right", "turn_left")
