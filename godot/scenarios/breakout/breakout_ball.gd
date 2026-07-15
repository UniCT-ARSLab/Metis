extends RigidBody2D
class_name BreakoutBall

var _reset_pending := false
var _pending_transform := Transform2D.IDENTITY
var _pending_velocity := Vector2.ZERO


func reset_ball(reset_transform:Transform2D, launch_velocity:Vector2) -> void:
	_pending_transform = reset_transform
	_pending_velocity = launch_velocity
	_reset_pending = true
	freeze = false
	sleeping = false


func stop_ball() -> void:
	_reset_pending = false
	set_deferred("freeze", true)
	call_deferred("reset_physics_interpolation")


func _integrate_forces(state:PhysicsDirectBodyState2D) -> void:
	if not _reset_pending:
		return

	state.transform = _pending_transform
	state.linear_velocity = _pending_velocity
	state.angular_velocity = 0.0
	_reset_pending = false
	call_deferred("_finish_reset_interpolation")


func _finish_reset_interpolation() -> void:
	reset_physics_interpolation()
