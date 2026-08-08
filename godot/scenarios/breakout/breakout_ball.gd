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
	# Move NOW, not just on the next _integrate_forces. Deferring the teleport left the ball at the
	# position where the previous episode ended for one physics frame, and the scenario re-enables
	# every brick during that same frame -- so the bricks reappeared on top of the stale ball,
	# reported contacts, and handed out one or two free bricks at the start of every episode after
	# the first. _integrate_forces still runs: it is the only place a RigidBody2D's velocity can be
	# set without the solver overwriting it.
	PhysicsServer2D.body_set_state(
		get_rid(), PhysicsServer2D.BODY_STATE_TRANSFORM, reset_transform)
	PhysicsServer2D.body_set_state(
		get_rid(), PhysicsServer2D.BODY_STATE_LINEAR_VELOCITY, launch_velocity)
	PhysicsServer2D.body_set_state(
		get_rid(), PhysicsServer2D.BODY_STATE_ANGULAR_VELOCITY, 0.0)
	global_transform = reset_transform


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


func park_outside(somewhere:Vector2) -> void:
	## Teleport the ball far off the playfield, immediately, with no velocity.
	##
	## Called before a scenario re-enables its obstacles: whatever reappears cannot collide with a
	## ball that is no longer there. Cancels any pending reset so _integrate_forces does not put the
	## ball back at the old target after this.
	_reset_pending = false
	freeze = false
	sleeping = false
	var parked := Transform2D(0.0, somewhere)
	PhysicsServer2D.body_set_state(get_rid(), PhysicsServer2D.BODY_STATE_TRANSFORM, parked)
	PhysicsServer2D.body_set_state(
		get_rid(), PhysicsServer2D.BODY_STATE_LINEAR_VELOCITY, Vector2.ZERO)
	PhysicsServer2D.body_set_state(get_rid(), PhysicsServer2D.BODY_STATE_ANGULAR_VELOCITY, 0.0)
	global_transform = parked
