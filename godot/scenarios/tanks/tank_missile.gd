extends CharacterBody3D
class_name TankMissile

@export var speed := 28.0
@export var damage := 34.0
@export var max_lifetime := 4.0

var shooter: BattleTank
var team_id := -1
var _lifetime := 0.0


func launch(owner_tank:BattleTank) -> void:
	shooter = owner_tank
	team_id = owner_tank.get_team_id()
	velocity = global_transform.basis.z.normalized() * speed
	add_collision_exception_with(owner_tank)


func _physics_process(delta:float) -> void:
	_lifetime += delta
	if _lifetime >= max_lifetime:
		queue_free()
		return

	var collision := move_and_collide(velocity * delta)
	if collision == null:
		return

	var collider := collision.get_collider()
	if collider is BattleTank:
		if collider.get_team_id() == team_id:
			add_collision_exception_with(collider)
			global_position += velocity.normalized() * 0.1
			return
		collider.take_damage(shooter, damage)
	queue_free()
