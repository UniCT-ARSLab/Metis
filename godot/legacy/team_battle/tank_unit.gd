extends CharacterBody3D

@export var move_speed := 2.0
@export var turn_speed := 1.0
@export var arena_half_extent := 9.0
@export var reload_time := 0.8
@export var max_hp := 100.0
@export var team_id := 0

var hp := 100.0
var alive := true
var current_throttle := 0.0
var current_turn := 0.0
var request_fire := false
var reload_left := 0.0
var hit_boundary := false

func _ready() -> void:
	_apply_team_material()

func reset_unit(new_transform: Transform3D, new_team_id: int) -> void:
	team_id = new_team_id
	global_transform = new_transform
	velocity = Vector3.ZERO
	current_throttle = 0.0
	current_turn = 0.0
	request_fire = false
	reload_left = 0.0
	hit_boundary = false
	hp = max_hp
	alive = true
	visible = true
	if has_node("CollisionShape3D"):
		$CollisionShape3D.disabled = false
	_apply_team_material()

func _apply_team_material() -> void:
	if not has_node("MeshInstance3D"):
		return
	var mat := StandardMaterial3D.new()
	if team_id == 0:
		mat.albedo_color = Color(0.2, 0.45, 1.0)
	else:
		mat.albedo_color = Color(1.0, 0.25, 0.25)
	if has_node("tank"):
		$tank.material_override = mat
	$MeshInstance3D.material_override = mat

func apply_action(action: int) -> void:
	current_throttle = 0.0
	current_turn = 0.0
	request_fire = false

	match action:
		0:
			pass
		1:
			current_throttle = 1.0
		2:
			current_turn = -1.0
		3:
			current_turn = 1.0
		4:
			current_throttle = 1.0
			current_turn = -1.0
		5:
			current_throttle = 1.0
			current_turn = 1.0
		6:
			request_fire = true
		7:
			current_throttle = 1.0
			request_fire = true
		_:
			pass

func sim_step(delta: float) -> void:
	hit_boundary = false
	if not alive:
		return
	reload_left = max(0.0, reload_left - delta)
	rotate_y(current_turn * turn_speed * delta)
	var forward := -global_transform.basis.z
	velocity = forward * (current_throttle * move_speed)
	move_and_slide()
	_clamp_to_arena()

func _clamp_to_arena() -> void:
	var pos := global_position
	var clamped_x := clampf(pos.x, -arena_half_extent, arena_half_extent)
	var clamped_z := clampf(pos.z, -arena_half_extent, arena_half_extent)
	if not is_equal_approx(pos.x, clamped_x) or not is_equal_approx(pos.z, clamped_z):
		global_position = Vector3(clamped_x, pos.y, clamped_z)
		velocity = Vector3.ZERO
		hit_boundary = true

func can_fire() -> bool:
	return alive and reload_left <= 0.0

func consume_fire() -> void:
	reload_left = reload_time

func take_damage(amount: float) -> bool:
	if not alive:
		return false
	hp -= amount
	if hp <= 0.0:
		hp = 0.0
		alive = false
		visible = false
		velocity = Vector3.ZERO
		if has_node("CollisionShape3D"):
			$CollisionShape3D.disabled = true
		return true
	return false

func hp_norm() -> float:
	return hp / max_hp

func reload_norm() -> float:
	return clamp(reload_left / reload_time, 0.0, 1.0)

func get_forward_speed() -> float:
	var forward := -global_transform.basis.z
	return velocity.dot(forward)
