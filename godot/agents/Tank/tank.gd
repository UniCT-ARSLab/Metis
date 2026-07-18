extends CharacterBody3D
class_name BattleTank

signal damage_received(victim:BattleTank, attacker:BattleTank, amount:float)
signal destroyed(victim:BattleTank, killer:BattleTank)

@export_category("Team")
@export var team_id := 0

@export_category("Movement")
@export var max_forward_speed := 12.0
@export var max_reverse_speed := 5.0
@export var acceleration := 18.0
@export var drag := 12.0
@export var turn_speed_degrees := 110.0

@export_category("Combat")
@export var max_health := 100.0
@export var fire_cooldown := 0.8
@export var projectile_scene: PackedScene
@export var projectile_parent: Node

@export_category("Control")
@export var manual_control := false

@onready var agent: Agent = $Agent
@onready var muzzle: Marker3D = $Muzzle

var _throttle_input := 0.0
var _rotation_input := 0.0
var _cooldown_left := 0.0
var _health := 100.0
var _alive := true
var _training_active := true
var _initial_collision_layer := 0
var _initial_collision_mask := 0


func _ready() -> void:
	_health = max_health
	_initial_collision_layer = collision_layer
	_initial_collision_mask = collision_mask


func _physics_process(delta:float) -> void:
	_cooldown_left = maxf(_cooldown_left - delta, 0.0)
	if not _training_active or not _alive:
		return

	if manual_control:
		_throttle_input = Input.get_axis("move_back", "move_forward")
		_rotation_input = Input.get_axis("turn_left", "turn_right")

	rotate_y(-_rotation_input * deg_to_rad(turn_speed_degrees) * delta)

	var speed_limit := max_forward_speed if _throttle_input >= 0.0 else max_reverse_speed
	var desired_velocity := global_transform.basis.z * _throttle_input * speed_limit
	var flat_velocity := Vector3(velocity.x, 0.0, velocity.z)
	if absf(_throttle_input) > 0.05:
		flat_velocity = flat_velocity.move_toward(desired_velocity, acceleration * delta)
	else:
		flat_velocity = flat_velocity.move_toward(Vector3.ZERO, drag * delta)

	velocity.x = flat_velocity.x
	velocity.z = flat_velocity.z
	if not is_on_floor():
		velocity += get_gravity() * delta
	move_and_slide()


func apply_action(action:Variant) -> Variant:
	if (typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME) and str(action) == "manual":
		return apply_manual_action()

	manual_control = false
	clear_inputs()
	if not _alive or typeof(action) != TYPE_DICTIONARY:
		return _zero_action()

	var action_map: Dictionary = action
	var movement := agent.decode_continuous_action(action_map)
	_throttle_input = clampf(float(movement[0]), -1.0, 1.0) if movement.size() > 0 else 0.0
	_rotation_input = clampf(float(movement[1]), -1.0, 1.0) if movement.size() > 1 else 0.0

	var weapon_action := int(action_map.get("weapon", 0))
	if agent.act_discrete(weapon_action, "weapon") != OK:
		weapon_action = 0

	return {
		"movement": [_throttle_input, _rotation_input],
		"weapon": weapon_action
	}


func apply_manual_action() -> Dictionary:
	return apply_action({
		"movement": [
			Input.get_axis("move_back", "move_forward"),
			Input.get_axis("turn_left", "turn_right")
		],
		"weapon": 1 if Input.is_action_just_pressed("fire") else 0
	})


func hold_fire() -> void:
	pass


func request_fire() -> void:
	if not try_fire():
		agent.add_reward_event("invalid_fire", 1.0)


func try_fire() -> bool:
	if not _alive or _cooldown_left > 0.0 or projectile_scene == null:
		return false

	var projectile := projectile_scene.instantiate()
	var parent := projectile_parent if projectile_parent != null else get_tree().current_scene
	parent.add_child(projectile)
	projectile.global_transform = muzzle.global_transform
	projectile.launch(self)
	_cooldown_left = fire_cooldown
	return true


func take_damage(attacker:BattleTank, amount:float) -> void:
	if not _alive:
		return
	_health = maxf(_health - amount, 0.0)
	damage_received.emit(self, attacker, amount)
	if _health > 0.0:
		return

	_alive = false
	visible = false
	collision_layer = 0
	collision_mask = 0
	clear_inputs()
	velocity = Vector3.ZERO
	destroyed.emit(self, attacker)


func get_team_id() -> int:
	return team_id


func get_control_input(input_name:String) -> float:
	if input_name == "throttle_input":
		return _throttle_input
	if input_name == "rotation_input":
		return _rotation_input
	return 0.0


func get_health_observation() -> float:
	return clampf(_health / maxf(max_health, 0.001), 0.0, 1.0)


func get_reload_ready_observation() -> float:
	return 1.0 if _cooldown_left <= 0.0 else 0.0


func get_alive_observation() -> float:
	return 1.0 if _alive else 0.0


func is_alive() -> bool:
	return _alive


func is_terminal() -> bool:
	return false


func clear_inputs() -> void:
	_throttle_input = 0.0
	_rotation_input = 0.0


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform
	_health = max_health
	_alive = true
	_cooldown_left = 0.0
	visible = true
	collision_layer = _initial_collision_layer
	collision_mask = _initial_collision_mask
	clear_inputs()
	velocity = Vector3.ZERO
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
		clear_inputs()
		velocity = Vector3.ZERO


func _zero_action() -> Dictionary:
	return {"movement": [0.0, 0.0], "weapon": 0}
