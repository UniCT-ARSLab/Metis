extends CharacterBody3D
class_name Car


@export var acceleration = 100.0
@export var steering_speed = 60
@export var friction = 4.0
@export var max_speed = 800.0
@export var manual_control:bool = true
@export var crash_floor_normal_threshold := 0.5
@export var update_raycast_debug_colors := true
@export var raycast_clear_debug_color := Color(0.0, 1.0, 0.0, 1.0)
@export var raycast_hit_debug_color := Color(1.0, 0.0, 0.0, 1.0)
@export var auto_manage_camera := false

@onready var camera: Camera3D = $Camera3D


var _rotate_input = 0.0
var _move_input = 0.0
var _raycasts: Array[RayCast3D] = []
var _crashed := false
@onready var status_text: Label3D = $StatusText

@onready var agent: Agent = $Agent

var _total_distance = 0.0
var _total_time = 0.0

func _ready() -> void:
	_register_observations()

func _physics_process(delta):
	
	if not _crashed:
	
		if manual_control:
			_move_input = Input.get_axis("move_back", "move_forward") 
			_rotate_input = Input.get_axis("turn_left", "turn_right")
			
		if not is_on_floor():
			velocity += get_gravity() * delta

		var steer_direction = -_rotate_input * min(velocity.length() / max_speed, 1.0)
		rotate_y(steer_direction * steering_speed * delta)
		
		var forward_dir = transform.basis.z.normalized() 
		velocity += forward_dir * _move_input * acceleration * delta
		velocity = velocity.lerp(Vector3.ZERO, friction * delta)
		velocity = velocity.limit_length(max_speed)
		_total_distance += velocity.length() * delta
		_total_time+= delta
		move_and_slide()
	_update_crash_state()
	status_text.text = "CRASHED %s | Reward %f" % [_crashed, get_reward()]

func reset_all(original_position:Transform3D, reset_rewards := true):
	_total_distance = 0.0
	_total_time = 0.0
	_crashed = false
	status_text.text = "OK"
	set_camera_current(false)
	
	if original_position!= null:
		transform = original_position
	clear_inputs()
	velocity = Vector3.ZERO
	if has_method("reset_physics_interpolation"):
		reset_physics_interpolation()
	reset_raycast_state()
	refresh_sensors()
	if reset_rewards:
		reset_reward()
		
func reset_reward() -> void:
	agent.reset_reward(_build_reward_context())

func apply_action(action:Variant) -> Variant:
	if _is_continuous_action(action):
		return apply_continuous_action(action)

	if typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME:
		var action_name := str(action)
		if action_name == "manual":
			return apply_manual_action()
		var named_action_id := _action_id_from_name(action_name)
		if named_action_id < 0:
			clear_inputs()
			return 0
		clear_inputs()
		agent.act(named_action_id)
		return named_action_id

	clear_inputs()
	var action_id := int(action)
	if agent.act(action_id) != OK:
		return 0
	return action_id

func apply_continuous_action(action:Variant) -> Array:
	var values: Array = _continuous_action_values(action)
	_move_input = clampf(float(values[0]), -1.0, 1.0)
	_rotate_input = clampf(float(values[1]), -1.0, 1.0)
	return [_move_input, _rotate_input]
	
func apply_manual_action() -> Array:
	var move_axis := Input.get_axis("move_back", "move_forward")
	var turn_axis := Input.get_axis("turn_left", "turn_right")
	return apply_continuous_action([move_axis, turn_axis])


func get_manual_action_id() -> int:
	var move_axis := Input.get_axis("move_back", "move_forward")
	var turn_axis := Input.get_axis("turn_left", "turn_right")
	var move := 0
	var turn := 0
	var deadzone = 0.2

	if move_axis > deadzone:
		move = 1
	elif move_axis < -deadzone:
		move = -1

	if turn_axis > deadzone:
		turn = 1
	elif turn_axis < -deadzone:
		turn = -1

	if move > 0 and turn > 0:
		return _action_id_from_name("forward_right")
	if move > 0 and turn < 0:
		return _action_id_from_name("forward_left")
	if move < 0 and turn > 0:
		return _action_id_from_name("backward_right")
	if move < 0 and turn < 0:
		return _action_id_from_name("backward_left")
	if move > 0:
		return _action_id_from_name("move_forward")
	if move < 0:
		return _action_id_from_name("move_backward")
	if turn > 0:
		return _action_id_from_name("turn_right")
	if turn < 0:
		return _action_id_from_name("turn_left")
	return _action_id_from_name("idle")


func _action_id_from_name(action_name:String) -> int:
	var action_names := agent.get_action_names()
	for idx in range(action_names.size()):
		if str(action_names[idx]) == action_name:
			return idx
	return -1
	
func _build_reward_context() -> Dictionary:
	return {
		"body": self,
		"observations": agent.get_observations()
	}

func clear_inputs() -> void:
	_move_input = 0.0
	_rotate_input = 0.0

func get_action_type() -> String:
	return "continuous"

func get_action_space() -> Dictionary:
	return {
		"move_input": {
			"size": 1,
			"action_type": "continuous",
			"low": -1.0,
			"high": 1.0
		},
		"rotation_input": {
			"size": 1,
			"action_type": "continuous",
			"low": -1.0,
			"high": 1.0
		}
	}

func get_action_size() -> int:
	return 2

func get_action_low() -> Array:
	return [-1.0, -1.0]

func get_action_high() -> Array:
	return [1.0, 1.0]

func get_action_count() -> int:
	return get_action_size()

func get_action_names() -> Array:
	return ["move_input", "rotation_input"]

func get_observation_vector() -> Array:
	return agent.get_observation_vector()

func get_observation_size() -> int:
	return agent.get_observation_size()

func get_observations() -> Dictionary:
	return agent.get_observations()

func get_reward() -> float:
	return agent.get_reward(_build_reward_context())

func get_reward_terms() -> Dictionary:
	return agent.get_reward_terms()

func is_terminal() -> bool:
	return _crashed

func is_crashed() -> bool:
	return _crashed

func get_total_distance() -> float:
	return _total_distance

func get_total_time() -> float:
	return _total_time

func get_forward_direction() -> Vector3:
	var forward := global_transform.basis.z
	forward.y = 0.0
	if forward.length() <= 0.000001:
		return Vector3.ZERO
	return forward.normalized()

func get_signed_forward_speed() -> float:
	var forward := get_forward_direction()
	if forward == Vector3.ZERO:
		return 0.0

	var flat_velocity := velocity
	flat_velocity.y = 0.0
	return flat_velocity.dot(forward)

func reset_raycast_state() -> void:
	for raycast in _raycasts:
		if is_instance_valid(raycast):
			raycast.clear_exceptions()
			_update_raycast_debug_color(raycast, false)
			raycast.enabled = false
			
func refresh_sensors() -> void:
	for raycast in _raycasts:
		if is_instance_valid(raycast):
			raycast.clear_exceptions()
			raycast.enabled = true
			raycast.force_raycast_update()
			_update_raycast_debug_color(raycast, raycast.is_colliding())
			
func _find_raycasts(node:Node, result:Array[RayCast3D]) -> void:
	for child in node.get_children():
		if child is RayCast3D:
			result.append(child)
		_find_raycasts(child, result)

func _make_unique_sensor_name(base_name:String, used_names:Dictionary) -> String:
	if base_name.is_empty():
		base_name = "sensor"

	if not used_names.has(base_name):
		used_names[base_name] = 1
		return base_name

	used_names[base_name] += 1
	return "%s_%d" % [base_name, used_names[base_name]]

func _get_sensor_base_name(raycast:RayCast3D) -> String:
	var parent_name := "sensor_group"
	if raycast.get_parent() != null:
		parent_name = str(raycast.get_parent().name)

	return "%s_%s" % [
		parent_name.to_lower().replace(" ", "_"),
		str(raycast.name).to_lower().replace(" ", "_")
	]

func _get_raycast_distance_observation(raycast:RayCast3D) -> float:
	if not is_instance_valid(raycast):
		return 1.0

	raycast.force_raycast_update()
	var is_hit := raycast.is_colliding()
	_update_raycast_debug_color(raycast, is_hit)

	var max_distance := raycast.target_position.length()
	if max_distance <= 0.001:
		return 1.0

	if not is_hit:
		return 1.0

	var hit_distance := raycast.global_position.distance_to(raycast.get_collision_point())
	return clamp(hit_distance / max_distance, 0.0, 1.0)

func _update_raycast_debug_color(raycast:RayCast3D, is_hit:bool) -> void:
	if not update_raycast_debug_colors:
		return
	if not is_instance_valid(raycast):
		return
	raycast.debug_shape_custom_color = raycast_hit_debug_color if is_hit else raycast_clear_debug_color

func _register_observations() -> void:
	_find_raycasts(self, _raycasts)

	var used_names := {}
	for raycast in _raycasts:
		raycast.enabled = true
		_update_raycast_debug_color(raycast, false)
		var sensor_name := _make_unique_sensor_name(_get_sensor_base_name(raycast), used_names)
		agent.add_observation("ray_%s" % sensor_name, Callable(self, "_get_raycast_distance_observation").bind(raycast))

func _update_crash_state() -> void:
	if _crashed:
		return

	for collision_idx in range(get_slide_collision_count()):
		var collision := get_slide_collision(collision_idx)
		if collision == null:
			continue
		var normal := collision.get_normal()
		if normal.dot(Vector3.UP) >= crash_floor_normal_threshold:
			continue
		_crashed = true
		if auto_manage_camera:
			set_camera_current(false)
		agent.add_reward_event("collision", 1.0)
		return

func set_camera_current(enabled:bool) -> void:
	if camera == null:
		return
	if enabled:
		camera.make_current()
	else:
		camera.current = false

func _is_continuous_action(action:Variant) -> bool:
	if typeof(action) == TYPE_ARRAY or typeof(action) == TYPE_PACKED_FLOAT32_ARRAY or typeof(action) == TYPE_PACKED_FLOAT64_ARRAY:
		return true
	if typeof(action) == TYPE_DICTIONARY:
		return action.has("move_input") or action.has("rotation_input")
	return false

func _continuous_action_values(action:Variant) -> Array:
	if typeof(action) == TYPE_DICTIONARY:
		var action_map: Dictionary = action
		return [
			_action_component_float(action_map.get("move_input", 0.0)),
			_action_component_float(action_map.get("rotation_input", 0.0))
		]

	if typeof(action) == TYPE_ARRAY or typeof(action) == TYPE_PACKED_FLOAT32_ARRAY or typeof(action) == TYPE_PACKED_FLOAT64_ARRAY:
		var values: Array = Array(action)
		var move_value := 0.0
		var rotation_value := 0.0
		if values.size() > 0:
			move_value = float(values[0])
		if values.size() > 1:
			rotation_value = float(values[1])
		return [move_value, rotation_value]

	return [0.0, 0.0]

func _action_component_float(value:Variant) -> float:
	if typeof(value) == TYPE_ARRAY or typeof(value) == TYPE_PACKED_FLOAT32_ARRAY or typeof(value) == TYPE_PACKED_FLOAT64_ARRAY:
		var values: Array = Array(value)
		if values.is_empty():
			return 0.0
		return float(values[0])
	return float(value)
