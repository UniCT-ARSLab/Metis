extends CharacterBody3D 
class_name Tank

@export_category("Tank Information")
@export var move_speed := 20.0
@export var turn_speed := 2.0
@export var gravity := 20.0
@export var manualControl:bool = true
@export var observation_position_scale := 10.0
@export var observation_speed_scale := 20.0

var _move_input = 0.0
var _turn_input = 0.0
var _raycasts: Array[RayCast3D] = []

@onready var agent:Agent = $Agent

func _ready() -> void:
	agent.add_action("idle", Callable(self, "clear_inputs"))
	agent.add_action("move_forward", Callable(self, "move_forward"))
	agent.add_action("move_backward", Callable(self, "move_backward"))
	agent.add_action("turn_right", Callable(self, "turn_right"))
	agent.add_action("turn_left", Callable(self, "turn_left"))
	agent.add_action("forward_right", Callable(self, "forward_right"))
	agent.add_action("forward_left", Callable(self, "forward_left"))
	agent.add_action("backward_right", Callable(self, "backward_right"))
	agent.add_action("backward_left", Callable(self, "backward_left"))
	
	_register_observations()

func _physics_process(delta):
	if not is_on_floor():
		velocity.y -= gravity * delta

	var y_velocity := velocity.y
	
	if manualControl:
		manual_control()

	velocity = global_transform.basis.z * _move_input * move_speed * delta
	velocity.y = y_velocity

	rotate_y(_turn_input * turn_speed * delta)
	move_and_slide()


func apply_action(action:Variant) -> int:
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


func apply_manual_action() -> int:
	var action_id := get_manual_action_id()
	clear_inputs()
	agent.act(action_id)
	return action_id


func get_manual_action_id() -> int:
	var move_axis := Input.get_axis("move_back", "move_forward")
	var turn_axis := Input.get_axis("turn_left", "turn_right")
	var deadzone := 0.2
	var move := 0
	var turn := 0

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


func get_observation_vector() -> Array:
	return agent.get_observation_vector()


func get_observation_size() -> int:
	return agent.get_observation_size()


func get_observations() -> Dictionary:
	return agent.get_observations()


func get_action_count() -> int:
	return agent.get_action_count()


func get_action_names() -> Array:
	return agent.get_action_names()


func reset_reward() -> void:
	agent.reset_reward(_build_reward_context())


func add_reward_event(term:String, value:float) -> void:
	agent.add_reward_event(term, value)


func get_reward() -> float:
	return agent.get_reward(_build_reward_context())


func get_reward_terms() -> Dictionary:
	return agent.get_reward_terms()


func clear_inputs() -> void:
	_move_input = 0.0
	_turn_input = 0.0

	
func move_forward() -> void:
	_move_input = 1
	

func move_backward() -> void:
	_move_input = -1


func turn_right() -> void:
	_turn_input = -1


func turn_left() -> void:
	_turn_input = 1


func forward_right() -> void:
	_move_input = 1
	_turn_input = -1


func forward_left() -> void:
	_move_input = 1
	_turn_input = 1


func backward_right() -> void:
	_move_input = -1
	_turn_input = -1


func backward_left() -> void:
	_move_input = -1
	_turn_input = 1


func manual_control() -> void:
	if manualControl:
		_move_input = Input.get_axis("move_back", "move_forward")
		_turn_input = -Input.get_axis("turn_left", "turn_right")

func reset_all(original_position:Transform3D, reset_rewards := true):
	if original_position!= null:
		transform = original_position
	clear_inputs()
	velocity = Vector3.ZERO
	if has_method("reset_physics_interpolation"):
		reset_physics_interpolation()
	reset_raycast_state()
	if reset_rewards:
		refresh_sensors()
		reset_reward()

func reset_raycast_state() -> void:
	for raycast in _raycasts:
		if is_instance_valid(raycast):
			raycast.clear_exceptions()
			raycast.enabled = false

func refresh_sensors() -> void:
	for raycast in _raycasts:
		if is_instance_valid(raycast):
			raycast.clear_exceptions()
			raycast.enabled = true
			raycast.force_raycast_update()

func _register_observations() -> void:
	_find_raycasts(self, _raycasts)

	agent.add_observation("position", Callable(self, "_get_position_observation"))
	agent.add_observation("forward", Callable(self, "_get_forward_observation"))
	agent.add_observation("local_velocity", Callable(self, "_get_local_velocity_observation"))
	agent.add_observation("move_input", Callable(self, "_get_move_input"))
	agent.add_observation("turn_input", Callable(self, "_get_turn_input"))
	
	agent.add_observation("target_visible", Callable(self, "_get_target_visible"))
	agent.add_observation("target_signal_left", Callable(self, "_get_target_signal").bind($ViewSensor/LineOfViewSensorLeft))
	agent.add_observation("target_signal_center", Callable(self, "_get_target_signal").bind($ViewSensor/LineOfViewSensor))
	agent.add_observation("target_signal_right", Callable(self, "_get_target_signal").bind($ViewSensor/LineOfViewSensorRight))

	var used_names := {}
	for raycast in _raycasts:
		raycast.enabled = true
		var sensor_name := _make_unique_sensor_name(_get_sensor_base_name(raycast), used_names)
		agent.add_observation("ray_%s" % sensor_name, Callable(self, "_get_raycast_distance_observation").bind(raycast))

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

func _action_id_from_name(action_name:String) -> int:
	var action_names := agent.get_action_names()
	for idx in range(action_names.size()):
		if str(action_names[idx]) == action_name:
			return idx
	return -1

func _build_reward_context() -> Dictionary:
	return {
		"body": self,
		"observations": get_observations()
	}


func _get_position_observation() -> Vector3:
	return global_position / max(observation_position_scale, 0.001)


func _get_forward_observation() -> Vector3:
	return global_transform.basis.z.normalized()


func _get_local_velocity_observation() -> Vector3:
	var local_velocity := global_transform.basis.inverse() * velocity
	return local_velocity / max(observation_speed_scale, 0.001)


func _get_move_input() -> float:
	return _move_input


func _get_turn_input() -> float:
	return _turn_input


func _get_raycast_distance_observation(raycast:RayCast3D) -> float:
	if not is_instance_valid(raycast):
		return 1.0

	raycast.force_raycast_update()

	var max_distance := raycast.target_position.length()
	if max_distance <= 0.001:
		return 1.0

	if not raycast.is_colliding():
		return 1.0

	var hit_distance := raycast.global_position.distance_to(raycast.get_collision_point())
	return clamp(hit_distance / max_distance, 0.0, 1.0)

func _raycast_hits_target(raycast: RayCast3D) -> bool:
	raycast.force_raycast_update()
	
	if not raycast.is_colliding():
		return false
	
	var collider = raycast.get_collider()
	if collider == null:
		return false
		
	return _node_or_parent_is_in_group(collider, "target")
	
func _get_target_signal(raycast: RayCast3D) -> float:
	if not _raycast_hits_target(raycast):
		return 0.0
	var max_distance = raycast.target_position.length()
	var hit_distance = raycast.global_position.distance_to(raycast.get_collision_point())
	var normalized_distance = clamp(hit_distance / max_distance, 0.0, 1.0)
	return 1.0 - normalized_distance
	
func _get_target_visible() -> float:
	if _raycast_hits_target($ViewSensor/LineOfViewSensorLeft):
		return 1.0
	if _raycast_hits_target($ViewSensor/LineOfViewSensor):
		return 1.0
	if _raycast_hits_target($ViewSensor/LineOfViewSensorRight):
		return 1.0
	return 0.0

func _node_or_parent_is_in_group(node: Node, group_name: String) -> bool:
	var current = node
	while current != null:
		if current.is_in_group(group_name):
			return true
		current = current.get_parent()
	return false
