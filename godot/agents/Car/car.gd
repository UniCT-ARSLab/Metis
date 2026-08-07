extends CharacterBody3D
class_name Car

@export_category("Driving")
@export var acceleration := 35.0
@export var brake_strength := 8.0
@export var steering_speed_degrees := 100.0
@export var min_speed_for_steering := 0.5
@export var friction := 1.6
@export var max_speed := 20.0
@export var manual_control := false

@export_category("RL Contract")
@export var path_aware_observations := false

@export_category("Episode")
@export var crash_floor_normal_threshold := 0.5
@export var crash_obstacle_group := "car_crash_obstacle"

@export_category("Presentation")
@export var auto_manage_camera := false
@export var update_status_text := true

@export_category("Training Optimization")
@export var auto_optimize_in_headless := true

@onready var agent: Agent = $Agent
@onready var camera: Camera3D = $Camera3D
@onready var status_text: Label3D = $StatusText
@onready var debug_bars: Sprite3D = $DebugBars
@onready var bars_viewport: SubViewport = $Bars
@onready var speed_bar: ProgressBar = $Bars/SpeedBar
@onready var right_steer_bar: ProgressBar = $Bars/RightSteerBar
@onready var left_steer_bar: ProgressBar = $Bars/LeftSteerBar

var _move_input := 0.0
var _rotate_input := 0.0
var _crashed := false
var _training_active := true


func _enter_tree() -> void:
	_configure_path_observations()


func _ready() -> void:
	if auto_optimize_in_headless and _is_headless():
		set_training_optimized(true)


func _physics_process(delta:float) -> void:
	if not _training_active:
		return

	if not _crashed:
		if manual_control:
			_move_input = Input.get_axis("move_back", "move_forward")
			_rotate_input = Input.get_axis("turn_left", "turn_right")

		if not is_on_floor():
			velocity += get_gravity() * delta

		var acceleration_input := clampf(_move_input, -1.0, 1.0)
		var forward_speed := absf(get_signed_forward_speed())
		var steering_authority := clampf(
			forward_speed / maxf(min_speed_for_steering * 4.0, 0.001),
			0.0,
			1.0
		)

		rotate_y(-_rotate_input * steering_authority * deg_to_rad(steering_speed_degrees) * delta)
		if acceleration_input >= 0.0:
			velocity += get_forward_direction() * acceleration_input * acceleration * delta
		else:
			velocity = velocity.lerp(Vector3.ZERO, minf(absf(acceleration_input) * brake_strength * delta, 1.0))

		velocity = velocity.lerp(Vector3.ZERO, minf(friction * delta, 1.0))
		velocity = velocity.limit_length(max_speed)
		move_and_slide()

	_update_crash_state()
	if not _is_headless():
		_update_status_text("CRASHED %s | Speed %.2f | Steering %.2f" % [_crashed, velocity.length(), _rotate_input])
		_update_bars(velocity.length(), _rotate_input)


func apply_action(action:Variant) -> Variant:
	if typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME:
		if str(action) == "manual":
			return apply_manual_action()
	return apply_continuous_action(action)


func apply_continuous_action(action:Variant) -> Array:
	var values := agent.decode_continuous_action(action)
	_move_input = clampf(float(values[0]), -1.0, 1.0) if values.size() > 0 else 0.0
	_rotate_input = clampf(float(values[1]), -1.0, 1.0) if values.size() > 1 else 0.0
	return [_move_input, _rotate_input]


func apply_manual_action() -> Array:
	return apply_continuous_action([
		Input.get_axis("move_back", "move_forward"),
		Input.get_axis("turn_left", "turn_right")
	])


func reset_all(original_transform:Variant, reset_rewards := true) -> void:
	set_training_active(true)
	_crashed = false
	_update_status_text("OK")
	set_camera_current(false)
	if typeof(original_transform) == TYPE_TRANSFORM3D:
		transform = original_transform

	clear_inputs()
	velocity = Vector3.ZERO
	if has_method("reset_physics_interpolation"):
		reset_physics_interpolation()
	agent.reset_observation_sources()
	if reset_rewards:
		agent.refresh_observation_sources()
		agent.reset_reward({"body": self})


func clear_inputs() -> void:
	_move_input = 0.0
	_rotate_input = 0.0


func get_control_input(input_name:String) -> float:
	if input_name == "throttle_input" or input_name == "move_input" or input_name == "drive_input":
		return _move_input
	if input_name == "rotation_input" or input_name == "steering_input":
		return _rotate_input
	return 0.0


func set_path_aware_observations(enabled:bool) -> void:
	path_aware_observations = enabled
	_configure_path_observations()


func is_terminal() -> bool:
	return _crashed


func is_crashed() -> bool:
	return _crashed


func get_forward_direction() -> Vector3:
	var forward := global_transform.basis.z
	forward.y = 0.0
	return forward.normalized() if forward.length() > 0.000001 else Vector3.ZERO


func get_signed_forward_speed() -> float:
	var flat_velocity := Vector3(velocity.x, 0.0, velocity.z)
	return flat_velocity.dot(get_forward_direction())


func set_camera_current(enabled:bool) -> void:
	if camera == null:
		return
	if enabled:
		camera.make_current()
	else:
		camera.current = false


func set_training_active(enabled:bool) -> void:
	_training_active = enabled
	set_physics_process(enabled)
	if not enabled:
		clear_inputs()
		velocity = Vector3.ZERO
		set_camera_current(false)
		agent.reset_observation_sources()


func set_training_optimized(enabled:bool) -> void:
	if not enabled:
		return
	update_status_text = false
	auto_manage_camera = false
	if status_text != null:
		status_text.visible = false
	if debug_bars != null:
		debug_bars.visible = false
	if bars_viewport != null:
		bars_viewport.process_mode = Node.PROCESS_MODE_DISABLED
	if camera != null:
		camera.current = false
		camera.process_mode = Node.PROCESS_MODE_DISABLED


func _configure_path_observations() -> void:
	var source := get_node_or_null("Agent/ObservationSystem/PathNavigation")
	if source != null:
		source.set("enabled", path_aware_observations)


func _update_crash_state() -> void:
	if _crashed:
		return
	for collision_idx in range(get_slide_collision_count()):
		var collision := get_slide_collision(collision_idx)
		if collision == null:
			continue
		var collider := collision.get_collider()
		var explicit_obstacle := (
			collider is Node
			and not crash_obstacle_group.is_empty()
			and _node_or_parent_is_in_group(collider, crash_obstacle_group)
		)
		var non_floor_contact := collision.get_normal().dot(Vector3.UP) < crash_floor_normal_threshold
		if not explicit_obstacle and not non_floor_contact:
			continue
		_crashed = true
		if auto_manage_camera:
			set_camera_current(false)
		agent.add_reward_event("collision", 1.0)
		return


func _node_or_parent_is_in_group(node:Node, group_name:String) -> bool:
	var current:Node = node
	while current != null:
		if current.is_in_group(group_name):
			return true
		current = current.get_parent()
	return false


func _update_status_text(text:String) -> void:
	if update_status_text and status_text != null:
		status_text.text = text


func _update_bars(speed:float, steering:float) -> void:
	speed_bar.value = speed
	right_steer_bar.value = maxf(steering, 0.0)
	left_steer_bar.value = maxf(-steering, 0.0)


func _is_headless() -> bool:
	return DisplayServer.get_name().to_lower() == "headless" or OS.has_feature("headless")
