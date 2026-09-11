extends ObservationSource
class_name RaycastObservationSource

@export var root_path: NodePath = NodePath("../..")
@export var observation_prefix := "ray"
@export var update_debug_colors := true
@export var auto_disable_debug_in_headless := true
@export var clear_debug_color := Color(0.0, 1.0, 0.0, 1.0)
@export var hit_debug_color := Color(1.0, 0.0, 0.0, 1.0)

var _raycasts: Array[RayCast3D] = []


func register_observations(agent:Agent, body:Node) -> void:
	if auto_disable_debug_in_headless and _is_headless():
		update_debug_colors = false

	var root := get_node_or_null(root_path)
	if root == null:
		root = body

	_raycasts.clear()
	_find_raycasts(root, _raycasts)

	var used_names := {}
	for raycast in _raycasts:
		raycast.enabled = true
		_update_raycast_debug_color(raycast, false)
		var sensor_name := _make_unique_sensor_name(_get_sensor_base_name(raycast), used_names)
		agent.add_observation("%s_%s" % [observation_prefix, sensor_name], Callable(self, "_get_raycast_distance_observation").bind(raycast))


func reset_source() -> void:
	for raycast in _raycasts:
		if is_instance_valid(raycast):
			raycast.clear_exceptions()
			_update_raycast_debug_color(raycast, false)
			raycast.enabled = false


func refresh_source() -> void:
	for raycast in _raycasts:
		if is_instance_valid(raycast):
			raycast.clear_exceptions()
			raycast.enabled = true
			raycast.force_raycast_update()
			_update_raycast_debug_color(raycast, raycast.is_colliding())


func get_forward_clearance(body:Node, dot_threshold:float = 0.86) -> float:
	if not (body is Node3D):
		return 1.0

	var forward: Vector3 = body.global_transform.basis.z
	forward.y = 0.0
	if forward.length() <= 0.000001:
		return 1.0
	forward = forward.normalized()

	var best_clearance := 1.0
	var found_forward_sensor := false
	for raycast in _raycasts:
		if not is_instance_valid(raycast):
			continue

		var ray_direction: Vector3 = raycast.global_transform.basis * raycast.target_position
		ray_direction.y = 0.0
		if ray_direction.length() <= 0.000001:
			continue
		ray_direction = ray_direction.normalized()
		if forward.dot(ray_direction) < dot_threshold:
			continue

		found_forward_sensor = true
		best_clearance = minf(best_clearance, _get_raycast_distance_observation(raycast))

	return best_clearance if found_forward_sensor else 1.0


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
	return clampf(hit_distance / max_distance, 0.0, 1.0)


func _update_raycast_debug_color(raycast:RayCast3D, is_hit:bool) -> void:
	if not update_debug_colors or not is_instance_valid(raycast):
		return
	raycast.debug_shape_custom_color = hit_debug_color if is_hit else clear_debug_color


func _is_headless() -> bool:
	return DisplayServer.get_name().to_lower() == "headless" or OS.has_feature("headless")
