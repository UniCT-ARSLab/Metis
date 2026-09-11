extends ObservationSource
class_name Path3DNavigationObservationSource

@export var path: Path3D
@export var path_group := "navigation_path"
@export var progress_observation_name := "path_progress"
@export var lateral_observation_name := "path_lateral_offset"
@export var heading_observation_name := "path_heading"
@export var lookahead_observation_prefix := "path_lookahead"
@export var lookahead_distances := PackedFloat32Array([5.0, 12.0, 25.0])
@export var lateral_scale := 6.0
@export var loop_path := false

var _body: Node3D


func register_observations(agent:Agent, body:Node) -> void:
	_body = body as Node3D
	if path == null and not path_group.is_empty():
		path = _find_path_in_group(path_group)
	if _body == null or path == null or path.curve == null:
		push_warning("Path3D navigation observations require a Node3D body and a Path3D")
		return

	if not progress_observation_name.is_empty():
		agent.add_observation(progress_observation_name, Callable(self, "_get_progress"))
	if not lateral_observation_name.is_empty():
		agent.add_observation(lateral_observation_name, Callable(self, "_get_lateral_offset"))
	if not heading_observation_name.is_empty():
		agent.add_observation(heading_observation_name, Callable(self, "_get_heading"))
	if not lookahead_observation_prefix.is_empty():
		for idx in range(lookahead_distances.size()):
			agent.add_observation(
				"%s_%d" % [lookahead_observation_prefix, idx],
				Callable(self, "_get_lookahead").bind(float(lookahead_distances[idx]))
			)


func _get_progress() -> float:
	var total_length := _path_length()
	return _closest_offset() / total_length if total_length > 0.001 else 0.0


func _get_lateral_offset() -> float:
	var total_length := _path_length()
	if total_length <= 0.001:
		return 0.0
	var offset := _closest_offset()
	var closest_global := path.to_global(path.curve.sample_baked(offset))
	var tangent := _tangent_at_offset(offset, total_length)
	var right := Vector3.UP.cross(tangent).normalized()
	var signed_offset := (_body.global_position - closest_global).dot(right)
	return clampf(signed_offset / maxf(lateral_scale, 0.001), -1.0, 1.0)


func _get_heading() -> Vector2:
	var total_length := _path_length()
	if total_length <= 0.001:
		return Vector2.ZERO
	var tangent := _tangent_at_offset(_closest_offset(), total_length)
	var local_tangent: Vector3 = _body.global_transform.basis.inverse() * tangent
	var flat := Vector2(local_tangent.x, local_tangent.z)
	return flat.normalized() if flat.length() > 0.000001 else Vector2.ZERO


func _get_lookahead(distance:float) -> Vector2:
	var total_length := _path_length()
	if total_length <= 0.001:
		return Vector2.ZERO
	var target_offset := _closest_offset() + maxf(distance, 0.0)
	if loop_path:
		target_offset = fposmod(target_offset, total_length)
	else:
		target_offset = clampf(target_offset, 0.0, total_length)
	var target_global := path.to_global(path.curve.sample_baked(target_offset))
	var local_delta: Vector3 = _body.global_transform.basis.inverse() * (target_global - _body.global_position)
	var flat := Vector2(local_delta.x, local_delta.z)
	return flat.normalized() if flat.length() > 0.000001 else Vector2.ZERO


func _closest_offset() -> float:
	if path == null or path.curve == null or _body == null:
		return 0.0
	return path.curve.get_closest_offset(path.to_local(_body.global_position))


func _path_length() -> float:
	return path.curve.get_baked_length() if path != null and path.curve != null else 0.0


func _tangent_at_offset(offset:float, total_length:float) -> Vector3:
	var sample_delta := minf(0.5, maxf(total_length * 0.005, 0.05))
	var before_offset := offset - sample_delta
	var after_offset := offset + sample_delta
	if loop_path:
		before_offset = fposmod(before_offset, total_length)
		after_offset = fposmod(after_offset, total_length)
	else:
		before_offset = clampf(before_offset, 0.0, total_length)
		after_offset = clampf(after_offset, 0.0, total_length)
	var before := path.to_global(path.curve.sample_baked(before_offset))
	var after := path.to_global(path.curve.sample_baked(after_offset))
	var tangent := after - before
	tangent.y = 0.0
	return tangent.normalized() if tangent.length() > 0.000001 else Vector3.FORWARD


func _find_path_in_group(group_name:String) -> Path3D:
	var tree := get_tree()
	if tree == null:
		return null
	for node in tree.get_nodes_in_group(group_name):
		if node is Path3D:
			return node
	return null
