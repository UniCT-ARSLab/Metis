extends ProgressProvider
class_name Path3DProgressProvider

@export var path:Path3D


func measure_progress(agent:Node, _context:Dictionary = {}) -> float:
	if path == null or path.curve == null or not agent is Node3D:
		return 0.0
	var total_length := path.curve.get_baked_length()
	if total_length <= 0.001:
		return 0.0
	var local_position := path.to_local((agent as Node3D).global_position)
	return path.curve.get_closest_offset(local_position) / total_length


func build_reset_transform(
	original_transform:Transform3D,
	rng:RandomNumberGenerator,
	options:Dictionary = {}
) -> Dictionary:
	if path == null or path.curve == null:
		return {}

	var total_length := path.curve.get_baked_length()
	if total_length <= 0.001:
		return {}

	var progress_min := clampf(float(options.get("progress_min", 0.0)), 0.0, 1.0)
	var progress_max := clampf(float(options.get("progress_max", progress_min)), 0.0, 1.0)
	if progress_min > progress_max:
		var swap := progress_min
		progress_min = progress_max
		progress_max = swap

	var progress := rng.randf_range(progress_min, progress_max)
	var offset := progress * total_length
	var local_position := path.curve.sample_baked(offset)
	var global_position := path.to_global(local_position)
	global_position.y = original_transform.origin.y

	var tangent := _tangent_at_offset(offset, total_length)
	var align_to_progress := bool(options.get("align_to_progress", true))
	var basis := Basis().rotated(Vector3.UP, _yaw_from_forward(tangent)) if align_to_progress else original_transform.basis
	var result_transform := Transform3D(basis, global_position)
	var lateral_jitter := maxf(float(options.get("lateral_jitter", 0.0)), 0.0)
	var lateral_offset := 0.0
	if lateral_jitter > 0.0:
		lateral_offset = rng.randf_range(-lateral_jitter, lateral_jitter)
		result_transform.origin += Vector3.UP.cross(tangent).normalized() * lateral_offset

	return {
		"transform": result_transform,
		"progress": progress,
		"lateral_offset": lateral_offset,
		"mode": "progress_provider"
	}


func _tangent_at_offset(offset:float, total_length:float) -> Vector3:
	var sample_delta := minf(0.5, maxf(total_length * 0.005, 0.05))
	var before := path.curve.sample_baked(clampf(offset - sample_delta, 0.0, total_length))
	var after := path.curve.sample_baked(clampf(offset + sample_delta, 0.0, total_length))
	var tangent := path.to_global(after) - path.to_global(before)
	tangent.y = 0.0
	return tangent.normalized() if tangent.length() > 0.000001 else Vector3.FORWARD


func _yaw_from_forward(forward:Vector3) -> float:
	var flat_forward := forward
	flat_forward.y = 0.0
	if flat_forward.length() <= 0.000001:
		return 0.0
	flat_forward = flat_forward.normalized()
	return atan2(flat_forward.x, flat_forward.z)
