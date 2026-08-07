class_name URDFUtils


static func normalize_file_path(
		path: String,
		relative_to: String = "res://") -> String:
	var clean_path := path.strip_edges().replace("\\", "/")
	if clean_path.is_empty():
		return ""
	if clean_path.begins_with("uid://"):
		var resource_id := ResourceUID.text_to_id(clean_path)
		clean_path = ResourceUID.get_id_path(resource_id)
	if clean_path.begins_with("res://") or clean_path.begins_with("user://"):
		return clean_path.simplify_path()
	if clean_path.is_absolute_path():
		var localized := ProjectSettings.localize_path(clean_path)
		if localized.begins_with("res://"):
			return localized.simplify_path()
		return clean_path.simplify_path()

	var base_path := relative_to.strip_edges().replace("\\", "/")
	if base_path.is_empty():
		base_path = "res://"
	if not (
		base_path.begins_with("res://")
		or base_path.begins_with("user://")
		or base_path.is_absolute_path()
	):
		base_path = "res://".path_join(base_path)
	return base_path.path_join(clean_path).simplify_path()


static func resolve_reference_path(
		reference_path: String,
		source_path: String,
		package_folder: String = "") -> String:
	var clean_reference := reference_path.strip_edges().replace("\\", "/")
	var normalized_source := normalize_file_path(source_path)
	if clean_reference.begins_with("package://"):
		var package_relative := clean_reference.trim_prefix("package://")
		var package_root := normalize_file_path(package_folder)
		if package_root.is_empty():
			# A common project layout is <package-root>/<package>/<robot.urdf>.
			package_root = normalized_source.get_base_dir().get_base_dir()
		return normalize_file_path(package_relative, package_root)
	if (
		clean_reference.begins_with("res://")
		or clean_reference.begins_with("user://")
		or clean_reference.begins_with("uid://")
		or clean_reference.is_absolute_path()
	):
		return normalize_file_path(clean_reference)
	return normalize_file_path(clean_reference, normalized_source.get_base_dir())


static func parse_xyz(xyz: String) -> Vector3:
	# URDF attributes such as origin/xyz are optional and default to zero.
	if xyz.is_empty():
		return Vector3.ZERO
	var xyz_split = xyz.split(" ", false)
	if xyz_split.size() < 3:
		push_error("not enough values for XYZ!")
		return Vector3(0, 0, 0)
	return Vector3(
		float(xyz_split[0]),
		float(xyz_split[2]),
		-float(xyz_split[1]))

static func parse_rpy(rpy: String) -> Vector3:
	if rpy.is_empty():
		return Vector3.ZERO
	var rpy_split = rpy.split(" ", false)
	if rpy_split.size() < 3:
		# throw error?
		push_error("not enough values for RPY: " + rpy)
		return Vector3.ZERO
	return Vector3(
			float(rpy_split[0]),
			float(rpy_split[2]),
			-float(rpy_split[1]))

static func xyz_rpy_to_transform3d(xyz: Vector3, rpy: Vector3) -> Transform3D:
	# Convert XYZ RPY vector to 3D transform
	var basis = Basis.from_euler(rpy, EULER_ORDER_XYZ)
	return Transform3D(basis, xyz)
