class_name URDFUtils

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
