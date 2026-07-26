extends SceneTree


func _initialize() -> void:
	var urdf_path := "res://assets/urdf/xarm/xarm_fixed.urdf"
	var absolute_urdf_path := ProjectSettings.globalize_path(urdf_path)
	var package_root := "res://assets/urdf"
	var absolute_package_root := ProjectSettings.globalize_path(package_root)
	var expected_mesh := "res://assets/urdf/xarm/meshes/base.stl"

	var passed := URDFUtils.normalize_file_path(absolute_urdf_path) == urdf_path
	passed = passed and URDFUtils.normalize_file_path(
		"assets/urdf/xarm/xarm_fixed.urdf") == urdf_path
	passed = passed and URDFUtils.resolve_reference_path(
		"package://xarm/meshes/base.stl",
		urdf_path,
		package_root) == expected_mesh
	passed = passed and URDFUtils.resolve_reference_path(
		"package://xarm/meshes/base.stl",
		urdf_path,
		absolute_package_root) == expected_mesh
	passed = passed and URDFUtils.resolve_reference_path(
		"meshes/base.stl",
		urdf_path) == expected_mesh

	var loader_script := load("res://addons/godot_urdf/urdf_loader.gd") as Script
	var loader := loader_script.new() as Node3D
	loader.urdf_file_path = absolute_urdf_path
	loader.package_folder = absolute_package_root
	passed = passed and loader.urdf_file_path == urdf_path
	passed = passed and loader.package_folder == package_root
	loader.free()

	var parser := URDFXMLParser.new()
	passed = passed and parser.parse(urdf_path) != null
	passed = passed and parser.parse(absolute_urdf_path) != null

	if not passed:
		push_error("URDF path test failed.")
		quit(1)
		return

	print(
		"URDF path test passed: source=%s package=%s mesh=%s" %
		[urdf_path, package_root, expected_mesh])
	quit(0)
