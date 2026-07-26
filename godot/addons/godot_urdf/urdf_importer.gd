@tool
class_name GodotURDFImporter
extends EditorImportPlugin

func _get_importer_name() -> String:
	return "godot_urdf"
	
func _get_visible_name() -> String:
	return "Godot URDF"
	
func _get_recognized_extensions() -> PackedStringArray:
	return ["urdf"]

func _get_save_extension() -> String:
	return "tscn"
	
func _get_import_options(path: String, _preset_index: int) -> Array[Dictionary]:
	var source_directory := URDFUtils.normalize_file_path(path).get_base_dir()
	var default_package_folder := source_directory.get_base_dir()
	return [
		{
			"name": "package_folder",
			"default_value": default_package_folder,
			"property_hint": PROPERTY_HINT_DIR,
			"hint_string": ""
		},
		{
			"name": "scale",
			"default_value": 0.001,
			"property_hint": PROPERTY_HINT_NONE,
			"hint_string": ""
		},
	]
	
func _get_import_order() -> int:
	return 0
	
func _get_resource_type() -> String:
	return "PackedScene"
	
func _get_preset_count() -> int:
	return 1
	
func _get_preset_name(_preset_index: int) -> String:
	return "Default preset"
	
func _get_option_visibility(_path: String, _option_name: StringName, _options: Dictionary) -> bool:
	return true
	
func _get_priority() -> float:
	return 1.0

func _import(
		source_file: String, save_path: String, options: Dictionary,
		_platform_variants: Array[String], _gen_files: Array[String]) -> Error:
	var scene = PackedScene.new()
	var urdf_parser = URDFXMLParser.new()
	var normalized_source := URDFUtils.normalize_file_path(source_file)
	var normalized_options := options.duplicate()
	normalized_options["package_folder"] = URDFUtils.normalize_file_path(
		str(options.get("package_folder", "")))
	var robot_node = urdf_parser.as_node3d(
		normalized_source, normalized_options, null, null)
	if not robot_node:
		push_error("Failed to generate robot from URDF: ", normalized_source)
		return ERR_PARSE_ERROR
	scene.pack(robot_node)
	var saved_path = save_path + "." + _get_save_extension()
	# Save the packed scene to the target path
	var save_result = ResourceSaver.save(scene, saved_path)
	if save_result != OK:
		push_error("Failed to save imported .urdf as a scene.")
		return ERR_CANT_CREATE
	return OK
