@tool
extends Node3D
# Script you can attach to your node to load a robots elements as child
# making it easier to modify/extend the robot

@export_group("URDF File")
@export_file("*.urdf", "*.xml") var urdf_file_path: String

@export_tool_button("Reload robot", "Godot") var _load_urdf_button = _load_urdf

@export_group("Package Directory")
# Change "package://robot_description/meshes/..." to "res://urdf/..."
@export_dir var package_folder: String = "res://"

@export_group("Transform")
@export var _position: Vector3 = Vector3(0, 0, 0)
@export var _rotation: Vector3 = Vector3(0, 0, 0)
@export var _scale: float = 0.001

func _load_urdf():
	for child in get_children():
		if child is GodotRobot:
			child.free()

	if urdf_file_path.is_empty():
		push_error("No URDF file selected!")
		return

	var parser = URDFXMLParser.new()

	var options = {
		"package_folder": package_folder,
		"scale": _scale,
		"create_physics": true
	}

	var _root = get_tree().edited_scene_root

	var robot_node = parser.as_node3d(
		urdf_file_path, options, self, _root)

	if robot_node:
		robot_node.set_position(_position)
		robot_node.set_rotation(_rotation / 180 * PI)

		print("Robot loaded successfully!")
	else:
		push_error("Failed to load robot node.")
