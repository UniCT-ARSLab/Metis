@tool
extends EditorPlugin

var godot_urdf = GodotURDFImporter.new()
var dock: VBoxContainer

func _enter_tree() -> void:
	add_import_plugin(godot_urdf)

	dock = preload("res://addons/godot_urdf/urdf_dock.gd").new()
	add_control_to_dock(DOCK_SLOT_RIGHT_UR, dock)

	EditorInterface.get_selection().selection_changed.connect(_on_selection_changed)

	# Check selection in case a GodotRobot is already selected
	_on_selection_changed()

func _exit_tree() -> void:
	remove_import_plugin(godot_urdf)

	EditorInterface.get_selection().selection_changed.disconnect(_on_selection_changed)
	remove_control_from_docks(dock)
	if dock:
		dock.free()

func _on_selection_changed():
	var selected_nodes = EditorInterface.get_selection().get_selected_nodes()
	
	var active_robot: GodotRobot = null
	
	if selected_nodes.size() > 0:
		var current_node = selected_nodes[0]
		
		# Walk up the scene tree to find a GodotRobot parent
		while current_node != null:
			if current_node is GodotRobot:
				active_robot = current_node
				break
			current_node = current_node.get_parent()

	if active_robot:
		if active_robot.urdf != null:
			dock.load_robot(active_robot.urdf)
			dock.set_ui_state("active")
		else:
			dock.set_ui_state("missing_urdf")
	else:
		dock.set_ui_state("hidden")
