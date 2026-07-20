@tool
extends VBoxContainer
class_name URDFDock

var main_content: VBoxContainer
var message_label: Label
var tree: Tree

var edit_panel: VBoxContainer
var title_label: Label
var joint_type_container: HBoxContainer
var joint_type_dropdown: OptionButton

var current_robot: URDFRobot
var selected_item_data: Resource 

func _init():
	name = "URDF Tree"
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	
	message_label = Label.new()
	message_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	message_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	message_label.size_flags_vertical = Control.SIZE_EXPAND_FILL
	message_label.autowrap_mode = TextServer.AUTOWRAP_WORD
	add_child(message_label)
	
	main_content = VBoxContainer.new()
	main_content.size_flags_vertical = Control.SIZE_EXPAND_FILL
	main_content.hide()
	add_child(main_content)
	
	tree = Tree.new()
	tree.size_flags_vertical = Control.SIZE_EXPAND_FILL
	tree.item_selected.connect(_on_tree_item_selected)
	main_content.add_child(tree)
	
func _show_message(text: String):
	# instead of the main content show the message
	main_content.hide()
	message_label.text = text
	message_label.show()
	tree.clear()
	current_robot = null

# Manage visibility of the UI
func set_ui_state(state: String):
	if state == "active":
		message_label.hide()
		main_content.show()
	elif state == "missing_urdf":
		_show_message("GodotRobot selected, but its 'urdf' property is empty.")
	elif state == "hidden":
		_show_message("Select a GodotRobot node in the scene to view its URDF tree.")
	else:
		_show_message("Unknown state " + state)

func load_robot(robot: URDFRobot):
	current_robot = robot
	tree.clear()
	
	var root = tree.create_item()
	root.set_text(0, robot.name)

	var root_links = robot.get_root_links()

	if root_links.is_empty():
		push_error("robot has no links.")

	for link in root_links:
		_add_tree_children(root, link)

func _add_tree_children(parent_item: TreeItem, link_name: String):
	var link = current_robot.get_link(link_name)
	if not link: return
	
	var link_item = tree.create_item(parent_item)
	link_item.set_text(0, " " + link.name)
	link_item.set_metadata(0, link) 
	
	var child_joints = current_robot.get_child_joints(link_name)
	for joint in child_joints:
		var joint_item = tree.create_item(link_item)
		joint_item.set_text(0, " " + joint.name)
		joint_item.set_metadata(0, joint) 
		_add_tree_children(joint_item, joint.child)

func _on_tree_item_selected():
	var selected = tree.get_selected()
	var data = selected.get_metadata(0)
	selected_item_data = data
	
	EditorInterface.edit_resource(data)
