@tool
extends EditorPlugin

const INSPECTOR_PLUGIN_SCRIPT := preload(
	"res://addons/metis/editor/inspector/rl_member_inspector.gd")
const URDF_IMPORTER_SCRIPT := preload(
	"res://addons/metis/integrations/urdf/urdf_importer.gd")
const URDF_DOCK_SCRIPT := preload(
	"res://addons/metis/integrations/urdf/urdf_dock.gd")
const STL_LOADER_SCRIPT := preload(
	"res://addons/metis/integrations/stl/stl-io.gd")
const RUNTIME_MANAGER_SCRIPT := preload(
	"res://addons/metis/editor/runtime/runtime_manager.gd")
const RUNTIME_SETUP_DIALOG_SCRIPT := preload(
	"res://addons/metis/editor/runtime/runtime_setup_dialog.gd")
const TRAIN_DIALOG_SCRIPT := preload(
	"res://addons/metis/editor/run/metis_train_dialog.gd")

var _inspector_plugin: EditorInspectorPlugin
var _urdf_importer: EditorImportPlugin
var _urdf_dock: VBoxContainer
var _stl_loader: ResourceFormatLoader
var _runtime_manager: MetisRuntimeManager
var _runtime_dialog: MetisRuntimeSetupDialog
var _metis_toolbar: HBoxContainer
var _toolbar_in_container := false
# Untyped on purpose (it is a MetisTrainDialog): keeps plugin.gd from depending on that class_name
# being registered before the project rescan, and lets us call its custom configure() dynamically.
var _train_dialog


func _enter_tree() -> void:
	_register_stl_loader()

	_inspector_plugin = INSPECTOR_PLUGIN_SCRIPT.new()
	add_inspector_plugin(_inspector_plugin)

	_urdf_importer = URDF_IMPORTER_SCRIPT.new()
	add_import_plugin(_urdf_importer)

	_urdf_dock = URDF_DOCK_SCRIPT.new()
	add_control_to_dock(DOCK_SLOT_RIGHT_UR, _urdf_dock)

	_runtime_manager = RUNTIME_MANAGER_SCRIPT.new()
	_runtime_dialog = RUNTIME_SETUP_DIALOG_SCRIPT.new()
	EditorInterface.get_base_control().add_child(_runtime_dialog)
	_runtime_dialog.configure(_runtime_manager)
	add_tool_menu_item("Metis Runtime Setup...", _show_runtime_setup)

	_train_dialog = TRAIN_DIALOG_SCRIPT.new()
	EditorInterface.get_base_control().add_child(_train_dialog)
	_train_dialog.configure(_runtime_manager)

	_metis_toolbar = HBoxContainer.new()
	var brand := TextureRect.new()
	var logo: Texture2D
	var logo_path := "res://addons/metis/logo.svg"
	if not ResourceLoader.exists(logo_path):
		logo_path = "res://icon.svg"
	if ResourceLoader.exists(logo_path):
		logo = load(logo_path)
	if logo != null:
		brand.texture = logo
	# EXPAND_IGNORE_SIZE: don't let the 576px source texture drive the control size (that blew the
	# whole top bar up). custom_minimum_size then fixes it to a small editor-icon size.
	brand.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	brand.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	brand.custom_minimum_size = Vector2(18, 18)
	brand.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	brand.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	brand.tooltip_text = "Metis"
	_metis_toolbar.add_child(brand)
	_add_toolbar_button("Train", _show_train, false)
	_add_toolbar_button("Run", _show_run, true)
	_add_toolbar_button("Record", _show_record, true)
	_metis_toolbar.add_child(VSeparator.new())
	# Preferred: drop the bar immediately LEFT of the editor's run/play buttons. If the run bar can't
	# be located (Godot internals change), fall back to the standard top toolbar container.
	var run_bar := _find_run_bar(EditorInterface.get_base_control())
	if run_bar != null and run_bar.get_parent() != null:
		var run_parent := run_bar.get_parent()
		run_parent.add_child(_metis_toolbar)
		run_parent.move_child(_metis_toolbar, run_bar.get_index())
		_toolbar_in_container = false
	else:
		add_control_to_container(CONTAINER_TOOLBAR, _metis_toolbar)
		_toolbar_in_container = true
		var toolbar_parent := _metis_toolbar.get_parent()
		if toolbar_parent != null:
			toolbar_parent.move_child(_metis_toolbar, 0)

	var selection := EditorInterface.get_selection()
	if not selection.selection_changed.is_connected(_on_selection_changed):
		selection.selection_changed.connect(_on_selection_changed)
	_on_selection_changed()
	if (
		DisplayServer.get_name() != "headless"
		and _runtime_manager.has_packaged_payload()
		and _runtime_manager.runtime_needs_update()
	):
		call_deferred("_show_runtime_setup")


func _exit_tree() -> void:
	remove_tool_menu_item("Metis Runtime Setup...")
	if _metis_toolbar != null:
		if _toolbar_in_container:
			remove_control_from_container(CONTAINER_TOOLBAR, _metis_toolbar)
		_metis_toolbar.queue_free()
	_metis_toolbar = null
	if _train_dialog != null:
		_train_dialog.queue_free()
	_train_dialog = null
	if _runtime_dialog != null:
		_runtime_dialog.queue_free()
	_runtime_dialog = null
	if _runtime_manager != null:
		_runtime_manager.stop_setup()
	_runtime_manager = null

	var selection := EditorInterface.get_selection()
	if selection.selection_changed.is_connected(_on_selection_changed):
		selection.selection_changed.disconnect(_on_selection_changed)

	if _urdf_dock != null:
		remove_control_from_docks(_urdf_dock)
		_urdf_dock.queue_free()
	_urdf_dock = null

	if _urdf_importer != null:
		remove_import_plugin(_urdf_importer)
	_urdf_importer = null

	if _inspector_plugin != null:
		remove_inspector_plugin(_inspector_plugin)
	_inspector_plugin = null

	if _stl_loader != null:
		# ResourceLoader may tear down its list before editor plugins receive
		# _exit_tree(). It owns this reference for the editor lifetime, and the
		# extension check in _register_stl_loader() prevents duplicate loaders if
		# the plugin is toggled off and on.
		_stl_loader = null


func _register_stl_loader() -> void:
	if "stl" in ResourceLoader.get_recognized_extensions_for_type("ArrayMesh"):
		return
	_stl_loader = STL_LOADER_SCRIPT.new()
	ResourceLoader.add_resource_format_loader(_stl_loader, true)


func _on_selection_changed() -> void:
	if _urdf_dock == null:
		return

	var active_robot: GodotRobot
	var selected_nodes := EditorInterface.get_selection().get_selected_nodes()
	if not selected_nodes.is_empty():
		var current_node := selected_nodes[0] as Node
		while current_node != null:
			if current_node is GodotRobot:
				active_robot = current_node
				break
			current_node = current_node.get_parent()

	if active_robot == null:
		_urdf_dock.set_ui_state("hidden")
	elif active_robot.urdf == null:
		_urdf_dock.set_ui_state("missing_urdf")
	else:
		_urdf_dock.load_robot(active_robot.urdf)
		_urdf_dock.set_ui_state("active")


func _show_runtime_setup() -> void:
	if _runtime_dialog == null:
		return
	# EXPLICIT compact size, centered, hard-clamped to the screen. An explicit WIDTH is essential:
	# the AUTOWRAP labels only report a short minimum height once they have a width to wrap at
	# (auto-sizing/reset_size measured them at ~0 width -> they wrapped tall and ran off-screen). The
	# 0.6 fallback ratio caps the window at 60% of the editor height, so it can NEVER exceed the
	# screen regardless of content or editor DPI scale.
	var editor_size := EditorInterface.get_base_control().size
	var dialog_size := Vector2i(
		clampi(int(editor_size.x * 0.4), 520, 600),
		clampi(int(editor_size.y * 0.4), 340, 400),
	)
	_runtime_dialog.popup_centered_clamped(dialog_size, 0.6)


func _find_run_bar(node: Node) -> Control:
	for child in node.get_children():
		if child is Control and child.get_class() == "EditorRunBar":
			return child
		var found := _find_run_bar(child)
		if found != null:
			return found
	return null


func _add_toolbar_button(text: String, handler: Callable, coming_soon: bool) -> void:
	var button := Button.new()
	button.text = text
	button.flat = true  # match the top-left menu-bar items (Scene / Project / …)
	if coming_soon:
		button.disabled = true
		button.tooltip_text = "Coming in the next step."
	else:
		button.pressed.connect(handler)
	_metis_toolbar.add_child(button)


func _show_train() -> void:
	if _train_dialog == null:
		return
	var editor_size := EditorInterface.get_base_control().size
	var dialog_size := Vector2i(
		clampi(int(editor_size.x * 0.5), 640, 820),
		clampi(int(editor_size.y * 0.6), 480, 680),
	)
	_train_dialog.popup_centered_clamped(dialog_size, 0.8)


func _show_run() -> void:
	pass


func _show_record() -> void:
	pass
