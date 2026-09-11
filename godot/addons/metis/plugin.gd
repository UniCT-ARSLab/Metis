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
const TRAIN_MONITOR_DIALOG_SCRIPT := preload(
	"res://addons/metis/editor/run/metis_train_monitor_dialog.gd")
const RUN_DIALOG_SCRIPT := preload(
	"res://addons/metis/editor/run/metis_run_dialog.gd")
const RECORD_DIALOG_SCRIPT := preload(
	"res://addons/metis/editor/run/metis_record_dialog.gd")
const RUN_STATE := preload("res://addons/metis/editor/run/metis_run_state.gd")
const BRANDING := preload("res://addons/metis/editor/metis_branding.gd")
const METIS_MENU_TRAIN := 0
const METIS_MENU_RUN := 1
const METIS_MENU_RECORD := 2
const METIS_MENU_RUNTIME_SETUP := 3
const METIS_MENU_HEADER := 100

var _inspector_plugin: EditorInspectorPlugin
var _urdf_importer: EditorImportPlugin
var _urdf_dock: VBoxContainer
var _stl_loader: ResourceFormatLoader
var _runtime_manager: MetisRuntimeManager
var _runtime_dialog: MetisRuntimeSetupDialog
var _metis_menu: PopupMenu
var _metis_tools_menu: PopupMenu
var _main_menu_bar: MenuBar
var _renderer_selector: OptionButton
var _main_screen_button_texts: Dictionary = {}
var _main_screen_buttons: Control
var _title_bar: Container
var _layout_refresh_serial := 0
# These dialogs may load before their class_name cache is refreshed.
var _train_dialog
var _monitor_dialog
var _run_dialog
var _record_dialog


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

	_train_dialog = TRAIN_DIALOG_SCRIPT.new()
	EditorInterface.get_base_control().add_child(_train_dialog)
	_train_dialog.configure(_runtime_manager)

	_monitor_dialog = TRAIN_MONITOR_DIALOG_SCRIPT.new()
	EditorInterface.get_base_control().add_child(_monitor_dialog)
	_train_dialog.training_started.connect(_on_training_started)

	_run_dialog = RUN_DIALOG_SCRIPT.new()
	EditorInterface.get_base_control().add_child(_run_dialog)
	_run_dialog.configure(_runtime_manager)

	_record_dialog = RECORD_DIALOG_SCRIPT.new()
	EditorInterface.get_base_control().add_child(_record_dialog)
	_record_dialog.configure(_runtime_manager)

	# Keep labels consistent with the CLI, regardless of the editor language.
	_metis_menu = _build_metis_popup()
	_metis_menu.title = "Metis"
	_metis_tools_menu = _build_metis_popup()
	add_tool_submenu_item("Metis", _metis_tools_menu)
	# Appending the popup places Metis after Help without replacing Godot's own toolbar.
	var run_bar := _find_run_bar(EditorInterface.get_base_control())
	if run_bar != null and run_bar.get_parent() != null:
		var run_parent := run_bar.get_parent()
		_title_bar = run_parent as Container
		var menu_bar := _find_menu_bar_child(run_parent)
		if menu_bar != null:
			menu_bar.add_child(_metis_menu)
			_main_menu_bar = menu_bar
		_renderer_selector = _find_renderer_selector(run_parent)
		_capture_main_screen_buttons(run_parent)
	if _metis_menu.get_parent() == null:
		push_warning("Metis could not locate Godot's main MenuBar; use Project > Tools > Metis.")
		_metis_menu.queue_free()
		_metis_menu = null
	var editor_root := EditorInterface.get_base_control()
	if not editor_root.resized.is_connected(_schedule_metis_menu_visibility):
		editor_root.resized.connect(_schedule_metis_menu_visibility)
	if not main_screen_changed.is_connected(_on_main_screen_changed):
		main_screen_changed.connect(_on_main_screen_changed)
	_schedule_metis_menu_visibility()

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
	remove_tool_menu_item("Metis")
	_set_main_screen_compact(false)
	var editor_root := EditorInterface.get_base_control()
	if editor_root.resized.is_connected(_schedule_metis_menu_visibility):
		editor_root.resized.disconnect(_schedule_metis_menu_visibility)
	if main_screen_changed.is_connected(_on_main_screen_changed):
		main_screen_changed.disconnect(_on_main_screen_changed)
	if _metis_menu != null:
		_metis_menu.queue_free()
	_metis_menu = null
	_metis_tools_menu = null
	_main_menu_bar = null
	_renderer_selector = null
	_main_screen_button_texts.clear()
	_main_screen_buttons = null
	_title_bar = null
	if _train_dialog != null:
		_train_dialog.queue_free()
	_train_dialog = null
	if _monitor_dialog != null:
		_monitor_dialog.queue_free()
	_monitor_dialog = null
	if _run_dialog != null:
		_run_dialog.queue_free()
	_run_dialog = null
	if _record_dialog != null:
		_record_dialog.queue_free()
	_record_dialog = null
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
		# ResourceLoader may shut down before editor plugins do.
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
	# A fixed width lets wrapped labels report a useful minimum height.
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


func _find_menu_bar_child(parent: Node) -> MenuBar:
	for child in parent.get_children():
		if child is MenuBar:
			return child
	return null


func _find_renderer_selector(parent: Node) -> OptionButton:
	for child in parent.get_children():
		if child is OptionButton and child.text in ["Forward+", "Mobile", "Compatibility"]:
			return child
		var nested := _find_renderer_selector(child)
		if nested != null:
			return nested
	return null


func _capture_main_screen_buttons(parent: Node) -> void:
	var container := parent.get_node_or_null("EditorMainScreenButtons")
	if container == null:
		return
	_main_screen_buttons = container as Control
	for child in container.get_children():
		if child is Button:
			_main_screen_button_texts[child] = child.text
			if child.tooltip_text.is_empty():
				child.tooltip_text = child.text


func _set_main_screen_compact(compact: bool) -> void:
	for control in _main_screen_button_texts:
		if not is_instance_valid(control):
			continue
		var button := control as Button
		button.text = "" if compact and button.icon != null else str(
			_main_screen_button_texts[control])
		button.update_minimum_size()
	if _main_screen_buttons != null:
		_main_screen_buttons.update_minimum_size()
		if _main_screen_buttons is Container:
			(_main_screen_buttons as Container).queue_sort()
	if _title_bar != null:
		_title_bar.update_minimum_size()
		_title_bar.queue_sort()


func _build_metis_popup() -> PopupMenu:
	var popup := PopupMenu.new()
	popup.auto_translate_mode = Node.AUTO_TRANSLATE_MODE_DISABLED
	popup.add_theme_constant_override("icon_max_width", 18)
	popup.add_icon_item(BRANDING.logo(), "Metis", METIS_MENU_HEADER)
	popup.set_item_disabled(0, true)
	popup.add_separator()
	popup.add_item("Train", METIS_MENU_TRAIN)
	popup.add_item("Run", METIS_MENU_RUN)
	popup.add_item("Record", METIS_MENU_RECORD)
	popup.add_separator()
	popup.add_item("Runtime Setup…", METIS_MENU_RUNTIME_SETUP)
	popup.id_pressed.connect(_on_metis_menu_pressed)
	return popup


func _schedule_metis_menu_visibility() -> void:
	if _main_menu_bar == null or _metis_menu == null:
		return
	var index := _metis_menu_index()
	if index < 0:
		return
	# On narrow windows, icon-only workspace buttons leave room for Metis and the renderer selector.
	_main_menu_bar.set_menu_hidden(index, false)
	_set_main_screen_compact(EditorInterface.get_base_control().size.x < 1500.0)
	_layout_refresh_serial += 1
	call_deferred("_finish_metis_menu_visibility", _layout_refresh_serial)


func _finish_metis_menu_visibility(serial: int) -> void:
	# Minimum-size changes need two layout passes to reach the title bar.
	await get_tree().process_frame
	await get_tree().process_frame
	if serial != _layout_refresh_serial or _main_menu_bar == null:
		return
	var index := _metis_menu_index()
	if index < 0:
		return
	_main_menu_bar.set_menu_hidden(index, false)
	_set_main_screen_compact(EditorInterface.get_base_control().size.x < 1500.0)


func _on_main_screen_changed(_screen_name: String) -> void:
	_schedule_metis_menu_visibility()


func _metis_menu_index() -> int:
	if _main_menu_bar == null or _metis_menu == null:
		return -1
	for index in range(_main_menu_bar.get_menu_count()):
		if _main_menu_bar.get_menu_popup(index) == _metis_menu:
			return index
	return -1


func _on_metis_menu_pressed(id: int) -> void:
	match id:
		METIS_MENU_TRAIN:
			_show_train()
		METIS_MENU_RUN:
			_show_run()
		METIS_MENU_RECORD:
			_show_record()
		METIS_MENU_RUNTIME_SETUP:
			_show_runtime_setup()


func _dialog_size() -> Vector2i:
	var editor_size := EditorInterface.get_base_control().size
	return Vector2i(
		clampi(int(editor_size.x * 0.5), 640, 820),
		clampi(int(editor_size.y * 0.6), 480, 680),
	)


func _show_train() -> void:
	# Reuse the Train entry as a shortcut to the active monitor.
	if _monitor_dialog != null and RUN_STATE.is_run_active():
		_monitor_dialog.popup_centered_clamped(_dialog_size(), 0.8)
		return
	if _train_dialog == null:
		return
	_train_dialog.popup_centered_clamped(_dialog_size(), 0.8)


func _on_training_started() -> void:
	if _monitor_dialog == null:
		return
	# Wait until the wizard is hidden before opening another exclusive window.
	_monitor_dialog.call_deferred("popup_centered_clamped", _dialog_size(), 0.8)


func _show_run() -> void:
	if _run_dialog != null:
		_run_dialog.popup_centered_clamped(_dialog_size(), 0.8)


func _show_record() -> void:
	if _record_dialog != null:
		_record_dialog.popup_centered_clamped(_dialog_size(), 0.8)
