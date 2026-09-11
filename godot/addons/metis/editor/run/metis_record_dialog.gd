@tool
class_name MetisRecordDialog
extends AcceptDialog

## Records manual expert demonstrations from a Godot scenario (python recorder.py).
##
## Recording stays attached to the editor and always opens a Godot window for manual control.

signal recording_finished

const BRANDING := preload("res://addons/metis/editor/metis_branding.gd")
const OPTION_FORM := preload("res://addons/metis/editor/run/metis_option_form.gd")
const SCENARIO := preload("res://addons/metis/editor/run/metis_scenario_scan.gd")

const ENTRY_POINT := "record"
const POLL_SECONDS := 0.5

# Options already shown as dedicated fields.
const CURATED := [
	["output", "Output .npz"],
	["append", "Append to it"],
	["godot_scene", "Godot scene"],
	["episodes", "Episodes"],
	["max_steps", "Max steps/episode"],
	["step_delay", "Step delay (s)"],
	["seed", "Seed"],
	["multi_agent", "Multi-agent scenario"],
	["agent_id", "Agent to observe"],
	["manual_agent_id", "Manually driven agent"],
	["record_all_agents", "Record every agent"],
]

var runtime_manager: MetisRuntimeManager
var _runner := MetisProcess.new()
var _form := OPTION_FORM.new()
var _scene_dialog: EditorFileDialog
var _output_dialog: EditorFileDialog
var _poll_timer: Timer
var _log_size := -1

var _curated_box: VBoxContainer
var _all_box: VBoxContainer
var _all_toggle: Button
var _scene_note: Label
var _preview: TextEdit
var _launch_button: Button
var _stop_button: Button
var _status_label: Label
var _log_toggle: Button
var _log: TextEdit


func _ready() -> void:
	title = "Metis — Record"
	auto_translate_mode = Node.AUTO_TRANSLATE_MODE_DISABLED
	min_size = Vector2i(660, 470)
	get_ok_button().text = "Close"
	about_to_popup.connect(_reset)

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)
	root.add_child(BRANDING.header("Metis — Record",
		"Capture manual demonstrations to train from"))

	var explain := Label.new()
	explain.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	explain.custom_minimum_size = Vector2(460, 0)
	explain.modulate = Color(1.0, 1.0, 1.0, 0.7)
	explain.text = ("A Godot window opens and you drive the agent yourself; every step is written to "
		+ "the .npz below. Feed it back with --demo-path when training. Recording is never headless — "
		+ "there would be nothing to drive.")
	root.add_child(explain)

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(scroll)
	var body := VBoxContainer.new()
	body.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	body.add_theme_constant_override("separation", 8)
	scroll.add_child(body)

	_curated_box = VBoxContainer.new()
	body.add_child(_curated_box)
	_scene_note = Label.new()
	_scene_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_scene_note.custom_minimum_size = Vector2(460, 0)
	_scene_note.add_theme_color_override("font_color", Color(0.78, 0.81, 0.88))
	body.add_child(_scene_note)

	_all_toggle = Button.new()
	_all_toggle.toggle_mode = true
	_all_toggle.alignment = HORIZONTAL_ALIGNMENT_LEFT
	_all_toggle.text = "▸ All options"
	_all_toggle.tooltip_text = "Every remaining recorder.py flag, with its CLI default filled in."
	_all_toggle.toggled.connect(func(pressed):
		_all_box.visible = pressed
		_all_toggle.text = ("▼ All options" if pressed else "▸ All options")
		_update_preview())
	body.add_child(_all_toggle)
	_all_box = VBoxContainer.new()
	_all_box.visible = false
	body.add_child(_all_box)

	root.add_child(HSeparator.new())
	_preview = TextEdit.new()
	_preview.editable = false
	_preview.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_preview.custom_minimum_size = Vector2(0, 70)
	root.add_child(_preview)

	var buttons := HBoxContainer.new()
	_launch_button = Button.new()
	_launch_button.text = "Record"
	_launch_button.custom_minimum_size = Vector2(150, 34)
	var theme := EditorInterface.get_editor_theme()
	if theme != null:
		if theme.has_icon("Play", "EditorIcons"):
			_launch_button.icon = theme.get_icon("Play", "EditorIcons")
		var accent := theme.get_color("accent_color", "Editor")
		_launch_button.add_theme_color_override("font_color", accent)
		_launch_button.add_theme_color_override("font_hover_color", accent)
	_launch_button.pressed.connect(_on_launch)
	buttons.add_child(_launch_button)
	_stop_button = Button.new()
	_stop_button.text = "Stop"
	_stop_button.visible = false
	_stop_button.custom_minimum_size = Vector2(110, 34)
	if theme != null and theme.has_icon("Stop", "EditorIcons"):
		_stop_button.icon = theme.get_icon("Stop", "EditorIcons")
	_stop_button.pressed.connect(_on_stop)
	buttons.add_child(_stop_button)
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	buttons.add_child(spacer)
	root.add_child(buttons)

	_status_label = Label.new()
	_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status_label.custom_minimum_size = Vector2(460, 0)
	root.add_child(_status_label)

	_log_toggle = Button.new()
	_log_toggle.toggle_mode = true
	_log_toggle.alignment = HORIZONTAL_ALIGNMENT_LEFT
	_log_toggle.text = "▸ Output"
	_log_toggle.toggled.connect(func(pressed):
		_log.visible = pressed
		_log_toggle.text = ("▼ Output" if pressed else "▸ Output"))
	root.add_child(_log_toggle)
	_log = TextEdit.new()
	_log.editable = false
	_log.visible = false
	_log.custom_minimum_size = Vector2(0, 150)
	root.add_child(_log)

	_poll_timer = Timer.new()
	_poll_timer.wait_time = POLL_SECONDS
	_poll_timer.timeout.connect(_poll)
	add_child(_poll_timer)


func configure(manager: MetisRuntimeManager) -> void:
	runtime_manager = manager
	if not is_node_ready():
		await ready


# form

func _reset() -> void:
	if _runner.is_running():
		return
	OPTION_FORM.ensure_spec(_python(), _source_python_root())
	_form = OPTION_FORM.new()
	OPTION_FORM.clear(_curated_box)
	OPTION_FORM.clear(_all_box)
	_all_toggle.button_pressed = false
	_all_box.visible = false
	_log.text = ""
	_log.visible = false
	_log_toggle.button_pressed = false
	_status_label.text = ""
	_stop_button.visible = false
	_launch_button.disabled = false
	_build_form()
	_update_preview()


func _build_form() -> void:
	var spec := OPTION_FORM.spec_by_dest(ENTRY_POINT)
	if spec.is_empty():
		var note := Label.new()
		note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		note.custom_minimum_size = Vector2(460, 0)
		note.text = ("Reading recorder.py's options from the Python CLI in the background — reopen "
			+ "this window in a few seconds.")
		_curated_box.add_child(note)
		return

	var grid := OPTION_FORM.make_grid()
	_curated_box.add_child(grid)
	for entry in CURATED:
		var dest := str(entry[0])
		if not spec.has(dest):
			continue
		var control := _form.add(grid, spec[dest], str(entry[1]))
		_attach_browse(dest, grid, control)
		if control is LineEdit:
			control.text_changed.connect(func(_t): _update_preview())
		elif control is CheckBox:
			control.toggled.connect(func(_p): _update_preview())
		elif control is OptionButton:
			control.item_selected.connect(func(_i): _update_preview())

	# A headless recording has no window to control.
	_form.add_all(_all_box, ENTRY_POINT, _skipped_dests())
	_apply_open_scene()


func _skipped_dests() -> PackedStringArray:
	var skip := PackedStringArray(["godot_bin", "godot_project", "headless"])
	for entry in CURATED:
		skip.append(str(entry[0]))
	return skip


func _attach_browse(dest: String, grid: GridContainer, control: Control) -> void:
	if dest not in ["godot_scene", "output"]:
		return
	var index := control.get_index()
	grid.remove_child(control)
	var box := HBoxContainer.new()
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	box.add_child(control)
	var browse := Button.new()
	browse.text = "Browse…"
	browse.pressed.connect(func(): _browse(dest, control))
	box.add_child(browse)
	grid.add_child(box)
	grid.move_child(box, index)


func _browse(dest: String, control: Control) -> void:
	var dialog: EditorFileDialog
	if dest == "godot_scene":
		if _scene_dialog == null:
			_scene_dialog = EditorFileDialog.new()
			_scene_dialog.file_mode = EditorFileDialog.FILE_MODE_OPEN_FILE
			_scene_dialog.access = EditorFileDialog.ACCESS_RESOURCES
			_scene_dialog.clear_filters()
			_scene_dialog.add_filter("*.tscn", "Godot scenes")
			add_child(_scene_dialog)
		dialog = _scene_dialog
	else:
		if _output_dialog == null:
			_output_dialog = EditorFileDialog.new()
			_output_dialog.file_mode = EditorFileDialog.FILE_MODE_SAVE_FILE
			_output_dialog.access = EditorFileDialog.ACCESS_FILESYSTEM
			_output_dialog.clear_filters()
			_output_dialog.add_filter("*.npz", "Demonstration archives")
			add_child(_output_dialog)
		dialog = _output_dialog
	for connection in dialog.file_selected.get_connections():
		dialog.file_selected.disconnect(connection["callable"])
	dialog.file_selected.connect(func(path):
		control.text = path
		_update_preview())
	dialog.popup_file_dialog()


func _apply_open_scene() -> void:
	var path := SCENARIO.open_scenario_path()
	if path.is_empty():
		_scene_note.text = ("Pick the scenario to record in — a scene containing a %s node."
			% SCENARIO.BRIDGE_CLASS)
		return
	_form.set_value("godot_scene", path)
	_scene_note.text = "Using the scene open in the editor."
	_update_preview()


# command

func _project_root() -> String:
	return ProjectSettings.globalize_path("res://").trim_suffix("/")


func _source_python_root() -> String:
	return runtime_manager.development_source_root() if runtime_manager != null else ""


func _working_directory() -> String:
	var source_root := _source_python_root()
	return source_root.get_base_dir() if not source_root.is_empty() else _project_root()


func _python() -> String:
	return runtime_manager.suggested_python() if runtime_manager != null else "python3"


func _command_prefix() -> PackedStringArray:
	var source_root := _source_python_root()
	if not source_root.is_empty():
		return PackedStringArray([_python(), source_root.path_join("recorder.py")])
	return PackedStringArray([_python(), "-m", "metis_cli", "record"])


func _build_args() -> PackedStringArray:
	var args := PackedStringArray()
	var godot_bin := OS.get_executable_path()
	if not godot_bin.is_empty():
		args.append_array(["--godot-bin", godot_bin])
	args.append_array(["--godot-project", _project_root()])
	args.append_array(_form.flags())
	return args


func _update_preview() -> void:
	if _preview == null:
		return
	var parts := _command_prefix()
	parts.append_array(_build_args())
	_preview.text = " ".join(parts)


# launch

func _on_launch() -> void:
	if _runner.is_running():
		return
	var python := _python()
	if not FileAccess.file_exists(python):
		_status_label.text = "Python not found: %s. Configure the runtime first." % python
		return
	var problems := _form.validation_errors()
	if not problems.is_empty():
		_status_label.text = "Fix these first — " + ", ".join(problems)
		return
	var output := str(_form.value_of("output")).strip_edges()
	if output.is_empty():
		_status_label.text = "Choose an output .npz first: without one the recorded episodes have nowhere to go."
		return
	var metis_dir := ProjectSettings.globalize_path("res://.metis")
	DirAccess.make_dir_recursive_absolute(metis_dir)
	var command := _command_prefix()
	var command_args := command.slice(1)
	command_args.append_array(_build_args())
	# Recording and training may run at the same time.
	var ok := _runner.start_command(
		command[0],
		command_args,
		_working_directory(),
		metis_dir.path_join("record.log"),
		metis_dir.path_join("record.pid"))
	if not ok:
		_status_label.text = "Could not start the recorder."
		return
	_launch_button.disabled = true
	_stop_button.visible = true
	_stop_button.disabled = false
	_status_label.text = "Recording — drive the agent in the Godot window that just opened."
	_log.visible = true
	_log_toggle.button_pressed = true
	_log_size = -1
	_poll_timer.start()


func _on_stop() -> void:
	_runner.stop()
	# Interrupted sessions do not publish an incomplete dataset.
	_status_label.text = "Stop requested. Any episodes not written yet are lost."
	_stop_button.disabled = true


func _poll() -> void:
	var size := _runner.log_size()
	if size != _log_size:
		_log_size = size
		_log.text = _runner.read_log_tail()
		_log.scroll_vertical = _log.get_line_count()
	if _runner.is_running():
		return
	if not _runner.has_pidfile():
		_status_label.text = "Starting…"
		return
	_poll_timer.stop()
	_status_label.text = "Recording finished. Demonstrations written to %s" % _form.value_of("output")
	_stop_button.visible = false
	_launch_button.disabled = false
	recording_finished.emit()
