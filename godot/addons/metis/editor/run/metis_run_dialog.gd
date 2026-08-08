@tool
class_name MetisRunDialog
extends AcceptDialog

## Runs or evaluates a trained policy against a Godot scenario (python run.py).
##
## Single page rather than a wizard: unlike training, running a policy is a short, repeated action --
## pick a checkpoint, pick a scene, watch it. Every remaining run.py flag is still reachable under
## "All options", built from the CLI specification exactly as the Train wizard's is.
##
## Unlike training, this launches ATTACHED and streams into the log panel: an evaluation is seconds
## to minutes, so surviving the editor buys nothing and a lingering detached viewer would be a
## nuisance. Stopping it therefore really does stop it.

signal run_finished

const BRANDING := preload("res://addons/metis/editor/metis_branding.gd")
const OPTION_FORM := preload("res://addons/metis/editor/run/metis_option_form.gd")
const SCENARIO := preload("res://addons/metis/editor/run/metis_scenario_scan.gd")

const ENTRY_POINT := "run"
const POLL_SECONDS := 0.5

# Shown as dedicated fields at the top; excluded from "All options" so no flag is editable twice.
const CURATED := [
	["checkpoint_dir", "Checkpoint directory"],
	["checkpoint_path", "Checkpoint (optional)"],
	["policy_path", "Exported policy (optional)"],
	["godot_scene", "Godot scene"],
	["episodes", "Episodes"],
	["max_steps", "Max steps/episode"],
	["seed", "Seed"],
	["evaluation_mode", "Evaluation mode"],
	["curriculum_level", "Curriculum level"],
	["headless", "Headless"],
	# Multi-agent is a mode, not a tweak: a scenario either drives several agents or it does not, and
	# burying that under thirty collapsed options made a whole class of scene look unsupported.
	["multi_agent", "Multi-agent scenario"],
	["agent_id", "Single agent to drive"],
	["multi_policy", "One policy per agent"],
	["policy_assignment", "Policy assignment"],
]

var runtime_manager: MetisRuntimeManager
var _runner := MetisProcess.new()
var _form := OPTION_FORM.new()
var _scene_dialog: EditorFileDialog
var _checkpoint_dialog: EditorFileDialog
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
	title = "Metis — Run"
	# English by design; see plugin.gd. Keeps the editor dictionary out of our labels.
	auto_translate_mode = Node.AUTO_TRANSLATE_MODE_DISABLED
	min_size = Vector2i(660, 470)
	get_ok_button().text = "Close"
	about_to_popup.connect(_reset)

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)
	root.add_child(BRANDING.header("Metis — Run", "Evaluate or watch a trained policy"))

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
	_all_toggle.tooltip_text = "Every remaining run.py flag, with its CLI default filled in."
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
	_launch_button.text = "Run"
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


# --- form ------------------------------------------------------------------------------------------

func _reset() -> void:
	if _runner.is_running():
		# A run is still streaming; leave the form and the output exactly as they are.
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
		note.text = ("Reading run.py's options from the Python CLI in the background — reopen this "
			+ "window in a few seconds.")
		_curated_box.add_child(note)
		return

	var grid := OPTION_FORM.make_grid()
	_curated_box.add_child(grid)
	for entry in CURATED:
		var dest := str(entry[0])
		if not spec.has(dest):
			# Renamed or dropped in Python: skip rather than emit something the CLI would reject. It
			# stays reachable under "All options" with its new name.
			continue
		var control := _form.add(grid, spec[dest], str(entry[1]))
		_attach_browse(dest, grid, control)
		if control is LineEdit:
			control.text_changed.connect(func(_t): _update_preview())
		elif control is CheckBox:
			control.toggled.connect(func(_p): _update_preview())
		elif control is OptionButton:
			control.item_selected.connect(func(_i): _update_preview())

	_form.add_all(_all_box, ENTRY_POINT, _skipped_dests())
	_wire_multi_agent_interlock()
	_apply_open_scene()


func _wire_multi_agent_interlock() -> void:
	## Grey out what the CLI would reject anyway.
	##
	## run.py raises "--multi-policy requires --multi-agent", and --policy-assignment means nothing
	## without one policy per agent. Showing all three as equals invites a combination that dies on
	## launch; a disabled control carrying the reason is the same information, earlier.
	var multi_agent := _control_for("multi_agent")
	var multi_policy := _control_for("multi_policy")
	var assignment := _control_for("policy_assignment")
	if multi_agent == null or multi_policy == null:
		return
	var refresh := func():
		var many: bool = multi_agent.button_pressed
		multi_policy.disabled = not many
		multi_policy.tooltip_text = ("Requires a multi-agent scenario." if not many
			else "Give every agent its own policy instead of sharing one.")
		if not many:
			multi_policy.button_pressed = false
		if assignment != null:
			assignment.disabled = not (many and multi_policy.button_pressed)
	multi_agent.toggled.connect(func(_pressed): refresh.call())
	multi_policy.toggled.connect(func(_pressed): refresh.call())
	refresh.call()


func _control_for(dest: String) -> Control:
	var entry: Variant = _form.controls.get(dest)
	return entry["control"] if entry is Dictionary else null


func _skipped_dests() -> PackedStringArray:
	## Curated fields plus the ones this window supplies itself, so neither is editable twice.
	var skip := PackedStringArray(["godot_bin", "godot_project"])
	for entry in CURATED:
		skip.append(str(entry[0]))
	return skip


func _attach_browse(dest: String, grid: GridContainer, control: Control) -> void:
	## Add a Browse… button beside the fields that name a file or directory. Done by wrapping the
	## control after the fact so the generic form builder stays free of per-field knowledge.
	if dest not in ["godot_scene", "checkpoint_dir", "checkpoint_path", "policy_path"]:
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
	if dest == "godot_scene":
		if _scene_dialog == null:
			_scene_dialog = EditorFileDialog.new()
			_scene_dialog.file_mode = EditorFileDialog.FILE_MODE_OPEN_FILE
			_scene_dialog.access = EditorFileDialog.ACCESS_RESOURCES
			_scene_dialog.clear_filters()
			_scene_dialog.add_filter("*.tscn", "Godot scenes")
			add_child(_scene_dialog)
		for connection in _scene_dialog.file_selected.get_connections():
			_scene_dialog.file_selected.disconnect(connection["callable"])
		_scene_dialog.file_selected.connect(func(path):
			control.text = path
			_update_preview())
		_scene_dialog.popup_file_dialog()
		return
	if _checkpoint_dialog == null:
		_checkpoint_dialog = EditorFileDialog.new()
		_checkpoint_dialog.access = EditorFileDialog.ACCESS_FILESYSTEM
		add_child(_checkpoint_dialog)
	_checkpoint_dialog.file_mode = (EditorFileDialog.FILE_MODE_OPEN_DIR if dest == "checkpoint_dir"
		else EditorFileDialog.FILE_MODE_OPEN_ANY)
	for connection in _checkpoint_dialog.file_selected.get_connections():
		_checkpoint_dialog.file_selected.disconnect(connection["callable"])
	for connection in _checkpoint_dialog.dir_selected.get_connections():
		_checkpoint_dialog.dir_selected.disconnect(connection["callable"])
	var apply := func(path):
		control.text = path
		_update_preview()
	_checkpoint_dialog.file_selected.connect(apply)
	_checkpoint_dialog.dir_selected.connect(apply)
	_checkpoint_dialog.popup_file_dialog()


func _apply_open_scene() -> void:
	## Prefill from the scene open in the editor, when it is a trainable scenario. Same rule as the
	## Train wizard: a scenario is a scene holding a BridgeServer, the TCP endpoint Python connects to.
	var path := SCENARIO.open_scenario_path()
	if path.is_empty():
		_scene_note.text = ("Pick the scenario to run in — a scene containing a %s node."
			% SCENARIO.BRIDGE_CLASS)
		return
	_form.set_value("godot_scene", path)
	_scene_note.text = "Using the scene open in the editor."
	_update_preview()


# --- command -----------------------------------------------------------------------------------

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
		return PackedStringArray([_python(), source_root.path_join("run.py")])
	return PackedStringArray([_python(), "-m", "metis_cli", "run"])


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


# --- launch ------------------------------------------------------------------------------------

func _on_launch() -> void:
	if _runner.is_running():
		return
	var python := _python()
	if not FileAccess.file_exists(python):
		_status_label.text = "Python not found: %s. Configure the runtime first." % python
		return
	var problems := _form.validation_errors()
	if not problems.is_empty():
		# Refused here rather than by argparse: a detached launch that dies on
		# `invalid int value` leaves nothing on screen to read.
		_status_label.text = "Fix these first — " + ", ".join(problems)
		return
	var metis_dir := ProjectSettings.globalize_path("res://.metis")
	DirAccess.make_dir_recursive_absolute(metis_dir)
	var command := _command_prefix()
	var command_args := command.slice(1)
	command_args.append_array(_build_args())
	# A separate log and pidfile from training's: an evaluation run alongside a training run must not
	# overwrite the file the training monitor is tailing.
	var ok := _runner.start_command(
		command[0],
		command_args,
		_working_directory(),
		metis_dir.path_join("run.log"),
		metis_dir.path_join("run.pid"))
	if not ok:
		_status_label.text = "Could not start the run."
		return
	_launch_button.disabled = true
	_stop_button.visible = true
	_stop_button.disabled = false
	_status_label.text = "Running…"
	_log.visible = true
	_log_toggle.button_pressed = true
	_log_size = -1
	_poll_timer.start()


func _on_stop() -> void:
	_runner.stop()
	_status_label.text = "Stop requested…"
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
		# The launcher writes the pidfile a moment after the spawn returns; until then "not running"
		# means "not up yet", not "finished".
		_status_label.text = "Starting…"
		return
	_poll_timer.stop()
	_status_label.text = "Finished."
	_stop_button.visible = false
	_launch_button.disabled = false
	run_finished.emit()
