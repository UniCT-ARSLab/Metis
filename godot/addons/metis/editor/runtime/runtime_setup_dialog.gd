@tool
class_name MetisRuntimeSetupDialog
extends AcceptDialog

signal runtime_ready(config: Dictionary)

var runtime_manager: MetisRuntimeManager
var _profile: OptionButton
var _python_path: LineEdit
var _existing_runtime: CheckBox
var _include_dashboard: CheckBox
var _include_export: CheckBox
var _include_sb3: CheckBox
var _recreate_runtime: CheckBox
var _status_label: Label
var _progress: ProgressBar
var _log: TextEdit
var _setup_started := false


func _ready() -> void:
	title = "Metis Runtime Setup"
	min_size = Vector2i(520, 320)
	dialog_hide_on_ok = false
	get_ok_button().text = "Set Up Runtime"
	get_ok_button().pressed.connect(_start_setup)
	canceled.connect(_on_cancelled)

	var content := VBoxContainer.new()
	content.add_theme_constant_override("separation", 8)
	add_child(content)

	var description := Label.new()
	description.text = (
		"Create a project-local Python environment for Metis, or validate an "
		+
		"environment that already contains the matching Metis runtime.")
	description.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	# Give autowrap a width to wrap at, otherwise the label reports its minimum height as if wrapped
	# at ~0px width (many lines) and blows the whole dialog's height past the screen.
	description.custom_minimum_size = Vector2(440, 0)
	content.add_child(description)

	var form := GridContainer.new()
	form.columns = 2
	form.add_theme_constant_override("h_separation", 12)
	form.add_theme_constant_override("v_separation", 8)
	content.add_child(form)

	_add_form_label(form, "Compute profile")
	_profile = OptionButton.new()
	_add_profile("Native CPU", "cpu")
	_add_profile("NVIDIA CUDA", "linux_cuda")
	_add_profile("Apple Metal", "macos_metal")
	form.add_child(_profile)

	_add_form_label(form, "Python executable")
	_python_path = LineEdit.new()
	_python_path.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	form.add_child(_python_path)

	_add_form_label(form, "Environment")
	_existing_runtime = CheckBox.new()
	_existing_runtime.text = "Use and validate the selected environment"
	_existing_runtime.toggled.connect(_on_existing_toggled)
	form.add_child(_existing_runtime)

	_add_form_label(form, "Optional components")
	var extras := HBoxContainer.new()
	_include_dashboard = CheckBox.new()
	_include_dashboard.text = "Dashboard"
	_include_dashboard.button_pressed = true
	extras.add_child(_include_dashboard)
	_include_export = CheckBox.new()
	_include_export.auto_translate_mode = Node.AUTO_TRANSLATE_MODE_DISABLED
	_include_export.text = "Policy export"
	_include_export.button_pressed = true
	extras.add_child(_include_export)
	_include_sb3 = CheckBox.new()
	_include_sb3.text = "Stable-Baselines3"
	extras.add_child(_include_sb3)
	form.add_child(extras)

	_add_form_label(form, "Reinstall")
	_recreate_runtime = CheckBox.new()
	_recreate_runtime.text = "Recreate the managed virtual environment"
	form.add_child(_recreate_runtime)

	_status_label = Label.new()
	_status_label.text = "Ready to configure the runtime."
	_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status_label.custom_minimum_size = Vector2(440, 0)
	content.add_child(_status_label)

	_progress = ProgressBar.new()
	_progress.max_value = 100.0
	_progress.show_percentage = true
	content.add_child(_progress)

	_log = TextEdit.new()
	_log.editable = false
	# Small minimum + expand-fill: the log takes whatever vertical space the compact, screen-clamped
	# dialog leaves after the form, and shrinks (never forcing the window taller than the screen). It
	# has its own scrollbar for longer output.
	_log.custom_minimum_size = Vector2(0, 90)
	_log.size_flags_vertical = Control.SIZE_EXPAND_FILL
	content.add_child(_log)
	set_process(false)


func configure(manager: MetisRuntimeManager) -> void:
	runtime_manager = manager
	if not is_node_ready():
		await ready
	_python_path.text = manager.suggested_python()
	_select_platform_profile()
	var packaged := manager.has_packaged_payload()
	_existing_runtime.button_pressed = not packaged
	_on_existing_toggled(_existing_runtime.button_pressed)
	if not packaged:
		_status_label.text = (
			"This source checkout has no packaged wheel. Select an existing "
			+
			"environment, or build a release payload first.")


func _process(_delta: float) -> void:
	if not _setup_started or runtime_manager == null:
		return
	var status := runtime_manager.setup_status()
	if status.is_empty():
		return
	_status_label.text = str(status.get("message", "Working..."))
	_progress.value = float(status.get("progress", 0.0)) * 100.0
	var log_text := runtime_manager.setup_log()
	if _log.text != log_text:
		_log.text = log_text
		_log.scroll_vertical = _log.get_line_count()

	var state := str(status.get("state", "running"))
	if state == "ready":
		_finish_setup(true)
		runtime_ready.emit(status.get("runtime", {}))
	elif state in ["error", "cancelled"]:
		_finish_setup(false)


func _start_setup() -> void:
	if _setup_started or runtime_manager == null:
		return
	var python_path := _python_path.text.strip_edges()
	if python_path.is_empty():
		_status_label.text = "Choose a Python executable first."
		return
	if not _existing_runtime.button_pressed and not runtime_manager.has_packaged_payload():
		_status_label.text = "A managed installation requires a packaged Metis wheel."
		return

	var profile_id := str(_profile.get_item_metadata(_profile.selected))
	var pid := runtime_manager.start_setup({
		"python": python_path,
		"profile": profile_id,
		"existing": _existing_runtime.button_pressed,
		"dashboard": _include_dashboard.button_pressed,
		"export": _include_export.button_pressed,
		"sb3": _include_sb3.button_pressed,
		"recreate": _recreate_runtime.button_pressed,
	})
	if pid <= 0:
		_status_label.text = "Could not start the Python runtime setup process."
		return

	_setup_started = true
	get_ok_button().disabled = true
	_set_inputs_disabled(true)
	_status_label.text = "Starting runtime setup..."
	set_process(true)


func _finish_setup(success: bool) -> void:
	_setup_started = false
	set_process(false)
	get_ok_button().disabled = success
	_set_inputs_disabled(success)
	if not success:
		get_ok_button().disabled = false
		_set_inputs_disabled(false)


func _on_cancelled() -> void:
	if _setup_started and runtime_manager != null:
		runtime_manager.stop_setup()
	_setup_started = false
	set_process(false)


func _on_existing_toggled(enabled: bool) -> void:
	if _recreate_runtime != null:
		_recreate_runtime.disabled = enabled
		if enabled:
			_recreate_runtime.button_pressed = false
	if get_ok_button() != null:
		get_ok_button().text = (
			"Validate Runtime" if enabled else "Set Up Runtime")


func _set_inputs_disabled(disabled: bool) -> void:
	_profile.disabled = disabled
	_python_path.editable = not disabled
	_existing_runtime.disabled = disabled
	_include_dashboard.disabled = disabled
	_include_export.disabled = disabled
	_include_sb3.disabled = disabled
	_recreate_runtime.disabled = disabled or _existing_runtime.button_pressed


func _add_form_label(form: GridContainer, text: String) -> void:
	var label := Label.new()
	label.text = text
	form.add_child(label)


func _add_profile(label: String, profile_id: String) -> void:
	var index := _profile.item_count
	_profile.add_item(label)
	_profile.set_item_metadata(index, profile_id)


func _select_platform_profile() -> void:
	var expected := "cpu"
	if OS.get_name() == "macOS" and Engine.get_architecture_name() == "arm64":
		expected = "macos_metal"
	elif OS.get_name() == "Linux" and not _resolve_command("nvidia-smi").is_empty():
		expected = "linux_cuda"
	for index in range(_profile.item_count):
		if str(_profile.get_item_metadata(index)) == expected:
			_profile.select(index)
			return


func _resolve_command(command_name: String) -> String:
	var output: Array = []
	if OS.execute("which", [command_name], output, true) != 0 or output.is_empty():
		return ""
	return str(output[0]).strip_edges().split("\n")[0]
