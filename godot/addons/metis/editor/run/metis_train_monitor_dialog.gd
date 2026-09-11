@tool
class_name MetisTrainMonitorDialog
extends AcceptDialog

## Monitors and stops the detached training run recorded in `res://.metis`.

const BRANDING := preload("res://addons/metis/editor/metis_branding.gd")
const RUN_STATE := preload("res://addons/metis/editor/run/metis_run_state.gd")

static var EPISODE_PATTERN := RegEx.create_from_string("episode=(\\d+)")

const POLL_SECONDS := 1.0

# TensorFlow may still be importing when the monitor first opens.
const STARTUP_GRACE_MS := 30_000

signal run_finished

var _runner := MetisProcess.new()
var _poll_timer: Timer
var _confirm_stop: ConfirmationDialog
var _log_size := -1
var _startup_deadline := 0
var _episode_total := 0
var _dashboard_url := ""

var _subtitle: Label
var _details: TextEdit
var _progress_bar: ProgressBar
var _progress_label: Label
var _dashboard_link: LinkButton
var _status_label: Label
var _stop_button: Button
var _log_toggle: Button
var _log: TextEdit


func _ready() -> void:
	title = "Metis — Training"
	auto_translate_mode = Node.AUTO_TRANSLATE_MODE_DISABLED
	min_size = Vector2i(660, 470)
	get_ok_button().text = "Close"
	about_to_popup.connect(_refresh_from_disk)

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)

	var header := BRANDING.header("Training in progress", "…")
	root.add_child(header)
	_subtitle = header.find_child(BRANDING.SUBTITLE_NAME, true, false)

	_details = TextEdit.new()
	_details.editable = false
	_details.custom_minimum_size = Vector2(0, 76)
	root.add_child(_details)

	_progress_label = Label.new()
	root.add_child(_progress_label)
	_progress_bar = ProgressBar.new()
	_progress_bar.show_percentage = false
	_progress_bar.custom_minimum_size = Vector2(0, 18)
	root.add_child(_progress_bar)

	_dashboard_link = LinkButton.new()
	_dashboard_link.visible = false
	_dashboard_link.underline = LinkButton.UNDERLINE_MODE_ON_HOVER
	_dashboard_link.pressed.connect(func(): OS.shell_open(_dashboard_url))
	root.add_child(_dashboard_link)

	var notice := Label.new()
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	notice.custom_minimum_size = Vector2(460, 0)
	notice.modulate = Color(1.0, 1.0, 1.0, 0.7)
	notice.text = ("Closing this window does not stop anything: the run is detached and survives "
		+ "closing the editor. Reopen Train to come back here.")
	root.add_child(notice)

	_status_label = Label.new()
	_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status_label.custom_minimum_size = Vector2(460, 0)
	root.add_child(_status_label)

	var buttons := HBoxContainer.new()
	_stop_button = Button.new()
	_stop_button.text = "Stop training"
	_stop_button.custom_minimum_size = Vector2(150, 34)
	var theme := EditorInterface.get_editor_theme()
	if theme != null and theme.has_icon("Stop", "EditorIcons"):
		_stop_button.icon = theme.get_icon("Stop", "EditorIcons")
	_stop_button.pressed.connect(_on_stop)
	buttons.add_child(_stop_button)
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	buttons.add_child(spacer)
	root.add_child(buttons)

	_log_toggle = Button.new()
	_log_toggle.toggle_mode = true
	_log_toggle.alignment = HORIZONTAL_ALIGNMENT_LEFT
	_log_toggle.text = "▸ Training log"
	_log_toggle.tooltip_text = "Last %d KB. The complete log is at %s" % [
		MetisProcess.LOG_TAIL_BYTES / 1024, RUN_STATE.log_path()]
	_log_toggle.toggled.connect(func(pressed):
		_log.visible = pressed
		_log_toggle.text = ("▼ Training log" if pressed else "▸ Training log"))
	root.add_child(_log_toggle)

	_log = TextEdit.new()
	_log.editable = false
	_log.visible = false
	_log.custom_minimum_size = Vector2(0, 160)
	_log.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(_log)

	_poll_timer = Timer.new()
	_poll_timer.wait_time = POLL_SECONDS
	_poll_timer.timeout.connect(_poll)
	add_child(_poll_timer)

	# Stopping may discard work since the last checkpoint, so ask first.
	_confirm_stop = ConfirmationDialog.new()
	_confirm_stop.title = "Stop training"
	_confirm_stop.ok_button_text = "Stop"
	_confirm_stop.confirmed.connect(_stop_confirmed)
	add_child(_confirm_stop)


func _refresh_from_disk() -> void:
	## Reconnects to the run currently described by `res://.metis`.
	_runner.attach(RUN_STATE.log_path(), RUN_STATE.pid_path())
	var state := RUN_STATE.read()
	_subtitle.text = RUN_STATE.summary(state)
	_episode_total = int(state.get("episodes", 0))

	if state.is_empty():
		_details.text = ("No %s was recorded for this run, so its configuration is unknown. This is "
			+ "normal for a run started from a terminal.") % RUN_STATE.STATE_FILE
	else:
		var lines := PackedStringArray()
		for key in ["algorithm", "scene", "checkpoint_dir", "episodes", "dashboard_port"]:
			if state.has(key):
				lines.append("%s: %s" % [key, RUN_STATE.format_value(state[key])])
		_details.text = "\n".join(lines)

	if _episode_total > 0:
		_progress_bar.visible = true
		_progress_bar.max_value = float(_episode_total)
		_progress_label.text = "episode 0 / %d" % _episode_total
	else:
		_progress_bar.visible = false
		_progress_label.text = "waiting for the first episode…"

	if bool(state.get("dashboard", false)):
		_dashboard_url = "http://127.0.0.1:%d" % int(state.get("dashboard_port", 8770))
		_dashboard_link.text = "Open the live dashboard — %s" % _dashboard_url
		_dashboard_link.visible = true
	else:
		_dashboard_link.visible = false

	_log_size = -1  # force a read: the log has grown since this dialog last looked at it
	_stop_button.disabled = false
	_status_label.text = ""
	_startup_deadline = Time.get_ticks_msec() + STARTUP_GRACE_MS
	_poll_timer.start()
	_poll()


func _poll() -> void:
	var size := _runner.log_size()
	if size != _log_size:
		_log_size = size
		var log_text := _runner.read_log_tail()
		_log.text = log_text
		_log.scroll_vertical = _log.get_line_count()
		_apply_progress(log_text)
	if _runner.is_running():
		_stop_button.disabled = false
		if _status_label.text.begins_with("Waiting"):
			_status_label.text = ""
		return
	if not _runner.has_pidfile() and Time.get_ticks_msec() < _startup_deadline:
		# The launcher may not have written its pidfile yet.
		_status_label.text = "Waiting for the trainer to start…"
		_stop_button.disabled = true
		return
	_poll_timer.stop()
	_status_label.text = ("Run finished or stopped." if _runner.has_pidfile()
		else "No training run was found.")
	_stop_button.disabled = true
	run_finished.emit()


func _apply_progress(log_text: String) -> void:
	## Updates progress from the latest `episode=NNNN` entry in the trainer log.
	var found := EPISODE_PATTERN.search_all(log_text)
	if found.is_empty():
		return
	var episode := int(found[found.size() - 1].get_string(1))
	if _episode_total > 0:
		_progress_bar.value = clampf(float(episode), 0.0, _progress_bar.max_value)
		_progress_label.text = "episode %d / %d" % [episode, _episode_total]
	else:
		_progress_label.text = "episode %d" % episode


func _on_stop() -> void:
	_confirm_stop.dialog_text = ("Stop the running training?\n\nThe trainer and every Godot "
		+ "environment it spawned are terminated. Progress since the last checkpoint is lost.")
	_confirm_stop.popup_centered()


func _stop_confirmed() -> void:
	_runner.stop()
	_status_label.text = "Stop requested. Waiting for the process group to exit…"
	_stop_button.disabled = true
	_poll_timer.start()
