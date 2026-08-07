@tool
class_name MetisTrainDialog
extends AcceptDialog

## A step-by-step wizard to configure and launch a Metis training run without leaving the editor:
##   1. Algorithm  ->  2. Common settings  ->  3. Algorithm options  ->  4. Review & launch.
## The run is spawned FULLY DETACHED (see MetisProcess): closing Godot does not stop training, and
## stopping training does not touch Godot. The review step shows the exact equivalent terminal
## command. The whole wizard resets to its defaults every time the window is (re)opened.

const LABEL_WIDTH := 160
const PAGE_TITLES := ["Algorithm", "Common settings", "Algorithm options", "Review & launch"]

# Args already exposed as curated fields on the Common / Algorithm-options pages — excluded from the
# auto-generated "All options" list so nothing is duplicated.
const CURATED_DESTS := [
	"algorithm", "godot_bin", "godot_project", "godot_scene", "num_envs", "base_port",
	"num_episodes", "max_steps_per_episode", "physics_frames_per_step", "batch_size",
	"replay_warmup", "collector_mode", "multi_agent", "checkpoint_dir", "resume",
	"resume_checkpoint", "best_metric", "auto_recovery", "best_checkpoint", "dashboard",
	"dashboard_port", "headless", "grad_clip_adaptive", "critic_learning_rate", "min_alpha",
	"policy_update_every",
]

const ALGO_INFO := {
	"auto": "Auto — inspects the scenario's action space and picks DQN (discrete), DDPG (continuous) "
		+ "or PPO (hybrid) for you. A safe starting point if you are unsure.",
	"sac": "Soft Actor-Critic — off-policy, CONTINUOUS actions. Sample-efficient and stable (with "
		+ "gradient clipping). Best default for continuous control: robot arms, driving.",
	"ppo": "Proximal Policy Optimization — on-policy, discrete/continuous/HYBRID. Robust and easy to "
		+ "tune; shines with many parallel envs. Use for hybrid action spaces or when SAC is unstable.",
	"dqn": "Deep Q-Network — off-policy, DISCRETE actions only. Use for discrete-action tasks "
		+ "(Atari-like, grid worlds, button presses).",
	"ddpg": "Deep Deterministic Policy Gradient — off-policy, continuous. Predecessor of TD3/SAC and "
		+ "less stable; prefer SAC or TD3 unless you have a reason.",
	"td3": "Twin Delayed DDPG — off-policy, continuous, DETERMINISTIC. Stable; pick it over SAC when "
		+ "you want a deterministic policy with less exploration noise.",
}

var runtime_manager: MetisRuntimeManager
var _runner := MetisProcess.new()
var _scene_dialog: EditorFileDialog

var _pages: Array[Control] = []
var _current := 0
var _step_label: Label
var _back_button: Button
var _next_button: Button

# Page 1
var _algorithm: OptionButton
var _algo_help: Label
# Page 2 (common)
var _scene: LineEdit
var _num_envs: SpinBox
var _base_port: SpinBox
var _episodes: SpinBox
var _max_steps: SpinBox
var _physics_frames: SpinBox
var _checkpoint_dir: LineEdit
var _resume: LineEdit
var _collector: OptionButton
var _batch_size: SpinBox
var _replay_warmup: SpinBox
var _multi_agent: CheckBox
var _best_metric: OptionButton
var _auto_recovery: CheckBox
var _dashboard: CheckBox
var _dashboard_port: SpinBox
var _headless: CheckBox
# Page 3 (algorithm-specific)
var _algo_options_title: Label
var _sac_group: VBoxContainer
var _no_options_note: Label
var _critic_lr: LineEdit
var _min_alpha: LineEdit
var _policy_update_every: SpinBox
var _grad_clip: CheckBox
var _extra: LineEdit
var _all_toggle: Button
var _all_box: VBoxContainer
var _all_controls := {}  # dest -> {"control": Control, "arg": Dictionary}
# Page 4 (review)
var _preview: TextEdit
var _train_button: Button
var _status_label: Label
var _log: TextEdit


func _ready() -> void:
	title = "Metis — Train"
	min_size = Vector2i(660, 470)
	get_ok_button().text = "Close"
	about_to_popup.connect(_reset)

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)

	_step_label = Label.new()
	_step_label.add_theme_color_override("font_color", Color(0.6, 0.72, 1.0))
	root.add_child(_step_label)
	root.add_child(HSeparator.new())

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(scroll)
	var holder := VBoxContainer.new()
	holder.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(holder)
	_pages = [_build_algorithm_page(), _build_common_page(), _build_algo_options_page(),
		_build_review_page()]
	for page in _pages:
		holder.add_child(page)

	root.add_child(HSeparator.new())
	var nav := HBoxContainer.new()
	_back_button = Button.new()
	_back_button.text = "◄ Back"
	_back_button.pressed.connect(func(): _show_page(_current - 1))
	nav.add_child(_back_button)
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	nav.add_child(spacer)
	_next_button = Button.new()
	_next_button.text = "Next ►"
	_next_button.pressed.connect(func(): _show_page(_current + 1))
	nav.add_child(_next_button)
	root.add_child(nav)

	set_process(false)
	_reset()


func configure(manager: MetisRuntimeManager) -> void:
	runtime_manager = manager
	if not is_node_ready():
		await ready


# --- pages ----------------------------------------------------------------------------------------

func _build_algorithm_page() -> Control:
	var page := VBoxContainer.new()
	page.add_theme_constant_override("separation", 8)
	page.add_child(_heading("Choose the learning algorithm"))
	var form := _make_grid()
	page.add_child(form)
	_algorithm = OptionButton.new()
	for algo in ["auto", "sac", "ppo", "dqn", "ddpg", "td3"]:
		_algorithm.add_item(algo)
	_algorithm.select(0)
	_algorithm.item_selected.connect(func(_i): _on_algorithm_changed())
	_row(form, "Algorithm", _algorithm)
	_algo_help = Label.new()
	_algo_help.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_algo_help.custom_minimum_size = Vector2(460, 0)
	_algo_help.add_theme_color_override("font_color", Color(0.78, 0.81, 0.88))
	page.add_child(_algo_help)
	return page


func _build_common_page() -> Control:
	var page := VBoxContainer.new()
	page.add_child(_heading("Settings shared by every algorithm"))
	var form := _make_grid()
	page.add_child(form)

	var scene_box := HBoxContainer.new()
	_scene = _line("res://scenarios/robotarms/XarmScenario.tscn")
	_scene.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scene_box.add_child(_scene)
	var browse := Button.new()
	browse.text = "Browse…"
	browse.pressed.connect(_open_scene_dialog)
	scene_box.add_child(browse)
	_row(form, "Godot scene", scene_box)

	_num_envs = _spin(1, 64, 8)
	_row(form, "Parallel envs", _num_envs)
	_base_port = _spin(1024, 65000, 7200)
	_row(form, "Base port", _base_port)
	_episodes = _spin(1, 1000000, 8000)
	_row(form, "Episodes", _episodes)
	_max_steps = _spin(1, 100000, 300)
	_row(form, "Max steps/episode", _max_steps)
	_physics_frames = _spin(1, 20, 3)
	_row(form, "Physics frames/step", _physics_frames)
	_checkpoint_dir = _line("checkpoints/metis_run")
	_row(form, "Checkpoint dir", _checkpoint_dir)
	_resume = _line("")
	_resume.placeholder_text = "(optional) checkpoints/<run>/ckpt-XXXX"
	_row(form, "Resume checkpoint", _resume)
	_collector = OptionButton.new()
	for mode in ["async", "sync"]:
		_collector.add_item(mode)
	_collector.select(0)
	_row(form, "Collector mode", _collector)
	_batch_size = _spin(16, 4096, 256)
	_row(form, "Batch size", _batch_size)
	_replay_warmup = _spin(0, 500000, 20000)
	_row(form, "Replay warmup", _replay_warmup)
	_best_metric = OptionButton.new()
	for metric in ["reward_mean", "success_rate"]:
		_best_metric.add_item(metric)
	_best_metric.select(0)
	_row(form, "Best metric", _best_metric)
	_multi_agent = CheckBox.new()
	_multi_agent.text = "Independent agents (--multi-agent)"
	_row(form, "Multi-agent", _multi_agent)
	_auto_recovery = CheckBox.new()
	_auto_recovery.text = "Restore best checkpoint on collapse"
	_row(form, "Auto-recovery", _auto_recovery)
	var dash_box := HBoxContainer.new()
	_dashboard = CheckBox.new()
	_dashboard.text = "Enable"
	dash_box.add_child(_dashboard)
	_dashboard_port = _spin(1024, 65000, 8770)
	dash_box.add_child(_dashboard_port)
	_row(form, "Live dashboard", dash_box)
	_headless = CheckBox.new()
	_headless.text = "Run Godot envs headless (recommended)"
	_headless.button_pressed = true
	_row(form, "Headless", _headless)
	return page


func _build_algo_options_page() -> Control:
	var page := VBoxContainer.new()
	_algo_options_title = _heading("Algorithm-specific options")
	page.add_child(_algo_options_title)

	_sac_group = VBoxContainer.new()
	page.add_child(_sac_group)
	var sac := _make_grid()
	_sac_group.add_child(sac)
	_critic_lr = _line("")
	_critic_lr.placeholder_text = "(default 3e-4) e.g. 1e-4"
	_row(sac, "Critic learning rate", _critic_lr)
	_min_alpha = _line("")
	_min_alpha.placeholder_text = "(default 0.0) e.g. 0.02"
	_row(sac, "Min alpha (entropy floor)", _min_alpha)
	_policy_update_every = _spin(1, 16, 2)
	_row(sac, "Policy update every", _policy_update_every)
	_grad_clip = CheckBox.new()
	_grad_clip.text = "Adaptive gradient clipping (recommended)"
	_grad_clip.button_pressed = true
	_row(sac, "Grad clip", _grad_clip)

	_no_options_note = Label.new()
	_no_options_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_no_options_note.custom_minimum_size = Vector2(460, 0)
	_no_options_note.add_theme_color_override("font_color", Color(0.78, 0.81, 0.88))
	_no_options_note.text = "No dedicated fields for this algorithm yet — add any specific flags in the box below."
	page.add_child(_no_options_note)

	var extra_form := _make_grid()
	page.add_child(extra_form)
	_extra = _line("")
	_extra.placeholder_text = "any other raw flags, appended verbatim"
	_row(extra_form, "Extra flags", _extra)

	_all_toggle = Button.new()
	_all_toggle.toggle_mode = true
	_all_toggle.text = "▸ All options"
	_all_toggle.tooltip_text = "Every remaining flag this algorithm accepts, with defaults filled in."
	_all_toggle.toggled.connect(func(pressed):
		_all_box.visible = pressed
		_all_toggle.text = ("▼ All options" if pressed else "▸ All options"))
	page.add_child(_all_toggle)
	_all_box = VBoxContainer.new()
	_all_box.visible = false
	page.add_child(_all_box)
	return page


func _build_review_page() -> Control:
	var page := VBoxContainer.new()
	page.size_flags_vertical = Control.SIZE_EXPAND_FILL
	page.add_child(_heading("Review the equivalent command, then launch"))
	_preview = TextEdit.new()
	_preview.editable = false
	_preview.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_preview.custom_minimum_size = Vector2(0, 90)
	page.add_child(_preview)

	var buttons := HBoxContainer.new()
	_train_button = Button.new()
	_train_button.text = "Train"
	_train_button.pressed.connect(_on_train)
	buttons.add_child(_train_button)
	page.add_child(buttons)

	_status_label = Label.new()
	_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status_label.custom_minimum_size = Vector2(460, 0)
	page.add_child(_status_label)

	_log = TextEdit.new()
	_log.editable = false
	_log.custom_minimum_size = Vector2(0, 120)
	_log.size_flags_vertical = Control.SIZE_EXPAND_FILL
	page.add_child(_log)
	return page


# --- navigation / reset ---------------------------------------------------------------------------

func _show_page(index: int) -> void:
	_current = clampi(index, 0, _pages.size() - 1)
	for i in _pages.size():
		_pages[i].visible = i == _current
	_step_label.text = "Step %d of %d — %s" % [_current + 1, _pages.size(), PAGE_TITLES[_current]]
	_back_button.disabled = _current == 0
	_next_button.visible = _current < _pages.size() - 1
	if _current == 2:
		_update_algo_options()
	elif _current == 3:
		_update_preview()


func _reset() -> void:
	_algorithm.select(0)
	_scene.text = "res://scenarios/robotarms/XarmScenario.tscn"
	_num_envs.value = 8
	_base_port.value = 7200
	_episodes.value = 8000
	_max_steps.value = 300
	_physics_frames.value = 3
	_checkpoint_dir.text = "checkpoints/metis_run"
	_resume.text = ""
	_collector.select(0)
	_batch_size.value = 256
	_replay_warmup.value = 20000
	_multi_agent.button_pressed = false
	_best_metric.select(0)
	_auto_recovery.button_pressed = false
	_dashboard.button_pressed = false
	_dashboard_port.value = 8770
	_headless.button_pressed = true
	_critic_lr.text = ""
	_min_alpha.text = ""
	_policy_update_every.value = 2
	_grad_clip.button_pressed = true
	_extra.text = ""
	_status_label.text = ""
	_log.text = ""
	_train_button.disabled = false
	if _all_toggle != null:
		_all_toggle.button_pressed = false
		_all_box.visible = false
	_ensure_argspec()
	_on_algorithm_changed()
	_show_page(0)


func _on_algorithm_changed() -> void:
	if _algo_help != null:
		_algo_help.text = str(ALGO_INFO.get(_current_algo(), ""))


func _update_algo_options() -> void:
	var is_sac := _current_algo() == "sac"
	_algo_options_title.text = "%s options" % _current_algo().to_upper()
	_sac_group.visible = is_sac
	_no_options_note.visible = not is_sac
	_rebuild_all_options(_current_algo())


func _current_algo() -> String:
	return _algorithm.get_item_text(_algorithm.selected)


# --- "All options" (auto-generated from python/core/argspec.py) ------------------------------------

func _argspec_path() -> String:
	return ProjectSettings.globalize_path("res://.metis").path_join("argspec.json")


func _ensure_argspec() -> void:
	# Generate .metis/argspec.json in the BACKGROUND (imports TF, ~15s) if missing. Delete the file
	# to force a refresh after changing an algorithm's arguments.
	var path := _argspec_path()
	if FileAccess.file_exists(path):
		return
	var python := _python()
	if not FileAccess.file_exists(python):
		return
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var source_root := _source_python_root()
	if not source_root.is_empty():
		var script := source_root.path_join("core/argspec.py")
		if FileAccess.file_exists(script):
			OS.create_process(
				python,
				PackedStringArray([script, "--all", "--output", path]))
	else:
		OS.create_process(
			python,
			PackedStringArray([
				"-m", "core.argspec", "--all", "--output", path]))


func _load_argspec() -> Dictionary:
	var path := _argspec_path()
	if not FileAccess.file_exists(path):
		return {}
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	return parsed if parsed is Dictionary else {}


func _rebuild_all_options(algo: String) -> void:
	if _all_box == null:
		return
	for child in _all_box.get_children():
		child.queue_free()
	_all_controls.clear()
	var spec: Variant = _load_argspec().get(algo, [])
	if not (spec is Array) or (spec as Array).is_empty():
		var note := Label.new()
		note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		note.custom_minimum_size = Vector2(460, 0)
		note.text = "Generating the full option list in the background — reopen the wizard shortly."
		_all_box.add_child(note)
		return
	var by_group := {}
	var order: Array[String] = []
	for arg in spec:
		if not (arg is Dictionary) or str(arg.get("dest", "")) in CURATED_DESTS:
			continue
		var group_name := str(arg.get("group", ""))
		if group_name.is_empty():
			group_name = "General"
		if not by_group.has(group_name):
			by_group[group_name] = []
			order.append(group_name)
		by_group[group_name].append(arg)
	for group_name in order:
		var header := Label.new()
		header.text = group_name
		header.add_theme_color_override("font_color", Color(0.6, 0.72, 1.0))
		_all_box.add_child(header)
		var grid := _make_grid()
		_all_box.add_child(grid)
		for arg in by_group[group_name]:
			_add_all_option(grid, arg)


func _add_all_option(grid: GridContainer, arg: Dictionary) -> void:
	var kind := str(arg.get("kind", "str"))
	var control: Control
	if kind in ["bool", "flag_true", "flag_false"]:
		var check := CheckBox.new()
		check.button_pressed = bool(arg.get("default", false))
		control = check
	elif kind == "choice":
		var option := OptionButton.new()
		var choices: Array = arg.get("choices", [])
		for i in choices.size():
			option.add_item(str(choices[i]))
			if str(choices[i]) == str(arg.get("default", "")):
				option.select(i)
		control = option
	else:
		var edit := LineEdit.new()
		var default_value: Variant = arg.get("default", null)
		if default_value != null:
			edit.text = str(default_value)
		control = edit
	control.tooltip_text = str(arg.get("help", ""))
	var flags: Array = arg.get("flags", [])
	var label := str(flags[0]).trim_prefix("--") if not flags.is_empty() else str(arg.get("dest", ""))
	_row(grid, label, control)
	_all_controls[str(arg.get("dest", ""))] = {"control": control, "arg": arg}


func _all_options_flags() -> PackedStringArray:
	var out := PackedStringArray()
	for dest in _all_controls:
		var entry: Dictionary = _all_controls[dest]
		var arg: Dictionary = entry["arg"]
		var control = entry["control"]
		var kind := str(arg.get("kind", "str"))
		var flags: Array = arg.get("flags", [])
		if flags.is_empty():
			continue
		if kind == "bool":
			var value: bool = control.button_pressed
			if value != bool(arg.get("default", false)):
				var positive := ""
				var negative := ""
				for flag in flags:
					if str(flag).begins_with("--no-"):
						negative = str(flag)
					else:
						positive = str(flag)
				out.append(positive if value else negative)
		elif kind == "flag_true":
			if control.button_pressed and not bool(arg.get("default", false)):
				out.append(str(flags[0]))
		elif kind == "flag_false":
			if not control.button_pressed and bool(arg.get("default", true)):
				out.append(str(flags[0]))
		elif kind == "choice":
			var selected: String = control.get_item_text(control.selected)
			if selected != str(arg.get("default", "")):
				out.append(str(flags[0]))
				out.append(selected)
		else:
			var text := str(control.text).strip_edges()
			if not text.is_empty() and text != str(arg.get("default", "")):
				out.append(str(flags[0]))
				out.append(text)
	return out


func _open_scene_dialog() -> void:
	if _scene_dialog == null:
		_scene_dialog = EditorFileDialog.new()
		_scene_dialog.file_mode = EditorFileDialog.FILE_MODE_OPEN_FILE
		_scene_dialog.access = EditorFileDialog.ACCESS_RESOURCES
		_scene_dialog.clear_filters()
		_scene_dialog.add_filter("*.tscn", "Godot scenes")
		_scene_dialog.file_selected.connect(func(path): _scene.text = path)
		add_child(_scene_dialog)
	_scene_dialog.popup_file_dialog()


# --- command construction (preview == what is launched) -------------------------------------------

func _project_root() -> String:
	return ProjectSettings.globalize_path("res://").trim_suffix("/")


func _source_python_root() -> String:
	if runtime_manager != null:
		return runtime_manager.development_source_root()
	return ""


func _working_directory() -> String:
	var source_root := _source_python_root()
	return source_root.get_base_dir() if not source_root.is_empty() else _project_root()


func _python() -> String:
	if runtime_manager != null:
		return runtime_manager.suggested_python()
	return "python3"


func _command_prefix() -> PackedStringArray:
	var source_root := _source_python_root()
	if not source_root.is_empty():
		return PackedStringArray([_python(), source_root.path_join("train.py")])
	return PackedStringArray([_python(), "-m", "metis_cli", "train"])


func _build_args() -> PackedStringArray:
	var args := PackedStringArray()
	var algo := _current_algo()
	args.append_array(["--algorithm", algo])
	var godot_bin := OS.get_executable_path()
	if not godot_bin.is_empty():
		args.append_array(["--godot-bin", godot_bin])
	args.append_array(["--godot-project", _project_root()])
	if not _scene.text.strip_edges().is_empty():
		args.append_array(["--godot-scene", _scene.text.strip_edges()])
	args.append_array(["--num-envs", str(int(_num_envs.value))])
	args.append_array(["--base-port", str(int(_base_port.value))])
	args.append_array(["--num-episodes", str(int(_episodes.value))])
	args.append_array(["--max-steps-per-episode", str(int(_max_steps.value))])
	args.append_array(["--physics-frames-per-step", str(int(_physics_frames.value))])
	args.append_array(["--batch-size", str(int(_batch_size.value))])
	args.append_array(["--replay-warmup", str(int(_replay_warmup.value))])
	args.append_array(["--collector-mode", _collector.get_item_text(_collector.selected)])
	if _multi_agent.button_pressed:
		args.append("--multi-agent")
	args.append_array(["--checkpoint-dir", _checkpoint_dir.text.strip_edges()])
	if not _resume.text.strip_edges().is_empty():
		args.append_array(["--resume", "--resume-checkpoint", _resume.text.strip_edges()])
	args.append_array(["--best-metric", _best_metric.get_item_text(_best_metric.selected)])
	if _auto_recovery.button_pressed:
		args.append_array(["--auto-recovery", "--best-checkpoint"])
	if algo == "sac":
		if _grad_clip.button_pressed:
			args.append("--grad-clip-adaptive")
		if not _critic_lr.text.strip_edges().is_empty():
			args.append_array(["--critic-learning-rate", _critic_lr.text.strip_edges()])
		if not _min_alpha.text.strip_edges().is_empty():
			args.append_array(["--min-alpha", _min_alpha.text.strip_edges()])
		args.append_array(["--policy-update-every", str(int(_policy_update_every.value))])
	if _dashboard.button_pressed:
		args.append_array(["--dashboard", "--dashboard-port", str(int(_dashboard_port.value))])
	args.append(("--headless" if _headless.button_pressed else "--no-headless"))
	args.append_array(_all_options_flags())
	for token in _extra.text.strip_edges().split(" ", false):
		if not token.strip_edges().is_empty():
			args.append(token.strip_edges())
	return args


func _update_preview() -> void:
	if _preview == null:
		return
	var parts := _command_prefix()
	parts.append_array(_build_args())
	_preview.text = " ".join(parts)


# --- launch / monitor -----------------------------------------------------------------------------

func _on_train() -> void:
	if _runner.is_running():
		_status_label.text = "A run is already active — stop it from the terminal first."
		return
	var python := _python()
	if not FileAccess.file_exists(python):
		_status_label.text = "Python not found: %s. Configure the runtime first." % python
		return
	var metis_dir := ProjectSettings.globalize_path("res://.metis")
	DirAccess.make_dir_recursive_absolute(metis_dir)
	var command := _command_prefix()
	var executable := command[0]
	var command_args := command.slice(1)
	command_args.append_array(_build_args())
	var ok := _runner.start_command(
		executable,
		command_args,
		_working_directory(),
		metis_dir.path_join("train.log"),
		metis_dir.path_join("train.pid"))
	if not ok:
		_status_label.text = "Failed to spawn the training process."
		return
	_train_button.disabled = true
	_status_label.text = "Training launched (detached). It keeps running if you close the editor."
	set_process(true)


func _process(_delta: float) -> void:
	var log_text := _runner.read_log()
	if _log.text != log_text:
		_log.text = log_text
		_log.scroll_vertical = _log.get_line_count()
	if not _runner.is_running():
		set_process(false)
		_status_label.text = "Run finished or stopped."


# --- tiny UI helpers ------------------------------------------------------------------------------

func _heading(text: String) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_color_override("font_color", Color(0.78, 0.81, 0.88))
	return label


func _make_grid() -> GridContainer:
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 6)
	grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	return grid


func _row(form: GridContainer, label_text: String, control: Control) -> void:
	var label := Label.new()
	label.text = label_text
	label.custom_minimum_size = Vector2(LABEL_WIDTH, 0)
	form.add_child(label)
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	form.add_child(control)


func _line(value: String) -> LineEdit:
	var edit := LineEdit.new()
	edit.text = value
	return edit


func _spin(min_value: float, max_value: float, value: float) -> SpinBox:
	var spin := SpinBox.new()
	spin.min_value = min_value
	spin.max_value = max_value
	spin.value = value
	spin.step = 1
	return spin
