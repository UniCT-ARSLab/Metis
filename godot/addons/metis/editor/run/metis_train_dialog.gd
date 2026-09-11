@tool
class_name MetisTrainDialog
extends AcceptDialog

## Configures and launches a detached Metis training run from the editor.
## The review page shows the equivalent terminal command.

signal training_started

const BRANDING := preload("res://addons/metis/editor/metis_branding.gd")
const RUN_STATE := preload("res://addons/metis/editor/run/metis_run_state.gd")
const SCENARIO := preload("res://addons/metis/editor/run/metis_scenario_scan.gd")
const OPTION_FORM := preload("res://addons/metis/editor/run/metis_option_form.gd")

const LABEL_WIDTH := 160
const PAGE_TITLES := ["Algorithm", "Common settings", "Algorithm options", "Review & launch"]

const SCENE_PLACEHOLDER := "res://path/to/your_scenario.tscn"


# Options already shown on the common page.
const CURATED_DESTS := [
	"algorithm", "godot_bin", "godot_project", "godot_scene", "num_envs", "base_port",
	"num_episodes", "max_steps_per_episode", "physics_frames_per_step", "batch_size",
	"replay_warmup", "collector_mode", "multi_agent", "checkpoint_dir", "resume",
	"resume_checkpoint", "best_metric", "auto_recovery", "best_checkpoint", "dashboard",
	"dashboard_port", "headless", "render_env_count", "render_mode",
]

# Prominent algorithm options. Types and defaults still come from argspec.json.
const ALGO_OPTIONS := {
	"sac": [
		["network_layers", "Hidden layers"],
		["actor_learning_rate", "Actor LR"],
		["critic_learning_rate", "Critic LR"],
		["alpha_learning_rate", "Alpha LR"],
		["initial_alpha", "Initial alpha"],
		["tune_alpha", "Auto-tune alpha"],
		["min_alpha", "Min alpha (entropy floor)"],
		["target_entropy", "Target entropy"],
		["policy_update_every", "Policy update every"],
		["tau", "Target smoothing (tau)"],
		["gamma", "Discount (gamma)"],
		["grad_clip_adaptive", "Adaptive grad clip"],
		["grad_clip_norm", "Grad clip norm"],
	],
	"ppo": [
		["network_layers", "Hidden layers"],
		["learning_rate", "Learning rate"],
		["ppo_rollout_steps", "Rollout steps"],
		["ppo_epochs", "Epochs per rollout"],
		["clip_ratio", "Clip ratio"],
		["gae_lambda", "GAE lambda"],
		["entropy_coef", "Entropy coefficient"],
		["value_loss_coef", "Value loss coefficient"],
		["initial_log_std", "Initial log std (continuous)"],
		["gamma", "Discount (gamma)"],
	],
	"dqn": [
		["network_layers", "Hidden layers"],
		["learning_rate", "Learning rate"],
		["gamma", "Discount (gamma)"],
		["target_update_every", "Target update every (episodes)"],
		["target_update_steps", "Target update every (steps)"],
		["epsilon_start", "Epsilon start"],
		["epsilon_min", "Epsilon min"],
		["epsilon_decay", "Epsilon decay"],
		["epsilon_decay_horizon_fraction", "Epsilon decay horizon"],
		["critic_loss", "Q loss"],
		["huber_delta", "Huber delta"],
		["grad_clip_norm", "Grad clip norm"],
		["grad_clip_adaptive", "Adaptive grad clip"],
	],
	"td3": [
		["network_layers", "Hidden layers"],
		["actor_learning_rate", "Actor LR"],
		["critic_learning_rate", "Critic LR"],
		["td3_policy_delay", "Policy delay"],
		["td3_target_policy_noise", "Target policy noise"],
		["td3_target_noise_clip", "Target noise clip"],
		["exploration_noise", "Exploration noise"],
		["exploration_noise_kind", "Noise kind"],
		["tau", "Target smoothing (tau)"],
		["gamma", "Discount (gamma)"],
		["grad_clip_adaptive", "Adaptive grad clip"],
	],
	"ddpg": [
		["network_layers", "Hidden layers"],
		["actor_learning_rate", "Actor LR"],
		["critic_learning_rate", "Critic LR"],
		["exploration_noise", "Exploration noise"],
		["exploration_noise_kind", "Noise kind"],
		["exploration_noise_min", "Exploration noise min"],
		["exploration_noise_decay", "Exploration noise decay"],
		["tau", "Target smoothing (tau)"],
		["gamma", "Discount (gamma)"],
		["grad_clip_adaptive", "Adaptive grad clip"],
	],
}

# Source labels shown beside each default network layout.
const ALGO_ARCH := {
	"sac": "Default hidden layers: 256 256 (Haarnoja et al. 2018).",
	"ppo": "Default hidden layers: 64 64 (Schulman et al. 2017).",
	"dqn": "Default hidden layers: 64 64 — common practice for vector observations; the 2015 DQN "
		+ "paper is convolutional and has no dense-only reference.",
	"ddpg": "Default hidden layers: 400 300 (Lillicrap et al. 2015).",
	"td3": "Default hidden layers: 400 300 (Fujimoto et al. 2018).",
}

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

var _algorithm: OptionButton
var _algo_help: Label
var _scene: LineEdit
var _scene_note: Label
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
var _show_envs: CheckBox
var _render_count: SpinBox
var _render_mode: OptionButton
var _render_note: Label
var _algo_options_title: Label
var _algo_box: VBoxContainer
var _algo_note: Label
var _algo_controls := {}  # dest -> {"control": Control, "arg": Dictionary}
var _options_algo := ""  # algorithm the option rows were last built for
var _refresh_button: Button
var _extra: LineEdit
var _all_toggle: Button
var _all_box: VBoxContainer
var _all_controls := {}  # dest -> {"control": Control, "arg": Dictionary}
var _preview: TextEdit
var _train_button: Button
var _status_label: Label


func _ready() -> void:
	title = "Metis — Train"
	auto_translate_mode = Node.AUTO_TRANSLATE_MODE_DISABLED
	min_size = Vector2i(660, 470)
	get_ok_button().text = "Close"
	about_to_popup.connect(_reset)

	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 8)
	add_child(root)

	root.add_child(BRANDING.header("Metis — Train", "Configure and launch a training run"))

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

	_reset()


func configure(manager: MetisRuntimeManager) -> void:
	runtime_manager = manager
	if not is_node_ready():
		await ready


# pages

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
	_scene = _line("")
	_scene.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_scene.placeholder_text = SCENE_PLACEHOLDER
	scene_box.add_child(_scene)
	var browse := Button.new()
	browse.text = "Browse…"
	browse.pressed.connect(_open_scene_dialog)
	scene_box.add_child(browse)
	_row(form, "Godot scene", scene_box)
	_scene_note = Label.new()
	_scene_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_scene_note.custom_minimum_size = Vector2(460, 0)
	_scene_note.add_theme_color_override("font_color", Color(0.78, 0.81, 0.88))
	page.add_child(_scene_note)

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
	_dashboard.text = "on port"
	dash_box.add_child(_dashboard)
	_dashboard_port = _spin(1024, 65000, 8770)
	dash_box.add_child(_dashboard_port)
	_row(form, "Live dashboard", dash_box)
	# One mode control avoids contradictory headless and render-count settings.
	var render_box := HBoxContainer.new()
	_show_envs = CheckBox.new()
	_show_envs.text = "Show"
	_show_envs.toggled.connect(func(pressed):
		_render_count.editable = pressed
		_render_mode.disabled = not pressed
		_update_render_note())
	render_box.add_child(_show_envs)
	_render_count = _spin(1, 64, 1)
	_render_count.editable = false
	_render_count.value_changed.connect(func(_v): _update_render_note())
	render_box.add_child(_render_count)
	var of_label := Label.new()
	of_label.text = "window(s), rendered with"
	render_box.add_child(of_label)
	_render_mode = OptionButton.new()
	for mode in ["light-gpu", "gpu", "cpu", "project"]:
		_render_mode.add_item(mode)
	_render_mode.select(0)
	_render_mode.disabled = true
	render_box.add_child(_render_mode)
	_row(form, "Environment windows", render_box)
	_render_note = Label.new()
	_render_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_render_note.custom_minimum_size = Vector2(460, 0)
	_render_note.modulate = Color(1.0, 1.0, 1.0, 0.7)
	page.add_child(_render_note)
	_num_envs.value_changed.connect(func(_v): _clamp_render_count())
	return page


func _clamp_render_count() -> void:
	## Keeps rendered windows within the environment count.
	_render_count.max_value = maxf(1.0, _num_envs.value)
	_render_count.value = minf(_render_count.value, _render_count.max_value)
	_update_render_note()


func _update_render_note() -> void:
	if _render_note == null:
		return
	var total := int(_num_envs.value)
	if not _show_envs.button_pressed:
		_render_note.text = ("All %d environments run headless — no windows, and Godot gets "
			+ "--fixed-fps so physics is not gated to wall-clock 60 Hz. Fastest by a wide margin.") % total
		return
	var shown := mini(int(_render_count.value), total)
	_render_note.text = ("%d of %d environments open a window; the other %d stay headless. Watching "
		+ "costs throughput — the rendered instances run at display rate.") % [shown, total,
		total - shown]


func _build_algo_options_page() -> Control:
	var page := VBoxContainer.new()
	_algo_options_title = _heading("Algorithm-specific options")
	page.add_child(_algo_options_title)

	_algo_note = Label.new()
	_algo_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_algo_note.custom_minimum_size = Vector2(460, 0)
	_algo_note.add_theme_color_override("font_color", Color(0.78, 0.81, 0.88))
	page.add_child(_algo_note)

	_algo_box = VBoxContainer.new()
	page.add_child(_algo_box)

	var extra_form := _make_grid()
	page.add_child(extra_form)
	_extra = _line("")
	_extra.placeholder_text = "any other raw flags, appended verbatim"
	_row(extra_form, "Extra flags", _extra)

	var more := HBoxContainer.new()
	page.add_child(more)
	_all_toggle = Button.new()
	_all_toggle.toggle_mode = true
	_all_toggle.text = "▸ All options"
	_all_toggle.tooltip_text = "Every remaining flag this algorithm accepts, with defaults filled in."
	_all_toggle.toggled.connect(func(pressed):
		_all_box.visible = pressed
		_all_toggle.text = ("▼ All options" if pressed else "▸ All options"))
	more.add_child(_all_toggle)
	var more_spacer := Control.new()
	more_spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	more.add_child(more_spacer)
	_refresh_button = Button.new()
	_refresh_button.text = "Refresh option list"
	_refresh_button.tooltip_text = ("Re-read the flags from the Python CLI. Done automatically when "
		+ "the sources are newer than the cached list; use this after editing a parser by hand.")
	_refresh_button.pressed.connect(_refresh_argspec)
	more.add_child(_refresh_button)
	_all_box = VBoxContainer.new()
	_all_box.visible = false
	page.add_child(_all_box)
	return page


func _build_review_page() -> Control:
	var page := VBoxContainer.new()
	page.add_theme_constant_override("separation", 8)
	page.size_flags_vertical = Control.SIZE_EXPAND_FILL
	page.add_child(_heading("Review the equivalent command, then launch"))
	_preview = TextEdit.new()
	_preview.editable = false
	_preview.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_preview.custom_minimum_size = Vector2(0, 80)
	page.add_child(_preview)

	var buttons := HBoxContainer.new()
	_train_button = Button.new()
	_train_button.text = "Train"
	_train_button.custom_minimum_size = Vector2(150, 34)
	var theme := EditorInterface.get_editor_theme()
	if theme != null:
		if theme.has_icon("Play", "EditorIcons"):
			_train_button.icon = theme.get_icon("Play", "EditorIcons")
		var accent := theme.get_color("accent_color", "Editor")
		_train_button.add_theme_color_override("font_color", accent)
		_train_button.add_theme_color_override("font_hover_color", accent)
		_train_button.add_theme_color_override("font_focus_color", accent)
	_train_button.pressed.connect(_on_train)
	buttons.add_child(_train_button)

	var buttons_spacer := Control.new()
	buttons_spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	buttons.add_child(buttons_spacer)
	page.add_child(buttons)

	var detached := Label.new()
	detached.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	detached.custom_minimum_size = Vector2(460, 0)
	detached.modulate = Color(1.0, 1.0, 1.0, 0.7)
	detached.text = ("The run is launched detached: you can close this window, or the editor itself, "
		+ "and training keeps going. Use the training monitor, which opens as soon as the run "
		+ "starts, to follow it or stop it.")
	page.add_child(detached)

	_status_label = Label.new()
	_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status_label.custom_minimum_size = Vector2(460, 0)
	page.add_child(_status_label)
	return page


# navigation / reset

func _show_page(index: int) -> void:
	_current = clampi(index, 0, _pages.size() - 1)
	for i in _pages.size():
		_pages[i].visible = i == _current
	_step_label.text = "Step %d of %d — %s" % [_current + 1, _pages.size(), PAGE_TITLES[_current]]
	_back_button.disabled = _current == 0
	_next_button.visible = _current < _pages.size() - 1
	if _current == 3:
		_update_preview()


func _reset() -> void:
	_algorithm.select(0)
	_apply_open_scene()
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
	_dashboard.button_pressed = true
	_dashboard_port.value = 8770
	_show_envs.button_pressed = false
	_render_count.value = 1
	_render_count.editable = false
	_render_mode.select(0)
	_render_mode.disabled = true
	_clamp_render_count()
	_extra.text = ""
	if _all_toggle != null:
		_all_toggle.button_pressed = false
		_all_box.visible = false
	_ensure_argspec()
	_options_algo = ""  # force a rebuild: reopening the wizard must clear the previous run's edits
	_on_algorithm_changed()

	_train_button.disabled = false
	_status_label.text = ""
	_show_page(0)


func _on_algorithm_changed() -> void:
	if _algo_help != null:
		_algo_help.text = str(ALGO_INFO.get(_current_algo(), ""))
	_update_algo_options()


func _update_algo_options() -> void:
	## Rebuilds option rows after the selected algorithm changes.
	if _algo_box == null:
		return
	var algo := _current_algo()
	if algo == _options_algo:
		return
	_options_algo = algo
	_algo_options_title.text = "%s options" % algo.to_upper()
	_rebuild_algo_options(algo)
	_rebuild_all_options(algo)


func _rebuild_algo_options(algo: String) -> void:
	_clear(_algo_box)
	_algo_controls.clear()

	if algo == "auto":
		_algo_note.text = ("\"auto\" resolves the algorithm from the scenario's action space when "
			+ "training starts, so its options are not known yet. Pick a concrete algorithm on step 1 "
			+ "to tune it, or leave the defaults and use the Extra flags box below.")
		return

	var spec := _spec_by_dest(algo)
	if spec.is_empty():
		_algo_note.text = ("Reading the option list from the Python CLI in the background — press "
			+ "\"Refresh option list\" in a few seconds, or use the Extra flags box below.")
		return

	_algo_note.text = str(ALGO_ARCH.get(algo, ""))
	var grid := _make_grid()
	_algo_box.add_child(grid)
	for entry in ALGO_OPTIONS.get(algo, []):
		var dest := str(entry[0])
		if not spec.has(dest):
			continue
		_add_option(grid, spec[dest], _algo_controls, str(entry[1]))


func _current_algo() -> String:
	return _algorithm.get_item_text(_algorithm.selected)


# option list (auto-generated from python/core/argspec.py)

# Parser changes invalidate the cached option list.
const ARGSPEC_SOURCES := [
	"core/argspec.py", "core/training.py", "algorithms/sac.py", "algorithms/ppo.py",
	"algorithms/dqn.py", "algorithms/common.py",
]


func _argspec_path() -> String:
	return ProjectSettings.globalize_path("res://.metis").path_join("argspec.json")


func _argspec_is_stale() -> bool:
	var path := _argspec_path()
	if not FileAccess.file_exists(path):
		return true
	var source_root := _source_python_root()
	if source_root.is_empty():
		return false
	var spec_time := FileAccess.get_modified_time(path)
	for relative in ARGSPEC_SOURCES:
		var source := source_root.path_join(relative)
		if FileAccess.file_exists(source) and FileAccess.get_modified_time(source) > spec_time:
			return true
	return false


func _ensure_argspec() -> void:
	# Python imports can take several seconds, so regenerate without blocking the editor.
	if not _argspec_is_stale():
		return
	var path := _argspec_path()
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


func _refresh_argspec() -> void:
	## Rebuilds option sections after the background specification job finishes.
	var path := _argspec_path()
	if FileAccess.file_exists(path):
		DirAccess.remove_absolute(path)
	_refresh_button.disabled = true
	_refresh_button.text = "Reading the CLI…"
	_ensure_argspec()
	for _attempt in 40:
		await get_tree().create_timer(1.0).timeout
		if not is_inside_tree():
			return
		if FileAccess.file_exists(path):
			break
	_refresh_button.disabled = false
	_refresh_button.text = "Refresh option list"
	_options_algo = ""  # the spec changed, so the rows must be rebuilt even for the same algorithm
	_update_algo_options()
	if _current == 3:
		_update_preview()


func _load_argspec() -> Dictionary:
	var path := _argspec_path()
	if not FileAccess.file_exists(path):
		return {}
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	return parsed if parsed is Dictionary else {}


func _spec_by_dest(algo: String) -> Dictionary:
	var out := {}
	var spec: Variant = _load_argspec().get(algo, [])
	if not (spec is Array):
		return out
	for arg in spec:
		if arg is Dictionary:
			out[str(arg.get("dest", ""))] = arg
	return out


func _rebuild_all_options(algo: String) -> void:
	if _all_box == null:
		return
	_clear(_all_box)
	_all_controls.clear()
	# Auto chooses an algorithm later, so it has no dedicated parser here.
	_all_toggle.visible = algo != "auto"
	_refresh_button.visible = algo != "auto"
	if algo == "auto":
		_all_toggle.button_pressed = false
		_all_box.visible = false
		return
	var spec: Variant = _load_argspec().get(algo, [])
	if not (spec is Array) or (spec as Array).is_empty():
		var note := Label.new()
		note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		note.custom_minimum_size = Vector2(460, 0)
		note.text = "Generating the full option list in the background — reopen the wizard shortly."
		_all_box.add_child(note)
		return
	var curated := PackedStringArray(CURATED_DESTS)
	for entry in ALGO_OPTIONS.get(algo, []):
		curated.append(str(entry[0]))
	var by_group := {}
	var order: Array[String] = []
	for arg in spec:
		if not (arg is Dictionary) or str(arg.get("dest", "")) in curated:
			continue
		var group_name := str(arg.get("group", ""))
		if group_name.is_empty():
			group_name = "General"
		if not by_group.has(group_name):
			by_group[group_name] = []
			order.append(group_name)
		by_group[group_name].append(arg)
	for group_name in order:
		var grid := _make_grid()
		grid.visible = false
		var section := Button.new()
		section.toggle_mode = true
		section.alignment = HORIZONTAL_ALIGNMENT_LEFT
		section.text = "▸ %s (%d)" % [group_name, by_group[group_name].size()]
		section.add_theme_color_override("font_color", Color(0.6, 0.72, 1.0))
		section.toggled.connect(func(pressed):
			grid.visible = pressed
			section.text = "%s %s (%d)" % [("▼" if pressed else "▸"), group_name,
				by_group[group_name].size()])
		_all_box.add_child(section)
		_all_box.add_child(grid)
		for arg in by_group[group_name]:
			_add_option(grid, arg, _all_controls)


func _add_option(grid: GridContainer, arg: Dictionary, controls: Dictionary,
		label_override: String = "") -> void:
	## Adds and registers one option row.
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
			edit.text = OPTION_FORM.format_default(arg, default_value)
		elif _takes_many_values(arg):
			edit.placeholder_text = "space-separated, e.g. 256 256"
		else:
			edit.placeholder_text = "(unset)"
		OPTION_FORM.mark_validity(edit, arg)
		control = edit
	control.tooltip_text = str(arg.get("help", ""))
	var label := label_override
	if label.is_empty():
		var flags: Array = arg.get("flags", [])
		label = str(flags[0]).trim_prefix("--") if not flags.is_empty() else str(arg.get("dest", ""))
	_row(grid, label, control)
	controls[str(arg.get("dest", ""))] = {"control": control, "arg": arg}


func _validation_errors() -> PackedStringArray:
	## Returns values the CLI would reject.
	var problems := PackedStringArray()
	for controls in [_algo_controls, _all_controls]:
		for dest in controls:
			var entry: Dictionary = controls[dest]
			if not (entry["control"] is LineEdit):
				continue
			var problem: String = OPTION_FORM.value_error(entry["arg"], str(entry["control"].text))
			if not problem.is_empty():
				problems.append(problem)
	return problems


func _takes_many_values(arg: Dictionary) -> bool:
	var nargs: Variant = arg.get("nargs", null)
	if nargs == null:
		return false
	var text := str(nargs)
	return text in ["+", "*"] or (text.is_valid_int() and text.to_int() > 0)


func _flags_from_controls(controls: Dictionary) -> PackedStringArray:
	## Emits changed values as command-line tokens.
	var out := PackedStringArray()
	for dest in controls:
		var entry: Dictionary = controls[dest]
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
			if not text.is_empty() and text != OPTION_FORM.format_default(arg, arg.get("default", "")):
				out.append(str(flags[0]))
				if _takes_many_values(arg):
					# argparse expects one token per item for nargs options.
					for token in text.split(" ", false):
						if not token.strip_edges().is_empty():
							out.append(token.strip_edges())
				else:
					out.append(text)
	return out


# scene detection

func scene_has_bridge_server(path: String) -> bool:
	return SCENARIO.has_bridge_server(path)


func _apply_open_scene() -> void:
	## Uses the trainable scene open when the wizard appears.
	var root := EditorInterface.get_edited_scene_root()
	var path := "" if root == null else root.scene_file_path
	if path.is_empty():
		_set_scene("", "No saved scene is open. Browse for a scenario, or open one first — a "
			+ "trainable scenario is a scene containing a %s node." % SCENARIO.BRIDGE_CLASS)
	elif not scene_has_bridge_server(path):
		_set_scene("", "\"%s\" has no %s node, so it cannot be trained on its own. Browse for the "
			% [path.get_file(), SCENARIO.BRIDGE_CLASS] + "scenario scene that instantiates it.")
	else:
		_set_scene(path, "Using the scene open in the editor.")


func _set_scene(path: String, note: String) -> void:
	_scene.text = path
	_scene_note.text = note


func _open_scene_dialog() -> void:
	if _scene_dialog == null:
		_scene_dialog = EditorFileDialog.new()
		_scene_dialog.file_mode = EditorFileDialog.FILE_MODE_OPEN_FILE
		_scene_dialog.access = EditorFileDialog.ACCESS_RESOURCES
		_scene_dialog.clear_filters()
		_scene_dialog.add_filter("*.tscn", "Godot scenes")
		_scene_dialog.file_selected.connect(func(path):
			if scene_has_bridge_server(path):
				_set_scene(path, "")
			else:
				_set_scene(path, "Warning: \"%s\" contains no %s node. Training will start Godot and "
					% [path.get_file(), SCENARIO.BRIDGE_CLASS]
					+ "then wait for a connection that never arrives."))
		add_child(_scene_dialog)
	_scene_dialog.popup_file_dialog()


# command construction (preview == what is launched)

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
	if _dashboard.button_pressed:
		args.append_array(["--dashboard", "--dashboard-port", str(int(_dashboard_port.value))])
	if _show_envs.button_pressed:
		args.append("--no-headless")
		args.append_array(["--render-env-count", str(int(_render_count.value))])
		args.append_array(["--render-mode", _render_mode.get_item_text(_render_mode.selected)])
	else:
		args.append("--headless")
	args.append_array(_flags_from_controls(_algo_controls))
	args.append_array(_flags_from_controls(_all_controls))
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


# launch

func _on_train() -> void:
	if RUN_STATE.is_run_active():
		_status_label.text = ("A run is already active. Open the training monitor to watch or stop "
			+ "it — Metis will not start another on top of it.")
		return
	var python := _python()
	if not FileAccess.file_exists(python):
		_status_label.text = "Python not found: %s. Configure the runtime first." % python
		return
	var problems := _validation_errors()
	if not problems.is_empty():
		_status_label.text = "Fix these first — " + ", ".join(problems)
		return
	DirAccess.make_dir_recursive_absolute(RUN_STATE.directory())
	var command := _command_prefix()
	var executable := command[0]
	var command_args := command.slice(1)
	command_args.append_array(_build_args())
	var ok := _runner.start_command(
		executable,
		command_args,
		_working_directory(),
		RUN_STATE.log_path(),
		RUN_STATE.pid_path())
	if not ok:
		_status_label.text = "Could not start training."
		return
	# The monitor needs values that are not present in the trainer log.
	RUN_STATE.write({
		"algorithm": _current_algo(),
		"scene": _scene.text.strip_edges(),
		"checkpoint_dir": _checkpoint_dir.text.strip_edges(),
		"episodes": int(_episodes.value),
		"dashboard": _dashboard.button_pressed,
		"dashboard_port": int(_dashboard_port.value),
	})
	_train_button.disabled = true
	_status_label.text = "Launched."
	# Godot allows only one exclusive child window at a time.
	hide()
	training_started.emit()



func _heading(text: String) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_color_override("font_color", Color(0.78, 0.81, 0.88))
	return label


func _clear(box: Control) -> void:
	## Detaches children immediately so the form can be rebuilt in the same frame.
	for child in box.get_children():
		box.remove_child(child)
		child.queue_free()


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
