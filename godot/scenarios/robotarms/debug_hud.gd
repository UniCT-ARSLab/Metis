extends CanvasLayer
## Live debug HUD (top-left): the three success gates with pass/fail, hold counter, per-step
## Reward, current-episode running total, and the previous episode's total, outcome and
## distances. Renders only in a windowed env; skipped in headless training.

@onready var _label: Label = $Label

var _arm: Node = null
var _controller: Node = null
var _scenario: Node = null

var _episode_reward := 0.0
var _last_ep_reward := 0.0
var _episode_last_distance := 0.0
var _episode_min_distance := INF
var _last_ep_final_distance := 0.0
var _last_ep_min_distance := 0.0
var _episode_last_angle := 0.0
var _episode_min_angle := INF
var _last_ep_final_angle := 0.0
var _last_ep_min_angle := 0.0
var _last_step := -1
var _last_outcome := "-"
var _ep_had_success := false
var _ep_had_collision := false
var _ep_collision_src := ""
## The controller publishes last_reward only after advancing physics, so _process sees the new
## step_count while last_reward still belongs to the previous step. This flag keeps that stale
## value from leaking into the next episode's running total.
var _episode_closed := true


func _ready() -> void:
	# No window in headless training envs: skip the per-frame work entirely.
	if DisplayServer.get_name() == "headless":
		visible = false
		set_process(false)
		return
	_scenario = get_parent()
	if _scenario != null:
		_arm = _scenario.get_node_or_null("OpenarmAgent")
		_controller = _scenario.get_node_or_null("ScenarioController")
	if _controller != null and _controller.has_signal("episode_reset_started"):
		_controller.episode_reset_started.connect(_on_episode_reset)


func _on_episode_reset(_seed: int = 0) -> void:
	_finalize_episode()


## Close out the episode that just ended: add its final step reward (still pending in
## last_reward), freeze total/outcome/distances, then zero the running state. The reset signal
## fires before the arm returns home, so the distances read here are the real end-of-episode ones.
func _finalize_episode() -> void:
	if _episode_closed:
		return
	_episode_closed = true
	if _controller != null:
		_episode_reward += _controller.last_reward
	_last_ep_reward = _episode_reward
	_last_ep_final_distance = _episode_last_distance
	_last_ep_min_distance = (
		_episode_last_distance
		if is_inf(_episode_min_distance)
		else _episode_min_distance)
	_last_ep_final_angle = _episode_last_angle
	_last_ep_min_angle = (
		_episode_last_angle
		if is_inf(_episode_min_angle)
		else _episode_min_angle)
	if _ep_had_collision:
		_last_outcome = "COLLISION [%s]" % ("self" if _ep_collision_src == "self_body" else "table/env")
	elif _ep_had_success:
		_last_outcome = "REACHED"
	else:
		_last_outcome = "STALL / TIMEOUT"
	_episode_reward = 0.0
	_episode_min_distance = INF
	_episode_min_angle = INF
	_last_step = _controller.step_count if _controller != null else -1
	_ep_had_success = false
	_ep_had_collision = false
	_ep_collision_src = ""


func _gate_flag(passed: bool) -> String:
	return "OK  " if passed else "MISS"


func _process(_delta: float) -> void:
	if _arm == null or _label == null:
		return

	# Latch the outcome the instant it happens (survives continue-after-success / truncation).
	if _arm._succeeded:
		_ep_had_success = true
	if _arm._collided:
		_ep_had_collision = true
		_ep_collision_src = str(_arm.get_last_collision_details().get("source", ""))

	# Tracked before the step bookkeeping below, because _finalize_episode() reads these to freeze
	# the ending episode's values.
	var distance: float = _arm.get_target_distance()
	_episode_last_distance = distance
	_episode_min_distance = minf(_episode_min_distance, distance)
	var angle_degrees: float = rad_to_deg(_arm._target_angle_error())
	_episode_last_angle = angle_degrees
	_episode_min_angle = minf(_episode_min_angle, angle_degrees)

	# Reset is normally driven by the episode_reset_started signal; the step_count-drop check is a
	# fallback in case a reset was missed. step_count resets to 0 at the start of every episode.
	if _controller != null:
		var sc: int = _controller.step_count
		if sc != _last_step:
			if sc < _last_step:
				_finalize_episode()
			if not _episode_closed:
				_episode_reward += _controller.last_reward
			_episode_closed = false
			_last_step = sc

	var joint_speed: float = _arm._max_joint_speed()
	var distance_gate: float = _arm.success_distance
	var angle_gate: float = _arm.success_angle_degrees
	var speed_gate: float = _arm.success_max_joint_speed
	var requires_still: bool = _arm.success_require_still
	var hold_total: int = _arm.success_hold_physics_frames
	var hold_frames: int = _arm._success_frames
	var progress: float = _arm.get_progress()
	var reward: float = _controller.last_reward if _controller != null else 0.0
	var step: int = _controller.step_count if _controller != null else 0
	var level := 0.0
	if _scenario != null and _scenario.has_method("_adaptive_level"):
		level = _scenario._adaptive_level()

	_label.text = (
		"dist    : %6.2f cm  %s gate %5.2f\n" % [
			distance * 100.0,
			_gate_flag(distance <= distance_gate),
			distance_gate * 100.0]
		+ "orient  : %6.1f deg %s gate %5.1f\n" % [
			angle_degrees,
			_gate_flag(angle_degrees <= angle_gate),
			angle_gate]
		+ "speed   : %6.3f     %s gate %5.3f\n" % [
			joint_speed,
			_gate_flag(not requires_still or joint_speed <= speed_gate),
			speed_gate]
		+ "hold    : %d / %d frames\n" % [hold_frames, hold_total]
		+ "progress: %.3f   level %.2f   step %d\n" % [progress, level, step]
		+ "Reward  : %+7.2f   Episode %+8.1f\n" % [reward, _episode_reward]
		+ "-- prev ep --\n"
		+ "outcome : %-14s rew %+8.1f\n" % [_last_outcome, _last_ep_reward]
		+ "dist    : fin %6.2f   min %6.2f  cm\n" % [
			_last_ep_final_distance * 100.0,
			_last_ep_min_distance * 100.0]
		+ "orient  : fin %6.1f   min %6.1f  deg" % [
			_last_ep_final_angle,
			_last_ep_min_angle]
	)
