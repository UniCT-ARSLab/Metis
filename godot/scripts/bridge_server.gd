extends Node
class_name BridgeServer

@export var port := 5555
@export var controller_path: NodePath
@export var verbose := false
@export var auto_silence_in_headless := true
@export_category("Simulation Clock")
@export var lockstep_enabled := true
@export_range(0, 10000, 100) var lockstep_idle_sleep_usec := 500
@export_range(0, 10000, 100) var lockstep_headless_idle_sleep_usec := 2000

var server := TCPServer.new()
var client: StreamPeerTCP = null
var controller
var _rx_buffer := ""
var _lockstep_active := false
var _tree_was_paused := false
var _realtime_frame_cap_active := false
var _max_fps_before_realtime := 0
var _realtime_simulation_fps := 60
var _realtime_next_frame_usec := 0

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	controller = get_node(controller_path)
	if auto_silence_in_headless and _is_headless():
		verbose = false
	if _is_headless():
		lockstep_idle_sleep_usec = maxi(lockstep_idle_sleep_usec, lockstep_headless_idle_sleep_usec)

	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--port="):
			port = int(arg.split("=")[1])

	var err := server.listen(port)
	if err != OK:
		push_error("Cannot listen on port %d" % port)
		return

	print("[BridgeServer] Listening on port %d" % port)


func _is_headless() -> bool:
	return DisplayServer.get_name().to_lower() == "headless" or OS.has_feature("headless")

func _process(_delta: float) -> void:
	if not lockstep_enabled and client != null:
		_pace_realtime_loop()
	if client == null:
		if server.is_connection_available():
			client = server.take_connection()
			_rx_buffer = ""
			_activate_lockstep()
			print("[BridgeServer] Client connected on port %d" % port)
		elif lockstep_enabled and lockstep_idle_sleep_usec > 0:
			OS.delay_usec(lockstep_idle_sleep_usec)
		return

	# A step reply is sent before _process. Give Python a brief chance to return the
	# next action so it can still be consumed in this render frame.
	if _lockstep_active and lockstep_idle_sleep_usec > 0:
		OS.delay_usec(lockstep_idle_sleep_usec)
	client.poll()
	var status := client.get_status()
	if status == StreamPeerTCP.STATUS_NONE or status == StreamPeerTCP.STATUS_ERROR:
		_disconnect_client()
		return

	var available := client.get_available_bytes()
	if available <= 0:
		return

	_rx_buffer += client.get_utf8_string(available)
	while _rx_buffer.contains("\n"):
		var line_end := _rx_buffer.find("\n")
		var line := _rx_buffer.substr(0, line_end).strip_edges()
		_rx_buffer = _rx_buffer.substr(line_end + 1)
		if line.is_empty():
			continue
		_handle_line(line)

func _handle_line(line: String) -> void:
	var request = JSON.parse_string(line)
	if typeof(request) != TYPE_DICTIONARY:
		_send({
			"ok": false,
			"error": "Expected JSON object"
		})
		return

	var cmd := str(request.get("cmd", ""))
	if verbose:
		print("[BridgeServer] cmd=%s port=%d" % [cmd, port])
	match cmd:
		"hello":
			_send({
				"ok": true,
				"version": 1,
				"lockstep": _lockstep_active,
				"execution_mode": "lockstep" if _lockstep_active else "realtime"
			})
		"spec":
			_send(_call_spec())
		"execution_mode":
			var mode := str(request.get("mode", "lockstep")).to_lower()
			var simulation_fps := maxi(1, int(request.get("simulation_fps", 60)))
			if mode != "lockstep" and mode != "realtime":
				_send({
					"ok": false,
					"error": "execution_mode must be 'lockstep' or 'realtime'"
				})
				return
			_set_execution_mode(mode, simulation_fps)
			_send({
				"ok": true,
				"mode": mode,
				"lockstep": _lockstep_active,
				"simulation_fps": Engine.max_fps if mode == "realtime" else 0
			})
		"reset":
			_begin_simulation_request()
			var reset_reply: Dictionary = await _call_reset(request)
			_end_simulation_request()
			var reset_agents: Variant = reset_reply.get("agents", [])
			if verbose:
				print("[BridgeServer] reset ok port=%d agents=%d" % [port, reset_agents.size()])
			_send(reset_reply)
		"config":
			var config: Variant = request.get("config", {})
			if typeof(config) != TYPE_DICTIONARY:
				_send({
					"ok": false,
					"error": "config requires an object"
				})
				return
			_send(_call_config(config as Dictionary))
		"step":
			_begin_simulation_request()
			var step_reply: Dictionary = await _call_step(request)
			_end_simulation_request()
			var info: Dictionary = step_reply.get("info", {})
			var terminated: Variant = step_reply.get("terminated", false)
			var truncated: Variant = step_reply.get("truncated", false)
			if verbose:
				print(
					"[BridgeServer] step ok port=%d step=%s terminated=%s truncated=%s" %
					[port, str(info.get("step", info.get("episode_step", "?"))), str(terminated), str(truncated)]
				)
			
			_send(step_reply)
		"close":
			_send({
				"ok": true
			})
			_disconnect_client()
		_:
			_send({
				"ok": false,
				"error": "Unknown command: %s" % cmd
			})

func _call_reset(request:Dictionary) -> Dictionary:
	if controller.has_method("reset_episode_with_request"):
		return await controller.reset_episode_with_request(request)

	if controller.has_method("reset_episode"):
		if controller.has_method("step_episode"):
			var seed := int(request.get("seed", 0))
			var teams: Variant = request.get("teams", [0])
			if typeof(teams) != TYPE_ARRAY:
				teams = [0]
			return await controller.reset_episode(seed, teams as Array)
		return await controller.reset_episode()

	return {"ok": false, "error": "Controller has no reset_episode"}


func _call_config(config:Dictionary) -> Dictionary:
	if controller.has_method("configure"):
		return controller.configure(config)
	return {
		"ok": true,
		"ignored": true,
		"reason": "Controller has no configure method"
	}


func _call_spec() -> Dictionary:
	if controller.has_method("get_spec"):
		return controller.get_spec()

	return {
		"ok": false,
		"error": "Controller has no get_spec method"
	}


func _call_step(request:Dictionary) -> Dictionary:
	var actions: Variant = request.get("actions", 0)

	if controller.has_method("step"):
		return await controller.step(actions)

	if controller.has_method("step_episode"):
		if typeof(actions) != TYPE_DICTIONARY:
			return {
				"ok": false,
				"error": "step_episode requires an actions object"
			}

		var controlled_teams: Variant = request.get("controlled_teams", [0])
		if typeof(controlled_teams) != TYPE_ARRAY:
			controlled_teams = [0]

		var teams: Variant = request.get("teams", controlled_teams)
		if typeof(teams) != TYPE_ARRAY:
			teams = controlled_teams

		return controller.step_episode(actions as Dictionary, controlled_teams as Array, teams as Array)

	return {"ok": false, "error": "Controller has no step method"}

func _send(payload: Dictionary) -> void:
	if client == null:
		return
	var bytes := (JSON.stringify(payload) + "\n").to_utf8_buffer()
	var err := client.put_data(bytes)
	if err != OK:
		push_warning("Failed to send TCP response on port %d: %s" % [port, error_string(err)])
		_disconnect_client()

func _disconnect_client() -> void:
	_restore_realtime_frame_cap()
	_deactivate_lockstep()
	if client != null:
		client.disconnect_from_host()
	client = null
	_rx_buffer = ""


func _activate_lockstep() -> void:
	if not lockstep_enabled or _lockstep_active:
		return
	_tree_was_paused = get_tree().paused
	_lockstep_active = true
	get_tree().paused = true


func _set_execution_mode(mode:String, simulation_fps:int = 60) -> void:
	if mode == "realtime":
		if not _realtime_frame_cap_active:
			_max_fps_before_realtime = Engine.max_fps
			_realtime_frame_cap_active = true
		Engine.max_fps = maxi(1, simulation_fps)
		_realtime_simulation_fps = maxi(1, simulation_fps)
		_realtime_next_frame_usec = Time.get_ticks_usec()
		lockstep_enabled = false
		_deactivate_lockstep()
		return
	_restore_realtime_frame_cap()
	lockstep_enabled = true
	_activate_lockstep()


func _restore_realtime_frame_cap() -> void:
	if not _realtime_frame_cap_active:
		return
	Engine.max_fps = _max_fps_before_realtime
	_realtime_frame_cap_active = false
	_realtime_next_frame_usec = 0


func _pace_realtime_loop() -> void:
	var frame_interval_usec := maxi(1, int(1_000_000.0 / float(_realtime_simulation_fps)))
	var now := Time.get_ticks_usec()
	if _realtime_next_frame_usec <= 0:
		_realtime_next_frame_usec = now
	_realtime_next_frame_usec += frame_interval_usec
	var remaining := _realtime_next_frame_usec - now
	if remaining > 0:
		OS.delay_usec(remaining)
		now = Time.get_ticks_usec()
	if now - _realtime_next_frame_usec > frame_interval_usec:
		_realtime_next_frame_usec = now


func _deactivate_lockstep() -> void:
	if not _lockstep_active:
		return
	get_tree().paused = _tree_was_paused
	_lockstep_active = false


func _begin_simulation_request() -> void:
	if _lockstep_active:
		get_tree().paused = false


func _end_simulation_request() -> void:
	if _lockstep_active:
		get_tree().paused = true


func _exit_tree() -> void:
	_deactivate_lockstep()
