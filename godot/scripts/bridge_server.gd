extends Node
class_name BridgeServer

@export var port := 5555
@export var controller_path: NodePath
@export var verbose := false
@export var auto_silence_in_headless := true

var server := TCPServer.new()
var client: StreamPeerTCP = null
var controller
var _rx_buffer := ""

func _ready() -> void:
	controller = get_node(controller_path)
	if auto_silence_in_headless and _is_headless():
		verbose = false

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
	if client == null:
		if server.is_connection_available():
			client = server.take_connection()
			_rx_buffer = ""
			print("[BridgeServer] Client connected on port %d" % port)
		return

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
				"version": 1
			})
		"spec":
			_send(_call_spec())
		"reset":
			var reset_reply: Dictionary = await _call_reset(request)
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
			var step_reply: Dictionary = await _call_step(request)
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
	if client != null:
		client.disconnect_from_host()
	client = null
	_rx_buffer = ""
