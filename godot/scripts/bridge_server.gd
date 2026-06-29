extends Node

@export var port := 5555
@export var controller_path: NodePath

var server := TCPServer.new()
var client: StreamPeerTCP = null
var controller
var _rx_buffer := ""

func _ready() -> void:
	controller = get_node(controller_path)

	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--port="):
			port = int(arg.split("=")[1])

	var err := server.listen(port)
	if err != OK:
		push_error("Cannot listen on port %d" % port)
		return

	print("[BridgeServer] Listening on port %d" % port)

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
	print("[BridgeServer] cmd=%s port=%d" % [cmd, port])
	match cmd:
		"hello":
			_send({
				"ok": true,
				"version": 1
			})
		"reset":
			var seed := int(request.get("seed", 0))
			var observed_teams: Variant = request.get("teams", [0])
			var reset_reply: Dictionary = controller.reset_episode(seed, observed_teams as Array)
			var reset_agents: Variant = reset_reply.get("agents", [])
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
			_send(controller.configure(config as Dictionary))
		"step":
			var actions: Variant = request.get("actions", {})
			if typeof(actions) != TYPE_DICTIONARY:
				_send({
					"ok": false,
					"error": "step requires an actions object"
				})
				return
			var controlled_teams: Variant = request.get("controlled_teams", [0])
			var observed_step_teams: Variant = request.get("teams", controlled_teams)
			var step_reply: Dictionary = controller.step_episode(
				actions as Dictionary,
				controlled_teams as Array,
				observed_step_teams as Array
			)
			var info: Dictionary = step_reply.get("info", {})
			var terminated: Variant = step_reply.get("terminated", false)
			var truncated: Variant = step_reply.get("truncated", false)
			print(
				"[BridgeServer] step ok port=%d episode_step=%s terminated=%s truncated=%s" %
				[port, str(info.get("episode_step", "?")), str(terminated), str(truncated)]
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
