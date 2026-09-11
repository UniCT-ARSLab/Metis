@tool
class_name MetisProcess
extends RefCounted

## Runs Metis commands outside the editor and tracks them through a log and pidfile.

var log_path := ""          # absolute path of the redirected log
var pidfile_path := ""      # absolute path of the pidfile (holds the detached child PID)
var _launcher_pid := -1     # PID of the transient setsid/bash launcher (not the training process)


static func _shq(value: String) -> String:
	# Quote one shell argument without changing its contents.
	return "'" + value.replace("'", "'\\''") + "'"


## Launches a Python script as a detached process.
func start(
		python: String,
		script_abs: String,
		args: PackedStringArray,
		project_dir: String,
		log_abs: String,
		pidfile_abs: String) -> bool:
	var command_args := PackedStringArray([script_abs])
	command_args.append_array(args)
	return start_command(
		python, command_args, project_dir, log_abs, pidfile_abs)


## Launches an arbitrary command as a detached process.
func start_command(
		executable: String,
		args: PackedStringArray,
		project_dir: String,
		log_abs: String,
		pidfile_abs: String) -> bool:
	log_path = log_abs
	pidfile_path = pidfile_abs
	DirAccess.make_dir_recursive_absolute(log_abs.get_base_dir())
	for stale in [log_abs, pidfile_abs]:
		if FileAccess.file_exists(stale):
			DirAccess.remove_absolute(stale)

	if OS.get_name() == "Windows":
		# Windows already detaches create_process; shell redirection is only used on Unix.
		_launcher_pid = OS.create_process(executable, args)
		return _launcher_pid > 0

	# exec preserves the recorded PID, while setsid lets the command outlive the editor.
	var exec_line := "exec " + _shq(executable)
	for argument in args:
		exec_line += " " + _shq(argument)
	exec_line += " > " + _shq(log_abs) + " 2>&1 < /dev/null"
	var script := "#!/usr/bin/env bash\n"
	script += "echo $$ > " + _shq(pidfile_abs) + "\n"
	script += "cd " + _shq(project_dir) + " || exit 1\n"
	script += exec_line + "\n"

	var script_path := pidfile_abs.get_base_dir().path_join("metis_launch.sh")
	var file := FileAccess.open(script_path, FileAccess.WRITE)
	if file == null:
		return false
	file.store_string(script)
	file.close()

	_launcher_pid = OS.create_process("setsid", PackedStringArray(["bash", script_path]))
	return _launcher_pid > 0


## Attaches to a detached run created by another editor session.
func attach(log_abs: String, pidfile_abs: String) -> void:
	log_path = log_abs
	pidfile_path = pidfile_abs


## The detached child PID once the launcher has written the pidfile (-1 until then).
func child_pid() -> int:
	if not FileAccess.file_exists(pidfile_path):
		return -1
	var text := FileAccess.get_file_as_string(pidfile_path).strip_edges()
	return int(text) if text.is_valid_int() else -1


func has_pidfile() -> bool:
	## Checks that the pidfile contains a usable process ID.
	return child_pid() > 0


func is_running() -> bool:
	var pid := child_pid()
	if pid > 0:
		return OS.execute("bash", ["-c", "kill -0 %d 2>/dev/null" % pid]) == 0
	# The launcher covers the short interval before the child writes its pidfile.
	return _launcher_pid > 0 and OS.is_process_running(_launcher_pid)


## Stops the whole process group (trainer + spawned Godot envs), gracefully then hard.
func stop() -> void:
	var pid := child_pid()
	if pid > 0:
		# A negative PID targets the setsid process group.
		OS.execute("bash", ["-c",
			"kill -TERM -%d 2>/dev/null || kill -TERM %d 2>/dev/null" % [pid, pid]])
		OS.execute("bash", ["-c",
			"sleep 2; kill -KILL -%d 2>/dev/null || kill -KILL %d 2>/dev/null || true" % [pid, pid]])
	elif _launcher_pid > 0 and OS.is_process_running(_launcher_pid):
		OS.kill(_launcher_pid)
	_launcher_pid = -1


const LOG_TAIL_BYTES := 64 * 1024


func read_log() -> String:
	if log_path.is_empty() or not FileAccess.file_exists(log_path):
		return ""
	return FileAccess.get_file_as_string(log_path)


func log_size() -> int:
	## Returns the current log size in bytes.
	if log_path.is_empty() or not FileAccess.file_exists(log_path):
		return 0
	var file := FileAccess.open(log_path, FileAccess.READ)
	if file == null:
		return 0
	var length := file.get_length()
	file.close()
	return int(length)


func read_log_tail(max_bytes := LOG_TAIL_BYTES) -> String:
	## Reads at most the last `max_bytes` bytes of the log.
	if log_path.is_empty() or not FileAccess.file_exists(log_path):
		return ""
	var file := FileAccess.open(log_path, FileAccess.READ)
	if file == null:
		return ""
	var length := int(file.get_length())
	var start := maxi(0, length - maxi(1, max_bytes))
	file.seek(start)
	var chunk := file.get_buffer(length - start).get_string_from_utf8()
	file.close()
	return chunk
