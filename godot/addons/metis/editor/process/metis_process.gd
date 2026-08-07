@tool
class_name MetisProcess
extends RefCounted

## Launches a Metis Python entry point (train.py / run.py / recorder.py) as a FULLY DETACHED process,
## independent of the editor: it keeps running if Godot is closed, and stopping it never touches
## Godot. stdout+stderr are redirected to a log file the editor can tail; the real child PID is
## written to a pidfile so the editor can poll/stop it. Stop targets the whole PROCESS GROUP so the
## Godot env subprocesses the async collector spawns are reaped too (the same reason a manual run
## uses `setsid ... & ; kill -TERM -<pgid>`).

var log_path := ""          # absolute path of the redirected log
var pidfile_path := ""      # absolute path of the pidfile (holds the detached child PID)
var _launcher_pid := -1     # PID of the transient setsid/bash launcher (not the training process)


static func _shq(value: String) -> String:
	# Single-quote for bash, escaping embedded single quotes.
	return "'" + value.replace("'", "'\\''") + "'"


## Launches `python script_abs args...` detached, cwd = project_dir. All paths absolute (globalized).
## Returns true on a successful spawn.
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


## Launches an arbitrary executable and argument list detached. This is used by installed add-ons
## to run `python -m metis_cli ...`; source checkouts keep using start() with python/train.py.
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
		# create_process is already detached on Windows; no shell redirect (users can tail the file
		# the trainer writes, or use the terminal). Kept minimal — Metis targets Linux.
		_launcher_pid = OS.create_process(executable, args)
		return _launcher_pid > 0

	# Generate a launch script: record our own PID (which `exec` then preserves for the Python
	# process), cd into the project, then exec the trainer with output redirected. `setsid` runs it in
	# a new session so it outlives the editor.
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


## The detached child PID once the launcher has written the pidfile (-1 until then).
func child_pid() -> int:
	if not FileAccess.file_exists(pidfile_path):
		return -1
	var text := FileAccess.get_file_as_string(pidfile_path).strip_edges()
	return int(text) if text.is_valid_int() else -1


func is_running() -> bool:
	var pid := child_pid()
	if pid > 0:
		# kill -0 probes liveness of ANY pid (the child was not created by Godot directly).
		return OS.execute("bash", ["-c", "kill -0 %d 2>/dev/null" % pid]) == 0
	# pidfile not written yet: fall back to the transient launcher still being alive.
	return _launcher_pid > 0 and OS.is_process_running(_launcher_pid)


## Stops the whole process group (trainer + spawned Godot envs), gracefully then hard.
func stop() -> void:
	var pid := child_pid()
	if pid > 0:
		# Negative pid = process group (setsid made the child a session/group leader).
		OS.execute("bash", ["-c",
			"kill -TERM -%d 2>/dev/null || kill -TERM %d 2>/dev/null" % [pid, pid]])
		OS.execute("bash", ["-c",
			"sleep 2; kill -KILL -%d 2>/dev/null || kill -KILL %d 2>/dev/null || true" % [pid, pid]])
	elif _launcher_pid > 0 and OS.is_process_running(_launcher_pid):
		OS.kill(_launcher_pid)
	_launcher_pid = -1


func read_log() -> String:
	if log_path.is_empty() or not FileAccess.file_exists(log_path):
		return ""
	return FileAccess.get_file_as_string(log_path)
