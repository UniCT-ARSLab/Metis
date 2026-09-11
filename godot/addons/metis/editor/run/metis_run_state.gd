@tool
extends RefCounted
## Persists enough state for the editor to reconnect to a detached training run.

const STATE_FILE := "train_run.json"
const LOG_FILE := "train.log"
const PID_FILE := "train.pid"

# The class cache may not be ready when a newly installed add-on first starts.
const PROCESS := preload("res://addons/metis/editor/process/metis_process.gd")


static func directory() -> String:
	return ProjectSettings.globalize_path("res://.metis")


static func log_path() -> String:
	return directory().path_join(LOG_FILE)


static func pid_path() -> String:
	return directory().path_join(PID_FILE)


static func state_path() -> String:
	return directory().path_join(STATE_FILE)


static func is_run_active() -> bool:
	## Checks whether the recorded detached process is still alive.
	var probe = PROCESS.new()
	probe.attach(log_path(), pid_path())
	return probe.is_running()


static func write(state: Dictionary) -> void:
	DirAccess.make_dir_recursive_absolute(directory())
	var file := FileAccess.open(state_path(), FileAccess.WRITE)
	if file != null:
		file.store_string(JSON.stringify(state, "\t"))
		file.close()


static func read() -> Dictionary:
	## Returns the recorded configuration, or an empty dictionary when unavailable.
	if not FileAccess.file_exists(state_path()):
		return {}
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(state_path()))
	return parsed if parsed is Dictionary else {}


static func format_value(value: Variant) -> String:
	## Avoids decimal notation for integer-valued JSON numbers.
	if value is float and value == floor(value):
		return str(int(value))
	return str(value)


static func summary(state: Dictionary) -> String:
	## One-line description of a run, for a window subtitle.
	if state.is_empty():
		return "configuration unknown"
	var algorithm := str(state.get("algorithm", "?"))
	var scene := str(state.get("scene", "")).get_file()
	return "%s on %s" % [algorithm, scene] if not scene.is_empty() else algorithm
