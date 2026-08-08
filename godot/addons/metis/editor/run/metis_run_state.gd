@tool
extends RefCounted
## The bookkeeping that lets the editor find a training run it did not launch.
##
## Runs are spawned FULLY DETACHED (see MetisProcess), so one survives closing the wizard, and the
## editor itself. Three files in `res://.metis` are what a later session has to go on:
##
##   train.log        the trainer's stdout, tailed for the log view and the episode counter
##   train.pid        the detached child PID, probed with `kill -0` for liveness
##   train_run.json   what it was LAUNCHED with -- written here
##
## The log holds output, not configuration, so without train_run.json the progress bar has no
## denominator and the dashboard link no port. Everything is addressed through this one place so the
## launcher and the monitor cannot disagree about a filename.

const STATE_FILE := "train_run.json"
const LOG_FILE := "train.log"
const PID_FILE := "train.pid"

# Preloaded rather than referenced as `MetisProcess`, matching plugin.gd: a global class_name is only
# available after the project has rescanned, which is not guaranteed the first time a freshly
# installed add-on is enabled.
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
	## Whether a detached run is alive right now, from the files alone.
	##
	## Static and stateless on purpose: the plugin asks this to decide which window to open, and the
	## wizard to refuse launching a second run over the first, both before any monitor exists.
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
	## The recorded configuration, or an empty dictionary when there is none to read.
	##
	## An empty result is a normal outcome, not an error: a run launched from a terminal writes no
	## state file at all, yet is still perfectly monitorable through the log and the pidfile.
	if not FileAccess.file_exists(state_path()):
		return {}
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(state_path()))
	return parsed if parsed is Dictionary else {}


static func format_value(value: Variant) -> String:
	## JSON has one number type, so every integer written here comes back as a float and str() renders
	## it "8000.0". Episode counts and port numbers are not fractional quantities.
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
