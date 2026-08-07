@tool
class_name MetisRuntimeManager
extends RefCounted

const PAYLOAD_PATH := "res://addons/metis/python_setup"
const MANIFEST_PATH := PAYLOAD_PATH + "/runtime_manifest.json"
const PROJECT_RUNTIME_DIR := "res://.metis"
const CONFIG_PATH := PROJECT_RUNTIME_DIR + "/runtime.json"
const STATUS_PATH := PROJECT_RUNTIME_DIR + "/setup_status.json"
const LOG_PATH := PROJECT_RUNTIME_DIR + "/setup.log"

var setup_pid := -1


func load_runtime_config() -> Dictionary:
	return _read_json(CONFIG_PATH)


func load_payload_manifest() -> Dictionary:
	return _read_json(MANIFEST_PATH)


func has_packaged_payload() -> bool:
	var manifest := load_payload_manifest()
	if manifest.is_empty():
		return false
	var wheel_path := PAYLOAD_PATH.path_join(str(manifest.get("wheel", "")))
	return FileAccess.file_exists(wheel_path)


func runtime_is_ready() -> bool:
	var config := load_runtime_config()
	if config.get("status") != "ready":
		return false
	var python_path := str(config.get("python", ""))
	return not python_path.is_empty() and FileAccess.file_exists(python_path)


func runtime_needs_update() -> bool:
	if not runtime_is_ready():
		return true
	var manifest := load_payload_manifest()
	if manifest.is_empty():
		return false
	var config := load_runtime_config()
	return (
		str(config.get("plugin_version", ""))
		!= str(manifest.get("plugin_version", ""))
		or str(config.get("python_version", ""))
		!= str(manifest.get("python_version", ""))
	)


func suggested_python() -> String:
	for variable in ["METIS_PYTHON", "PYTHON"]:
		var value := OS.get_environment(variable)
		if not value.is_empty() and FileAccess.file_exists(value):
			return value

	var configured_python := str(load_runtime_config().get("python", ""))
	if not configured_python.is_empty() and FileAccess.file_exists(configured_python):
		return configured_python

	var project_root := ProjectSettings.globalize_path("res://").trim_suffix("/")
	var repository_root := project_root.get_base_dir()
	var development_candidates: Array[String]
	if OS.get_name() == "Windows":
		development_candidates = [
			repository_root.path_join("python/.venv/Scripts/python.exe"),
		]
	else:
		development_candidates = [
			repository_root.path_join("python/.venv/bin/python"),
		]
	for candidate in development_candidates:
		if FileAccess.file_exists(candidate):
			return candidate

	var executable_names := ["python", "python3"]
	if OS.get_name() != "Windows":
		executable_names.reverse()
	for executable_name in executable_names:
		var resolved := _resolve_executable(executable_name)
		if not resolved.is_empty():
			return resolved
	return executable_names[0]


func start_setup(options: Dictionary) -> int:
	if setup_is_running():
		return setup_pid

	var python_path := str(options.get("python", "")).strip_edges()
	if python_path.is_empty():
		return -1
	if not FileAccess.file_exists(python_path):
		var resolved_python := _resolve_executable(python_path)
		if resolved_python.is_empty():
			return -1
		python_path = resolved_python

	DirAccess.make_dir_recursive_absolute(
		ProjectSettings.globalize_path(PROJECT_RUNTIME_DIR))
	for path in [STATUS_PATH, LOG_PATH]:
		var absolute_path := ProjectSettings.globalize_path(path)
		if FileAccess.file_exists(absolute_path):
			DirAccess.remove_absolute(absolute_path)
	var helper_path := ProjectSettings.globalize_path(
		PAYLOAD_PATH.path_join("setup_runtime.py"))
	var arguments := PackedStringArray([
		helper_path,
		"--project-dir",
		ProjectSettings.globalize_path("res://"),
		"--payload-dir",
		ProjectSettings.globalize_path(PAYLOAD_PATH),
		"--profile",
		str(options.get("profile", "cpu")),
		"--status-path",
		ProjectSettings.globalize_path(STATUS_PATH),
		"--log-path",
		ProjectSettings.globalize_path(LOG_PATH),
	])
	if bool(options.get("dashboard", false)):
		arguments.append("--dashboard")
	if bool(options.get("export", false)):
		arguments.append("--export")
	if bool(options.get("sb3", false)):
		arguments.append("--sb3")
	if bool(options.get("existing", false)):
		arguments.append("--existing")
	if bool(options.get("recreate", false)):
		arguments.append("--recreate")
	var godot_path := OS.get_executable_path()
	if not godot_path.is_empty():
		arguments.append_array(["--godot-bin", godot_path])
	var source_root := development_source_root()
	if bool(options.get("existing", false)) and not source_root.is_empty():
		arguments.append_array(["--source-root", source_root])

	setup_pid = OS.create_process(python_path, arguments)
	return setup_pid


func setup_is_running() -> bool:
	return setup_pid > 0 and OS.is_process_running(setup_pid)


func stop_setup() -> void:
	var was_running := setup_is_running()
	if was_running:
		OS.kill(setup_pid)
	setup_pid = -1
	if was_running:
		_write_json(STATUS_PATH, {
			"format": "metis-runtime-setup",
			"format_version": 1,
			"state": "cancelled",
			"step": "cancelled",
			"progress": 0.0,
			"message": "Runtime setup was cancelled.",
		})


func setup_status() -> Dictionary:
	var status := _read_json(STATUS_PATH)
	if setup_pid > 0 and not OS.is_process_running(setup_pid):
		setup_pid = -1
		if status.is_empty() or str(status.get("state", "")) == "running":
			status = {
				"format": "metis-runtime-setup",
				"format_version": 1,
				"state": "error",
				"step": "failed",
				"progress": 0.0,
				"message": "The setup process exited before reporting completion.",
			}
			_write_json(STATUS_PATH, status)
	return status


func setup_log() -> String:
	if not FileAccess.file_exists(LOG_PATH):
		return ""
	return FileAccess.get_file_as_string(LOG_PATH)


func development_source_root() -> String:
	var project_root := ProjectSettings.globalize_path("res://").trim_suffix("/")
	var candidate := project_root.get_base_dir().path_join("python")
	return candidate if FileAccess.file_exists(candidate.path_join("pyproject.toml")) else ""


func _read_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	return parsed if parsed is Dictionary else {}


func _write_json(path: String, payload: Dictionary) -> void:
	var absolute_path := ProjectSettings.globalize_path(path)
	DirAccess.make_dir_recursive_absolute(absolute_path.get_base_dir())
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		return
	file.store_string(JSON.stringify(payload, "\t") + "\n")


func _resolve_executable(name: String) -> String:
	var command := "where" if OS.get_name() == "Windows" else "which"
	var output: Array = []
	if OS.execute(command, [name], output, true) != 0 or output.is_empty():
		return ""
	return str(output[0]).strip_edges().split("\n")[0]
