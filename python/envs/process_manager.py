import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path


_GPU_OFFLOAD_ENV_KEYS = {
    "DRI_PRIME",
    "__GLX_VENDOR_LIBRARY_NAME",
    "__NV_PRIME_RENDER_OFFLOAD",
    "__VK_LAYER_NV_optimus",
}


def _pid_is_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True


def port_is_available(port, host="127.0.0.1"):
    port = int(port)
    if not 1 <= port <= 65535:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((str(host), port))
        return True
    except OSError:
        return False


class PortLeaseSet:
    """Process-local coordination for ports that will shortly be bound by Godot."""

    def __init__(self, ports, directory, host="127.0.0.1"):
        clean_ports = [int(port) for port in ports]
        if not clean_ports or len(clean_ports) != len(set(clean_ports)):
            raise RuntimeError("Godot ports must be a non-empty set of unique values")
        if any(not 1 <= port <= 65535 for port in clean_ports):
            raise RuntimeError("Godot ports must be between 1 and 65535")
        self.ports = clean_ports
        self.directory = Path(directory)
        self.host = str(host)
        self.token = uuid.uuid4().hex
        self.paths = []

    def acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            for port in self.ports:
                self._acquire_one(port)
            return self
        except Exception:
            self.release()
            raise

    def _acquire_one(self, port):
        path = self.directory / f"port-{port}.json"
        for _attempt in range(2):
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                self._reclaim_stale(path, port)
                continue
            payload = {
                "host": self.host,
                "port": port,
                "pid": os.getpid(),
                "token": self.token,
                "created_at_unix_ns": time.time_ns(),
            }
            try:
                os.write(
                    descriptor,
                    (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.paths.append(path)
            if not port_is_available(port, self.host):
                raise RuntimeError(f"Godot port {port} is already occupied")
            return
        raise RuntimeError(f"Godot port {port} is leased by another Metis process")

    def _reclaim_stale(self, path, port):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            owner_pid = int(payload["pid"])
            owner_port = int(payload["port"])
            owner_token = str(payload["token"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid port lease blocks Godot port {port}: {path}") from exc
        if owner_port != port or not owner_token:
            raise RuntimeError(f"Mismatched port lease blocks Godot port {port}: {path}")
        if _pid_is_alive(owner_pid):
            raise RuntimeError(
                f"Godot port {port} is leased by live process {owner_pid}: {path}"
            )
        if not port_is_available(port, self.host):
            raise RuntimeError(f"Godot port {port} is occupied despite stale lease {path}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def release(self):
        unreleased = []
        for path in reversed(self.paths):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("token") == self.token:
                    path.unlink()
                elif path.exists():
                    unreleased.append(path)
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError):
                unreleased.append(path)
        self.paths.clear()
        return unreleased


def _is_software_opengl(glxinfo_output):
    output = str(glxinfo_output).lower()
    return any(
        marker in output
        for marker in ("llvmpipe", "softpipe", "software rasterizer", "accelerated: no")
    )


def _parse_switcheroo_offload_environment(switcheroo_output):
    blocks = []
    current = []
    for line in str(switcheroo_output).splitlines():
        if line.startswith("Device:") and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)

    for block in blocks:
        is_default = any(
            line.strip().lower().startswith("default:")
            and line.split(":", 1)[1].strip().lower() == "yes"
            for line in block
        )
        if is_default:
            continue
        for line in block:
            stripped = line.strip()
            if not stripped.startswith("Environment:"):
                continue
            result = {}
            for assignment in shlex.split(stripped.removeprefix("Environment:").strip()):
                key, separator, value = assignment.partition("=")
                if separator and key in _GPU_OFFLOAD_ENV_KEYS:
                    result[key] = value
            if result:
                return result
    return {}


def _light_gpu_fallback_environment(system_name=None, command_runner=subprocess.run):
    """Select a discrete Linux GPU only when default OpenGL is software-rendered."""
    system_name = system_name or platform.system()
    if system_name != "Linux" or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return {}
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1":
        return {}
    if any(os.environ.get(key) for key in _GPU_OFFLOAD_ENV_KEYS):
        return {}

    try:
        glxinfo = command_runner(
            ["glxinfo", "-B"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {}
    glxinfo_output = (glxinfo.stdout or "") + (glxinfo.stderr or "")
    if glxinfo.returncode != 0 or not _is_software_opengl(glxinfo_output):
        return {}

    try:
        switcheroo = command_runner(
            ["switcherooctl", "list"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {}
    if switcheroo.returncode != 0:
        return {}
    return _parse_switcheroo_offload_environment(switcheroo.stdout)


def godot_render_args(render_mode):
    """Map a --render-mode value to (extra Godot engine flags, env overrides) for a
    non-headless instance. 'project' (or None) leaves the project renderer untouched."""
    if render_mode in (None, "project"):
        return [], {}
    if render_mode == "light-gpu":
        return ["--rendering-driver", "opengl3", "--rendering-method", "gl_compatibility"], {}
    if render_mode == "cpu":
        # OpenGL compat on top of Mesa llvmpipe -> software rasterizer on CPU, GPU stays free.
        return (
            ["--rendering-driver", "opengl3", "--rendering-method", "gl_compatibility"],
            {"LIBGL_ALWAYS_SOFTWARE": "1"},
        )
    if render_mode == "gpu":
        return ["--rendering-driver", "vulkan", "--rendering-method", "forward_plus"], {}
    return [], {}


class GodotProcessManager:
    def __init__(
        self,
        godot_bin=None,
        project_dir=None,
        logs_dir=None,
        scene_path=None,
        port_lease_dir=None,
    ):
        self.godot_bin = godot_bin or os.environ.get("GODOT_BIN")
        if not self.godot_bin:
            raise RuntimeError("Set GODOT_BIN or pass godot_bin explicitly.")
        self.project_dir = Path(project_dir or Path(__file__).resolve().parents[2] / "godot")
        self.logs_dir = Path(logs_dir or Path(__file__).resolve().parents[2] / ".runtime" / "godot_logs")
        self.port_lease_dir = Path(
            port_lease_dir or self.logs_dir.parent / "port_leases"
        )
        self.scene_path = scene_path
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.processes = []
        self._port_leases = None
        self.last_stop_report = None

    def _wait_for_log_ready(self, log_path, expected, timeout=20.0):
        start = time.time()
        while time.time() - start < timeout:
            if log_path.exists():
                text = log_path.read_text(errors="ignore")
                if expected in text:
                    return True
            time.sleep(0.2)
        return False

    def _log_tail(self, log_path, lines=20):
        if not log_path.exists():
            return "<log file not created>"
        text = log_path.read_text(errors="ignore").splitlines()
        return "\n".join(text[-lines:]) if text else "<empty log file>"

    def start_many(
        self,
        ports,
        headless=True,
        debug=False,
        user_args=None,
        fixed_fps=60,
        render_env_count=None,
        render_mode="project",
    ):
        if self.processes or self._port_leases is not None:
            raise RuntimeError("This GodotProcessManager already owns running processes")
        self.last_stop_report = None
        ports = [int(port) for port in ports]
        self._port_leases = PortLeaseSet(ports, self.port_lease_dir).acquire()
        user_args = list(user_args or [])
        rendered_count = len(ports) if render_env_count is None else max(0, int(render_env_count))
        light_gpu_fallback_env = {}
        if not headless and rendered_count > 0 and render_mode == "light-gpu":
            light_gpu_fallback_env = _light_gpu_fallback_environment()
            if light_gpu_fallback_env:
                print(
                    "light-gpu: software OpenGL detected; applying discrete GPU offload "
                    f"({', '.join(sorted(light_gpu_fallback_env))}).",
                    flush=True,
                )

        try:
            for env_index, port in enumerate(ports):
                instance_headless = bool(headless) or env_index >= rendered_count
                args_prefix = [self.godot_bin]
                proc_env = None
                if instance_headless:
                    args_prefix.append("--headless")
                    if fixed_fps is not None and int(fixed_fps) > 0:
                        args_prefix += ["--fixed-fps", str(int(fixed_fps))]
                else:
                    render_flags, render_env = godot_render_args(render_mode)
                    if render_mode == "light-gpu" and light_gpu_fallback_env:
                        render_env.update(light_gpu_fallback_env)
                    args_prefix += render_flags
                    if render_env:
                        proc_env = dict(os.environ)
                        proc_env.update(render_env)
                if debug:
                    args_prefix.append("--debug")
                    args_prefix.append("--debug-collisions")
                    args_prefix.append("--debug-paths")
                    args_prefix.append("--debug-navigation")
                    args_prefix.append("--debug-avoidance")

                engine_log_path = self.logs_dir / f"godot_engine_{port}.log"
                args_prefix += ["--log-file", str(engine_log_path)]
                args_prefix += ["--path", str(self.project_dir)]
                if self.scene_path:
                    args_prefix.append(str(self.scene_path))

                log_path = self.logs_dir / f"godot_{port}.log"
                log_file = open(log_path, "w", buffering=1)
                cmd = args_prefix + ["--", f"--port={port}"] + user_args
                popen_kwargs = {"start_new_session": True} if os.name == "posix" else {}
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=proc_env,
                        **popen_kwargs,
                    )
                except Exception:
                    log_file.close()
                    raise
                self.processes.append({
                    "port": port,
                    "proc": proc,
                    "process_group_id": (
                        int(proc.pid)
                        if os.name == "posix" and getattr(proc, "pid", None) is not None
                        else None
                    ),
                    "log_file": log_file,
                    "log_path": log_path,
                    "engine_log_path": engine_log_path,
                })

            for item in self.processes:
                port = item["port"]
                proc = item["proc"]
                log_path = item["log_path"]

                time.sleep(0.5)
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Godot exited immediately on port {port}. Log: {log_path}\n"
                        f"{self._log_tail(log_path)}"
                    )

                if not self._wait_for_log_ready(
                    log_path,
                    f"[BridgeServer] Listening on port {port}",
                    timeout=20.0,
                ):
                    raise RuntimeError(
                        f"Godot did not become ready on port {port}. Log: {log_path}\n"
                        f"{self._log_tail(log_path)}"
                    )
        except Exception:
            self.stop_all()
            raise

    def stop_all(self):
        if not self.processes and self._port_leases is None and self.last_stop_report is not None:
            return dict(self.last_stop_report)
        items = list(self.processes)
        leased_ports = (
            list(self._port_leases.ports) if self._port_leases is not None else []
        )
        ports = list(
            dict.fromkeys([int(item["port"]) for item in items] + leased_ports)
        )
        killed_groups = []
        surviving_groups = []
        cleanup_errors = []
        for item in items:
            proc = item["proc"]
            if proc.poll() is None:
                try:
                    self._signal_process(item, signal.SIGTERM)
                except Exception as exc:
                    cleanup_errors.append(f"SIGTERM port {item['port']}: {exc}")

        for item in items:
            proc = item["proc"]
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._signal_process(item, signal.SIGKILL)
                    killed_groups.append(item.get("process_group_id"))
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    cleanup_errors.append(f"process on port {item['port']} ignored SIGKILL")
                except Exception as exc:
                    cleanup_errors.append(f"SIGKILL port {item['port']}: {exc}")
            except Exception as exc:
                cleanup_errors.append(f"wait port {item['port']}: {exc}")

        for item in items:
            group_id = item.get("process_group_id")
            if group_id is not None and self._process_group_alive(group_id):
                try:
                    self._signal_process(item, signal.SIGKILL)
                    time.sleep(0.05)
                except Exception as exc:
                    cleanup_errors.append(f"process-group cleanup {group_id}: {exc}")
                if self._process_group_alive(group_id):
                    surviving_groups.append(group_id)
            try:
                item["log_file"].close()
            except Exception:
                pass

        self.processes.clear()
        if self._port_leases is not None:
            try:
                unreleased = self._port_leases.release()
                if unreleased:
                    cleanup_errors.append(
                        "port lease files remain: "
                        + ", ".join(os.fspath(path) for path in unreleased)
                    )
            except Exception as exc:
                cleanup_errors.append(f"port lease release: {exc}")
            self._port_leases = None
        released_ports = [port for port in ports if port_is_available(port)]
        self.last_stop_report = {
            "ports": ports,
            "ports_released": released_ports,
            "surviving_process_groups": surviving_groups,
            "forced_process_groups": [value for value in killed_groups if value is not None],
            "errors": cleanup_errors,
            "clean": (
                not surviving_groups
                and not cleanup_errors
                and len(released_ports) == len(ports)
            ),
        }
        return self.last_stop_report

    @staticmethod
    def _signal_process(item, signal_number):
        proc = item["proc"]
        group_id = item.get("process_group_id")
        if os.name == "posix" and group_id is not None:
            try:
                os.killpg(group_id, signal_number)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if signal_number == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()

    @staticmethod
    def _process_group_alive(group_id):
        if os.name != "posix" or group_id is None:
            return False
        try:
            os.killpg(int(group_id), 0)
            return True
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True
