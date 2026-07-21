import os
import platform
import shlex
import socket
import subprocess
import time
from pathlib import Path


_GPU_OFFLOAD_ENV_KEYS = {
    "DRI_PRIME",
    "__GLX_VENDOR_LIBRARY_NAME",
    "__NV_PRIME_RENDER_OFFLOAD",
    "__VK_LAYER_NV_optimus",
}


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
    def __init__(self, godot_bin=None, project_dir=None, logs_dir=None, scene_path=None):
        self.godot_bin = godot_bin or os.environ.get("GODOT_BIN")
        if not self.godot_bin:
            raise RuntimeError("Set GODOT_BIN or pass godot_bin explicitly.")
        self.project_dir = Path(project_dir or Path(__file__).resolve().parents[2] / "godot")
        self.logs_dir = Path(logs_dir or Path(__file__).resolve().parents[2] / ".runtime" / "godot_logs")
        self.scene_path = scene_path
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.processes = []

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

            args_prefix += ["--path", str(self.project_dir)]
            if self.scene_path:
                args_prefix.append(str(self.scene_path))

            log_path = self.logs_dir / f"godot_{port}.log"
            log_file = open(log_path, "w", buffering=1)

            cmd = args_prefix + ["--", f"--port={int(port)}"] + user_args
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=proc_env,
            )
            self.processes.append({
                "port": port,
                "proc": proc,
                "log_file": log_file,
                "log_path": log_path,
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

            if not self._wait_for_log_ready(log_path, f"[BridgeServer] Listening on port {port}", timeout=20.0):
                raise RuntimeError(
                    f"Godot did not become ready on port {port}. Log: {log_path}\n"
                    f"{self._log_tail(log_path)}"
                )

    def stop_all(self):
        for item in self.processes:
            proc = item["proc"]
            if proc.poll() is None:
                proc.terminate()

        for item in self.processes:
            proc = item["proc"]
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        for item in self.processes:
            try:
                item["log_file"].close()
            except Exception:
                pass

        self.processes.clear()
