import os
import socket
import subprocess
import time
from pathlib import Path


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
    ):
        user_args = list(user_args or [])
        rendered_count = len(ports) if render_env_count is None else max(0, int(render_env_count))

        for env_index, port in enumerate(ports):
            instance_headless = bool(headless) or env_index >= rendered_count
            args_prefix = [self.godot_bin]
            if instance_headless:
                args_prefix.append("--headless")
                if fixed_fps is not None and int(fixed_fps) > 0:
                    args_prefix += ["--fixed-fps", str(int(fixed_fps))]
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
                    f"Godot non risulta pronto sulla porta {port}. Log: {log_path}\n"
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
