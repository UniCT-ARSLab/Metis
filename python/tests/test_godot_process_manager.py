import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from envs.process_manager import (
    GodotProcessManager,
    _is_software_opengl,
    _parse_switcheroo_offload_environment,
)


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


class GodotProcessManagerTests(unittest.TestCase):
    def test_detects_software_opengl_renderers(self):
        self.assertTrue(_is_software_opengl("OpenGL renderer string: llvmpipe"))
        self.assertTrue(_is_software_opengl("Accelerated: no"))
        self.assertFalse(_is_software_opengl("OpenGL renderer string: NVIDIA GeForce RTX"))

    def test_parses_non_default_switcheroo_environment(self):
        output = """Device: 0
  Name:        Intel Corporation Iris Xe
  Default:     yes
  Environment: DRI_PRIME=pci-0000_00_02_0

Device: 1
  Name:        NVIDIA Corporation RTX 4060
  Default:     no
  Environment: __GLX_VENDOR_LIBRARY_NAME=nvidia __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only
"""
        self.assertEqual(
            _parse_switcheroo_offload_environment(output),
            {
                "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
                "__NV_PRIME_RENDER_OFFLOAD": "1",
                "__VK_LAYER_NV_optimus": "NVIDIA_only",
            },
        )

    def test_default_runtime_paths_live_outside_the_python_package(self):
        manager = GodotProcessManager(godot_bin="godot")
        repository_root = Path(__file__).resolve().parents[2]
        self.assertEqual(manager.project_dir, repository_root / "godot")
        self.assertEqual(manager.logs_dir, repository_root / ".runtime" / "godot_logs")

    def start_and_capture(self, *, headless, render_env_count):
        commands = []

        def fake_popen(command, **_kwargs):
            commands.append(command)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
            )
            with (
                patch("envs.process_manager.subprocess.Popen", side_effect=fake_popen),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", return_value=True),
            ):
                manager.start_many(
                    [6200, 6201, 6202],
                    headless=headless,
                    render_env_count=render_env_count,
                )
            manager.stop_all()
        return commands

    def test_omitted_render_count_renders_every_environment(self):
        commands = self.start_and_capture(headless=False, render_env_count=None)
        self.assertEqual(["--headless" in command for command in commands], [False, False, False])

    def test_render_count_limits_visible_environments(self):
        commands = self.start_and_capture(headless=False, render_env_count=1)
        self.assertEqual(["--headless" in command for command in commands], [False, True, True])

    def test_headless_flag_overrides_render_count(self):
        commands = self.start_and_capture(headless=True, render_env_count=None)
        self.assertEqual(["--headless" in command for command in commands], [True, True, True])

    def test_each_environment_uses_a_distinct_engine_log(self):
        commands = self.start_and_capture(headless=True, render_env_count=None)
        log_paths = [
            Path(command[command.index("--log-file") + 1]).name
            for command in commands
        ]
        self.assertEqual(
            log_paths,
            ["godot_engine_6200.log", "godot_engine_6201.log", "godot_engine_6202.log"],
        )

    def test_light_gpu_passes_detected_offload_environment_to_godot(self):
        popen_calls = []

        def fake_popen(command, **kwargs):
            popen_calls.append((command, kwargs))
            return FakeProcess()

        offload_env = {
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__NV_PRIME_RENDER_OFFLOAD": "1",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GodotProcessManager(
                godot_bin="godot",
                project_dir=temp_dir,
                logs_dir=temp_dir,
            )
            with (
                patch("envs.process_manager.subprocess.Popen", side_effect=fake_popen),
                patch("envs.process_manager.time.sleep"),
                patch.object(manager, "_wait_for_log_ready", return_value=True),
                patch(
                    "envs.process_manager._light_gpu_fallback_environment",
                    return_value=offload_env,
                ),
            ):
                manager.start_many(
                    [6200, 6201],
                    headless=False,
                    render_env_count=1,
                    render_mode="light-gpu",
                )
            manager.stop_all()

        rendered_command, rendered_kwargs = popen_calls[0]
        self.assertIn("opengl3", rendered_command)
        self.assertEqual(rendered_kwargs["env"]["__GLX_VENDOR_LIBRARY_NAME"], "nvidia")
        self.assertEqual(rendered_kwargs["env"]["__NV_PRIME_RENDER_OFFLOAD"], "1")
        self.assertIn("--headless", popen_calls[1][0])
        self.assertIsNone(popen_calls[1][1]["env"])


if __name__ == "__main__":
    unittest.main()
