import tempfile
import unittest
from unittest.mock import patch

from godot_process_manager import GodotProcessManager


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
                patch("godot_process_manager.subprocess.Popen", side_effect=fake_popen),
                patch("godot_process_manager.time.sleep"),
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


if __name__ == "__main__":
    unittest.main()
