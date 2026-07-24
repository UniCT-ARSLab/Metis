import json
import tempfile
import unittest
from pathlib import Path

from backends.sb3_state import latest_checkpoint


class SB3CheckpointSelectionTests(unittest.TestCase):
    def test_latest_checkpoint_can_select_newer_final_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "ckpt-100.zip"
            final_model = directory / "final_model.zip"
            checkpoint.write_bytes(b"checkpoint")
            final_model.write_bytes(b"final")
            checkpoint.with_suffix(".json").write_text(
                json.dumps({"num_timesteps": 1000, "completed_episodes": 100}),
                encoding="utf-8",
            )
            final_model.with_suffix(".json").write_text(
                json.dumps({"num_timesteps": 1250, "completed_episodes": 123}),
                encoding="utf-8",
            )

            self.assertEqual(latest_checkpoint(directory), final_model)


if __name__ == "__main__":
    unittest.main()
