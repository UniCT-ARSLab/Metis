import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.server import DashboardServer


class DashboardCurriculumTests(unittest.TestCase):
    def test_record_curriculum_stores_and_is_sent_on_connect(self):
        server = DashboardServer(port=0)
        # No client yet -> stored for the connect handshake.
        server.record_curriculum(
            {"stage_index": 5, "stage_name": "F", "cooldown_active": False,
             "confirmations": 2, "promotion_required": 3})
        self.assertEqual(server._curriculum["stage_name"], "F")
        self.assertEqual(server._curriculum["confirmations"], 2)
        # A later snapshot replaces it (latest canonical state wins).
        server.record_curriculum({"stage_index": 0, "stage_name": "A", "cooldown_active": True})
        self.assertEqual(server._curriculum["stage_name"], "A")
        self.assertTrue(server._curriculum["cooldown_active"])

    def test_record_curriculum_copies_input(self):
        server = DashboardServer(port=0)
        payload = {"stage_name": "B"}
        server.record_curriculum(payload)
        payload["stage_name"] = "MUTATED"
        self.assertEqual(server._curriculum["stage_name"], "B")


if __name__ == "__main__":
    unittest.main()
