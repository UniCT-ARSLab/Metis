import json
import unittest

from dashboard.server import DashboardServer


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    def send(self, payload):
        self.messages.append(json.loads(payload))


class DashboardHealthTests(unittest.TestCase):
    def test_health_event_updates_meta_history_and_connected_clients(self):
        server = DashboardServer(history=10)
        client = _FakeWebSocket()
        server._clients[client] = {"batch": 1, "buf": []}
        event = {
            "event": "recovery_started",
            "state": "critical",
            "phase": "recovering",
            "reason": "evaluation collapse",
            "auto_recovery": True,
            "recovery_count": 1,
            "recovery_cycle": 2,
            "recovery_cycle_attempt": 1,
            "recovery_max_attempts": 3,
            "last_recovery_mode": "soft",
            "verification_pending": True,
        }

        server.record_health(event)

        self.assertEqual(server._meta["health_state"], "critical")
        self.assertEqual(server._meta["training_phase"], "recovering")
        self.assertEqual(server._meta["recovery_cycle"], 2)
        self.assertEqual(server._meta["recovery_cycle_attempt"], 1)
        self.assertTrue(server._meta["verification_pending"])
        self.assertEqual(list(server._health_history), [event])
        self.assertEqual(client.messages[-1], {"type": "health", "data": event})


if __name__ == "__main__":
    unittest.main()
