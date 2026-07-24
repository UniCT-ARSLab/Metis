"""Live training dashboard: a Flask + flask-sock server, started in a daemon thread by the
trainer when --dashboard is set. It serves a static client (see ./static), an HTTP history
API, and a WebSocket stream that pushes per-episode metrics.

The trainer feeds metrics through DashboardServer.record(dict) (registered as a metrics sink
in core.training.print_episode_metrics). record() is called from the learner thread, so it
must stay fast and never raise back into training -- exceptions are swallowed by the sink.

WebSocket protocol (JSON text frames):
  server -> client: {"type": "snapshot", "data": [<episode>, ...]}   on connect
                    {"type": "episodes", "data": [<episode>, ...]}   live, batched per client
  client -> server: {"batch": N}   set how many episodes to buffer before a push (default 1)
"""
import json
import threading
import time
from collections import deque
from pathlib import Path


class DashboardServer:
    def __init__(self, port=8770, host="127.0.0.1", history=20000, meta=None):
        self.port = int(port)
        self.host = host
        self._history = deque(maxlen=int(history))
        self._clients = {}  # ws -> {"batch": int, "buf": list}
        self._meta = dict(meta or {})  # run info: algorithm, scenario, agents, envs, ...
        self._meta.setdefault("started_at", time.time())  # epoch; client ticks a live timer off it
        self._meta.setdefault("status", "running")
        self._lock = threading.Lock()
        self._thread = None

    def set_meta(self, **kwargs):
        """Merge run metadata (e.g. agent count once the scenario is probed) and push it live."""
        with self._lock:
            self._meta.update({k: v for k, v in kwargs.items() if v is not None})
            snapshot = dict(self._meta)
            clients = list(self._clients)
        for ws in clients:
            self._safe_send(ws, {"type": "meta", "data": snapshot})

    # -- called from the training thread, once per episode --------------------------------
    def record(self, metrics):
        to_send = []
        with self._lock:
            self._history.append(metrics)
            for ws, state in self._clients.items():
                state["buf"].append(metrics)
                if len(state["buf"]) >= state["batch"]:
                    to_send.append((ws, state["buf"]))
                    state["buf"] = []
        for ws, batch in to_send:
            self._safe_send(ws, {"type": "episodes", "data": batch})

    def _safe_send(self, ws, payload):
        try:
            ws.send(json.dumps(payload))
        except Exception:
            with self._lock:
                self._clients.pop(ws, None)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="dashboard", daemon=True)
        self._thread.start()

    def _run(self):
        from flask import Flask, jsonify, request, send_from_directory
        from flask_sock import Sock

        static_dir = Path(__file__).resolve().parent / "static"
        app = Flask(__name__, static_folder=None)
        # Keep WebSocket connections open indefinitely (metrics can be sparse between episodes).
        app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
        sock = Sock(app)
        server = self

        @app.route("/")
        def index():
            return send_from_directory(static_dir, "index.html")

        @app.route("/<path:path>")
        def static_files(path):
            return send_from_directory(static_dir, path)

        @app.route("/api/meta")
        def api_meta():
            with server._lock:
                return jsonify(dict(server._meta))

        @app.route("/api/metrics")
        def api_metrics():
            with server._lock:
                data = list(server._history)
            since = request.args.get("since")
            if since is not None:
                try:
                    cutoff = int(since)
                    data = [row for row in data if int(row.get("episode", 0)) > cutoff]
                except ValueError:
                    pass
            return jsonify({"metrics": data, "count": len(data)})

        @sock.route("/ws")
        def ws_route(ws):
            with server._lock:
                server._clients[ws] = {"batch": 1, "buf": []}
                snapshot = list(server._history)[-1000:]
                meta = dict(server._meta)
            server._safe_send(ws, {"type": "meta", "data": meta})
            server._safe_send(ws, {"type": "snapshot", "data": snapshot})
            try:
                while True:
                    message = ws.receive()
                    if message is None:
                        break
                    try:
                        config = json.loads(message)
                    except (ValueError, TypeError):
                        continue
                    if "batch" in config:
                        with server._lock:
                            state = server._clients.get(ws)
                            if state is not None:
                                state["batch"] = max(1, int(config["batch"]))
            except Exception:
                pass
            finally:
                with server._lock:
                    server._clients.pop(ws, None)

        # threaded=True: one thread per HTTP/WS connection so record() can broadcast while a
        # WS handler blocks in receive(). use_reloader=False: never fork inside training.
        app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)
