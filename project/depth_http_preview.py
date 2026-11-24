"""LAN preview + compatibility API for roi-site/ar.html."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import socket
import json
import time
import cv2
import numpy as np

SITE_ORIGIN = "https://tfhguk.github.io"


def _default_settings(rect, width, height, height_max=0.5, gamma=0.5, interval=1.0):
    l, t, r, b = rect
    return {
        "schema_version": 1,
        "height_range_m": [0.0, float(height_max)],
        "above_range": "black",
        "palette_gamma": float(gamma),
        "update_interval_s": float(interval),
        "roi": {
            "coordinate_space": "projector_normalized",
            "vertices": [[l / width, t / height], [r / width, t / height],
                         [r / width, b / height], [l / width, b / height]],
        },
    }


class DepthPreview:
    def __init__(self, host="0.0.0.0", port=8765, calibrated=False,
                 projection_size=(1280, 800), available_rect=None,
                 projector_ready=False, stage_hint=None, initial_settings=None,
                 on_settings=None, on_action=None):
        self._lock = threading.RLock()
        self._jpeg = None
        self.calibrated = bool(calibrated)
        self.projector_ready = bool(projector_ready)
        self.stage_hint = stage_hint
        self.width, self.height = map(int, projection_size)
        self.available_rect = tuple(available_rect or (0, 0, self.width, self.height))
        self.projection_revision = "local-" + str(time.time_ns())
        self.settings_revision = 0
        self.snapshot_id = 0
        self.snapshot_time = 0.0
        self.snapshot_settings_revision = 0
        self.update_count = 0
        self.settings = initial_settings or _default_settings(
            self.available_rect, self.width, self.height)
        self.on_settings = on_settings
        self.on_action = on_action
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _cors(self):
                origin = self.headers.get("Origin")
                if origin == SITE_ORIGIN:
                    self.send_header("Access-Control-Allow-Origin", SITE_ORIGIN)
                    self.send_header("Vary", "Origin")
                # Chrome may ask permission for a public HTTPS page to reach a
                # private/local service. Only the known page origin is allowed.
                if origin == SITE_ORIGIN:
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Expose-Headers",
                                 "X-Snapshot-Id, X-Projection-Revision, X-Settings-Revision")

            def _send(self, code, body, content_type="application/json", extra=None):
                try:
                    self.send_response(code)
                    self._cors()
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    if extra:
                        for key, value in extra.items():
                            self.send_header(key, str(value))
                    self.end_headers()
                    self.wfile.write(body)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    # Browser closed/reloaded. This is a normal disconnect.
                    pass

            def _json(self, code, data):
                self._send(code, json.dumps(data, separators=(",", ":")).encode("utf-8"))

            def do_OPTIONS(self):
                try:
                    self.send_response(204)
                    self._cors()
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                except OSError:
                    pass

            def do_GET(self):
                try:
                    if self.path == "/" or self.path.startswith("/?"):
                        self._send(200, owner.PAGE, "text/html; charset=utf-8")
                    elif self.path.startswith("/status"):
                        with owner._lock:
                            ready = owner._jpeg is not None
                            data = {"calibrated": owner.calibrated, "frame_ready": ready}
                        self._json(200, data)
                    elif self.path.startswith("/frame.jpg"):
                        with owner._lock:
                            body = owner._jpeg
                        if body is None:
                            self._send(503, b"Frame not ready", "text/plain")
                        else:
                            self._send(200, body, "image/jpeg")
                    elif self.path == "/api/v1/state":
                        with owner._lock:
                            data = owner._state_locked()
                        self._json(200, data)
                    elif self.path == "/api/v1/preview.jpg":
                        with owner._lock:
                            body = owner._jpeg
                            sid = owner.snapshot_id
                            rev = owner.projection_revision
                            srev = owner.snapshot_settings_revision
                        if body is None:
                            self._send(503, json.dumps({"error":{"message":"Кадр ещё не готов"}}).encode(), "application/json")
                        else:
                            self._send(200, body, "image/jpeg", {
                                "X-Snapshot-Id": sid,
                                "X-Projection-Revision": rev,
                                "X-Settings-Revision": srev,
                            })
                    elif self.path.startswith("/api/v1/actions/"):
                        self._json(404, {"error":{"message":"Action endpoint not found","field":"action"}})
                    else:
                        self._send(404, b"Not found", "text/plain")
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    pass

            def do_POST(self):
                if self.path.startswith("/api/v1/actions/"):
                    action = self.path.rsplit("/", 1)[-1]
                    if action == "calibrate-floor":
                        self._json(501, {"error":{"message":"Калибровку пола запускайте в локальном окне AR, не через веб-сайт","field":"action"}})
                        return
                    if action != "rescan":
                        self._json(404, {"error":{"message":"Unknown action","field":"action"}})
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        request = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                        if request.get("projection_revision") != owner.projection_revision:
                            self._json(409, {"error":{"message":"Калибровка изменилась","field":"projection_revision"}})
                            return
                        if owner.on_action:
                            owner.on_action("rescan")
                        self._json(200, {"request_id":request.get("request_id"), "accepted":True,
                                         "action_id":"rescan-"+str(time.time_ns())})
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                        pass
                    except Exception as error:
                        self._json(500, {"error":{"message":str(error)}})
                    return
                if self.path != "/api/v1/settings":
                    self._json(404, {"error":{"message":"Endpoint not implemented","field":"action"}})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 64 * 1024:
                        self._json(400, {"error":{"message":"Некорректный размер запроса"}})
                        return
                    request = json.loads(self.rfile.read(length).decode("utf-8"))
                    with owner._lock:
                        expected = owner.settings_revision
                        if request.get("projection_revision") != owner.projection_revision:
                            self._json(409, {"error":{"message":"Калибровка проектора изменилась","field":"projection_revision"}})
                            return
                        if request.get("expected_settings_revision") != expected:
                            self._json(409, {"error":{"message":"Настройки изменились; обновите их перед применением","field":"expected_settings_revision"}})
                            return
                        if not owner.projector_ready:
                            self._json(409, {"error":{"message":"3D-калибровка проектора ещё не подтверждена","field":"projection"}})
                            return
                        validated = owner._validate_settings(request)
                        callback = owner.on_settings
                        if callback is not None:
                            callback(validated)
                        owner.settings = validated
                        owner.settings_revision += 1
                        # Snapshot revision is advanced only when a new image is
                        # rendered, so the page waits for a matching preview.
                        reply = {"request_id": request.get("request_id"),
                                 "settings_revision": owner.settings_revision,
                                 "snapshot_id": owner.snapshot_id}
                    self._json(200, reply)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    pass
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    self._json(400, {"error":{"message":str(error),"field":"settings"}})
                except Exception as error:
                    self._json(500, {"error":{"message":"Не удалось применить настройки: "+str(error)}})

            def log_message(self, *_):
                pass

        # The simple standalone viewer page remains available at /.
        self.PAGE = ('''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth map preview</title><style>body{font:16px system-ui;background:#111;color:#eee;margin:1rem}img{display:block;max-width:100%;height:auto;background:#222}#status{margin:.5rem 0;color:#aaa}</style>
<h2>Карта глубин / выход проектора</h2><div id="status">Подключение…</div><img id="map" alt="Карта глубин">
<script>const img=document.querySelector('#map'),s=document.querySelector('#status');let busy=false;async function tick(){if(busy)return;busy=true;try{const r=await fetch('/status',{cache:'no-store'});const j=await r.json();s.textContent=j.calibrated?'Поток активен · метрическая глубина':'Поток активен · диагностическая disparity';if(j.frame_ready)img.src='/frame.jpg?t='+Date.now()}catch(e){s.textContent='Нет связи'}finally{busy=false}}setInterval(tick,500);tick();</script></html>''').encode("utf-8")

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        print(f"Depth preview local: http://localhost:{self.port}/", flush=True)
        addresses = self._lan_ipv4_addresses()
        for ip in addresses:
            print(f"Depth preview LAN: http://{ip}:{self.port}/", flush=True)
        if not addresses:
            print(f"Depth preview LAN: use this PC IPv4 address at port {self.port}", flush=True)

    def _state_locked(self):
        stage = self.stage_hint or ("stereo_required" if not self.calibrated else
                 "floor_required" if not self.projector_ready else "ready")
        projection = {
            "width": self.width, "height": self.height,
            "available_rect": list(self.available_rect),
            "calibration_3d_accepted": self.projector_ready,
        }
        snapshot = None
        if self._jpeg is not None:
            snapshot = {
                "id": self.snapshot_id,
                "age_s": max(0.0, time.time() - self.snapshot_time),
                "projection_revision": self.projection_revision,
                "settings_revision": self.snapshot_settings_revision,
            }
        return {
            "schema_version": 1, "stage": stage,
            "projection_revision": self.projection_revision,
            "projection": projection,
            "settings_revision": self.settings_revision,
            "settings": self.settings,
            "snapshot": snapshot,
            "metrics": {"check_hz": 0.0, "update_count": self.update_count,
                        "measured_fraction": 0.0, "filled_fraction": 0.0},
            "errors": [],
        }

    def _validate_settings(self, request):
        if request.get("schema_version") != 1:
            raise ValueError("Unsupported settings schema")
        h = request.get("height_range_m")
        if not isinstance(h, list) or len(h) != 2:
            raise ValueError("height_range_m must have two values")
        lo, hi = map(float, h)
        gamma = float(request.get("palette_gamma"))
        interval = float(request.get("update_interval_s"))
        if not (0 <= lo < hi <= 0.5 and 0.1 <= gamma <= 3 and 0.5 <= interval <= 10):
            raise ValueError("Settings out of supported range")
        if request.get("above_range") != "black":
            raise ValueError("above_range must be black")
        roi = request.get("roi") or {}
        if roi.get("coordinate_space") != "projector_normalized":
            raise ValueError("ROI coordinate space is invalid")
        vertices = roi.get("vertices")
        if not isinstance(vertices, list) or not 3 <= len(vertices) <= 32:
            raise ValueError("ROI needs 3 to 32 vertices")
        l, t, r, b = self.available_rect
        for point in vertices:
            if (not isinstance(point, list) or len(point) != 2 or
                    not all(np.isfinite(float(v)) for v in point) or
                    not (0 <= float(point[0]) <= 1 and 0 <= float(point[1]) <= 1) or
                    not (l <= float(point[0]) * self.width <= r and
                         t <= float(point[1]) * self.height <= b)):
                raise ValueError("ROI vertices must stay inside the available projector area")
        return {
            "schema_version": 1, "height_range_m": [lo, hi],
            "above_range": "black", "palette_gamma": gamma,
            "update_interval_s": interval,
            "roi": {"coordinate_space":"projector_normalized", "vertices":vertices},
        }

    @staticmethod
    def _lan_ipv4_addresses():
        addresses = set()
        try:
            for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
                ip = item[4][0]
                if not ip.startswith("127."):
                    addresses.add(ip)
        except OSError:
            pass
        return sorted(addresses)

    def update(self, image):
        if image is None:
            return
        try:
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with self._lock:
                    self._jpeg = data.tobytes()
                    self.snapshot_id += 1
                    self.snapshot_time = time.time()
                    self.snapshot_settings_revision = self.settings_revision
                    self.update_count += 1
        except cv2.error:
            pass

    def close(self):
        self.server.shutdown()
        self.server.server_close()
