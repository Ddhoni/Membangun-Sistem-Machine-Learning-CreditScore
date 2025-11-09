from flask import Flask, request, jsonify, Response
import os, time, json
import requests
import psutil
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

# =======================
# Config (via env var)
# =======================
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:5005/invocations")  # MLflow serve
HOST      = os.getenv("HOST", "0.0.0.0")
PORT      = int(os.getenv("PORT", "8000"))
TIMEOUT_S = float(os.getenv("TIMEOUT_S", "10"))

app = Flask(__name__)

# =======================
# Metrics
# =======================
# HTTP-level (pakai label method/path/status untuk fleksibel)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
    labelnames=["method", "path", "status"]
)

HTTP_REQUEST_SIZE_BYTES = Histogram(
    "http_request_size_bytes", "Request payload size (bytes)"
)
HTTP_RESPONSE_SIZE_BYTES = Histogram(
    "http_response_size_bytes", "Response payload size (bytes)"
)

HTTP_INFLIGHT = Gauge(
    "http_requests_in_flight", "Number of in-flight HTTP requests"
)

# Business / model-level (opsional, jika model kirim probabilitas)
MODEL_CONFIDENCE = Histogram(
    "model_confidence", "Max probability of prediction",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# System metrics (ringan, jangan blocking)
SYSTEM_CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percent")
SYSTEM_RAM_USAGE = Gauge("system_ram_usage_percent", "RAM usage percent")
SYSTEM_DISK_USAGE = Gauge("system_disk_usage_percent", "Disk usage percent")
SYSTEM_NET_BYTES_SENT = Gauge("system_net_bytes_sent_total", "Total bytes sent")
SYSTEM_NET_BYTES_RECV = Gauge("system_net_bytes_recv_total", "Total bytes received")

# =======================
# Helpers
# =======================
def _update_system_metrics():
    # non-blocking: psutil.cpu_percent(interval=0) -> rolling
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=0))
    SYSTEM_RAM_USAGE.set(psutil.virtual_memory().percent)
    SYSTEM_DISK_USAGE.set(psutil.disk_usage("/").percent)
    net = psutil.net_io_counters()
    SYSTEM_NET_BYTES_SENT.set(net.bytes_sent)
    SYSTEM_NET_BYTES_RECV.set(net.bytes_recv)

def _call_model(payload: dict) -> requests.Response:
    return requests.post(
        MODEL_URL, headers={"Content-Type": "application/json"},
        data=json.dumps(payload), timeout=TIMEOUT_S
    )

# =======================
# Routes
# =======================
@app.route("/metrics", methods=["GET"])
def metrics():
    _update_system_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    path = "/predict"
    method = "POST"
    HTTP_INFLIGHT.inc()
    start = time.time()
    status = "500"

    try:
        req_json = request.get_json(silent=True) or {}
        # ukuran request
        HTTP_REQUEST_SIZE_BYTES.observe(len(request.data or b""))

        # proxy ke model
        r = _call_model(req_json)
        status = str(r.status_code)

        # catat response size
        HTTP_RESPONSE_SIZE_BYTES.observe(len(r.content or b""))

        # jika model return proba: {"predictions": [[p1,p2,...]]}
        try:
            out = r.json()
            preds = out.get("predictions", [])
            if preds and isinstance(preds[0], list):
                MODEL_CONFIDENCE.observe(max(preds[0]))
        except Exception:
            pass

        return (r.content, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        dt = time.time() - start
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path, status=status).observe(dt)
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
        HTTP_INFLIGHT.dec()

# =======================
# Main
# =======================
if __name__ == "__main__":
    print(f"🚀 Exporter+Proxy running on http://{HOST}:{PORT}  → MODEL_URL={MODEL_URL}")
    app.run(host=HOST, port=PORT, threaded=True)