from flask import Flask, request, jsonify, Response
import os
import time
import json
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
# METRIK HTTP / API MODEL
# =======================

# Total request (pakai label biar bisa dipecah method/path/status)
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
    labelnames=["method", "path", "status"],
)

# Throughput: ini sebenarnya total requests juga,
# 'per second'-nya nanti dihitung di Prometheus pakai rate()
THROUGHPUT = Counter(
    "http_requests_throughput_total",
    "Total HTTP requests (for throughput calculation)"
)

ERROR_COUNT = Counter(
    "http_requests_error_total",
    "Total number of failed requests",
    ["method", "path"]
)

SUCCESS_COUNT = Counter(
    "http_requests_success_total",
    "Total number of successful requests",
    ["method", "path"]
)

REQUEST_SIZE = Histogram(
    "http_request_size_bytes",
    "Size of HTTP request payload (bytes)"
)

RESPONSE_SIZE = Histogram(
    "http_response_size_bytes",
    "Size of HTTP response payload (bytes)"
)

HTTP_INFLIGHT = Gauge(
    "http_requests_in_flight",
    "Number of in-flight HTTP requests"
)

# Business / model-level (opsional, jika model kirim probabilitas)
MODEL_CONFIDENCE = Histogram(
    "model_confidence",
    "Max probability of prediction",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# =======================
# METRIK SISTEM
# =======================
CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage Percentage")
DISK_USAGE = Gauge("system_disk_usage", "Disk Usage Percentage")
NET_BYTES_SENT = Gauge("system_net_bytes_sent", "Total Bytes Sent (OS counter)")
NET_BYTES_RECV = Gauge("system_net_bytes_recv", "Total Bytes Received (OS counter)")

# =======================
# Helpers
# =======================
def _update_system_metrics():
    # interval=0 → non-blocking, pakai rolling average
    CPU_USAGE.set(psutil.cpu_percent(interval=0))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage("/").percent)
    net = psutil.net_io_counters()
    NET_BYTES_SENT.set(net.bytes_sent)
    NET_BYTES_RECV.set(net.bytes_recv)

def _call_model(payload: dict) -> requests.Response:
    return requests.post(
        MODEL_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=TIMEOUT_S,
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
    THROUGHPUT.inc()  # total request untuk dihitung rate() di Prometheus

    start = time.time()
    status = "500"

    try:
        req_json = request.get_json(silent=True) or {}

        # ukuran request (pakai raw body biar akurat)
        REQUEST_SIZE.observe(len(request.data or b""))

        # proxy ke model
        r = _call_model(req_json)
        status = str(r.status_code)

        # catat response size
        RESPONSE_SIZE.observe(len(r.content or b""))

        # jika model return proba: {"predictions": [[p1,p2,...]]}
        try:
            out = r.json()
            preds = out.get("predictions", [])
            if preds and isinstance(preds[0], list):
                MODEL_CONFIDENCE.observe(max(preds[0]))
        except Exception:
            # kalau parsing JSON gagal, abaikan saja untuk metrik confidence
            pass

        # hitung sukses/error
        if 200 <= r.status_code < 400:
            SUCCESS_COUNT.labels(method=method, path=path).inc()
        else:
            ERROR_COUNT.labels(method=method, path=path).inc()

        return (
            r.content,
            r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")},
        )

    except Exception as e:
        # error di sisi proxy (bukan model)
        ERROR_COUNT.labels(method=method, path=path).inc()
        return jsonify({"error": str(e)}), 500

    finally:
        dt = time.time() - start
        REQUEST_LATENCY.labels(method=method, path=path, status=status).observe(dt)
        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        HTTP_INFLIGHT.dec()

# =======================
# Main
# =======================
if __name__ == "__main__":
    print(f"🚀 Exporter+Proxy running on http://{HOST}:{PORT}  → MODEL_URL={MODEL_URL}")
    app.run(host=HOST, port=PORT, threaded=True)