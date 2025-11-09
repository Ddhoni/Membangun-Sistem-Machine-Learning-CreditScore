import requests
import json
import time
import random
from prometheus_client import start_http_server, Summary, Counter, Histogram, Gauge

# =========================================================
# CONFIG
# =========================================================
PREDICT_URL = "http://127.0.0.1:8000/predict"   # endpoint Flask wrapper
PROM_PORT = 9000                                # port exporter Prometheus

# contoh payload (disesuaikan dengan model kamu)
sample_input = {
    "instances": [
        [0.5869, 0.2479, 0.4, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.8, 0.35, 0.6, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    ]
}

# =========================================================
# PROMETHEUS METRICS
# =========================================================
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Latency per prediction (s)",
                            buckets=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5])
REQUEST_COUNT   = Counter("inference_requests_total", "Total prediction requests")
ERROR_COUNT     = Counter("inference_errors_total", "Total failed predictions")
CONFIDENCE      = Summary("inference_confidence", "Predicted confidence average")
REQ_INFLIGHT    = Gauge("inference_inflight", "Requests currently in flight")
CPU_SIMULATED   = Gauge("cpu_usage_percent", "Simulated CPU usage (for Grafana demo)")
MEM_SIMULATED   = Gauge("memory_usage_mb", "Simulated memory usage (for Grafana demo)")

# =========================================================
# MAIN LOOP (continuous load generation)
# =========================================================
def send_request():
    """Send prediction request and update Prometheus metrics."""
    headers = {"Content-Type": "application/json"}
    REQ_INFLIGHT.inc()
    t0 = time.time()
    try:
        response = requests.post(PREDICT_URL, headers=headers, data=json.dumps(sample_input), timeout=5)
        dt = time.time() - t0
        REQUEST_LATENCY.observe(dt)
        REQUEST_COUNT.inc()

        if response.status_code == 200:
            data = response.json()
            # jika model return probabilitas, ambil confidence (opsional)
            if isinstance(data, dict) and "predictions" in data:
                preds = data["predictions"]
                if preds and isinstance(preds[0], list):
                    CONFIDENCE.observe(max(preds[0]))
            print("✅ Prediction:", data)
        else:
            ERROR_COUNT.inc()
            print("❌ Error:", response.status_code, response.text)
    except Exception as e:
        ERROR_COUNT.inc()
        print("⚠️ Exception:", e)
    finally:
        REQ_INFLIGHT.dec()


def main():
    print(f"🚀 Serving monitoring exporter at :{PROM_PORT}/metrics")
    start_http_server(PROM_PORT)

    while True:
        # setiap iterasi kirim 1 request (simulate ~2 rps)
        send_request()

        # dummy resource metrics (buat panel CPU/MEM di Grafana)
        CPU_SIMULATED.set(random.uniform(10, 40))
        MEM_SIMULATED.set(random.uniform(150, 400))
        time.sleep(0.5)  # 2 request per detik


if __name__ == "__main__":
    main()