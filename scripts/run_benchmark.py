#!/usr/bin/env python3
"""
FLUXOR 27-Trial Automated Benchmark Campaign Orchestrator.
Executes 3 scenarios x 3 algorithms x 3 repetitions = 27 trials.
Collects k6 latency, throughput, error rates, and Prometheus CPU time-series.
"""

import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Configuration
RUNNER_URL   = os.getenv("RUNNER_URL", "http://localhost:5006")
RL_AGENT_URL = os.getenv("RL_AGENT_URL", "http://localhost:5005")
LB_URL       = os.getenv("LB_URL", "http://localhost:8080")
PROM_URL     = os.getenv("PROM_URL", "http://localhost:9090")

SCENARIOS  = ["steady", "spike", "failure"]
ALGORITHMS = ["rr", "lc", "rl"]
TRIALS     = ["trial1", "trial2", "trial3"]

RESULTS_DIR = Path("results")


def request_json(url: str, method: str = "GET", payload: dict = None, timeout: float = 30.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def set_rl_mode(mode: str):
    """Switch RL agent between 'evaluation' (frozen weights, eps=0) and 'training'."""
    url = f"{RL_AGENT_URL}/eval_mode"
    try:
        res = request_json(url, method="POST", payload={"mode": mode})
        print(f"[RL] Mode switched to: {res.get('mode')} (eps={res.get('epsilon')}, training={res.get('training_enabled')})")
        return res
    except Exception as e:
        print(f"[RL] Warning: Failed to set RL mode {mode}: {e}")
        return None


def wait_until_idle(timeout_seconds: float = 120.0):
    """Wait for any running experiment in experiment-runner to complete."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            st = request_json(f"{RUNNER_URL}/status", timeout=5.0)
            if not st.get("running"):
                return True
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError("Experiment runner did not become idle in time")


def query_prometheus_cpu(start_time: float, end_time: float) -> dict:
    """
    Query Prometheus for backend_cpu_utilization over [start_time, end_time].
    Computes mean CPU per backend and sample counts.
    Distinguishes available backends from missing/stopped backends.
    """
    params = {
        "query": "backend_cpu_utilization",
        "start": f"{start_time:.3f}",
        "end": f"{end_time:.3f}",
        "step": "2s",
    }
    url = f"{PROM_URL}/api/v1/query_range?{urlencode(params)}"
    data = request_json(url, timeout=15.0)

    per_server_samples = {"1": [], "2": [], "3": []}

    results = data.get("data", {}).get("result", [])
    for entry in results:
        srv_id = entry.get("metric", {}).get("server_id")
        if srv_id in per_server_samples:
            vals = [float(v[1]) for v in entry.get("values", [])]
            per_server_samples[srv_id].extend(vals)

    server_means = {}
    server_counts = {}
    valid_means = []

    for srv_id in ["1", "2", "3"]:
        vals = per_server_samples[srv_id]
        server_counts[f"cpu_samples_srv_{srv_id}"] = len(vals)
        if vals:
            m = sum(vals) / len(vals)
            server_means[f"mean_cpu_srv_{srv_id}"] = round(m, 2)
            valid_means.append(m)
        else:
            server_means[f"mean_cpu_srv_{srv_id}"] = None

    # Calculate CPU standard deviation across available backends
    if len(valid_means) > 1:
        avg_mean = sum(valid_means) / len(valid_means)
        variance = sum((x - avg_mean) ** 2 for x in valid_means) / len(valid_means)
        cpu_std_dev = round(math.sqrt(variance), 3)
    elif len(valid_means) == 1:
        cpu_std_dev = 0.0
    else:
        cpu_std_dev = None

    return {
        **server_means,
        **server_counts,
        "available_backends_count": len(valid_means),
        "cpu_std_dev": cpu_std_dev,
    }


def main():
    print("=" * 60)
    print("FLUXOR 27-Trial Automated Benchmark Campaign")
    print(f"Scenarios:   {SCENARIOS}")
    print(f"Algorithms:  {ALGORITHMS}")
    print(f"Repetitions: {TRIALS}")
    print("=" * 60)

    timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = RESULTS_DIR / f"benchmark_campaign_{timestamp_str}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    trial_idx = 0
    total_trials = len(SCENARIOS) * len(ALGORITHMS) * len(TRIALS)

    try:
        # Pre-flight check: ensure cluster is clean
        wait_until_idle()

        for scenario in SCENARIOS:
            for algo in ALGORITHMS:
                for trial in TRIALS:
                    trial_idx += 1
                    print(f"\n[{trial_idx}/{total_trials}] Starting {scenario.upper()} - {algo.upper()} - {trial}...")

                    # Wait for clean state before each trial
                    wait_until_idle()
                    time.sleep(2.0)

                    # Manage RL mode
                    if algo == "rl":
                        set_rl_mode("evaluation")
                    else:
                        set_rl_mode("training")

                    start_wall_time = time.time()

                    # Trigger trial
                    start_resp = request_json(
                        f"{RUNNER_URL}/start",
                        method="POST",
                        payload={"scenario": scenario, "algorithm": algo, "trial": trial},
                        timeout=15.0,
                    )
                    run_id = start_resp.get("run_id")
                    print(f"  -> Run ID: {run_id}")

                    # Wait for experiment to finish
                    # Normal: 60s, Spike: 80s, Failure: 90s
                    expected_duration = 60 if scenario == "steady" else (80 if scenario == "spike" else 90)
                    time.sleep(expected_duration - 5)

                    wait_until_idle(timeout_seconds=90.0)
                    end_wall_time = time.time()
                    print(f"  -> Finished in {end_wall_time - start_wall_time:.1f}s")

                    # Small settle window for Prometheus metrics
                    time.sleep(4.0)

                    # Query Prometheus CPU over [start_wall_time, end_wall_time]
                    cpu_metrics = query_prometheus_cpu(start_wall_time, end_wall_time)

                    # Locate and load k6 summary
                    summary_file = RESULTS_DIR / f"{run_id}_summary.json"
                    if not summary_file.exists():
                        # Try finding by glob if filename prefix slightly differed
                        matches = list(RESULTS_DIR.glob(f"*{scenario}_{algo}_{trial}_summary.json"))
                        if matches:
                            summary_file = matches[0]

                    if not summary_file.exists():
                        raise FileNotFoundError(f"Summary file not found for run {run_id}")

                    with summary_file.open() as f:
                        summary_data = json.load(f)

                    metrics = summary_data.get("metrics", {})
                    http_reqs = metrics.get("http_reqs", {})
                    req_duration = metrics.get("http_req_duration", {})
                    req_failed = metrics.get("http_req_failed", {})

                    thresholds = {}
                    for metric_key in ["http_req_duration", "http_req_failed"]:
                        m = metrics.get(metric_key, {})
                        if "thresholds" in m:
                            thresholds.update(m["thresholds"])

                    # Build consolidated record
                    result_record = {
                        "run_id": run_id,
                        "scenario": scenario,
                        "algorithm": algo,
                        "trial": trial,
                        "timestamp_start": datetime.utcfromtimestamp(start_wall_time).isoformat() + "Z",
                        "timestamp_end": datetime.utcfromtimestamp(end_wall_time).isoformat() + "Z",
                        "duration_seconds": round(end_wall_time - start_wall_time, 2),
                        "total_requests": int(http_reqs.get("count", 0)),
                        "throughput_rps": round(float(http_reqs.get("rate", 0.0)), 2),
                        "mean_latency_ms": round(float(req_duration.get("avg", 0.0)), 2),
                        "median_p50_ms": round(float(req_duration.get("med", 0.0)), 2),
                        "p90_latency_ms": round(float(req_duration.get("p(90)", 0.0)), 2),
                        "p95_latency_ms": round(float(req_duration.get("p(95)", 0.0)), 2),
                        "max_latency_ms": round(float(req_duration.get("max", 0.0)), 2),
                        "http_failed_count": int(req_failed.get("passes", 0)),
                        "http_failed_rate": round(float(req_failed.get("value", 0.0)), 4),
                        "threshold_results": thresholds,
                        **cpu_metrics,
                    }

                    all_results.append(result_record)

                    # Persist per-run record and intermediate progress
                    with (campaign_dir / f"{run_id}_metrics.json").open("w") as f:
                        json.dump(result_record, f, indent=2)

                    with (campaign_dir / "progress.json").open("w") as f:
                        json.dump(all_results, f, indent=2)

                    print(f"  -> Req: {result_record['total_requests']} | "
                          f"RPS: {result_record['throughput_rps']} | "
                          f"Mean Lat: {result_record['mean_latency_ms']}ms | "
                          f"P95: {result_record['p95_latency_ms']}ms | "
                          f"CPU StdDev: {result_record['cpu_std_dev']}")

    except KeyboardInterrupt:
        print("\n[WARNING] Benchmark interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Trial execution error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: Ensure RL training mode is restored and backend-2 is running
        print("\nCleaning up...")
        try:
            set_rl_mode("training")
        except Exception as error:
            print(f"[WARNING] Could not restore RL training mode: {error}")

        try:
            request_json(f"{RUNNER_URL}/stop", method="POST", timeout=60)
        except Exception as error:
            print(f"[WARNING] Runner cleanup issue: {error}")

    if len(all_results) != 27:
        print(f"[WARNING] Campaign incomplete: {len(all_results)}/27 trials completed.")

    if not all_results:
        return

    csv_path = campaign_dir / "benchmark_results.csv"
    json_path = campaign_dir / "benchmark_results.json"

    fields = list(all_results[0].keys())
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in all_results:
            row = result.copy()
            row["threshold_results"] = json.dumps(row["threshold_results"])
            writer.writerow(row)

    with json_path.open("w") as file:
        json.dump(all_results, file, indent=2)

    print("\nCampaign finished.")
    print(f"Completed: {len(all_results)}/27")
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
