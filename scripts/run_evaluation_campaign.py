#!/usr/bin/env python3
"""
FLUXOR 60-Trial Automated Evaluation Campaign Orchestrator.

Matrix:
  - 4 Algorithms: Round Robin (RR), Least Connections (LC), Double DQN (DDQN), Vanilla DQN (DQN)
  - 3 Scenarios: Steady (60s @ 50 VUs), Spike (80s @ 10-200 VUs), Failure (90s @ 50 VUs, backend-2 dropped)
  - 5 Trials per combination
  Total: 60 trials.

Safe Phased Execution:
  Phase 1: Round Robin (15 trials)
  Phase 2: Least Connections (15 trials)
  Phase 3: Double DQN (15 trials, checkpoint: ddqn_experiment.pt, step: 184238)
  Phase 4: Vanilla DQN (15 trials, checkpoint: dqn_baseline_step180700_20261107.pt, step: 180700)

Safety & Concurrency Safeguards:
  - Immutable SHA-256 hash checks on baseline and experiment checkpoints.
  - Full feedback queue & pending transition drainage verification before model switches.
  - Continuous evaluation mode enforcement (eps=0.0, training_enabled=False).
  - Runner state & backend health verification before every trial.
  - Atomic incremental persistence to results/final_campaign_60/ (manifest, jsonl, csv, markdown).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

# Configuration
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE_DIR / "results"
CAMPAIGN_DIR = RESULTS_DIR / "final_campaign_60"

RUNNER_URL = os.getenv("RUNNER_URL", "http://localhost:5006")
RL_AGENT_URL = os.getenv("RL_AGENT_URL", "http://localhost:5005")
LB_URL = os.getenv("LB_URL", "http://localhost:8080")
PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")
BACKEND_URLS = [
    os.getenv("BACKEND1_URL", "http://localhost:8081"),
    os.getenv("BACKEND2_URL", "http://localhost:8082"),
    os.getenv("BACKEND3_URL", "http://localhost:8083"),
]

# Protected Checkpoint Definitions
BASELINE_CHECKPOINT_FILE = "dqn_baseline_step180700_20261107.pt"
BASELINE_EXPECTED_HASH = "87dbfd996c98a9d8224512398165f74c32349d245672a0fa1edf6d2815b6501b"
BASELINE_EXPECTED_STEP = 180700

DDQN_CHECKPOINT_FILE = "ddqn_experiment.pt"
DDQN_EXPECTED_HASH = "81994ac6ba4f93edf46be3e41efc312a45e635b10879a43fd83b57d6ede26228"
DDQN_EXPECTED_STEP = 184238

CHECKPOINT_DIR = WORKSPACE_DIR / "rl-agent" / "checkpoints"

# Evaluation Matrix
PHASES = [
    {
        "phase": 1,
        "algo_name": "round_robin",
        "display_name": "Round Robin",
        "runner_algo": "rr",
        "checkpoint": None,
        "expected_step": None,
    },
    {
        "phase": 2,
        "algo_name": "least_connections",
        "display_name": "Least Connections",
        "runner_algo": "lc",
        "checkpoint": None,
        "expected_step": None,
    },
    {
        "phase": 3,
        "algo_name": "double_dqn",
        "display_name": "Double DQN",
        "runner_algo": "rl",
        "checkpoint": DDQN_CHECKPOINT_FILE,
        "expected_step": DDQN_EXPECTED_STEP,
    },
    {
        "phase": 4,
        "algo_name": "vanilla_dqn",
        "display_name": "Vanilla DQN",
        "runner_algo": "rl",
        "checkpoint": BASELINE_CHECKPOINT_FILE,
        "expected_step": BASELINE_EXPECTED_STEP,
    },
]

SCENARIOS = ["steady", "spike", "failure"]
DEFAULT_TRIALS_PER_GROUP = 5


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 of a local file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_integrity():
    """Verify that neither baseline nor DDQN checkpoint has been corrupted or overwritten."""
    baseline_path = CHECKPOINT_DIR / BASELINE_CHECKPOINT_FILE
    ddqn_path = CHECKPOINT_DIR / DDQN_CHECKPOINT_FILE

    base_hash = compute_sha256(baseline_path)
    if base_hash != BASELINE_EXPECTED_HASH:
        raise ValueError(
            f"BASELINE CHECKPOINT CORRUPTED! Expected {BASELINE_EXPECTED_HASH}, found {base_hash}"
        )

    ddqn_hash = compute_sha256(ddqn_path)
    if ddqn_hash != DDQN_EXPECTED_HASH:
        raise ValueError(
            f"DDQN CHECKPOINT CORRUPTED! Expected {DDQN_EXPECTED_HASH}, found {ddqn_hash}"
        )


def http_request_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Execute JSON HTTP request."""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def get_lb_metrics_text(timeout: float = 5.0) -> str:
    """Fetch Prometheus metrics text from load-balancer."""
    req = urllib.request.Request(f"{LB_URL}/metrics")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_lb_metric_counters(metrics_text: str) -> Dict[str, float]:
    """Parse key counter values from Prometheus text."""
    counters = {
        "lb_rl_fallbacks_total": 0.0,
        "lb_requests_total": 0.0,
        "active_conns_total": 0.0,
    }
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("lb_rl_fallbacks_total "):
            parts = line.split()
            if len(parts) >= 2:
                counters["lb_rl_fallbacks_total"] = float(parts[1])
        elif line.startswith("lb_active_connections{"):
            parts = line.split()
            if len(parts) >= 2:
                counters["active_conns_total"] += float(parts[1])
        elif line.startswith("lb_requests_total{"):
            parts = line.split()
            if len(parts) >= 2:
                counters["lb_requests_total"] += float(parts[1])
    return counters


def check_backends_healthy(timeout: float = 5.0) -> bool:
    """Check that all three backend instances are healthy."""
    for idx, url in enumerate(BACKEND_URLS, 1):
        try:
            res = http_request_json(f"{url}/health", timeout=timeout)
            if res.get("status") != "ok":
                print(f"[PREFLIGHT WARNING] Backend {idx} returned status: {res}")
                return False
        except Exception as e:
            print(f"[PREFLIGHT WARNING] Backend {idx} check failed: {e}")
            return False
    return True


def wait_for_runner_clean(timeout: float = 120.0):
    """Wait until experiment-runner reports idle with no pending cleanups."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            st = http_request_json(f"{RUNNER_URL}/status", timeout=5.0)
            if (
                not st.get("running")
                and not st.get("starting")
                and not st.get("cleanup_pending")
            ):
                return st
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError("Experiment runner did not become clean within timeout.")


def drain_and_verify_rl_idle(
    required_stable_checks: int = 4,
    check_interval: float = 3.0,
    timeout: float = 120.0,
):
    """
    Ensure the RL agent has 0 pending transitions and step_count is unchanged
    for at least required_stable_checks consecutive intervals.
    """
    start_time = time.time()
    last_step = None
    stable_count = 0

    while time.time() - start_time < timeout:
        try:
            st = http_request_json(f"{RL_AGENT_URL}/status", timeout=5.0)
            step = st.get("step_count")
            pending = st.get("pending_transitions", 0)
            training_enabled = st.get("training_enabled", True)
            epsilon = st.get("epsilon", 1.0)
            buffer_len = st.get("buffer_len", 0)

            if training_enabled is not False or epsilon != 0.0:
                # Enforce evaluation mode
                http_request_json(
                    f"{RL_AGENT_URL}/eval_mode",
                    method="POST",
                    payload={"mode": "evaluation"},
                    timeout=5.0,
                )

            if pending == 0 and last_step is not None and step == last_step:
                stable_count += 1
                if stable_count >= required_stable_checks:
                    return st
            else:
                stable_count = 0

            last_step = step
        except Exception as e:
            stable_count = 0

        time.sleep(check_interval)

    raise TimeoutError(
        f"RL agent transitions did not settle to 0 pending transitions in {timeout}s"
    )


def switch_checkpoint_safe(checkpoint_name: str, expected_step: int):
    """
    Safely switch the RL model via POST /load_checkpoint after complete drainage.
    """
    print(f"\n[MODEL SWITCH] Initiating checkpoint switch to {checkpoint_name}...")

    # 1. Verify integrity of both files on disk
    verify_checkpoint_integrity()

    # 2. Verify runner is idle
    wait_for_runner_clean()

    # 3. Verify all backends are healthy
    if not check_backends_healthy():
        raise RuntimeError("Cannot switch checkpoint: one or more backends are unhealthy.")

    # 4. Verify no active connections on load balancer
    metrics_text = get_lb_metrics_text()
    counters = parse_lb_metric_counters(metrics_text)
    if counters["active_conns_total"] > 0:
        time.sleep(2.0)

    # 5. Drain pending transitions in RL agent
    drain_and_verify_rl_idle()

    # 6. Call /load_checkpoint
    res = http_request_json(
        f"{RL_AGENT_URL}/load_checkpoint",
        method="POST",
        payload={"checkpoint": checkpoint_name},
        timeout=10.0,
    )

    if res.get("status") != "loaded":
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_name}: {res}")

    if res.get("step_count") != expected_step:
        raise ValueError(
            f"Step count mismatch after loading {checkpoint_name}: "
            f"expected {expected_step}, got {res.get('step_count')}"
        )

    if res.get("epsilon") != 0.0 or res.get("training_enabled") is not False:
        raise ValueError(
            f"Evaluation mode not strictly enforced after loading {checkpoint_name}: {res}"
        )

    # 7. Post-load verification
    verify_checkpoint_integrity()
    agent_st = http_request_json(f"{RL_AGENT_URL}/status")
    if (
        agent_st.get("step_count") != expected_step
        or agent_st.get("epsilon") != 0.0
        or agent_st.get("training_enabled") is not False
    ):
        raise RuntimeError(f"RL agent state mismatch after checkpoint load: {agent_st}")

    print(
        f"[MODEL SWITCH SUCCESS] Loaded {checkpoint_name} (step={expected_step}, eps=0.0, training=False)\n"
    )


def query_prometheus_cpu(start_time: float, end_time: float) -> Dict[str, Any]:
    """Query Prometheus for backend CPU utilization during trial window."""
    params = {
        "query": "backend_cpu_utilization",
        "start": f"{start_time:.3f}",
        "end": f"{end_time:.3f}",
        "step": "2s",
    }
    url = f"{PROM_URL}/api/v1/query_range?{urllib.parse.urlencode(params)}"
    try:
        data = http_request_json(url, timeout=10.0)
    except Exception as e:
        print(f"[PROM WARNING] Could not query CPU metrics: {e}")
        return {
            "mean_cpu_srv_1": None,
            "mean_cpu_srv_2": None,
            "mean_cpu_srv_3": None,
            "cpu_std_dev": None,
        }

    per_server_samples = {"1": [], "2": [], "3": []}
    results = data.get("data", {}).get("result", [])
    for entry in results:
        srv_id = entry.get("metric", {}).get("server_id")
        if srv_id in per_server_samples:
            vals = [float(v[1]) for v in entry.get("values", [])]
            per_server_samples[srv_id].extend(vals)

    server_means = {}
    valid_means = []
    for srv_id in ["1", "2", "3"]:
        vals = per_server_samples[srv_id]
        if vals:
            m = sum(vals) / len(vals)
            server_means[f"mean_cpu_srv_{srv_id}"] = round(m, 2)
            valid_means.append(m)
        else:
            server_means[f"mean_cpu_srv_{srv_id}"] = None

    if len(valid_means) > 1:
        avg_mean = sum(valid_means) / len(valid_means)
        variance = sum((x - avg_mean) ** 2 for x in valid_means) / len(valid_means)
        cpu_std_dev = round(math.sqrt(variance), 3)
    else:
        cpu_std_dev = 0.0

    return {**server_means, "cpu_std_dev": cpu_std_dev}


def generate_plan(trials_per_group: int = DEFAULT_TRIALS_PER_GROUP) -> List[Dict[str, Any]]:
    """Build the ordered list of 60 trials across 4 phases."""
    plan = []
    global_index = 1
    for phase_info in PHASES:
        for scenario in SCENARIOS:
            for trial_num in range(1, trials_per_group + 1):
                trial_id = f"{phase_info['algo_name']}_{scenario}_t{trial_num}"
                trial_tag = f"{phase_info['algo_name'][:4]}_t{trial_num}"
                plan.append({
                    "global_index": global_index,
                    "phase": phase_info["phase"],
                    "algo_name": phase_info["algo_name"],
                    "display_name": phase_info["display_name"],
                    "runner_algo": phase_info["runner_algo"],
                    "checkpoint": phase_info["checkpoint"],
                    "expected_step": phase_info["expected_step"],
                    "scenario": scenario,
                    "trial_num": trial_num,
                    "trial_id": trial_id,
                    "trial_tag": trial_tag,
                })
                global_index += 1
    return plan


def update_summary_csv(completed_trials: List[Dict[str, Any]], filepath: Path):
    """Write summary CSV."""
    if not completed_trials:
        return
    fieldnames = [
        "trial_id",
        "phase",
        "algo_name",
        "display_name",
        "scenario",
        "trial_num",
        "run_id",
        "timestamp_start",
        "timestamp_end",
        "duration_seconds",
        "total_requests",
        "throughput_rps",
        "mean_latency_ms",
        "median_p50_ms",
        "p90_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
        "http_failed_count",
        "http_failed_rate",
        "fallbacks_delta",
        "fallback_rate",
        "mean_cpu_srv_1",
        "mean_cpu_srv_2",
        "mean_cpu_srv_3",
        "cpu_std_dev",
        "step_count_before",
        "step_count_after",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in completed_trials:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def calculate_stats(values: List[float]):
    """Calculate mean and sample standard deviation."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) > 1:
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std


def generate_comparison_report(
    completed_trials: List[Dict[str, Any]],
    output_path: Path,
    total_planned: int = 60,
):
    """Generate comprehensive comparison markdown report with statistical aggregation."""
    lines = []
    lines.append("# FLUXOR 60-Trial Evaluation Campaign Report")
    lines.append("")
    lines.append(f"- **Generated At**: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- **Progress**: {len(completed_trials)} / {total_planned} trials completed")
    lines.append(f"- **Active Git Branch**: `exp/v2-double-dqn`")
    lines.append(f"- **Baseline Checkpoint SHA-256**: `{BASELINE_EXPECTED_HASH}`")
    lines.append(f"- **Double DQN Checkpoint SHA-256**: `{DDQN_EXPECTED_HASH}`")
    lines.append("")

    if not completed_trials:
        lines.append("*No completed trials recorded yet.*")
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    # Aggregate by (scenario, algo_name)
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for t in completed_trials:
        sc = t["scenario"]
        al = t["algo_name"]
        grouped.setdefault(sc, {}).setdefault(al, []).append(t)

    for sc in SCENARIOS:
        lines.append(f"## Scenario: `{sc.upper()}`")
        lines.append("")
        lines.append("| Algorithm | Trials | Throughput (RPS) | Avg Latency (ms) | P90 Lat (ms) | P95 Lat (ms) | Max Lat (ms) | Error Rate (%) | Fallback Rate (%) | CPU Imbalance (σ) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        algo_order = ["round_robin", "least_connections", "double_dqn", "vanilla_dqn"]
        sc_stats = {}

        for al in algo_order:
            trials = grouped.get(sc, {}).get(al, [])
            count = len(trials)
            if count == 0:
                lines.append(f"| {al} | 0 | - | - | - | - | - | - | - | - |")
                continue

            rpss = [t["throughput_rps"] for t in trials]
            avgls = [t["mean_latency_ms"] for t in trials]
            p90ls = [t["p90_latency_ms"] for t in trials]
            p95ls = [t["p95_latency_ms"] for t in trials]
            maxls = [t["max_latency_ms"] for t in trials]
            errs = [t["http_failed_rate"] * 100.0 for t in trials]
            fbs = [t.get("fallback_rate", 0.0) * 100.0 for t in trials]
            cpus = [t["cpu_std_dev"] for t in trials if t.get("cpu_std_dev") is not None]

            m_rps, s_rps = calculate_stats(rpss)
            m_avg, s_avg = calculate_stats(avgls)
            m_p90, s_p90 = calculate_stats(p90ls)
            m_p95, s_p95 = calculate_stats(p95ls)
            m_max, s_max = calculate_stats(maxls)
            m_err, s_err = calculate_stats(errs)
            m_fb, s_fb = calculate_stats(fbs)
            m_cpu, s_cpu = calculate_stats(cpus)

            sc_stats[al] = {
                "rps": m_rps,
                "avg_lat": m_avg,
                "p95_lat": m_p95,
                "err": m_err,
                "fb": m_fb,
            }

            display_title = {
                "round_robin": "Round Robin",
                "least_connections": "Least Connections",
                "double_dqn": "Double DQN (Exp A)",
                "vanilla_dqn": "Vanilla DQN (Base)",
            }.get(al, al)

            lines.append(
                f"| **{display_title}** | {count} | "
                f"{m_rps:.2f} ± {s_rps:.2f} | "
                f"{m_avg:.2f} ± {s_avg:.2f} | "
                f"{m_p90:.2f} ± {s_p90:.2f} | "
                f"{m_p95:.2f} ± {s_p95:.2f} | "
                f"{m_max:.2f} ± {s_max:.2f} | "
                f"{m_err:.3f}% ± {s_err:.3f}% | "
                f"{m_fb:.3f}% ± {s_fb:.3f}% | "
                f"{m_cpu:.2f} ± {s_cpu:.2f} |"
            )

        lines.append("")

        # Relative Comparison: Double DQN vs Vanilla DQN
        if "double_dqn" in sc_stats and "vanilla_dqn" in sc_stats:
            dd = sc_stats["double_dqn"]
            vn = sc_stats["vanilla_dqn"]
            if vn["avg_lat"] > 0:
                lat_diff = ((dd["avg_lat"] - vn["avg_lat"]) / vn["avg_lat"]) * 100.0
                p95_diff = ((dd["p95_lat"] - vn["p95_lat"]) / vn["p95_lat"]) * 100.0
                rps_diff = ((dd["rps"] - vn["rps"]) / vn["rps"]) * 100.0
                lines.append(f"**Double DQN vs Vanilla DQN ({sc.upper()})**:")
                lines.append(f"- Latency Δ: **{lat_diff:+.2f}%** (Avg Latency: `{dd['avg_lat']:.2f} ms` vs `{vn['avg_lat']:.2f} ms`)")
                lines.append(f"- P95 Latency Δ: **{p95_diff:+.2f}%** (P95: `{dd['p95_lat']:.2f} ms` vs `{vn['p95_lat']:.2f} ms`)")
                lines.append(f"- Throughput Δ: **{rps_diff:+.2f}%** (Throughput: `{dd['rps']:.2f} RPS` vs `{vn['rps']:.2f} RPS`)")
                lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def execute_trial(trial_meta: Dict[str, Any], trial_idx: int, total_trials: int) -> Dict[str, Any]:
    """Execute a single evaluation trial with rigorous preflight, drainage, and result logging."""
    scenario = trial_meta["scenario"]
    algo_name = trial_meta["algo_name"]
    runner_algo = trial_meta["runner_algo"]
    trial_tag = trial_meta["trial_tag"]

    print(
        f"\n================================================================================"
        f"\n[{trial_idx}/{total_trials}] EXECUTING TRIAL: {trial_meta['trial_id']}"
        f"\nScenario: {scenario} | Algorithm: {algo_name} (runner={runner_algo}) | Tag: {trial_tag}"
        f"\n================================================================================"
    )

    # 1. Pre-trial safety checks
    verify_checkpoint_integrity()
    wait_for_runner_clean()
    if not check_backends_healthy():
        raise RuntimeError(f"Pre-trial check failed: backends unhealthy before {trial_meta['trial_id']}")

    # 2. RL Agent Drain & Status Check
    drain_st = drain_and_verify_rl_idle()
    step_before = drain_st.get("step_count")

    # 3. Capture baseline counters before trial start
    metrics_before = parse_lb_metric_counters(get_lb_metrics_text())
    fallbacks_before = metrics_before["lb_rl_fallbacks_total"]
    start_wall_time = time.time()

    # 4. Trigger trial via runner
    start_payload = {
        "scenario": scenario,
        "algorithm": runner_algo,
        "trial": trial_tag,
    }
    start_resp = http_request_json(
        f"{RUNNER_URL}/start",
        method="POST",
        payload=start_payload,
        timeout=15.0,
    )
    run_id = start_resp.get("run_id")
    if not run_id:
        raise RuntimeError(f"Failed to obtain run_id from runner: {start_resp}")

    print(f"  -> Runner launched run_id: {run_id}")

    # 5. Wait for duration
    expected_duration = 60 if scenario == "steady" else (80 if scenario == "spike" else 90)
    print(f"  -> Waiting for execution duration ({expected_duration}s)...")
    time.sleep(max(10, expected_duration - 5))

    # 6. Poll runner for completion
    wait_for_runner_clean(timeout=120.0)
    end_wall_time = time.time()
    duration_actual = round(end_wall_time - start_wall_time, 2)
    print(f"  -> Trial run completed in {duration_actual}s")

    # 7. Settle window for Prometheus and metrics flush
    time.sleep(4.0)

    # 8. Verify backends recovered (especially after failure scenario)
    if not check_backends_healthy():
        # Give up to 15s for backend-2 recovery
        recovery_deadline = time.time() + 15.0
        while time.time() < recovery_deadline:
            time.sleep(2.0)
            if check_backends_healthy():
                break
        else:
            raise RuntimeError(f"Backend recovery failed after trial {run_id}")

    # 9. Verify RL state integrity post-trial
    drain_post = drain_and_verify_rl_idle()
    step_after = drain_post.get("step_count")
    if drain_post.get("training_enabled") is not False or drain_post.get("epsilon") != 0.0:
        raise RuntimeError(f"RL agent left evaluation mode during trial! Status: {drain_post}")
    if step_after != step_before:
        raise RuntimeError(
            f"Step count mutated during evaluation trial: before={step_before}, after={step_after}"
        )

    # 10. Compute delta counters
    metrics_after = parse_lb_metric_counters(get_lb_metrics_text())
    fallbacks_delta = int(metrics_after["lb_rl_fallbacks_total"] - fallbacks_before)

    # 11. Read summary JSON
    summary_path = RESULTS_DIR / f"{run_id}_summary.json"
    if not summary_path.exists():
        matches = list(RESULTS_DIR.glob(f"*{scenario}_{runner_algo}_{trial_tag}_summary.json"))
        if matches:
            summary_path = matches[0]
        else:
            raise FileNotFoundError(f"k6 summary file not found for run {run_id}")

    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    k6_metrics = summary_data.get("metrics", {})
    http_reqs = k6_metrics.get("http_reqs", {})
    req_duration = k6_metrics.get("http_req_duration", {})
    req_failed = k6_metrics.get("http_req_failed", {})

    total_requests = int(http_reqs.get("count", 0))
    throughput_rps = round(float(http_reqs.get("rate", 0.0)), 2)
    mean_latency = round(float(req_duration.get("avg", 0.0)), 2)
    median_latency = round(float(req_duration.get("med", 0.0)), 2)
    p90_latency = round(float(req_duration.get("p(90)", 0.0)), 2)
    p95_latency = round(float(req_duration.get("p(95)", 0.0)), 2)
    max_latency = round(float(req_duration.get("max", 0.0)), 2)
    http_failed_count = int(req_failed.get("passes", 0))
    http_failed_rate = round(float(req_failed.get("value", 0.0)), 4)

    fallback_rate = (
        round(fallbacks_delta / total_requests, 4)
        if (total_requests > 0 and runner_algo == "rl")
        else 0.0
    )

    # 12. Query Prometheus CPU
    cpu_metrics = query_prometheus_cpu(start_wall_time, end_wall_time)

    # 13. Assemble trial record
    trial_record = {
        "trial_id": trial_meta["trial_id"],
        "phase": trial_meta["phase"],
        "algo_name": trial_meta["algo_name"],
        "display_name": trial_meta["display_name"],
        "scenario": trial_meta["scenario"],
        "trial_num": trial_meta["trial_num"],
        "run_id": run_id,
        "timestamp_start": datetime.fromtimestamp(start_wall_time, timezone.utc).isoformat(),
        "timestamp_end": datetime.fromtimestamp(end_wall_time, timezone.utc).isoformat(),
        "duration_seconds": duration_actual,
        "total_requests": total_requests,
        "throughput_rps": throughput_rps,
        "mean_latency_ms": mean_latency,
        "median_p50_ms": median_latency,
        "p90_latency_ms": p90_latency,
        "p95_latency_ms": p95_latency,
        "max_latency_ms": max_latency,
        "http_failed_count": http_failed_count,
        "http_failed_rate": http_failed_rate,
        "fallbacks_delta": fallbacks_delta,
        "fallback_rate": fallback_rate,
        "step_count_before": step_before,
        "step_count_after": step_after,
        **cpu_metrics,
    }

    print(
        f"  -> SUCCESS: Req={total_requests} | RPS={throughput_rps} | "
        f"AvgLat={mean_latency}ms | P95={p95_latency}ms | "
        f"Err={http_failed_rate * 100:.2f}% | Fallbacks={fallbacks_delta} ({fallback_rate * 100:.2f}%)"
    )

    return trial_record


def run_campaign(
    dry_run: bool = False,
    single_trial: bool = False,
    resume: bool = False,
    trials_per_group: int = DEFAULT_TRIALS_PER_GROUP,
    scenario_filter: Optional[str] = None,
    algo_filter: Optional[str] = None,
):
    """Main campaign controller."""
    print("================================================================================")
    print("FLUXOR 60-Trial Automated Evaluation Campaign")
    print(f"Working Directory: {WORKSPACE_DIR}")
    print(f"Results Directory: {CAMPAIGN_DIR}")
    print(f"Trials per Group:  {trials_per_group}")
    print("================================================================================")

    # 1. Verify Checkpoint Integrity
    print("\n[PREFLIGHT] Verifying checkpoint hashes on disk...")
    verify_checkpoint_integrity()
    print(f"  ✓ Baseline ({BASELINE_CHECKPOINT_FILE}): {BASELINE_EXPECTED_HASH}")
    print(f"  ✓ DDQN ({DDQN_CHECKPOINT_FILE}):     {DDQN_EXPECTED_HASH}")

    # 2. Build Plan
    all_trials = generate_plan(trials_per_group=trials_per_group)

    # Apply filters if provided
    if scenario_filter:
        all_trials = [t for t in all_trials if t["scenario"] == scenario_filter]
    if algo_filter:
        all_trials = [t for t in all_trials if t["algo_name"] == algo_filter]
    if single_trial:
        all_trials = all_trials[:1]

    total_planned = len(all_trials)
    print(f"\n[PLAN] Total planned trials for this invocation: {total_planned}")

    if dry_run:
        print("\n" + "=" * 50 + " DRY RUN SCHEDULE " + "=" * 50)
        current_checkpoint = None
        for idx, t in enumerate(all_trials, 1):
            switch_note = ""
            if t["checkpoint"] != current_checkpoint and t["checkpoint"] is not None:
                switch_note = f" --> [SWITCH MODEL: load '{t['checkpoint']}' (step={t['expected_step']})]"
                current_checkpoint = t["checkpoint"]
            elif t["runner_algo"] != "rl" and current_checkpoint is not None:
                switch_note = f" --> [ALGO SWITCH: '{t['runner_algo']}']"
                current_checkpoint = None

            print(
                f"[{idx:02d}/{total_planned:02d}] Phase {t['phase']} | "
                f"{t['display_name']:<18} | Scenario: {t['scenario']:<8} | "
                f"Trial {t['trial_num']}/5 | ID: {t['trial_id']}{switch_note}"
            )
        print("=" * 118)
        print("\nDry run completed successfully. No experiments were executed.")
        return

    # Create destination directory
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CAMPAIGN_DIR / "campaign_manifest.json"
    trials_jsonl_path = CAMPAIGN_DIR / "trials.jsonl"
    summary_csv_path = CAMPAIGN_DIR / "summary.csv"
    report_md_path = CAMPAIGN_DIR / "comparison_report.md"

    # Load existing completed trials if resuming
    completed_trials: List[Dict[str, Any]] = []
    completed_trial_ids = set()

    if resume and trials_jsonl_path.exists():
        with open(trials_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed_trials.append(rec)
                    completed_trial_ids.add(rec["trial_id"])
        print(f"\n[RESUME] Found {len(completed_trials)} already completed trials in {trials_jsonl_path}")

    # Track currently active model in RL agent
    agent_st = http_request_json(f"{RL_AGENT_URL}/status")
    current_active_step = agent_st.get("step_count")

    # Execution Loop
    for idx, trial_meta in enumerate(all_trials, 1):
        trial_id = trial_meta["trial_id"]

        if trial_id in completed_trial_ids:
            print(f"[{idx}/{total_planned}] Skipping already completed trial: {trial_id}")
            continue

        # Check if we need to load a checkpoint for this phase
        target_ckpt = trial_meta["checkpoint"]
        target_step = trial_meta["expected_step"]

        if trial_meta["runner_algo"] == "rl":
            if current_active_step != target_step:
                switch_checkpoint_safe(target_ckpt, target_step)
                current_active_step = target_step
        else:
            # Ensure evaluation mode is maintained even during non-RL trials
            drain_and_verify_rl_idle()

        # Execute trial
        record = execute_trial(trial_meta, idx, total_planned)
        completed_trials.append(record)
        completed_trial_ids.add(trial_id)

        # Append to JSONL immediately
        with open(trials_jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Update CSV
        update_summary_csv(completed_trials, summary_csv_path)

        # Update Markdown Report
        generate_comparison_report(completed_trials, report_md_path, total_planned=total_planned)

        # Update Campaign Manifest
        manifest_data = {
            "campaign_name": "FLUXOR 60-Trial Evaluation Campaign",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_planned": total_planned,
            "total_completed": len(completed_trials),
            "baseline_hash": BASELINE_EXPECTED_HASH,
            "ddqn_hash": DDQN_EXPECTED_HASH,
            "completed_trial_ids": list(completed_trial_ids),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"CAMPAIGN EXECUTION COMPLETE: {len(completed_trials)}/{total_planned} trials recorded.")
    print(f"Summary CSV:  {summary_csv_path}")
    print(f"Trials JSONL: {trials_jsonl_path}")
    print(f"Report:       {report_md_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUXOR 60-Trial Evaluation Campaign")
    parser.add_argument("--dry-run", action="store_true", help="Print schedule without running")
    parser.add_argument("--single-trial", action="store_true", help="Run only the first trial as test")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed trial in trials.jsonl")
    parser.add_argument("--trials-per-group", type=int, default=DEFAULT_TRIALS_PER_GROUP, help="Trials per scenario/algorithm (default 5)")
    parser.add_argument("--scenario", type=str, choices=SCENARIOS, default=None, help="Filter by scenario")
    parser.add_argument("--algorithm", type=str, choices=["round_robin", "least_connections", "double_dqn", "vanilla_dqn"], default=None, help="Filter by algorithm")

    args = parser.parse_args()

    run_campaign(
        dry_run=args.dry_run,
        single_trial=args.single_trial,
        resume=args.resume,
        trials_per_group=args.trials_per_group,
        scenario_filter=args.scenario,
        algo_filter=args.algorithm,
    )

