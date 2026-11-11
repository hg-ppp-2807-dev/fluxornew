import csv
import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(".")
RESULTS = ROOT / "results"
BASELINE = RESULTS / "historical_60_trial" / "summary.csv"
OUT = RESULTS / "independent_audit"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ("steady", "spike", "failure")


def number(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def nested(obj, *keys, default=None):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def find_status_check_fails(obj):
    """Find the k6 check named 'status is 200', including nested group layouts."""
    if isinstance(obj, dict):
        checks = obj.get("checks")
        if isinstance(checks, dict):
            for name, value in checks.items():
                if name == "status is 200" and isinstance(value, dict):
                    return number(value.get("fails"), 0)
        for value in obj.values():
            found = find_status_check_fails(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_status_check_fails(value)
            if found is not None:
                return found
    return None


def parse_prometheus(path):
    """Return {(metric_name, sorted_labels): value} from a Prometheus text snapshot."""
    result = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing Prometheus snapshot: {path}")

    pattern = re.compile(
        r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+'
        r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]Inf)'
    )
    label_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if not match:
                continue

            metric, raw_labels, raw_value = match.groups()
            labels = {}
            if raw_labels:
                for key, value in label_pattern.findall(raw_labels):
                    labels[key] = value.replace(r'\"', '"').replace(r'\\', '\\')

            result[(metric, tuple(sorted(labels.items())))] = number(raw_value)

    return result


def prom_sum(snapshot, metric, label_filters=None):
    label_filters = label_filters or {}
    values = []
    for (name, labels), value in snapshot.items():
        if name != metric:
            continue
        labels = dict(labels)
        if all(labels.get(k) == v for k, v in label_filters.items()):
            values.append(value)
    return sum(values) if values else 0.0


def prom_delta(before, after, metric, label_filters=None):
    a = prom_sum(before, metric, label_filters)
    b = prom_sum(after, metric, label_filters)
    # Counters should be monotonic during a trial. A decrease indicates reset.
    return (b - a) if b >= a else b


def snapshot_pair(scenario, trial, suffix):
    stem = f"{scenario}_{trial:02d}"
    before = list((RESULTS / "latest_gunicorn_evaluation" / "telemetry").glob(f"{stem}_before_{suffix}.prom"))
    after = list((RESULTS / "latest_gunicorn_evaluation" / "telemetry").glob(f"{stem}_after_{suffix}.prom"))
    if len(before) != 1 or len(after) != 1:
        raise FileNotFoundError(
            f"Expected one before/after {suffix} snapshot for {stem}; "
            f"found before={len(before)}, after={len(after)}"
        )
    return parse_prometheus(before[0]), parse_prometheus(after[0])


def find_summary_files():
    found = {}
    pattern = re.compile(r"_(steady|spike|failure)_(\d+)_summary\.json$")
    for path in (RESULTS / "latest_gunicorn_evaluation" / "summaries").glob("*_summary.json"):
        match = pattern.search(path.name)
        if match:
            scenario, trial = match.group(1), int(match.group(2))
            key = (scenario, trial)
            if key in found:
                raise RuntimeError(f"Duplicate summary files for {key}: {found[key]} and {path}")
            found[key] = path
    return found


def read_gunicorn_trial(scenario, trial, path):
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})

    reqs = nested(metrics, "http_reqs", "count")
    rate = nested(metrics, "http_reqs", "rate")
    duration = nested(metrics, "http_req_duration", default={})
    failed_rate = nested(metrics, "http_req_failed", "value")
    status_fails = find_status_check_fails(data)

    reqs = number(reqs, 0)
    rate = number(rate)
    avg = number(nested(duration, "avg"))
    p50 = number(nested(duration, "med"))
    p90 = number(nested(duration, "p(90)"))
    p95 = number(nested(duration, "p(95)"))
    max_latency = number(nested(duration, "max"))
    failed_rate = number(failed_rate)

    # Derive active test duration from k6's request count/rate.
    active_seconds = reqs / rate if rate and rate > 0 else float("nan")

    lb_before, lb_after = snapshot_pair(scenario, trial, "lb")
    agent_before, agent_after = snapshot_pair(scenario, trial, "agent")

    routed_rl = prom_delta(
        lb_before, lb_after, "lb_requests_total", {"algorithm": "rl"}
    )
    routed_fallback = prom_delta(
        lb_before, lb_after, "lb_requests_total", {"algorithm": "rr_fallback"}
    )
    routed_total = routed_rl + routed_fallback

    fallback_counter = prom_delta(
        lb_before, lb_after, "lb_rl_fallbacks_total"
    )
    timeout_counter = prom_delta(
        lb_before, lb_after, "lb_rl_timeouts_total"
    )
    decisions = prom_delta(
        agent_before, agent_after, "rl_decisions_total"
    )

    go_count = prom_delta(
        lb_before, lb_after, "lb_rl_decision_latency_ms_count"
    )
    go_sum = prom_delta(
        lb_before, lb_after, "lb_rl_decision_latency_ms_sum"
    )
    go_mean_ms = go_sum / go_count if go_count > 0 else float("nan")

    py_count = prom_delta(agent_before, agent_after, "rl_decide_total_seconds_count")
    py_sum = prom_delta(agent_before, agent_after, "rl_decide_total_seconds_sum")
    py_mean_ms = (py_sum / py_count * 1000) if py_count > 0 else float("nan")

    inf_count = prom_delta(agent_before, agent_after, "rl_inference_seconds_count")
    inf_sum = prom_delta(agent_before, agent_after, "rl_inference_seconds_sum")
    inf_mean_ms = (inf_sum / inf_count * 1000) if inf_count > 0 else float("nan")

    return {
        "scenario": scenario,
        "trial": trial,
        "summary_file": path.name,
        "requests_k6": reqs,
        "throughput_k6_rps": rate,
        "active_seconds_derived": active_seconds,
        "mean_latency_ms": avg,
        "p50_latency_ms": p50,
        "p90_latency_ms": p90,
        "p95_latency_ms": p95,
        "max_latency_ms": max_latency,
        "http_failed_rate_k6": failed_rate,
        "http_status_check_fails": status_fails,
        "prom_rl_routed_delta": routed_rl,
        "prom_rr_fallback_routed_delta": routed_fallback,
        "prom_total_routed_delta": routed_total,
        "prom_fallback_counter_delta": fallback_counter,
        "prom_timeout_counter_delta": timeout_counter,
        "prom_rl_decisions_delta": decisions,
        "go_decision_mean_ms": go_mean_ms,
        "python_handler_mean_ms": py_mean_ms,
        "python_inference_mean_ms": inf_mean_ms,
        "request_counter_matches": math.isclose(reqs, routed_total, abs_tol=1),
        "fallback_counters_match": math.isclose(
            fallback_counter, routed_fallback, abs_tol=1
        ),
    }


def mean_sd(series):
    values = [number(x) for x in series]
    values = [x for x in values if math.isfinite(x)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def summarize_group(df, label):
    reqs = df["requests"].sum()
    fallbacks = df["fallbacks"].sum()
    duration = df["active_seconds"].sum()

    fields = [
        "throughput_rps", "mean_latency_ms", "p95_latency_ms",
        "trial_fallback_pct", "go_decision_mean_ms",
        "python_handler_mean_ms", "python_inference_mean_ms",
    ]
    row = {"dataset": label, "trials": len(df), "requests": int(reqs)}
    for field in fields:
        values = df[field] if field in df.columns else pd.Series(float("nan"), index=df.index)
        mean, sd = mean_sd(values)
        row[f"{field}_mean"] = mean
        row[f"{field}_sd"] = sd

    row["pooled_throughput_rps"] = reqs / duration if duration > 0 else float("nan")
    row["total_fallbacks"] = int(fallbacks)
    row["pooled_fallback_pct"] = fallbacks / reqs * 100 if reqs else float("nan")
    row["http_errors_total"] = int(df["http_errors"].sum())
    return row


# ---- Load and validate baseline DDQN rows ----
baseline = pd.read_csv(BASELINE)
baseline = baseline[baseline["algo_name"] == "double_dqn"].copy()

if len(baseline) != 15:
    raise RuntimeError(f"Expected 15 baseline DDQN rows; found {len(baseline)}")

baseline_rows = []
for _, r in baseline.iterrows():
    baseline_rows.append({
        "scenario": r["scenario"],
        "trial": int(r["trial_num"]),
        "requests": number(r["total_requests"], 0),
        "throughput_rps": number(r["throughput_rps"]),
        "mean_latency_ms": number(r["mean_latency_ms"]),
        "p95_latency_ms": number(r["p95_latency_ms"]),
        "fallbacks": number(r["fallbacks_delta"], 0),
        "trial_fallback_pct": number(r["fallback_rate"]) * 100,
        "http_errors": number(r["http_failed_count"], 0),
        "active_seconds": (
            number(r["total_requests"], 0) / number(r["throughput_rps"])
            if number(r["throughput_rps"]) > 0 else float("nan")
        ),
    })
baseline_df = pd.DataFrame(baseline_rows)

# ---- Load and validate Gunicorn DDQN trials ----
files = find_summary_files()
expected = {(s, t) for s in SCENARIOS for t in range(1, 6)}
if set(files) != expected:
    missing = sorted(expected - set(files))
    extra = sorted(set(files) - expected)
    raise RuntimeError(f"Gunicorn summary inventory mismatch. Missing={missing}, extra={extra}")

gunicorn_rows = [
    read_gunicorn_trial(s, t, files[(s, t)])
    for s in SCENARIOS
    for t in range(1, 6)
]
gunicorn_raw = pd.DataFrame(gunicorn_rows)

gunicorn_rows_for_summary = []
for _, r in gunicorn_raw.iterrows():
    requests = number(r["requests_k6"], 0)
    rate = number(r["throughput_k6_rps"])
    gunicorn_rows_for_summary.append({
        "scenario": r["scenario"],
        "trial": int(r["trial"]),
        "requests": requests,
        "throughput_rps": rate,
        "mean_latency_ms": r["mean_latency_ms"],
        "p95_latency_ms": r["p95_latency_ms"],
        "fallbacks": r["prom_fallback_counter_delta"],
        "trial_fallback_pct": (
            r["prom_fallback_counter_delta"] / requests * 100
            if requests else float("nan")
        ),
        "http_errors": (
            r["http_status_check_fails"]
            if r["http_status_check_fails"] is not None else 0
        ),
        "active_seconds": r["active_seconds_derived"],
        "go_decision_mean_ms": r["go_decision_mean_ms"],
        "python_handler_mean_ms": r["python_handler_mean_ms"],
        "python_inference_mean_ms": r["python_inference_mean_ms"],
    })
gunicorn_df = pd.DataFrame(gunicorn_rows_for_summary)

# ---- Trial-level comparison ----
baseline_df["dataset"] = "Flask baseline"
gunicorn_df["dataset"] = "Gunicorn"
trial_comparison = pd.concat([baseline_df, gunicorn_df], ignore_index=True)
trial_comparison.to_csv(OUT / "independent_trial_results.csv", index=False)

# ---- Scenario summaries ----
summary_rows = []
for scenario in SCENARIOS:
    b = baseline_df[baseline_df["scenario"] == scenario]
    g = gunicorn_df[gunicorn_df["scenario"] == scenario]
    summary_rows.append(summarize_group(b, f"Flask baseline / {scenario}"))
    summary_rows.append(summarize_group(g, f"Gunicorn / {scenario}"))

scenario_summary = pd.DataFrame(summary_rows)
scenario_summary.to_csv(OUT / "independent_scenario_comparison.csv", index=False)

# ---- Audit report ----
lines = [
    "FLUXOR INDEPENDENT NUMERICAL AUDIT",
    "=" * 42,
    f"Baseline DDQN trials: {len(baseline_df)}",
    f"Gunicorn DDQN trials: {len(gunicorn_raw)}",
    "",
    "PROMETHEUS / K6 CROSS-CHECKS",
    "-" * 42,
    f"Request counter matches all 15 trials: {gunicorn_raw['request_counter_matches'].all()}",
    f"Fallback counter matches all 15 trials: {gunicorn_raw['fallback_counters_match'].all()}",
    f"Trials with request-counter mismatch: {int((~gunicorn_raw['request_counter_matches']).sum())}",
    f"Trials with fallback-counter mismatch: {int((~gunicorn_raw['fallback_counters_match']).sum())}",
    "",
    "Gunicorn trial-level checks:",
]
for _, r in gunicorn_raw.iterrows():
    lines.append(
        f"{r['scenario']} T{int(r['trial'])}: "
        f"k6 requests={int(r['requests_k6'])}, "
        f"Prom routed={int(r['prom_total_routed_delta'])}, "
        f"fallback counter={int(r['prom_fallback_counter_delta'])}, "
        f"fallback-routed={int(r['prom_rr_fallback_routed_delta'])}, "
        f"HTTP status-check failures={r['http_status_check_fails']}, "
        f"k6 failure rate={r['http_failed_rate_k6']}"
    )

lines += ["", "SCENARIO SUMMARY", "-" * 42]
for _, r in scenario_summary.iterrows():
    lines.append(
        f"{r['dataset']}: n={int(r['trials'])}, "
        f"requests={int(r['requests'])}, "
        f"mean throughput={r['throughput_rps_mean']:.2f}±{r['throughput_rps_sd']:.2f} RPS, "
        f"pooled throughput={r['pooled_throughput_rps']:.2f} RPS, "
        f"mean latency={r['mean_latency_ms_mean']:.2f}±{r['mean_latency_ms_sd']:.2f} ms, "
        f"mean trial P95={r['p95_latency_ms_mean']:.2f}±{r['p95_latency_ms_sd']:.2f} ms, "
        f"mean trial fallback={r['trial_fallback_pct_mean']:.3f}%, "
        f"pooled fallback={r['pooled_fallback_pct']:.3f}%, "
        f"HTTP errors={int(r['http_errors_total'])}"
    )

lines += [
    "",
    "INTERPRETATION NOTES",
    "-" * 42,
    "- Latency and P95 summaries are means and sample SDs across five trial summaries.",
    "- They are not pooled request-level latency percentiles.",
    "- Trial fallback mean is unweighted; pooled fallback is total fallbacks / total requests.",
    "- Pooled throughput is total requests / sum of per-trial active durations inferred from k6 count/rate.",
    "- HTTP errors use the explicit 'status is 200' check failure count.",
    "- A mismatch in Prometheus counters requires investigation before reporting the affected metric.",
    "- n=5 per scenario; do not claim statistical significance from these descriptive summaries.",
]

(OUT / "independent_audit_report.txt").write_text("\n".join(lines) + "\n")

print("\n".join(lines))
print("\nCreated:")
for path in sorted(OUT.iterdir()):
    print(" -", path)

