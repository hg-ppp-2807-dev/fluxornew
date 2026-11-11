# FLUXOR Technical Audit — Repository Context

**Audit date:** 2026-11-11
**Review branch:** `review/fluxor-technical-audit-2026-11-11`
**Starting commit:** `5522796`

## 1. Project Overview

FLUXOR is an adaptive reinforcement-learning-based load balancer for a distributed system with three backend servers.

Core technologies:

- Go load balancer
- Python PyTorch DQN/DDQN agent
- Node.js backend servers
- Docker Compose
- Prometheus and Grafana
- k6 workload generation
- React dashboard

The published FLUXOR research paper is the source of truth for the original methodology. Its projected outcomes must not be presented as measured experimental results.

The implementation includes subsequent extensions, including Double DQN experimentation and Gunicorn-based RL-agent serving. These extensions must be distinguished from the original paper methodology.

## 2. Research Documents

See `docs/research/` for:

- Published FLUXOR research paper
- Phase 2 implementation and results review
- Benchmark summary

Use these documents to understand the original research objectives, implementation decisions, and reported experimental context.

## 3. Historical Algorithm Benchmark — November 10

**Location:** `results/historical_60_trial/`

Contains:

- `campaign_manifest.json`
- `trials.jsonl`
- `summary.csv`
- `comparison_report.md`

This campaign contains 60 trials:

- 4 algorithms: Round Robin (RR), Least Connections (LC), vanilla DQN, and Double DQN (DDQN).
- 3 scenarios: steady traffic, traffic spike, and partial backend failure.
- 5 trials per algorithm/scenario combination.

This is the historical four-algorithm routing benchmark. It must remain distinct from the subsequent Gunicorn deployment evaluation.

The benchmark has a limited sample size of five trials per group. Do not claim statistical significance without an appropriate statistical analysis.

## 4. Latest Gunicorn DDQN Deployment Evaluation — November 11

**Location:** `results/latest_gunicorn_evaluation/`

Directory structure:

- `summaries/`: 15 k6 trial summary JSON files.
- `telemetry/`: 60 Prometheus `.prom` snapshots and 30 timestamp `.txt` markers.

Campaign composition:

- 5 steady-traffic trials
- 5 traffic-spike trials
- 5 partial-failure trials

The campaign evaluates the DDQN deployment using Gunicorn. It is a serving/deployment comparison against the earlier Flask DDQN baseline, not a new comparison of four routing algorithms.

### 4.1 Independently Audited Results

The independent audit reports the following scenario-level results:

| Scenario | Deployment | Trials | Requests | Throughput (RPS) | Mean latency (ms) | Mean trial P95 (ms) | Pooled fallback |
|---|---|---:|---:|---:|---:|---:|---:|
| Steady | Flask baseline | 5 | 105,306 | 350.15 ± 3.49 | 41.69 ± 1.43 | 55.39 ± 1.85 | 0.082% |
| Steady | Gunicorn | 5 | 105,380 | 350.41 ± 3.90 | 41.55 ± 1.56 | 55.80 ± 1.59 | 0.131% |
| Spike | Flask baseline | 5 | 301,870 | 753.41 ± 11.95 | 85.78 ± 2.95 | 119.26 ± 7.43 | 37.567% |
| Spike | Gunicorn | 5 | 324,198 | 809.10 ± 19.90 | 72.99 ± 4.42 | 100.25 ± 12.06 | 8.158% |
| Failure | Flask baseline | 5 | 155,587 | 345.18 ± 2.07 | 43.79 ± 0.86 | 57.59 ± 0.34 | 0.093% |
| Failure | Gunicorn | 5 | 157,089 | 348.50 ± 2.44 | 42.39 ± 1.01 | 56.61 ± 0.81 | 0.083% |

Values after ± represent the standard deviation across five trial-level measurements. Mean trial P95 is the mean of the five individual trial P95 values, not a pooled request-level P95.

The audit also reports mean trial fallback rates separately from pooled fallback rates. These are different aggregation methods and must not be interchanged.

### 4.2 Request and Fallback Counter Verification

The independent audit cross-checks the 15 Gunicorn k6 summaries against Prometheus telemetry:

- Total k6 requests: 586,667.
- Request-counter mismatches: 0.
- Fallback-counter mismatches: 0.
- Total recorded fallbacks: 26,716.
- Pooled fallback rate: approximately 4.55%.
- Total HTTP status-check failures: 1.

The single HTTP status-check failure occurred in failure scenario Trial 4. Its k6 failure rate was approximately 0.00316%.

These checks establish agreement between the recorded k6 request/fallback counts and the corresponding Prometheus counters. They do not independently establish the correctness of every application metric or the absence of unmeasured system-level effects.

### 4.3 Important Experimental Observations

- Steady traffic: Gunicorn and Flask have broadly similar throughput and latency; Gunicorn has a slightly higher pooled fallback rate.
- Traffic spike: Gunicorn shows higher throughput, lower mean latency, lower mean trial P95, and substantially lower pooled fallback than the Flask baseline.
- Partial failure: Gunicorn shows modestly higher throughput and lower latency, with one HTTP error compared with two in the Flask baseline.

The spike campaign includes a trial with a 32% fallback rate. Its cause has not been conclusively established. Do not present a suspected cause as a verified finding.

The evaluation also recorded persistent avoidance of backend 3 in direct DDQN decisions. This behavior requires further investigation of state mapping, action mapping, Q-values, health masking, and telemetry before drawing conclusions about policy quality.

## 5. Independent Numerical Audit

**Location:** `results/independent_audit/`
**Script:** `scripts/independent_audit.py`

Generated outputs:

- `independent_audit_report.txt`
- `independent_scenario_comparison.csv`
- `independent_trial_results.csv`

The audit consolidates:

1. The 15 Flask DDQN baseline trials extracted from the historical 60-trial campaign.
2. The 15 Gunicorn DDQN deployment trials.
3. The 60 Prometheus snapshots.
4. The 30 timestamp markers.
5. Request-counter and fallback-counter consistency checks.
6. Trial-level, pooled, and profiling metrics where instrumentation is available.

The script writes generated outputs under `results/independent_audit/`. It is not read-only.

### 5.1 Reproducibility Verification

After reorganizing the results directories, the audit script input paths were updated and the audit was rerun successfully.

All three regenerated outputs were bit-for-bit identical to the pre-reorganization files.

Verified SHA-256 hashes:

| Output | SHA-256 |
|---|---|
| `independent_audit_report.txt` | `15214e515e7299fec1fe82ad3d1e76843af45dac205a80da0e423aeffa29388a` |
| `independent_scenario_comparison.csv` | `3099ce308e55837df3482d534a91738f1122cd6af1a9dd49d403c3d63665ea56` |
| `independent_trial_results.csv` | `07b955fb6ed090b0a579338322e9ee17b94c52e158739e565171243b7d1a56f0` |

This verifies that the directory reorganization and path changes did not alter the generated audit outputs.

## 6. Checkpoints

Verified DDQN checkpoint:

`rl-agent/checkpoints/ddqn_experiment.pt`

SHA-256:

`81994ac6ba4f93edf46be3e41efc312a45e635b10879a43fd83b57d6ede26228`

Verified vanilla DQN baseline:

`rl-agent/checkpoints/dqn_baseline_step180700_20261107.pt`

SHA-256:

`87dbfd996c98a9d8224512398165f74c32349d245672a0fa1edf6d2815b6501b`

The local `dqn_latest.pt` has the same SHA-256 as the named vanilla DQN baseline. It is intentionally excluded from this review package.

## 7. Current Implementation Changes

At review preparation, the working tree contains:

- Modified `rl-agent/Dockerfile`
- Modified `rl-agent/app.py`
- New `rl-agent/gunicorn.conf.py`
- New `scripts/independent_audit.py`

The Gunicorn configuration uses one worker with eight threads. The single-worker choice is relevant because the agent maintains in-memory state and pending transition queues.

The Dockerfile uses the local base image:

`fluxornew-rl-agent:flask-baseline`

That image is not included in the repository package. Verify its origin and reproducibility before relying on it for a clean build.

The Gunicorn serving changes are deployment optimizations. They must not be described as changes to the DDQN routing policy unless supported by code and experimental evidence.

## 8. Audit Questions and Known Investigation Areas

The reviewer should inspect the exact local branch and current code rather than assume that remote branches contain the latest implementation.

Areas requiring technical verification include:

- DDQN policy and target-network implementation.
- State-vector construction and consistency with the paper.
- Action-index-to-backend mapping.
- Health masking and fallback behavior.
- Persistent backend avoidance in DDQN decisions.
- Spike Trial 2 fallback outlier.
- Backend CPU telemetry validity and measurement methodology.
- Latency measurement windows and whether metrics represent recent or lifetime averages.
- Gunicorn worker/thread behavior and shared in-memory state.
- Reproducibility of the local Docker base image.
- Alignment between the experimental scripts, raw evidence, and reported results.

Separate confirmed defects from hypotheses. A suspicious metric or unusual behavior is an investigation item until verified.

## 9. Preservation and Interpretation Rules

- Preserve all existing experimental branches and checkpoints.
- Do not overwrite or delete historical results.
- Keep the historical 60-trial algorithm comparison separate from the latest 15-trial Gunicorn deployment evaluation.
- Do not tune metrics to match paper projections.
- Do not claim statistical significance from five trials per group.
- Distinguish mean trial metrics from pooled metrics.
- Distinguish the original paper's DQN methodology from later DDQN and Gunicorn extensions.
- Verify the exact local branch and code before making conclusions.
- Treat the published paper as the source of truth for the original methodology.
- Treat the independent audit outputs and underlying trial evidence as the source of truth for the November 11 deployment comparison.
- Identify unsupported conclusions, measurement limitations, and reproducibility risks explicitly.

## 10. Review Objective

Conduct a rigorous technical audit of the FLUXOR repository, research methodology, implementation, checkpoints, benchmark orchestration, experimental evidence, and reported conclusions.

The goal is to establish:

1. Whether the implementation faithfully represents the stated research methodology.
2. Whether the DDQN extension is correctly implemented.
3. Whether the benchmark results are reproducible and internally consistent.
4. Whether the Gunicorn deployment changes improve serving behavior under the tested workloads.
5. Which results are defensible, which require qualification, and which require additional experiments.
6. What technical corrections or additional evidence are necessary before final academic reporting.
