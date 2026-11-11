# FLUXOR 60-Trial Evaluation Campaign Report

- **Generated At**: `2026-11-10T12:04:45.744860+00:00`
- **Progress**: 60 / 60 trials completed
- **Active Git Branch**: `exp/v2-double-dqn`
- **Baseline Checkpoint SHA-256**: `87dbfd996c98a9d8224512398165f74c32349d245672a0fa1edf6d2815b6501b`
- **Double DQN Checkpoint SHA-256**: `81994ac6ba4f93edf46be3e41efc312a45e635b10879a43fd83b57d6ede26228`

## Scenario: `STEADY`

| Algorithm | Trials | Throughput (RPS) | Avg Latency (ms) | P90 Lat (ms) | P95 Lat (ms) | Max Lat (ms) | Error Rate (%) | Fallback Rate (%) | CPU Imbalance (σ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Round Robin** | 5 | 320.63 ± 1.93 | 54.72 ± 0.93 | 74.28 ± 1.35 | 78.61 ± 1.29 | 120.37 ± 35.79 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.64 ± 0.41 |
| **Least Connections** | 5 | 328.58 ± 1.91 | 50.90 ± 0.88 | 70.74 ± 0.96 | 75.75 ± 1.31 | 90.66 ± 1.18 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.94 ± 0.33 |
| **Double DQN (Exp A)** | 5 | 350.15 ± 3.49 | 41.69 ± 1.43 | 52.57 ± 2.01 | 55.39 ± 1.85 | 119.07 ± 19.78 | 0.000% ± 0.000% | 0.082% ± 0.122% | 0.80 ± 0.32 |
| **Vanilla DQN (Base)** | 5 | 343.67 ± 1.86 | 44.35 ± 0.80 | 55.54 ± 0.69 | 58.49 ± 0.79 | 135.62 ± 33.61 | 0.000% ± 0.000% | 0.140% ± 0.190% | 0.42 ± 0.20 |

**Double DQN vs Vanilla DQN (STEADY)**:
- Latency Δ: **-6.01%** (Avg Latency: `41.69 ms` vs `44.35 ms`)
- P95 Latency Δ: **-5.30%** (P95: `55.39 ms` vs `58.49 ms`)
- Throughput Δ: **+1.89%** (Throughput: `350.15 RPS` vs `343.67 RPS`)

## Scenario: `SPIKE`

| Algorithm | Trials | Throughput (RPS) | Avg Latency (ms) | P90 Lat (ms) | P95 Lat (ms) | Max Lat (ms) | Error Rate (%) | Fallback Rate (%) | CPU Imbalance (σ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Round Robin** | 5 | 906.86 ± 2.05 | 54.10 ± 0.35 | 73.88 ± 1.30 | 78.20 ± 1.15 | 89.97 ± 1.20 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.69 ± 0.38 |
| **Least Connections** | 5 | 927.47 ± 3.86 | 50.67 ± 0.65 | 70.89 ± 0.61 | 75.84 ± 1.02 | 89.74 ± 0.31 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.36 ± 0.25 |
| **Double DQN (Exp A)** | 5 | 753.41 ± 11.95 | 85.78 ± 2.95 | 114.13 ± 7.59 | 119.26 ± 7.43 | 152.57 ± 18.39 | 0.000% ± 0.000% | 37.862% ± 24.849% | 0.69 ± 0.34 |
| **Vanilla DQN (Base)** | 5 | 732.74 ± 9.13 | 91.06 ± 2.41 | 122.02 ± 2.15 | 127.32 ± 2.33 | 158.68 ± 17.71 | 0.000% ± 0.000% | 48.520% ± 15.243% | 0.52 ± 0.26 |

**Double DQN vs Vanilla DQN (SPIKE)**:
- Latency Δ: **-5.80%** (Avg Latency: `85.78 ms` vs `91.06 ms`)
- P95 Latency Δ: **-6.33%** (P95: `119.26 ms` vs `127.32 ms`)
- Throughput Δ: **+2.82%** (Throughput: `753.41 RPS` vs `732.74 RPS`)

## Scenario: `FAILURE`

| Algorithm | Trials | Throughput (RPS) | Avg Latency (ms) | P90 Lat (ms) | P95 Lat (ms) | Max Lat (ms) | Error Rate (%) | Fallback Rate (%) | CPU Imbalance (σ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Round Robin** | 5 | 323.06 ± 1.34 | 53.58 ± 0.65 | 75.05 ± 1.31 | 78.97 ± 1.28 | 90.21 ± 0.76 | 0.020% ± 0.000% | 0.000% ± 0.000% | 0.71 ± 0.67 |
| **Least Connections** | 5 | 331.86 ± 1.49 | 49.50 ± 0.67 | 72.62 ± 0.70 | 77.14 ± 0.92 | 96.85 ± 15.02 | 0.022% ± 0.004% | 0.000% ± 0.000% | 0.58 ± 0.10 |
| **Double DQN (Exp A)** | 5 | 345.18 ± 2.07 | 43.79 ± 0.86 | 54.79 ± 0.49 | 57.59 ± 0.34 | 130.47 ± 17.96 | 0.002% ± 0.004% | 0.092% ± 0.149% | 0.64 ± 0.59 |
| **Vanilla DQN (Base)** | 5 | 341.02 ± 1.76 | 45.54 ± 0.76 | 56.68 ± 1.17 | 59.72 ± 0.94 | 166.68 ± 86.42 | 0.000% ± 0.000% | 0.284% ± 0.481% | 0.70 ± 0.28 |

**Double DQN vs Vanilla DQN (FAILURE)**:
- Latency Δ: **-3.84%** (Avg Latency: `43.79 ms` vs `45.54 ms`)
- P95 Latency Δ: **-3.57%** (P95: `57.59 ms` vs `59.72 ms`)
- Throughput Δ: **+1.22%** (Throughput: `345.18 RPS` vs `341.02 RPS`)

