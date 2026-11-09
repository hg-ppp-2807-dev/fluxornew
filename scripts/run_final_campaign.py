import json
import random
import time
import urllib.request
import urllib.error
import sys

BASE = "http://localhost:5006"
ALGORITHMS = ["rr", "lc", "rl"]
SCENARIOS = ["steady", "spike", "failure"]
TRIALS = 3

def get_json(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

def post_json(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())

jobs = [
    (scenario, algorithm, trial)
    for scenario in SCENARIOS
    for algorithm in ALGORITHMS
    for trial in range(1, TRIALS + 1)
]
random.Random(20261109).shuffle(jobs)

initial = get_json("/status")
if initial.get("running") or initial.get("starting") or initial.get("cleanup_pending"):
    raise SystemExit(f"Runner is not clean before campaign: {initial}")

rl_status = json.load(
    urllib.request.urlopen("http://localhost:5005/status", timeout=10)
)
if (
    rl_status.get("mode") != "evaluation"
    or rl_status.get("epsilon") != 0
    or rl_status.get("training_enabled") is not False
    or rl_status.get("step_count") != 180700
):
    raise SystemExit(f"RL agent is not in verified evaluation state: {rl_status}")

print(f"Starting campaign: {len(jobs)} runs", flush=True)

completed = []

for index, (scenario, algorithm, trial) in enumerate(jobs, 1):
    trial_name = f"final_t{trial}"

    print(
        f"\n[{index}/{len(jobs)}] "
        f"Starting scenario={scenario}, algorithm={algorithm}, trial={trial}",
        flush=True
    )

    try:
        result = post_json("/start", {
            "scenario": scenario,
            "algorithm": algorithm,
            "trial": trial_name,
        })
    except Exception as e:
        raise SystemExit(f"Could not start run; campaign halted: {e}")

    run_id = result.get("run_id")
    if not run_id:
        raise SystemExit(f"Start response did not contain run_id: {result}")

    print(f"Run ID: {run_id}", flush=True)

    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(3)
        status = get_json("/status")

        if not status.get("running") and not status.get("starting"):
            break
    else:
        raise SystemExit(f"Timed out waiting for {run_id}; inspect runner status.")

    if (
        status.get("cleanup_pending")
        or status.get("cleanup_error")
        or status.get("failure_injection_error")
        or status.get("backend2_stopped")
    ):
        raise SystemExit(
            f"Cleanup/failure issue after {run_id}; campaign halted: {status}"
        )

    algorithm_status = json.load(
        urllib.request.urlopen(
            "http://localhost:8080/admin/algorithm", timeout=10
        )
    )
    if algorithm_status.get("algorithm") != "rl":
        raise SystemExit(
            f"Original algorithm was not restored after {run_id}: "
            f"{algorithm_status}"
        )

    completed.append({
        "run_id": run_id,
        "scenario": scenario,
        "algorithm": algorithm,
        "trial": trial,
    })
    print(f"Completed successfully: {run_id}", flush=True)

    # Confirm the agent remains in fixed evaluation mode.
    agent = json.load(
        urllib.request.urlopen("http://localhost:5005/status", timeout=10)
    )
    if (
        agent.get("mode") != "evaluation"
        or agent.get("training_enabled") is not False
        or agent.get("step_count") != 180700
        or agent.get("buffer_len") != 0
    ):
        raise SystemExit(f"Agent evaluation state changed; campaign halted: {agent}")

print("\nCAMPAIGN COMPLETE", flush=True)
print(f"Completed runs: {len(completed)}/27", flush=True)
for item in completed:
    print(
        f"{item['run_id']} | {item['scenario']} | "
        f"{item['algorithm']} | trial {item['trial']}",
        flush=True
    )
