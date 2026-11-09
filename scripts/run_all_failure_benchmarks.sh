#!/bin/bash
set -euo pipefail

BASE="http://localhost:8080"
RUNNER="http://localhost:5006"
ALGOS=("rl" "rr" "lc")

mkdir -p results

for algo in "${ALGOS[@]}"; do
  echo ""
  echo "======================================"
  echo "Starting FAILURE benchmark: $algo"
  echo "======================================"

  # Ensure the previous experiment has finished.
  status=$(curl -fsS "$RUNNER/status")
  if ! echo "$status" | python3 -c \
    'import json,sys; sys.exit(0 if json.load(sys.stdin).get("running") is False else 1)'; then
    echo "ERROR: An experiment is already running: $status"
    exit 1
  fi

  # Ensure backend 2 is running before the next test.
  if [ "$(docker inspect -f '{{.State.Running}}' backend-2)" != "true" ]; then
    docker start backend-2
  fi

  # Wait up to 60 seconds for backend 2 to become healthy.
  healthy=0
  for i in $(seq 1 30); do
    health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' backend-2)
    if [ "$health" = "healthy" ]; then
      healthy=1
      break
    fi
    sleep 2
  done

  if [ "$healthy" != "1" ]; then
    echo "ERROR: backend-2 did not become healthy."
    exit 1
  fi

  # Set the algorithm.
  curl -fsS -X POST "$BASE/admin/algorithm" \
    -H "Content-Type: application/json" \
    -d "{\"algorithm\":\"$algo\"}"
  echo ""

  # Confirm the selected algorithm.
  curl -fsS "$BASE/admin/algorithm"
  echo ""

  # Record the log start time and start the test.
  start_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  curl -fsS -X POST "$RUNNER/start" \
    -H "Content-Type: application/json" \
    -d '{"scenario":"failure"}'
  echo ""

  # Wait for the experiment to finish (maximum 180 seconds).
  finished=0
  for i in $(seq 1 90); do
    status=$(curl -fsS "$RUNNER/status")
    running=$(echo "$status" | python3 -c \
      'import json,sys; print(str(json.load(sys.stdin).get("running", False)).lower())')

    if [ "$running" = "false" ]; then
      finished=1
      break
    fi
    sleep 2
  done

  # Save this run's runner logs.
  docker logs --since "$start_time" experiment-runner \
    > "results/failure-${algo}.log" 2>&1 || true

  if [ "$finished" != "1" ]; then
    echo "ERROR: $algo experiment timed out. Inspect results/failure-${algo}.log"
    exit 1
  fi

  # Verify backend 2 recovered.
  if [ "$(docker inspect -f '{{.State.Running}}' backend-2)" != "true" ]; then
    echo "WARNING: backend-2 is stopped; attempting to start it."
    docker start backend-2
  fi

  echo "Finished $algo. Log: results/failure-${algo}.log"
done

echo ""
echo "All three failure experiments completed."
echo "Review results/failure-rl.log, results/failure-rr.log, and results/failure-lc.log"
