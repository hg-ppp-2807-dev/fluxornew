#!/bin/bash
set -euo pipefail

URL="http://localhost:5005/status"
TIMEOUT_SECONDS=120
INTERVAL_SECONDS=3
REQUIRED_STABLE_CHECKS=4

start=$(date +%s)
previous_step=""
previous_pending=""
stable_checks=0

echo "Waiting for RL agent to settle..."

while true; do
  status=$(curl -fsS "$URL") || {
    echo "ERROR: RL status endpoint unavailable."
    exit 1
  }

  step=$(printf '%s' "$status" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["step_count"])')
  pending=$(printf '%s' "$status" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["pending_transitions"])')

  echo "$(date '+%H:%M:%S') step_count=$step pending_transitions=$pending"

  if [[ "$step" == "$previous_step" &&
        "$pending" == "$previous_pending" &&
        "$pending" == "0" ]]; then
    stable_checks=$((stable_checks + 1))
  else
    stable_checks=0
  fi

  if (( stable_checks >= REQUIRED_STABLE_CHECKS )); then
    echo "SUCCESS: RL state remained stable with zero pending transitions."
    exit 0
  fi

  now=$(date +%s)
  if (( now - start >= TIMEOUT_SECONDS )); then
    echo "ERROR: RL state did not settle within ${TIMEOUT_SECONDS}s."
    echo "Do not start the benchmark yet."
    exit 2
  fi

  previous_step="$step"
  previous_pending="$pending"
  sleep "$INTERVAL_SECONDS"
done
