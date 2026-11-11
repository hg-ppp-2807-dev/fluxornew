"""
RL Agent — DQN-based load balancing decision maker.
Exposes REST API for the Go load balancer to call.
"""

import os, json, time, random, logging
import threading
from collections import deque
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from flask import Flask, request, jsonify
from prometheus_client import (
    Gauge,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logging.basicConfig(level=logging.INFO, format="[RL] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({
        "message": "RL Agent API is running.",
        "endpoints": ["/decide (POST)", "/feedback (POST)", "/status (GET)", "/metrics (GET)", "/health (GET)"]
    })

# ── Config ────────────────────────────────────────────────────
N_SERVERS   = int(os.getenv("N_SERVERS", 3))
STATE_DIM   = N_SERVERS * 3 + 2   # cpu, conns, latency per server + rate + queue
ACTION_DIM  = N_SERVERS
PROM_URL    = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
CKPT_DIR    = os.getenv("CKPT_DIR", "./checkpoints")
CKPT_LOAD_PATH = os.getenv(
    "RL_CHECKPOINT_LOAD_PATH",
    f"{CKPT_DIR}/dqn_latest.pt"
)
CKPT_SAVE_PATH = os.getenv(
    "RL_CHECKPOINT_SAVE_PATH",
    f"{CKPT_DIR}/dqn_latest.pt"
)
SAVE_CHECKPOINTS = os.getenv(
    "RL_SAVE_CHECKPOINTS", "true"
).lower() == "true"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── DQN Model ─────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.net(x)

def compute_double_dqn_targets(
    policy_net,
    target_net,
    next_states,
    rewards,
    gamma,
):
    """
    Calculate Double DQN bootstrap targets.
    Policy network selects the next action.
    Target network evaluates the selected action.
    """
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(
            dim=1,
            keepdim=True
        )
        next_q_values = target_net(next_states).gather(
            1,
            next_actions
        ).squeeze(1)
        return rewards + gamma * next_q_values

# ── Replay Buffer ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buf.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32))

    def __len__(self):
        return len(self.buf)

# ── Reader-Writer Lock ─────────────────────────────────────────
class ReadWriteLock:
    """A reader-writer lock with writer priority and re-entrant writer support.
    Multiple readers can hold read_lock concurrently.
    Writers acquire exclusive write_lock.
    """
    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_owner = None
        self._writer_depth = 0

    def acquire_read(self):
        tid = threading.get_ident()
        with self._cond:
            while (self._writer_owner is not None and self._writer_owner != tid) or (self._writers_waiting > 0 and self._writer_owner != tid):
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self):
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner == tid:
                self._writer_depth += 1
                return
            self._writers_waiting += 1
            while self._writer_owner is not None or self._readers > 0:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer_owner = tid
            self._writer_depth = 1

    def release_write(self):
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner != tid:
                raise RuntimeError("release_write called by non-owner thread")
            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer_owner = None
                self._cond.notify_all()

    def __enter__(self):
        self.acquire_write()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_write()

    class _ReadContext:
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            self._lock.acquire_read()
            return self._lock

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._lock.release_read()

    class _WriteContext:
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            self._lock.acquire_write()
            return self._lock

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._lock.release_write()

    def read_lock(self):
        return self._ReadContext(self)

    def write_lock(self):
        return self._WriteContext(self)

# ── DQN Agent ─────────────────────────────────────────────────
class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(STATE_DIM, ACTION_DIM)
        self.target_net = DQN(STATE_DIM, ACTION_DIM)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn   = nn.SmoothL1Loss()
        self.buffer    = ReplayBuffer()

        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995
        self.gamma         = 0.95
        self.batch_size    = 64
        self.target_sync   = 50
        self.step_count    = 0
        self.total_reward  = 0.0
        self.episode_steps = 0
        self.rwlock        = ReadWriteLock()
        self.lock          = self.rwlock
        self.training_enabled = True
        self.eval_saved_epsilon = None

        self._last_state   = None
        self._last_action  = None

        # Try to load existing checkpoint
        self.load_checkpoint()

    def select_action(self, state: np.ndarray, healthy_mask=None) -> tuple[int, float]:
        if healthy_mask is None:
            healthy_mask = [True] * ACTION_DIM
        if len(healthy_mask) != ACTION_DIM:
            raise ValueError("healthy_mask must contain exactly 3 values")
        if not all(isinstance(h, (bool, np.bool_)) for h in healthy_mask):
            raise ValueError("All elements of healthy_mask must be boolean")
        available = [
            i for i, healthy in enumerate(healthy_mask)
            if healthy
        ]
        if not available:
            raise ValueError("No healthy backend available")

        t = torch.as_tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)
        mask = torch.as_tensor(
            healthy_mask,
            dtype=torch.bool
        )

        lock_wait_start = time.perf_counter()
        with self.rwlock.read_lock():
            rl_inference_lock_wait_seconds.observe(
                time.perf_counter() - lock_wait_start
            )
            inference_start = time.perf_counter()
            with torch.no_grad():
                q = self.policy_net(t).squeeze(0)
                if random.random() < self.epsilon:
                    action = random.choice(available)
                    q_value = float(q[action].item())
                else:
                    q_masked = q.clone()
                    q_masked[~mask] = -1e9
                    action = int(q_masked.argmax().item())
                    q_value = float(q_masked[action].item())
            rl_inference_seconds.observe(
                time.perf_counter() - inference_start
            )
            return action, q_value

    def q_values(self, state: np.ndarray) -> list:
        t = torch.FloatTensor(state).unsqueeze(0)
        with self.rwlock.read_lock():
            with torch.no_grad():
                return self.policy_net(t).squeeze().tolist()

    def store_and_train(self, state, action, reward, next_state):
        # Fast path during evaluation.
        if not self.training_enabled:
            return
        with self.rwlock.write_lock():
            # Recheck after acquiring the lock because evaluation
            # mode may have been enabled while this thread was waiting.
            if not self.training_enabled:
                return
            self.buffer.push(state, action, reward, next_state)
            self.total_reward += reward
            self.episode_steps += 1

            if len(self.buffer) < self.batch_size:
                return

            s, a, r, ns = self.buffer.sample(self.batch_size)
            s  = torch.FloatTensor(s)
            a  = torch.LongTensor(a).unsqueeze(1)
            r  = torch.FloatTensor(r)
            ns = torch.FloatTensor(ns)

            current_q = self.policy_net(s).gather(1, a).squeeze()
            target_q = compute_double_dqn_targets(
                self.policy_net,
                self.target_net,
                ns,
                r,
                self.gamma,
            )

            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            self.step_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            prom_epsilon.set(self.epsilon)
            prom_reward.set(self.total_reward / max(1, self.episode_steps))

            if self.step_count % self.target_sync == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.step_count % 100 == 0:
                self.save_checkpoint()
                log.info(f"step={self.step_count} eps={self.epsilon:.3f} "
                         f"avg_reward={self.total_reward/max(1,self.episode_steps):.3f}")

    def save_checkpoint(self):
        if not SAVE_CHECKPOINTS:
            log.info("Checkpoint saving disabled; skipping save.")
            return
        with self.rwlock.write_lock():
            torch.save({
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            }, CKPT_SAVE_PATH)

    def load_checkpoint(self):
        if os.path.exists(CKPT_LOAD_PATH):
            with self.rwlock.write_lock():
                ck = torch.load(CKPT_LOAD_PATH, map_location="cpu")
                self.policy_net.load_state_dict(ck["policy"])
                self.target_net.load_state_dict(ck["target"])
                self.epsilon = ck.get("epsilon", self.epsilon)
                self.step_count = ck.get("step_count", 0)
            log.info(
                f"Checkpoint loaded from {CKPT_LOAD_PATH}: "
                f"step={self.step_count} eps={self.epsilon:.3f}"
            )
        else:
            log.warning(f"Checkpoint not found: {CKPT_LOAD_PATH}")

    def load_checkpoint_from_path(self, checkpoint_path: str):
        """
        Dynamically load a checkpoint from path with atomic model replacement,
        parameter validation, evaluation enforcement, and optimizer reinitialization.
        """
        with self.rwlock.write_lock():
            ck = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False
            )
            if not isinstance(ck, dict):
                raise ValueError("Invalid checkpoint format: expected dictionary")
            if "policy" not in ck or "target" not in ck:
                raise ValueError("Checkpoint missing model weights ('policy' or 'target')")
            if "step_count" not in ck or not isinstance(ck["step_count"], int) or ck["step_count"] < 0:
                raise ValueError("Checkpoint step_count must be a non-negative integer")
            if "epsilon" not in ck or not isinstance(ck["epsilon"], (int, float)):
                raise ValueError("Checkpoint missing valid epsilon value")

            # Validate using temporary model instances
            new_policy = DQN(STATE_DIM, ACTION_DIM)
            new_target = DQN(STATE_DIM, ACTION_DIM)
            new_policy.load_state_dict(ck["policy"], strict=True)
            new_target.load_state_dict(ck["target"], strict=True)

            # Atomic replacement of active networks
            self.policy_net = new_policy
            self.target_net = new_target

            # Reinitialize Adam optimizer against the newly loaded policy parameters
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

            # Restore step count from checkpoint
            self.step_count = int(ck["step_count"])

            # Force evaluation mode (override exploration to 0.0)
            self.training_enabled = False
            self.epsilon = 0.0
            self.eval_saved_epsilon = None
            self.policy_net.eval()
            self.target_net.eval()
            prom_epsilon.set(0.0)

            # Reset replay buffer and episodic reward metrics
            self.buffer = ReplayBuffer()
            self.total_reward = 0.0
            self.episode_steps = 0
            log.info(
                f"Dynamically loaded checkpoint {checkpoint_path}: "
                f"step={self.step_count} (evaluation mode enforced, eps=0.0)"
            )

    def enable_evaluation(self):
        with self.rwlock.write_lock():
            if not self.training_enabled:
                return
            self.eval_saved_epsilon = self.epsilon
            self.epsilon = 0.0
            self.training_enabled = False
            self.policy_net.eval()
            prom_epsilon.set(0.0)

    def enable_training(self, epsilon: float | None = None):
        with self.rwlock.write_lock():
            if epsilon is not None:
                epsilon = float(epsilon)
                if not 0.0 <= epsilon <= 1.0:
                    raise ValueError("epsilon must be between 0.0 and 1.0")
            self.training_enabled = True
            if epsilon is not None:
                self.epsilon = epsilon
            elif self.eval_saved_epsilon is not None:
                self.epsilon = self.eval_saved_epsilon
            else:
                self.epsilon = self.epsilon_min
            self.eval_saved_epsilon = None
            self.policy_net.train()
            prom_epsilon.set(self.epsilon)

# ── State Builder ─────────────────────────────────────────────
def build_state(payload: dict) -> np.ndarray:
    """
    Build normalised state vector from Go payload or Prometheus metrics.
    State: [cpu_0, conns_0, lat_0, cpu_1, conns_1, lat_1, ..., req_rate, queue]
    """
    state = []
    cpu_util    = payload.get("cpu_util", [0.0] * N_SERVERS)
    active_conn = payload.get("active_conns", [0.0] * N_SERVERS)
    avg_lat     = payload.get("avg_latency_ms", [100.0] * N_SERVERS)

    for i in range(N_SERVERS):
        cpu  = float(cpu_util[i])    if i < len(cpu_util)    else 0.0
        conn = float(active_conn[i]) if i < len(active_conn) else 0.0
        lat  = float(avg_lat[i])     if i < len(avg_lat)     else 100.0

        # Enrich with Prometheus if available
        prom_cpu = _prom_metric(i)
        if prom_cpu is not None:
            cpu = prom_cpu

        state.append(np.clip(cpu / 100.0, 0, 1))
        state.append(np.clip(conn / 100.0, 0, 1))
        state.append(np.clip(lat / 2000.0, 0, 1))

    req_rate = float(payload.get("request_rate", 0))
    state.append(np.clip(req_rate / 500.0, 0, 1))
    queue_depth = float(payload.get("queue_depth", 0))
    state.append(np.clip(queue_depth / 100.0, 0, 1))
    return np.array(state, dtype=np.float32)

def compute_reward(state: np.ndarray, action: int, latency_ms: float) -> float:
    cpu_utils = [state[i * 3] for i in range(N_SERVERS)]   # normalised [0,1]
    chosen_cpu = cpu_utils[action] if action < len(cpu_utils) else 0.5

    r_latency  = -(latency_ms / 1000.0) * 0.5
    r_balance  = -float(np.std(cpu_utils)) * 0.3
    r_capacity = -10.0 if chosen_cpu > 0.85 else 0.0
    r_capacity *= 0.2

    return float(np.clip(r_latency + r_balance + r_capacity, -15.0, 1.0))

# ── Background Prometheus Telemetry Cache ─────────────────────
_prom_cache = {}
_prom_cache_lock = threading.Lock()
PROM_POLL_INTERVAL = 1.0
PROM_REQUEST_TIMEOUT = 0.5

def poll_prometheus():
    """Poll backend CPU metrics outside the request-handling path."""
    while True:
        for i in range(N_SERVERS):
            query = f'backend_cpu_utilization{{server_id="{i+1}"}}'
            try:
                response = requests.get(
                    f"{PROM_URL}/api/v1/query",
                    params={"query": query},
                    timeout=PROM_REQUEST_TIMEOUT
                )
                response.raise_for_status()
                results = response.json()["data"]["result"]
                if results:
                    value = float(results[0]["value"][1])
                    with _prom_cache_lock:
                        _prom_cache[i] = value
            except Exception as exc:
                log.warning(
                    "Prometheus polling failed for backend %d: %s",
                    i + 1, exc
                )
        time.sleep(PROM_POLL_INTERVAL)

def _prom_metric(server_idx: int):
    with _prom_cache_lock:
        return _prom_cache.get(server_idx, None)

_telemetry_thread = None
_telemetry_start_lock = threading.Lock()

def start_telemetry_thread():
    """Start exactly one Prometheus polling thread per worker process."""
    global _telemetry_thread
    with _telemetry_start_lock:
        if _telemetry_thread is not None and _telemetry_thread.is_alive():
            log.info("Prometheus telemetry thread already running")
            return
        _telemetry_thread = threading.Thread(
            target=poll_prometheus,
            name="prometheus-telemetry",
            daemon=True,
        )
        _telemetry_thread.start()
        log.info("Prometheus telemetry thread started")

# ── Prometheus Metrics ────────────────────────────────────────
prom_epsilon = Gauge("rl_epsilon", "Current exploration rate")
prom_reward  = Gauge("rl_avg_reward", "Average episodic reward")
prom_decides = Counter("rl_decisions_total", "Total routing decisions made")

# RL decision profiling
rl_state_build_seconds = Histogram(
    "rl_state_build_seconds",
    "Time spent constructing the RL state",
    buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05),
)
rl_inference_lock_wait_seconds = Histogram(
    "rl_inference_lock_wait_seconds",
    "Time waiting for the DQN inference lock",
    buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05),
)
rl_inference_seconds = Histogram(
    "rl_inference_seconds",
    "DQN forward-pass and action-selection duration",
    buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05),
)
rl_decide_total_seconds = Histogram(
    "rl_decide_total_seconds",
    "Total successful Python decision-handler duration",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

# ── Flask API ─────────────────────────────────────────────────
agent = DQNAgent()

# Store RL transitions separately for each request.
TRANSITION_TTL_SECONDS = 10.0
pending_transitions = {}
transition_expiry_queue = deque()
transition_lock = threading.Lock()
expired_transitions = 0


def expire_stale_transitions_locked(now=None):
    """
    Expire transitions using an ordered expiry queue.
    Caller must hold transition_lock.
    """
    global expired_transitions

    if now is None:
        now = time.monotonic()

    expired_count = 0
    while (
        transition_expiry_queue
        and transition_expiry_queue[0][0] <= now
    ):
        expires_at, request_id = transition_expiry_queue.popleft()
        transition = pending_transitions.get(request_id)
        # Ignore entries already consumed by feedback or replaced
        # by a newer transition with the same request ID.
        if (
            transition is not None
            and transition["expires_at"] == expires_at
        ):
            pending_transitions.pop(request_id, None)
            expired_count += 1

    expired_transitions += expired_count
    if expired_count:
        log.info("Expired %d stale RL transitions", expired_count)

    return expired_count

@app.route("/decide", methods=["POST"])
def decide():
    """
    POST /decide — called by Go load balancer per request.
    Body: { request_id: str, cpu_util: [], active_conns: [], avg_latency_ms: [], request_rate: N, healthy_mask: [bool] }
    Returns: { action: int, q_value: float }
    """
    decision_start = time.perf_counter()
    payload = request.get_json(force=True) or {}
    request_id = payload.get("request_id")

    state_start = time.perf_counter()
    state = build_state(payload)
    rl_state_build_seconds.observe(
        time.perf_counter() - state_start
    )

    healthy_mask = payload.get(
        "healthy_mask",
        [True] * ACTION_DIM
    )
    try:
        action, q_val = agent.select_action(state, healthy_mask)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503

    if request_id:
        with transition_lock:
            now = time.monotonic()
            expire_stale_transitions_locked(now)
            expires_at = now + TRANSITION_TTL_SECONDS
            pending_transitions[request_id] = {
                "state": state,
                "action": action,
                "expires_at": expires_at,
            }
            transition_expiry_queue.append(
                (expires_at, request_id)
            )

    prom_decides.inc()

    rl_decide_total_seconds.observe(
        time.perf_counter() - decision_start
    )

    return jsonify({
        "action": action,
        "q_value": q_val
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    """
    POST /feedback — called by Go after request completes.
    Body: { request_id: str, server_id: int, latency_ms: float }
    """
    payload = request.get_json(force=True) or {}

    request_id = payload.get("request_id")
    latency_ms = float(payload.get("latency_ms", 200))
    server_id = int(payload.get("server_id", 0))

    with transition_lock:
        expire_stale_transitions_locked()
        transition = pending_transitions.pop(request_id, None)

    if transition is None:
        return jsonify({
            "status": "skipped",
            "reason": "transition_not_found"
        })

    prev_state = transition["state"]
    prev_action = transition["action"]

    reward = compute_reward(
        prev_state,
        prev_action,
        latency_ms
    )

    # Use the actual post-request cluster measurements supplied by the load balancer.
    # CPU may still be enriched from Prometheus inside build_state().
    next_payload = {
        "active_conns": payload.get("active_conns", []),
        "avg_latency_ms": payload.get("avg_latency_ms", []),
        "request_rate": payload.get("request_rate", 0.0),
        "queue_depth": payload.get("queue_depth", 0.0),
    }

    next_state = build_state(next_payload)

    agent.store_and_train(
        prev_state,
        prev_action,
        reward,
        next_state
    )

    return jsonify({
        "status": "ok",
        "reward": reward,
        "server_id": server_id
    })

@app.route("/save_checkpoint", methods=["POST"])
def save_checkpoint():
    agent.save_checkpoint()
    return jsonify({
        "status": "saved",
        "step_count": agent.step_count
    })

@app.route("/load_checkpoint", methods=["POST"])
def load_checkpoint_route():
    payload = request.get_json(silent=True) or {}
    checkpoint_name = payload.get("checkpoint")
    if not checkpoint_name or not isinstance(checkpoint_name, str):
        return jsonify({
            "status": "error",
            "message": "checkpoint filename is required"
        }), 400

    base_name = os.path.basename(checkpoint_name)
    if (
        base_name != checkpoint_name
        or "/" in checkpoint_name
        or "\\" in checkpoint_name
        or ".." in checkpoint_name
        or not checkpoint_name.endswith(".pt")
    ):
        return jsonify({
            "status": "error",
            "message": "invalid filename or path traversal rejected"
        }), 400

    checkpoint_path = os.path.join(CKPT_DIR, base_name)
    if not os.path.isfile(checkpoint_path):
        return jsonify({
            "status": "error",
            "message": f"checkpoint file not found: {base_name}"
        }), 404

    with transition_lock:
        pending_transitions.clear()
        transition_expiry_queue.clear()

    try:
        agent.load_checkpoint_from_path(checkpoint_path)
    except Exception as exc:
        log.error(f"Failed to load checkpoint {base_name}: {exc}")
        return jsonify({
            "status": "error",
            "message": f"failed to load checkpoint: {str(exc)}"
        }), 400

    return jsonify({
        "status": "loaded",
        "checkpoint": base_name,
        "step_count": agent.step_count,
        "epsilon": agent.epsilon,
        "training_enabled": agent.training_enabled
    })

@app.route("/eval_mode", methods=["POST"])
def eval_mode():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode")
    if mode == "evaluation":
        agent.enable_evaluation()
    elif mode == "training":
        raw_eps = payload.get("epsilon")
        try:
            agent.enable_training(epsilon=raw_eps)
        except (ValueError, TypeError) as exc:
            return jsonify({
                "status": "error",
                "message": str(exc)
            }), 400
    else:
        return jsonify({
            "status": "error",
            "message": "mode must be evaluation or training"
        }), 400
    return jsonify({
        "status": "ok",
        "mode": mode,
        "epsilon": agent.epsilon,
        "training_enabled": agent.training_enabled
    })

@app.route("/status", methods=["GET"])
def status():
    with transition_lock:
        expire_stale_transitions_locked()
        pending_count = len(pending_transitions)
        expired_count = expired_transitions

    return jsonify({
        "training_enabled": agent.training_enabled,
        "mode": "training" if agent.training_enabled else "evaluation",
        "epsilon": round(agent.epsilon, 4),
        "step_count": agent.step_count,
        "buffer_len": len(agent.buffer),
        "avg_reward": round(
            agent.total_reward / max(1, agent.episode_steps), 4
        ),
        "pending_transitions": pending_count,
        "expired_transitions": expired_count,
    })

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    log.info(
        f"RL Agent starting — state_dim={STATE_DIM} "
        f"action_dim={ACTION_DIM}"
    )
    start_telemetry_thread()
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True,
        use_reloader=False
    )
