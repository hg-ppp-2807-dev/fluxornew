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
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

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
        self.lock          = Lock()
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
        available = [i for i, healthy in enumerate(healthy_mask) if healthy]
        if not available:
            raise ValueError("No healthy backend available")
        with self.lock:
            with torch.no_grad():
                t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                q = self.policy_net(t).squeeze(0)
                # Exploration is restricted to healthy backends.
                if random.random() < self.epsilon:
                    action = random.choice(available)
                    return action, float(q[action].item())
                # Prevent selection of unhealthy backends.
                mask = torch.as_tensor(healthy_mask, dtype=torch.bool)
                q[~mask] = -1e9
                action = int(q.argmax().item())
                return action, float(q[action].item())

    def q_values(self, state: np.ndarray) -> list:
        with self.lock:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0)
                return self.policy_net(t).squeeze().tolist()

    def store_and_train(self, state, action, reward, next_state):
        with self.lock:
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
            with torch.no_grad():
                max_next_q = self.target_net(ns).max(1)[0]
            target_q = r + self.gamma * max_next_q

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
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }, CKPT_SAVE_PATH)

    def load_checkpoint(self):
        if os.path.exists(CKPT_LOAD_PATH):
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

    def enable_evaluation(self):
        with self.lock:
            if not self.training_enabled:
                return
            self.eval_saved_epsilon = self.epsilon
            self.epsilon = 0.0
            self.training_enabled = False
            self.policy_net.eval()
            prom_epsilon.set(0.0)

    def enable_training(self):
        with self.lock:
            if self.training_enabled:
                return
            self.training_enabled = True
            self.epsilon = (
                self.eval_saved_epsilon
                if self.eval_saved_epsilon is not None
                else self.epsilon_min
            )
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

# ── Prometheus Metrics ────────────────────────────────────────
prom_epsilon = Gauge("rl_epsilon", "Current exploration rate")
prom_reward  = Gauge("rl_avg_reward", "Average episodic reward")
prom_decides = Counter("rl_decisions_total", "Total routing decisions made")

# ── Flask API ─────────────────────────────────────────────────
agent = DQNAgent()

# Store RL transitions separately for each request.
TRANSITION_TTL_SECONDS = 10.0
pending_transitions = {}
transition_lock = threading.Lock()
expired_transitions = 0


def expire_stale_transitions_locked(now=None):
    """Expire stale transitions. Caller must hold transition_lock."""
    global expired_transitions

    if now is None:
        now = time.time()

    stale_ids = [
        request_id
        for request_id, transition in pending_transitions.items()
        if now - transition.get("created_at", now) > TRANSITION_TTL_SECONDS
    ]

    for request_id in stale_ids:
        pending_transitions.pop(request_id, None)
        expired_transitions += 1
        log.warning("Expired stale transition request_id=%s", request_id)

    return len(stale_ids)

@app.route("/decide", methods=["POST"])
def decide():
    """
    POST /decide — called by Go load balancer per request.
    Body: { request_id: str, cpu_util: [], active_conns: [], avg_latency_ms: [], request_rate: N, healthy_mask: [bool] }
    Returns: { action: int, q_value: float }
    """
    payload = request.get_json(force=True) or {}
    request_id = payload.get("request_id")

    state = build_state(payload)
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
            expire_stale_transitions_locked()
            pending_transitions[request_id] = {
                "state": state,
                "action": action,
                "created_at": time.time(),
            }

    prom_decides.inc()

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
    with agent.lock:
        agent.save_checkpoint()
    return jsonify({
        "status": "saved",
        "step_count": agent.step_count
    })

@app.route("/eval_mode", methods=["POST"])
def eval_mode():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode")
    if mode == "evaluation":
        agent.enable_evaluation()
    elif mode == "training":
        agent.enable_training()
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
    telemetry_thread = threading.Thread(
        target=poll_prometheus,
        daemon=True
    )
    telemetry_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True,
        use_reloader=False
    )
