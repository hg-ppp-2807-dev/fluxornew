import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

# Add rl-agent to Python module path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "rl-agent")
)

from app import compute_double_dqn_targets


class FixedQNetwork(nn.Module):
    """Network returning fixed Q-values for testing target calculation."""
    def __init__(self, values):
        super().__init__()
        self.register_buffer(
            "values",
            torch.tensor(values, dtype=torch.float32)
        )

    def forward(self, x):
        if self.values.dim() == 1:
            return self.values.unsqueeze(0).expand(x.shape[0], -1)
        elif self.values.shape[0] == 1:
            return self.values.expand(x.shape[0], -1)
        return self.values


class DoubleDQNTests(unittest.TestCase):
    def test_policy_selects_and_target_evaluates(self):
        # Policy favors action 1 (Q=5.0) over actions 0 and 2
        policy = FixedQNetwork([[1.0, 5.0, 2.0]])
        # Target has higher value for action 2 (Q=20.0) than action 1 (Q=3.0)
        target = FixedQNetwork([[10.0, 3.0, 20.0]])

        next_states = torch.zeros((1, 11))
        rewards = torch.tensor([2.0])
        gamma = 0.95

        target_q = compute_double_dqn_targets(
            policy,
            target,
            next_states,
            rewards,
            gamma,
        )

        # In standard DQN: target_q = 2.0 + 0.95 * 20.0 = 21.0
        # In Double DQN: policy selects action 1, target evaluates action 1 -> 3.0
        # target_q = 2.0 + 0.95 * 3.0 = 4.85
        expected = 2.0 + 0.95 * 3.0
        self.assertEqual(target_q.shape, torch.Size([1]))
        self.assertAlmostEqual(target_q.item(), expected, places=5)
        self.assertNotAlmostEqual(target_q.item(), 2.0 + 0.95 * 20.0, places=2)

    def test_batch_calculation(self):
        # 3 batch items with distinct chosen actions
        # Item 0: policy picks action 0 (Q=9.0), target evaluates action 0 -> 4.0
        # Item 1: policy picks action 2 (Q=8.0), target evaluates action 2 -> 7.0
        # Item 2: policy picks action 1 (Q=6.0), target evaluates action 1 -> 2.0
        policy_vals = [
            [9.0, 1.0, 2.0],
            [1.0, 3.0, 8.0],
            [2.0, 6.0, 4.0],
        ]
        target_vals = [
            [4.0, 15.0, 1.0],
            [10.0, 2.0, 7.0],
            [5.0, 2.0, 8.0],
        ]
        policy = FixedQNetwork(policy_vals)
        target = FixedQNetwork(target_vals)

        next_states = torch.zeros((3, 11))
        rewards = torch.tensor([1.0, -0.5, 3.0])
        gamma = 0.90

        target_q = compute_double_dqn_targets(
            policy,
            target,
            next_states,
            rewards,
            gamma,
        )

        expected = [
            1.0 + 0.90 * 4.0,   # 4.60
            -0.5 + 0.90 * 7.0,  # 5.80
            3.0 + 0.90 * 2.0,   # 4.80
        ]
        self.assertEqual(target_q.shape, torch.Size([3]))
        for i in range(3):
            self.assertAlmostEqual(target_q[i].item(), expected[i], places=5)

    def test_no_gradients_created(self):
        policy = FixedQNetwork([[1.0, 2.0, 3.0]])
        target = FixedQNetwork([[4.0, 5.0, 6.0]])
        next_states = torch.zeros((2, 11), requires_grad=True)
        rewards = torch.tensor([1.0, 2.0])
        gamma = 0.95

        target_q = compute_double_dqn_targets(
            policy,
            target,
            next_states,
            rewards,
            gamma,
        )

        self.assertFalse(target_q.requires_grad)
        self.assertIsNone(target_q.grad_fn)


if __name__ == "__main__":
    unittest.main()
