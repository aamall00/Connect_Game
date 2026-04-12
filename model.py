"""
Dual-head ResNet for AlphaZero Connect 4.

Input : (3, ROWS, COLS)
Output: (policy_logits [7], value [1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import ROWS, COLS, NUM_RES_BLOCKS, NUM_FILTERS


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------
class ResBlock(nn.Module):
    """Single residual block: Conv → BN → ReLU → Conv → BN → (+ residual) → ReLU."""

    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


# ------------------------------------------------------------------
# Full network
# ------------------------------------------------------------------
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_actions = COLS

        # Initial convolution
        self.conv_input = nn.Conv2d(3, NUM_FILTERS, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(NUM_FILTERS)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(NUM_FILTERS) for _ in range(NUM_RES_BLOCKS)])

        # Policy head
        self.policy_conv = nn.Conv2d(NUM_FILTERS, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, self.num_actions)

        # Value head
        self.value_conv = nn.Conv2d(NUM_FILTERS, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor shape (batch, 3, ROWS, COLS)
        Returns:
            policy_logits: (batch, COLS)  — raw logits over actions
            value:         (batch, 1)     — tanh-scaled
        """
        # Shared trunk
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)  # raw logits

        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(self, state_np):
        """Convenience inference method.

        Args:
            state_np: np.ndarray from get_canonical_state(), shape (3, ROWS, COLS)
                      or batched (N, 3, ROWS, COLS)
        Returns:
            policy: np.ndarray  (COLS,) or (N, COLS)  — softmax over legal cols
            value:  float or np.ndarray
        """
        self.eval()
        with torch.no_grad():
            single = False
            if state_np.ndim == 3:
                state_np = state_np[np.newaxis, ...]
                single = True
            x = torch.from_numpy(state_np).float()
            # Ensure device consistency
            device = next(self.parameters()).device
            x = x.to(device)
            policy_logits, value = self(x)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()
            value = value.cpu().numpy()
        if single:
            return policy[0], value[0, 0]
        return policy, value.squeeze(-1)
