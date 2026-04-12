"""
Self-play loop + training for AlphaZero Connect 4.

Usage:
    python train.py                  # full training run
    python train.py --iterations 10  # quick test run
"""

import os
import copy
import random
from collections import deque
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse

from config import (
    NUM_ITERATIONS, SELF_PLAY_GAMES, MCTS_SIMULATIONS,
    BATCH_SIZE, TRAIN_BATCHES, REPLAY_BUFFER_SIZE,
    TEMP_THRESHOLD, TEMP_START,
    LR, WEIGHT_DECAY,
    CHECKPOINT_DIR, BEST_NET_FILE,
    WIN_THRESHOLD, EVAL_GAMES,
    ROWS, COLS,
)
from game import ConnectFour
from model import AlphaZeroNet
from mcts import MCTS
from evaluate import pit


# ------------------------------------------------------------------
# Self-play
# ------------------------------------------------------------------
def self_play_game(net, mcts_sims=MCTS_SIMULATIONS):
    """Play one complete game using MCTS.

    Returns:
        trajectory: list of (canonical_state, pi, player_who_moved)
    """
    game = ConnectFour()
    trajectory = []
    move_count = 0

    while not game.is_terminal():
        mcts = MCTS(game, net, add_dirichlet=True)
        pi = mcts.search(mcts_sims)

        # Temperature schedule
        if move_count < TEMP_THRESHOLD:
            tau = TEMP_START
        else:
            tau = 0.0

        action = mcts.select_action(pi, temperature=tau)
        if action is None:
            break

        canonical = game.get_canonical_state()
        trajectory.append((canonical.copy(), pi.copy(), game.current_player))

        game = game.make_move(action)
        move_count += 1

    # Determine outcome
    winner = game.check_winner()  # +1, -1, 0, or None (draw)

    # Assign z values
    data = []
    for state, pi, player in trajectory:
        if winner is None:
            z = 0
        elif winner == player:
            z = 1
        else:
            z = -1
        data.append((state, pi, z))

    # Augmentation: horizontal flip (doubles data)
    augmented = []
    for state, pi, z in data:
        flipped_state = np.flip(state, axis=2).copy()  # flip columns
        flipped_pi = np.flip(pi).copy()
        augmented.append((flipped_state, flipped_pi, z))

    return data + augmented


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train_iteration(net, replay_buffer, device):
    """Sample mini-batches from replay_buffer and train for TRAIN_BATCHES steps."""
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    net.train()

    total_loss = 0.0
    for _ in range(TRAIN_BATCHES):
        if len(replay_buffer) < BATCH_SIZE:
            break

        batch = random.sample(replay_buffer, BATCH_SIZE)
        states, policies, values = zip(*batch)

        states_t = torch.from_numpy(np.array(states)).float().to(device)
        policies_t = torch.from_numpy(np.array(policies)).float().to(device)
        values_t = torch.from_numpy(np.array(values)).float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        policy_logits, value_pred = net(states_t)

        # Policy loss: cross-entropy
        policy_loss = -torch.sum(policies_t * F.log_softmax(policy_logits, dim=1)) / BATCH_SIZE

        # Value loss: MSE
        value_loss = F.mse_loss(value_pred.squeeze(-1), values_t.squeeze(-1))

        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(TRAIN_BATCHES, 1)


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------
def save_checkpoint(net, optimizer, iteration, path):
    torch.save({
        "net_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }, path)


def load_checkpoint(net, optimizer, path):
    chk = torch.load(path, map_location="cpu")
    net.load_state_dict(chk["net_state"])
    if optimizer is not None:
        optimizer.load_state_dict(chk["optimizer_state"])
    return chk.get("iteration", 0)


def save_best_net(net):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, BEST_NET_FILE))


def load_best_net(net):
    path = os.path.join(CHECKPOINT_DIR, BEST_NET_FILE)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return True
    return False


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main(num_iterations=None):
    if num_iterations is None:
        num_iterations = NUM_ITERATIONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = AlphaZeroNet().to(device)
    best_net = AlphaZeroNet().to(device)

    # Try to load existing best network
    start_iter = 0
    if load_best_net(best_net):
        best_net.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, BEST_NET_FILE), map_location=device
        ))
        net.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, BEST_NET_FILE), map_location=device
        ))
        print("Loaded existing best network.")

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    for iteration in range(start_iter, num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # 1. Self-play
        print("Running self-play games...")
        for _ in tqdm(range(SELF_PLAY_GAMES), desc="Self-play"):
            experiences = self_play_game(net)
            replay_buffer.extend(experiences)

        print(f"Replay buffer size: {len(replay_buffer)}")

        # 2. Training
        if len(replay_buffer) >= BATCH_SIZE:
            print("Training...")
            avg_loss = train_iteration(net, replay_buffer, device)
            print(f"Average loss: {avg_loss:.4f}")

        # 3. Evaluation
        if iteration > 0:
            print("Evaluating new network vs best...")
            win_rate = pit(net, best_net, EVAL_GAMES, device)
            print(f"Win rate of new net: {win_rate:.3f}")

            if win_rate > WIN_THRESHOLD:
                print(">>> New network promoted! <<<")
                best_net.load_state_dict(net.state_dict())
                save_best_net(best_net)
            else:
                print("New network not promoted.")

        # 4. Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_checkpoint(net, torch.optim.Adam(net.parameters()), iteration,
                        os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{iteration+1}.pth"))
        save_best_net(best_net)

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlphaZero Connect 4")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations (overrides config)")
    args = parser.parse_args()
    main(num_iterations=args.iterations)
