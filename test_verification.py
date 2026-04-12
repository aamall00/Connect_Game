"""Verification tests for AlphaZero Connect 4."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from game import ConnectFour
from model import AlphaZeroNet
from mcts import MCTS


def test_game_engine():
    print("=" * 50)
    print("TEST 1: Game Engine")
    print("=" * 50)

    # Horizontal win
    g = ConnectFour()
    assert g.get_legal_moves() == list(range(7)), "Empty board should have 7 legal moves"
    assert g.check_winner() == 0, "Empty board: no winner"
    assert not g.is_terminal()

    # Vertical win: 4 pieces in column 0
    g2 = ConnectFour()
    g2 = g2.make_move(0)  # P1
    g2 = g2.make_move(3)  # P2
    g2 = g2.make_move(0)  # P1
    g2 = g2.make_move(3)  # P2
    g2 = g2.make_move(0)  # P1
    g2 = g2.make_move(3)  # P2
    g2 = g2.make_move(0)  # P1 wins
    assert g2.check_winner() == 1, f"Vertical win expected winner=1, got {g2.check_winner()}"
    print("  Vertical win: PASS")

    # Canonical state
    g3 = ConnectFour()
    g3 = g3.make_move(0)
    cs = g3.get_canonical_state()
    assert cs.shape == (3, 6, 7), f"Canonical shape expected (3,6,7), got {cs.shape}"
    print("  Canonical state shape: PASS")

    print("Game Engine: ALL PASS\n")


def test_model():
    print("=" * 50)
    print("TEST 2: Model")
    print("=" * 50)

    net = AlphaZeroNet()
    g = ConnectFour()
    state = g.get_canonical_state()

    policy, value = net.predict(state)
    assert policy.shape == (7,), f"Policy shape expected (7,), got {policy.shape}"
    assert abs(policy.sum() - 1.0) < 0.01, f"Policy should sum to ~1, got {policy.sum()}"
    assert -1.0 <= value <= 1.0, f"Value should be in [-1,1], got {value}"
    print(f"  Policy: {policy}")
    print(f"  Value: {value:.4f}")

    # Batched
    states = np.array([state, state])
    policy_b, value_b = net.predict(states)
    assert policy_b.shape == (2, 7), f"Batched policy shape expected (2,7), got {policy_b.shape}"
    print("  Batched inference: PASS")

    print("Model: ALL PASS\n")


def test_mcts():
    print("=" * 50)
    print("TEST 3: MCTS")
    print("=" * 50)

    net = AlphaZeroNet()
    g = ConnectFour()
    mcts = MCTS(g, net, add_dirichlet=False)
    pi = mcts.search(10)

    assert pi.shape == (7,), f"pi shape expected (7,), got {pi.shape}"
    assert abs(pi.sum() - 1.0) < 1e-5, f"pi should sum to 1, got {pi.sum()}"
    assert all(pi[c] >= 0 for c in range(7)), "All pi values should be non-negative"
    print(f"  pi = {pi}")
    print(f"  pi.sum() = {pi.sum():.6f}")
    print("  MCTS 10 sims: PASS")

    # Test select_action with temperature=0
    action = mcts.select_action(pi, temperature=0.0)
    assert action in g.get_legal_moves(), f"Action {action} should be legal"
    print(f"  select_action (tau=0): {action}")

    print("MCTS: ALL PASS\n")


def test_self_play():
    print("=" * 50)
    print("TEST 4: Self-play (short game)")
    print("=" * 50)

    from train import self_play_game
    net = AlphaZeroNet()
    trajectory = self_play_game(net, mcts_sims=20)
    print(f"  Trajectory length: {len(trajectory)}")
    state, pi, z = trajectory[0]
    assert state.shape == (3, 6, 7), f"State shape expected (3,6,7), got {state.shape}"
    assert pi.shape == (7,), f"pi shape expected (7,), got {pi.shape}"
    assert z in [-1, 0, 1], f"z should be -1, 0, or 1, got {z}"
    print(f"  Sample z: {z}")
    print("Self-play: ALL PASS\n")


if __name__ == "__main__":
    test_game_engine()
    test_model()
    test_mcts()
    test_self_play()
    print("=" * 50)
    print("ALL VERIFICATION TESTS PASSED!")
    print("=" * 50)
