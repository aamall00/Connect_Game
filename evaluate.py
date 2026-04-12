"""
Evaluation — pit two networks against each other.

Usage (standalone):
    python evaluate.py --net_a checkpoint_a.pth --net_b checkpoint_b.pth --games 40
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm

from config import MCTS_SIMULATIONS, ROWS, COLS
from game import ConnectFour
from model import AlphaZeroNet
from mcts import MCTS


def play_one_game(net_a, net_b, device, mcts_sims=MCTS_SIMULATIONS):
    """Play one game: net_a goes first.

    Returns:
        +1 if net_a wins, -1 if net_b wins, 0 if draw.
    """
    game = ConnectFour()
    nets = [net_a, net_b]  # index 0 = player +1, index 1 = player -1

    while not game.is_terminal():
        # Determine which net is moving
        idx = 0 if game.current_player == 1 else 1
        current_net = nets[idx]

        mcts = MCTS(game, current_net, add_dirichlet=False)
        pi = mcts.search(mcts_sims)
        action = mcts.select_action(pi, temperature=0.0)  # deterministic
        if action is None:
            break
        game = game.make_move(action)

    winner = game.check_winner()
    if winner is None:
        return 0
    # winner is from perspective of raw players (+1 or -1)
    # net_a is player +1, so:
    return winner  # +1 = net_a wins, -1 = net_b wins


def pit(net_new, net_old, n_games, device):
    """Play n_games alternating sides. Return win rate of net_new.

    net_new plays as +1 in even games, -1 in odd games.
    Returns: fraction of games net_new wins (draws = 0.5).
    """
    net_new.eval()
    net_old.eval()

    score = 0.0
    for i in tqdm(range(n_games), desc="Pitting"):
        if i % 2 == 0:
            result = play_one_game(net_new, net_old, device)
            # result: +1 = net_new wins, -1 = net_old wins
            if result == 1:
                score += 1.0
            elif result == 0:
                score += 0.5
        else:
            result = play_one_game(net_old, net_new, device)
            # result: +1 = net_old wins, -1 = net_new wins
            if result == -1:
                score += 1.0
            elif result == 0:
                score += 0.5

    return score / n_games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pit two networks against each other")
    parser.add_argument("--net_a", type=str, required=True, help="Path to network A checkpoint")
    parser.add_argument("--net_b", type=str, required=True, help="Path to network B checkpoint")
    parser.add_argument("--games", type=int, default=40, help="Number of games to play")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_a = AlphaZeroNet().to(device)
    net_b = AlphaZeroNet().to(device)

    net_a.load_state_dict(torch.load(args.net_a, map_location=device))
    net_b.load_state_dict(torch.load(args.net_b, map_location=device))

    win_rate = pit(net_a, net_b, args.games, device)
    print(f"Network A win rate: {win_rate:.3f}")
