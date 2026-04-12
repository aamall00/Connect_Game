"""
Human vs AI / AI vs AI interface for AlphaZero Connect 4.

Usage:
    python play.py --human-vs-ai          # human plays first
    python play.py --human-vs-ai --ai-first  # AI plays first
    python play.py --ai-vs-ai             # watch two AIs play
    python play.py --ai-vs-random         # AI vs random player (sanity check)
    python play.py --human-vs-ai --checkpoint checkpoints/best_net.pth
"""

import argparse
import torch
import numpy as np
import random
import sys
import os

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MCTS_SIMULATIONS, CHECKPOINT_DIR, BEST_NET_FILE
from game import ConnectFour
from model import AlphaZeroNet
from mcts import MCTS


def load_net(path=None, device=None):
    """Load a network from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet().to(device)
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, BEST_NET_FILE)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        print(f"Loaded network from {path}")
    else:
        print("No checkpoint found. Using untrained network.")
    return net


def ai_move(game, net, mcts_sims=200):
    """Return the best action according to MCTS."""
    mcts = MCTS(game, net, add_dirichlet=False)
    pi = mcts.search(mcts_sims)
    return mcts.select_action(pi, temperature=0.0)


def random_move(game):
    """Return a random legal move."""
    return random.choice(game.get_legal_moves())


def human_move(game):
    """Prompt human for a column choice."""
    legal = game.get_legal_moves()
    while True:
        try:
            col = int(input(f"Select column {legal}: "))
            if col in legal:
                return col
            print(f"Invalid move. Choose from {legal}.")
        except ValueError:
            print("Please enter a number.")


def human_vs_ai(net, human_first=True, mcts_sims=200):
    """Interactive game: human vs AI."""
    device = next(net.parameters()).device
    game = ConnectFour()

    print("\n=== Connect 4: Human vs AI ===")
    print("You are X, AI is O")
    print(game)

    while not game.is_terminal():
        if game.current_player == 1 and human_first or game.current_player == -1 and not human_first:
            # Human's turn (human is always +1)
            col = human_move(game)
        else:
            # AI's turn
            print("AI is thinking...")
            col = ai_move(game, net, mcts_sims)
            print(f"AI plays column {col}")

        game = game.make_move(col)
        print(game)

    # Result
    winner = game.check_winner()
    if winner is None:
        print("\nIt's a draw!")
    elif winner == 1:
        if human_first:
            print("\nYou win! (Or the AI let you win...)")
        else:
            print("\nAI wins!")
    else:
        if human_first:
            print("\nAI wins!")
        else:
            print("\nYou win!")


def ai_vs_ai(mcts_sims=200):
    """Watch two instances of the AI play each other."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(device=device)

    game = ConnectFour()
    print("\n=== Connect 4: AI vs AI ===")
    print(game)

    while not game.is_terminal():
        col = ai_move(game, net, mcts_sims)
        player = "X" if game.current_player == 1 else "O"
        print(f"Player {player} plays column {col}")
        game = game.make_move(col)
        print(game)

    winner = game.check_winner()
    if winner is None:
        print("\nIt's a draw!")
    elif winner == 1:
        print("\nPlayer X (AI) wins!")
    else:
        print("\nPlayer O (AI) wins!")


def ai_vs_random(mcts_sims=200):
    """AI vs random player — sanity check."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(device=device)

    game = ConnectFour()
    print("\n=== Connect 4: AI vs Random ===")
    print("AI is X, Random is O")
    print(game)

    while not game.is_terminal():
        if game.current_player == 1:
            col = ai_move(game, net, mcts_sims)
            print(f"AI plays column {col}")
        else:
            col = random_move(game)
            print(f"Random plays column {col}")
        game = game.make_move(col)
        print(game)

    winner = game.check_winner()
    if winner is None:
        print("\nIt's a draw!")
    elif winner == 1:
        print("\nAI wins!")
    else:
        print("\nRandom wins! (This shouldn't happen after training.)")


def main():
    parser = argparse.ArgumentParser(description="Play Connect 4 against AlphaZero AI")
    parser.add_argument("--human-vs-ai", action="store_true", help="Human vs AI")
    parser.add_argument("--ai-vs-ai", action="store_true", help="Watch AI play itself")
    parser.add_argument("--ai-vs-random", action="store_true", help="AI vs random player")
    parser.add_argument("--ai-first", action="store_true", help="AI goes first in human vs AI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to network checkpoint")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations per move")
    args = parser.parse_args()

    if args.human_vs_ai:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load_net(path=args.checkpoint, device=device)
        human_vs_ai(net, human_first=not args.ai_first, mcts_sims=args.mcts_sims)
    elif args.ai_vs_ai:
        ai_vs_ai(mcts_sims=args.mcts_sims)
    elif args.ai_vs_random:
        ai_vs_random(mcts_sims=args.mcts_sims)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
