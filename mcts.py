"""
PUCT-based Monte Carlo Tree Search for AlphaZero Connect 4.
"""

import numpy as np
import copy
from config import C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON


class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = ["N", "W", "Q", "P", "children", "terminal"]

    def __init__(self, P=0.0):
        self.N = 0          # visit count
        self.W = 0.0        # total value
        self.Q = 0.0        # mean value
        self.P = P          # prior probability
        self.children = {}  # action → MCTSNode
        self.terminal = False

    @property
    def expanded(self):
        return len(self.children) > 0


class MCTS:
    def __init__(self, game, net, config=None, add_dirichlet=True, temperature=1.0):
        """
        Args:
            game: ConnectFour instance (current state).
            net: AlphaZeroNet (eval mode).
            add_dirichlet: whether to add Dirichlet noise at root.
            temperature: initial temperature (not used during search, used when returning pi).
        """
        self.root = MCTSNode()
        self.game = game
        self.net = net
        self.add_dirichlet = add_dirichlet

    def search(self, num_simulations):
        """Run *num_simulations* MCTS iterations and return visit-count distribution π.

        Returns:
            pi: np.ndarray shape (COLS,) with probabilities over legal actions.
        """
        from config import COLS

        for _ in range(num_simulations):
            game = copy.deepcopy(self.game)
            self._simulate(self.root, game)

        # Build visit-count distribution
        pi = np.zeros(COLS, dtype=np.float32)
        for action, child in self.root.children.items():
            pi[action] = child.N

        # Mask to only legal moves
        legal = self.game.get_legal_moves()
        if sum(pi) > 0:
            pi = pi / pi.sum()
        else:
            # Fallback: uniform over legal moves
            for a in legal:
                pi[a] = 1.0 / len(legal)
        return pi

    def _simulate(self, node, game):
        """One complete simulation: select → expand → backup."""
        if node.terminal:
            return 0.0  # shouldn't happen if called correctly

        # ---- Selection ----
        while node.expanded and not node.terminal:
            action, node = self._select_child(node)
            game = game.make_move(action)

        winner = game.check_winner()
        if winner == 0:
            # Not terminal — expand
            value = self._expand_and_evaluate(node, game)
        else:
            # Terminal node
            node.terminal = True
            if winner is None:  # draw
                value = 0.0
            else:
                value = 1.0 if winner == game.current_player else -1.0

        # ---- Backup (value from perspective of the player at this node) ----
        self._backup(node, value)
        return value

    def _select_child(self, node):
        """Select child with highest PUCT score."""
        total_visits = sum(child.N for child in node.children.values())
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():
            score = self._puct_score(child, total_visits)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    @staticmethod
    def _puct_score(child, total_visits):
        """PUCT = Q + U."""
        q = child.Q
        u = C_PUCT * child.P * np.sqrt(total_visits) / (1 + child.N)
        return q + u

    def _expand_and_evaluate(self, node, game):
        """Expand leaf node using network, return value from network."""
        canonical = game.get_canonical_state()
        policy_probs, value = self.net.predict(canonical)

        legal_moves = game.get_legal_moves()

        # Mask policy to legal moves and renormalize
        masked_policy = np.zeros_like(policy_probs)
        for a in legal_moves:
            masked_policy[a] = policy_probs[a]
        if masked_policy.sum() > 0:
            masked_policy /= masked_policy.sum()
        else:
            # Uniform fallback
            for a in legal_moves:
                masked_policy[a] = 1.0 / len(legal_moves)

        # Add Dirichlet noise at root
        if node is self.root and self.add_dirichlet:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_moves))
            for i, a in enumerate(legal_moves):
                masked_policy[a] = (1 - DIRICHLET_EPSILON) * masked_policy[a] + DIRICHLET_EPSILON * noise[i]

        # Create children
        for a in legal_moves:
            node.children[a] = MCTSNode(P=masked_policy[a])

        return value

    def _backup(self, node, value):
        """Propagate value up the tree."""
        if node is not self.root and node.terminal:
            # Already handled during expansion
            pass
        node.N += 1
        node.W += value
        node.Q = node.W / node.N

    def select_action(self, pi, temperature=1.0):
        """Select an action from the visit-count distribution, optionally with temperature.

        Args:
            pi: visit-count distribution from search().
            temperature: τ=1 for exploration, τ→0 for argmax.
        Returns:
            action: int
        """
        legal = self.game.get_legal_moves()
        if not legal:
            return None

        if temperature == 0 or temperature < 1e-3:
            # Deterministic: pick most visited legal action
            return max(legal, key=lambda a: pi[a])

        # Sample with temperature
        probs = np.array([pi[a] for a in legal], dtype=np.float64)
        probs = probs ** (1.0 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(legal, p=probs)
