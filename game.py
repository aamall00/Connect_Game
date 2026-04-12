"""
Connect 4 rules engine — immutable state representation for AlphaZero.

Board encoding (canonical, from current player's POV):
  plane 0: current player's pieces
  plane 1: opponent's pieces
  Shape: (2, ROWS, COLS)
"""

import numpy as np
import copy
from config import ROWS, COLS, WIN_LENGTH


class ConnectFour:
    """Immutable Connect 4 game state.

    Internal board (self.board) is ALWAYS from the perspective of player +1:
      plane 0: player +1's pieces
      plane 1: player -1's pieces

    Canonical form (for the network) is computed on-demand via get_canonical_state().
    """

    def __init__(self, board=None, current_player=1):
        """
        Args:
            board: np.ndarray shape (2, ROWS, COLS) or None for empty board.
                   If provided, must be from player +1's perspective.
            current_player: +1 or -1 (who moves next).
        """
        if board is None:
            self.board = np.zeros((2, ROWS, COLS), dtype=np.float32)
        else:
            self.board = board.copy()
        self.current_player = current_player

    # ------------------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------------------
    def get_legal_moves(self):
        """Return list of legal column indices (0..COLS-1)."""
        # A column is legal if any row is empty (check total occupancy)
        occupied = self.board[0] + self.board[1]  # both players' pieces
        return [c for c in range(COLS) if occupied[:, c].sum() < ROWS]

    # ------------------------------------------------------------------
    # Make a move — returns a NEW ConnectFour (immutable)
    # ------------------------------------------------------------------
    def make_move(self, col):
        """Place a piece in *col* and return a new ConnectFour state.

        The internal board is always from player +1's perspective.
        Raises ValueError if the column is full.
        """
        if col not in self.get_legal_moves():
            raise ValueError(f"Column {col} is full.")

        # Find the lowest empty row in this column
        occupied = self.board[0] + self.board[1]
        for row in range(ROWS):
            if occupied[row, col] == 0:
                break

        new_board = self.board.copy()
        # Place on the correct player's plane
        plane_idx = 0 if self.current_player == 1 else 1
        new_board[plane_idx, row, col] = 1

        # Board stays in player +1's perspective; just switch who moves
        return ConnectFour(board=new_board, current_player=-self.current_player)

    # ------------------------------------------------------------------
    # Winner / terminal detection  (works on raw board, not canonical)
    # ------------------------------------------------------------------
    def _get_raw_board(self):
        """Return raw board from perspective of player +1 (already stored that way)."""
        return self.board

    def check_winner(self):
        """Return +1 if player 1 wins, -1 if player 2 wins, 0 if ongoing, None if draw."""
        raw = self._get_raw_board()
        # plane 0 = player +1, plane 1 = player -1
        for player_idx in range(2):
            pieces = raw[player_idx]
            if self._has_four(pieces):
                return 1 if player_idx == 0 else -1
        # Draw?
        if self._is_full():
            return None
        return 0

    @staticmethod
    def _has_four(pieces):
        """Check if there are 4 contiguous pieces in any direction."""
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if pieces[r, c:c + WIN_LENGTH].sum() == WIN_LENGTH:
                    return True
        # Vertical
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS):
                if pieces[r:r + WIN_LENGTH, c].sum() == WIN_LENGTH:
                    return True
        # Diagonal /
        for r in range(WIN_LENGTH - 1, ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if pieces[r, c] and pieces[r - 1, c + 1] and pieces[r - 2, c + 2] and pieces[r - 3, c + 3]:
                    return True
        # Diagonal \
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS - WIN_LENGTH + 1):
                if pieces[r, c] and pieces[r + 1, c + 1] and pieces[r + 2, c + 2] and pieces[r + 3, c + 3]:
                    return True
        return False

    def _is_full(self):
        return not self.get_legal_moves()

    def is_terminal(self):
        return self.check_winner() != 0

    # ------------------------------------------------------------------
    # Canonical state for the neural net
    # ------------------------------------------------------------------
    def get_canonical_state(self):
        """Return np.ndarray (3, ROWS, COLS) ready for network input.

        The board is stored from player +1's perspective.
        If current_player == -1, we flip planes so the network always sees
        "my pieces" in plane 0.
        Plane 2: all ones if current_player == +1, else all zeros.
        """
        if self.current_player == 1:
            board = self.board
        else:
            board = self.board[::-1]  # swap planes
        turn_plane = np.full((ROWS, COLS), 1.0 if self.current_player == 1 else 0.0, dtype=np.float32)
        return np.vstack([board, turn_plane[np.newaxis, ...]])

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        raw = self._get_raw_board()
        lines = []
        lines.append("  0   1   2   3   4   5   6")
        for r in range(ROWS):
            row_str = "|"
            for c in range(COLS):
                if raw[0, r, c] == 1:
                    row_str += " X |"
                elif raw[1, r, c] == 1:
                    row_str += " O |"
                else:
                    row_str += "   |"
            lines.append(row_str)
        lines.append("+" + "---+" * COLS)
        winner = self.check_winner()
        if winner == 1:
            lines.append("Player X wins!")
        elif winner == -1:
            lines.append("Player O wins!")
        elif winner is None:
            lines.append("Draw!")
        return "\n".join(lines)
