# Connect_Game Application Flow

## Overview

This project implements an AlphaZero-style Connect 4 system. The application has four main parts:

1. `game.py` defines the Connect 4 rules and board state.
2. `model.py` defines the neural network that predicts moves and board value.
3. `mcts.py` uses Monte Carlo Tree Search to choose actions with help from the network.
4. `train.py`, `evaluate.py`, and `play.py` run training, evaluation, and interactive gameplay.

## Core Flow

### 1. Game State Management

The flow starts in [`game.py`](./game.py):

- `ConnectFour()` creates an empty board.
- `get_legal_moves()` returns the columns where a move can still be played.
- `make_move(col)` returns a new game state after placing a piece.
- `check_winner()` determines whether player `+1`, player `-1`, or nobody has won.
- `get_canonical_state()` converts the board into a 3-channel tensor for the model.

Important detail:

- The board is stored internally from player `+1`'s perspective.
- The canonical state is flipped when needed so the network always sees the current player as "self".

### 2. Neural Network Prediction

The model is defined in [`model.py`](./model.py) as `AlphaZeroNet`.

Input:

- Shape: `(3, ROWS, COLS)`
- Channel 0: current player's pieces
- Channel 1: opponent's pieces
- Channel 2: turn indicator plane

Output:

- `policy_logits`: scores for each of the 7 columns
- `value`: estimated outcome of the position in the range `[-1, 1]`

`predict()` is the main helper used by MCTS. It:

- converts NumPy input to a PyTorch tensor
- runs inference
- returns softmax policy probabilities and a scalar value

### 3. Move Selection with MCTS

[`mcts.py`](./mcts.py) wraps the network in search logic.

For each move:

1. `MCTS(game, net)` starts a tree from the current board.
2. `search(num_simulations)` runs repeated simulations.
3. `_select_child()` chooses promising child nodes using the PUCT formula.
4. `_expand_and_evaluate()` calls the neural network on the leaf state.
5. The returned policy is masked to legal moves only.
6. Child nodes are created with those prior probabilities.
7. `_backup()` updates visit counts and values.
8. `select_action()` converts visit counts into the final chosen move.

Training-time behavior:

- Dirichlet noise is added at the root for exploration.

Evaluation/play-time behavior:

- No Dirichlet noise.
- Temperature is usually `0.0` for deterministic best-move selection.

## Training Flow

The main pipeline lives in [`train.py`](./train.py).

### Step 1. Self-Play

`self_play_game(net)` runs one full game:

1. Start with a fresh `ConnectFour` board.
2. For each turn, run MCTS using the current network.
3. Store:
   - canonical board state
   - MCTS visit distribution `pi`
   - player who made the move
4. Apply the chosen action and continue until terminal state.
5. After the game ends, assign outcome labels:
   - `1` for positions belonging to the winner
   - `-1` for positions belonging to the loser
   - `0` for draws
6. Augment the data with horizontal flips.

Each training sample becomes:

- `state`
- `pi`
- `z`

### Step 2. Replay Buffer

Self-play samples are added into a replay buffer:

- implemented with `deque(maxlen=REPLAY_BUFFER_SIZE)`
- old samples are automatically dropped when the buffer is full

### Step 3. Network Training

`train_iteration(net, replay_buffer, device)`:

1. Randomly samples mini-batches from the replay buffer.
2. Runs the network forward pass.
3. Computes two losses:
   - policy loss from MCTS target distribution
   - value loss from game outcome target
4. Adds them together and updates the model using Adam.

### Step 4. Evaluation and Promotion

After training, the current network is compared against the saved best network.

- `pit(net, best_net, EVAL_GAMES, device)` from [`evaluate.py`](./evaluate.py) plays multiple games.
- The networks alternate first-player advantage.
- If the new network's win rate is above `WIN_THRESHOLD`, it becomes the new best model.

### Step 5. Checkpointing

Training saves:

- iteration checkpoints in `checkpoints/checkpoint_iter_*.pth`
- the best model in `checkpoints/best_net.pth`

## Evaluation Flow

[`evaluate.py`](./evaluate.py) is responsible for head-to-head testing.

Flow:

1. Load two networks.
2. Start a game with `ConnectFour()`.
3. On each turn, choose the active network based on `current_player`.
4. Use MCTS with deterministic action selection.
5. Repeat until the game ends.
6. Return win/draw/loss result.
7. Aggregate scores across many games to compute win rate.

## Interactive Play Flow

[`play.py`](./play.py) provides three modes:

- Human vs AI
- AI vs AI
- AI vs random

Shared flow:

1. Load a checkpointed model with `load_net()`.
2. Create a `ConnectFour` game.
3. On each turn, choose the move source:
   - human input
   - MCTS-powered AI
   - random move
4. Apply the move with `make_move()`.
5. Print the board.
6. Stop when `is_terminal()` becomes true and display the result.

## Configuration Flow

[`config.py`](./config.py) is the central place for hyperparameters and settings:

- board dimensions
- network size
- MCTS constants
- self-play counts
- training batch sizes
- evaluation thresholds
- checkpoint paths

Most of the application imports values from this file, so changing `config.py` changes behavior across the project.

## Quick End-to-End Summary

The full application loop is:

1. Start from the game rules in `game.py`.
2. Convert each board into canonical model input.
3. Use `AlphaZeroNet` to predict policy and value.
4. Use MCTS to improve move selection.
5. Generate self-play data.
6. Train the network on that data.
7. Evaluate the new network against the best saved network.
8. Promote and save the model if it performs better.
9. Use the saved model in `play.py` for interactive games.

## Why AlphaZeroNet And MCTS Work Together

The main intuition behind the architecture is that the neural network and MCTS solve different parts of the problem.

`AlphaZeroNet` is fast and gives an immediate estimate for a position:

- `policy` says which moves look promising
- `value` says how good the board looks for the current player

But the network alone is only an approximation. It does not explicitly search ahead through many move sequences before choosing.

MCTS adds that search process. Starting from the current board, it explores future move sequences and uses the network to guide which branches deserve attention.

So the partnership works like this:

- the network provides intuition
- MCTS provides look-ahead reasoning

This combination is useful because:

- without the network, MCTS would have to explore too many weak moves
- without MCTS, the network would rely only on its first guess

Together they improve each other:

- the network helps MCTS focus search on promising moves
- MCTS improves move selection beyond the network's raw prediction
- the improved MCTS move distribution becomes the training target for the policy head
- the final game result becomes the training target for the value head

Over time, this creates a learning loop:

1. the current network guides self-play search
2. MCTS produces stronger move targets than the raw network alone
3. the network trains on those improved targets
4. the improved network guides stronger future searches

So the system becomes stronger through repeated cycles of:

- prediction
- search
- self-play
- training
