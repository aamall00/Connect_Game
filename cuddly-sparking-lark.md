# Plan: AlphaZero Connect 4

## Context

Build a self-play AlphaZero-style AI for Connect 4, trained entirely on a laptop CPU/GPU. The pipeline mirrors DeepMind's AlphaZero: a dual-head ResNet generates policy + value estimates, Monte Carlo Tree Search (MCTS) uses these estimates to guide game-tree exploration during self-play, and the resulting game data trains the network iteratively. Target location: `Connect_Game/` (currently empty).

---

## Architecture Overview

```
Connect_Game/
├── game.py          # Connect 4 rules engine
├── model.py         # Dual-head ResNet (policy + value)
├── mcts.py          # PUCT-based MCTS
├── train.py         # Self-play loop + training
├── play.py          # Human vs AI / AI vs AI interface
├── config.py        # Hyperparameters (single source of truth)
├── evaluate.py      # Pit new vs old network, track Elo
└── requirements.txt
```

---

## Phase 1 — Game Engine (`game.py`)

**Class: `ConnectFour`**

State representation:
- `board`: `np.ndarray` shape `(2, 6, 7)` — plane 0 = current player's pieces, plane 1 = opponent's pieces (canonical form, always from current player's POV)
- `current_player`: `+1` or `-1`

Methods:
- `get_legal_moves() → list[int]` — columns 0-6 that are not full
- `make_move(col) → ConnectFour` — returns new state (immutable)
- `check_winner() → int` — returns `+1`, `-1`, or `0` (ongoing), `None` (draw)
- `get_canonical_state() → np.ndarray` — flips planes if `current_player == -1` so network always sees "my pieces" in plane 0
- `is_terminal() → bool`
- `__repr__()` — ASCII board for debugging

Win detection: check horizontal, vertical, diagonal (4-in-a-row).

---

## Phase 2 — Neural Network (`model.py`)

**Architecture: small ResNet**
- Input: `(3, 6, 7)` — 2 board planes + 1 constant plane (whose turn: all 1s = current player)
- Residual tower: 5 blocks × 64 filters, 3×3 convolutions, batch norm, ReLU
- Policy head: Conv(2 filters) → Flatten → Linear → softmax over 7 actions
- Value head: Conv(1 filter) → Flatten → Linear(64) → ReLU → Linear(1) → tanh

**Class: `AlphaZeroNet(nn.Module)`**
- `forward(x) → (policy_logits, value)` — raw logits (apply softmax/masking outside)
- `predict(state: np.ndarray) → (policy: np.ndarray, value: float)` — inference convenience method, handles batching + device

Loss functions:
- Policy: cross-entropy against MCTS visit-count distribution π
- Value: MSE against game outcome z ∈ {-1, 0, +1}
- Combined: `L = (z - v)² - π·log(p) + c·||θ||²` (weight decay = L2 reg)

---

## Phase 3 — MCTS (`mcts.py`)

**Class: `MCTSNode`**
```
N: visit count
W: total value
Q: mean value = W/N
P: prior probability from network
children: dict[action → MCTSNode]
```

**Class: `MCTS`**
- `__init__(game, net, config)` — holds root node, reuses tree across moves
- `search(num_simulations) → np.ndarray` — returns π (visit-count distribution)

**Per simulation:**
1. **Select** — traverse from root using PUCT score:  
   `U(s,a) = Q(s,a) + c_puct · P(s,a) · √(ΣN(s)) / (1 + N(s,a))`
2. **Expand** — at leaf: call `net.predict(state)`, create child nodes with priors P
3. **Backup** — propagate value up the path, alternating sign at each level

**Temperature τ:**
- First 30 moves: sample action proportional to `N^(1/τ)` with `τ=1` (exploration)
- After move 30: `τ→0` (argmax — deterministic best play)

**Dirichlet noise** at root: `P_root = (1-ε)·P_net + ε·Dir(α)` with `α=0.3, ε=0.25`

---

## Phase 4 — Training Loop (`train.py`)

**Self-play data format:** list of `(canonical_state, π, z)` tuples

**`SelfPlayWorker`:**
- Plays one full game using MCTS (400 simulations per move)
- Returns trajectory: `[(state, π, player), ...]`
- After game ends, assigns `z` to each state: `+1` if that player won, `-1` if lost, `0` if draw
- Augmentation: horizontal flip of board + π (doubles data for free)

**Replay buffer:** deque of last `500_000` examples

**Training iteration:**
1. Generate `N_self_play=100` games using current network
2. Add to replay buffer
3. Sample `N_batches=1000` mini-batches of size 512 from buffer
4. Train with Adam (lr=0.001, weight decay=1e-4)
5. Save checkpoint; run evaluation vs previous best
6. If win rate > 55% → promote new network as "best"

**`config.py` parameters:**
```python
NUM_ITERATIONS    = 100
SELF_PLAY_GAMES   = 100
MCTS_SIMULATIONS  = 400
BATCH_SIZE        = 512
TRAIN_BATCHES     = 1000
C_PUCT            = 1.0
DIRICHLET_ALPHA   = 0.3
DIRICHLET_EPSILON = 0.25
TEMP_THRESHOLD    = 30     # moves before τ→0
REPLAY_BUFFER_SIZE= 500_000
WIN_THRESHOLD     = 0.55
EVAL_GAMES        = 40
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
```

---

## Phase 5 — Evaluation (`evaluate.py`)

**`pit(net_new, net_old, n_games) → float`** — returns win rate of new vs old
- Play `n_games` games alternating sides; no temperature (τ=0), no Dirichlet noise
- Used inside training loop to decide whether to promote new network

Optional: track approximate Elo over iterations using a log file.

---

## Phase 6 — Human Interface (`play.py`)

- ASCII board rendered each turn
- Human selects column (0-6) via stdin
- AI uses best-network MCTS (200 simulations for interactive speed)
- Modes: `--human-vs-ai`, `--ai-vs-ai`, `--ai-vs-random` (sanity check)
- Load checkpoint by path argument

---

## Phase 7 — `requirements.txt`

```
torch>=2.0
numpy>=1.24
tqdm
```

No other dependencies needed.

---

## Implementation Order

| Step | File | Key deliverable |
|------|------|----------------|
| 1 | `config.py` | All hyperparameters in one place |
| 2 | `game.py` | Full rules engine with tests |
| 3 | `model.py` | ResNet forward pass + inference |
| 4 | `mcts.py` | PUCT search, Dirichlet noise, temperature |
| 5 | `train.py` | Self-play + training loop + checkpointing |
| 6 | `evaluate.py` | Network pit evaluation |
| 7 | `play.py` | Human vs AI interface |

---

## Verification

1. **Unit test game engine:** `python -c "from game import ConnectFour; ..."` — check win detection all 4 directions, legal moves at full board, canonical flip.
2. **Sanity check model:** random input through ResNet, verify output shapes `(7,)` and `(1,)`.
3. **MCTS smoke test:** run 10 simulations from start state, confirm π sums to 1 over legal moves.
4. **Overfit test:** train on a single state for 100 steps — value head should converge to ground truth.
5. **Random baseline:** AI vs random player should achieve >95% win rate within 10 iterations.
6. **Play test:** `python play.py --human-vs-ai` — game should be unwinnable for a human after ~50 iterations.
7. **Benchmark:** solved Connect 4 — first player wins with optimal play. After full training, AI should always win as player 1 vs a random opponent.
