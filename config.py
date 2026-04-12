"""
Hyperparameters — single source of truth for AlphaZero Connect 4.
"""

# === Self-play & Training ===
NUM_ITERATIONS = 100          # total training iterations
SELF_PLAY_GAMES = 100         # games per iteration
MCTS_SIMULATIONS = 400        # MCTS sims per move during self-play
BATCH_SIZE = 512              # mini-batch size
TRAIN_BATCHES = 1000          # training steps per iteration
REPLAY_BUFFER_SIZE = 500_000  # max experiences kept in memory

# === MCTS ===
C_PUCT = 1.0                  # exploration constant
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMP_THRESHOLD = 30            # move number after which τ → 0
TEMP_START = 1.0               # initial temperature

# === Evaluation ===
WIN_THRESHOLD = 0.55          # new net promoted if win rate > this
EVAL_GAMES = 40               # games played to evaluate new net

# === Optimizer ===
LR = 1e-3
WEIGHT_DECAY = 1e-4

# === Model Architecture ===
NUM_RES_BLOCKS = 5
NUM_FILTERS = 64

# === Board ===
ROWS = 6
COLS = 7
WIN_LENGTH = 4

# === Checkpointing ===
CHECKPOINT_DIR = "checkpoints"
BEST_NET_FILE = "best_net.pth"
