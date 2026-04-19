# MCTS Flow In Connect_Game

## Overview

This project uses Monte Carlo Tree Search (MCTS) to choose moves from the current Connect 4 board position.

The search starts from the current board and uses the neural network (`AlphaZeroNet`) to:

- suggest which moves look promising (`policy`)
- estimate how good a position is (`value`)

MCTS then turns those predictions into a stronger move distribution by running many simulations.

## Main Entry Point

The process starts in [`mcts.py`](./mcts.py) when an `MCTS` object is created:

- `game` is the current `ConnectFour` board state
- `net` is the current `AlphaZeroNet`
- `root` is the search-tree node representing the current board

Then `search(num_simulations)` is called to run multiple simulations from that root position.

## High-Level Search Loop

For each simulation, MCTS does the following:

1. Copy the current board state.
2. Start from the root node.
3. Repeatedly select the most promising child node.
4. Stop when it reaches a leaf node or a terminal position.
5. If the leaf is not terminal, call the neural network.
6. Expand the node by creating children for legal moves.
7. Use the returned value to update node statistics.
8. Repeat this process many times.

After all simulations are complete, MCTS converts the root children visit counts into a probability distribution `pi`.

## Step 1. Root Node Represents The Current Board

When `MCTS(game, net)` is created:

- `self.game` stores the current position
- `self.root` starts as an empty `MCTSNode`

At this point:

- no moves have been expanded yet
- no visit counts have been collected yet
- no values have been backed up yet

The root node is just a placeholder for the current board state.

## Step 2. Each Simulation Starts From A Fresh Copy

Inside `search(num_simulations)`, each simulation begins by copying the current board.

This matters because:

- each simulation explores one hypothetical future path
- the original root game state should not be changed

So every simulation starts clean from the same board position, then explores one path through the tree.

## Step 3. Selection

Selection means walking down the already-expanded part of the tree.

As long as the current node already has children, MCTS chooses one child and applies that move to the copied game state.

The child is chosen using the PUCT score:

`PUCT = Q + U`

Where:

- `Q` = the child node's average value from previous simulations
- `U` = an exploration bonus

The exploration bonus depends on:

- `P`, the prior probability from the neural network
- `N`, the child visit count
- total visits across the parent node's children

This creates a balance:

- high `Q` favors moves that have performed well so far
- high `U` favors moves that look promising but have not been explored much yet

Important clarification:

- MCTS selection is based on `Q + U`, not `Q + W`
- `W` is only the running total of backed-up values
- `Q` is derived from `W` as the average value: `Q = W / N`

So MCTS does not always follow the currently best-known move. It also keeps testing alternatives.

## Step 4. Reach A Leaf Or Terminal Position

Selection stops when one of these happens:

- the current node has no children yet, so it is a leaf
- the position is terminal, meaning the game is already over

At that point, MCTS decides how to evaluate the position.

## Step 5. If Terminal, Assign The Value Directly

If the copied game state is already finished:

- draw -> value `0.0`
- win for the side to evaluate -> value `1.0`
- loss for the side to evaluate -> value `-1.0`

No neural-network call is needed for a terminal state because the result is already known exactly.

## Step 6. If Leaf And Non-Terminal, Use AlphaZeroNet

If the leaf is not terminal, MCTS calls:

- `game.get_canonical_state()`
- `self.net.predict(canonical)`

This gives two outputs:

- `policy_probs`: the network's move probabilities for the current player
- `value`: the network's estimate of how favorable the position is

The canonical state is important because the network always sees the board from the perspective of the player whose turn it is.

## Step 7. How The Policy Output Is Used

The policy output is not used as the final answer directly.

Instead, MCTS uses it to initialize the children of the leaf node.

The flow is:

1. Get the legal moves from the game state.
2. Mask out illegal columns from `policy_probs`.
3. Renormalize the remaining probabilities so they sum to 1.
4. Create one child node per legal move.
5. Store the probability for each move as that child node's prior `P`.

This means the network gives search an informed starting guess:

- moves with larger priors get explored more readily
- moves with very small priors are still possible, but less favored

Important detail:

- leaf expansion only creates the immediate next child nodes from that position
- it does not expand grandchildren or deeper descendants in the same step

So expansion is one ply at a time. The tree grows gradually across simulations:

1. a leaf is expanded into its direct legal moves
2. later simulations may descend into one of those children
3. that child can then become the next leaf to expand

This is why the search tree becomes deeper over time instead of being built all at once.

## Step 8. Dirichlet Noise At The Root

During self-play, this implementation adds Dirichlet noise to the root priors.

That only happens:

- at the root node
- when `add_dirichlet=True`

This encourages exploration in training games so the model does not keep repeating the same openings too confidently.

During evaluation or human play, this noise is usually disabled.

## Step 9. How The Value Output Is Used

The `value` returned by the network is the evaluation of the reached leaf position.

MCTS uses that value to update search statistics:

- `N`: number of times the node has been visited
- `W`: accumulated value
- `Q`: average value, computed as `W / N`

This is how search gradually learns which branches of the tree appear stronger.

In this implementation, the update happens in `_backup()`:

1. `N` is incremented by 1
2. `W` adds the new simulation value
3. `Q` is recomputed as the running mean

So the formula is:

- `Q = W / N`

Example:

- after one backup with value `0.8`: `N = 1`, `W = 0.8`, `Q = 0.8`
- after a second backup with value `0.2`: `N = 2`, `W = 1.0`, `Q = 0.5`

So `Q` is simply the average of all values that have been backed up into that node.

Important clarification:

- the leaf `policy` does not directly update `Q`
- the leaf `policy` is used to initialize child priors `P`
- the leaf `value` is what updates `N`, `W`, and `Q`

So the responsibilities are split clearly:

- `policy` influences future selection through the `U` term
- `value` influences node quality through the `Q` term

Over many simulations:

- good branches tend to keep attractive `Q` values
- weak branches tend to become less attractive

## Step 10. Repeating Simulations Improves The Move Distribution

One simulation gives only a tiny amount of information.

Many simulations together produce a stronger signal because:

- the network keeps proposing priors for new leaves
- PUCT keeps balancing exploration and exploitation
- node statistics keep accumulating evidence

Eventually, the root node's children have different visit counts depending on how often search preferred them.

## Step 11. Building The Final Policy Distribution `pi`

After all simulations are finished, MCTS builds `pi` from the root children.

For each legal move:

- `pi[action] = child.N`

So `pi` is based on visit counts, not directly on the raw neural-network output.

Then the counts are normalized into probabilities.

This is the key idea:

- the network gives the initial guidance
- MCTS refines that guidance through search
- the final distribution `pi` reflects the result of search, not just the original guess

## Step 12. Choosing The Actual Move

Once `pi` has been built, the move is selected from that distribution.

This implementation supports two styles:

- exploratory selection: sample from `pi` using temperature
- deterministic selection: choose the move with the highest probability

Typical usage:

- self-play uses more exploration early in the game
- evaluation and gameplay usually use deterministic selection

## Why `pi` Is Used As The Policy Target During Training

The network is not trained to copy only the final chosen move.

Instead, it is trained to match `pi`, the visit-count distribution from MCTS.

That makes the policy target stronger because it comes from:

- the network's prior knowledge
- plus many search simulations from the current board

So training teaches the network to imitate the improved search policy.

## End-To-End Summary

Starting from the current board position, the MCTS flow is:

1. Create a root node for the current board.
2. Run many simulations.
3. In each simulation, follow child nodes using PUCT.
4. Stop at a new leaf or terminal state.
5. If terminal, assign the exact result as the value.
6. If not terminal, call `AlphaZeroNet`.
7. Use `policy` to create child priors.
8. Use `value` to update node statistics.
9. Repeat until all simulations are done.
10. Convert root visit counts into `pi`.
11. Choose a move from `pi`.

## Important Note About This Implementation

This repository's MCTS is conceptually AlphaZero-style, but the current `_backup()` implementation only updates the current node rather than propagating values back through the full simulation path.

So the overall structure matches AlphaZero:

- network-guided search
- priors from policy
- evaluation from value
- final move distribution from visit counts

But the backup stage is simpler than a full production AlphaZero implementation.

In a full AlphaZero-style implementation, the simulation value is usually propagated through every node on the selected path, often with alternating sign because players take turns. Here, `Q` is updated only on the node passed into `_backup()`, so the value update is more local than standard AlphaZero MCTS.
