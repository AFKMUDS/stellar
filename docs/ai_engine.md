---
name: AI Engine
scope: stellar
status: living_document
---

# AI Engine (Living Notes)

## Purpose

## Core Objects

## Inference-Time Control (Backspace / Editing)

## Commitment / Anchor Concept

## Banned-Next Sets (Avoid Repeating Failures)

## Hint Injection (Inverse / Reverse Search)

## Symmetry Expansion

### Idea

Work at the byte level (0-255), but provide the model with a structured view of each byte by exposing a fixed family of bitwise symmetries. The intent is that the network does less "discovering" of equivalent transforms and more "routing/alignment" of what it already sees.

Treat a byte as an 8-bit vector:

```
 b ∈ {0..255}
 bits(b) ∈ {0,1}^8
```

Define these transforms on bytes:

- `inv(b) = b XOR 0xFF`
- `rev(b)` = bit-reversal (mirror bits)
- `rotl_k(b)` = rotate-left by k bits (k=1..7)

### Feature construction

When `--use_symmetry_expansion` is enabled, the raw 8-bit representation is expanded into a concatenation of multiple symmetry views:

```
 φ(b) = concat(
   bits(b),
   bits(inv(b)),
   bits(rev(b)),
   bits(rotl_1(b)), ... , bits(rotl_7(b))
 )
```

This yields 10 * 8 = 80 binary features per byte.

The model then learns a linear projection into the model dimension:

```
 x_emb = W_bit · φ(b) + bias
```

Notes:

- The symmetry family is fixed; learning is in `W_bit` and downstream dynamics.
- The symmetries are not assumed to be "semantic" in the human language sense; they provide a dense, multi-view coordinate system over bytes.

### Why this matters (engine intuition)

In a conventional token embedding, the model must learn many equivalences implicitly. Here, the model gets multiple coordinate charts of the same byte upfront.

The working hypothesis:

- The hard part becomes deciding **when/where** a symmetry-aligned feature should dominate (routing), not learning the transform itself.
- This supports the "already has the pieces" viewpoint: the set of transforms is predefined; the model learns which transform view to use to stay on-manifold.

### Connection to "drift" / "same frequency"

Small perturbations in the prompt (e.g., single-byte corruption) should ideally produce *small* changes in:

- the hidden state trajectory
- the next-byte distribution

Symmetry expansion gives the model a richer local basis to represent near-misses and structural relationships, which can reduce brittle sensitivity to tiny byte-level changes.

## Retrieval (Symmetry / Path)

### Concept

Retrieval is a causal, differentiable-by-forward-pass mechanism that injects a summary of *previous states* when the current byte is related (under a transform family) to earlier bytes.

The key idea is: "if I've seen a transformed neighbor of this byte before, pull that past state forward as extra context".

There are two retrieval families:

### Symmetry retrieval

When `--use_symmetry_retrieval` is enabled, for each timestep `t` (after the first), define:

- `b_t` = current byte
- candidate transforms: `inv(b_t)` and `rev(b_t)`

Maintain a causal table `last_pos[byte] = most recent timestep where that byte appeared`.

If `inv(b_t)` or `rev(b_t)` appeared earlier, gather their saved states from the history buffer `state_hist` and average them:

```
 retrieval_t = mean( state_hist[last_pos[inv(b_t)]], state_hist[last_pos[rev(b_t)]] )
```

If neither exists, `retrieval_t = 0`.

Then the controller input becomes:

```
 ctrl_in = concat(x_emb(b_t), state_{t-1}, retrieval_t)
```

This is causal (only looks back) and acts like a "Ctrl+F" over symmetry-related bytes.

### Path retrieval (byte-move graph)

When `--use_path_retrieval` is enabled, bytes are nodes in a graph whose edges are the primitive moves:

- rotate, invert, reverse, increment, decrement

Precompute a k-hop neighbor table for each byte: `neigh(b)` and corresponding graph distance `dist(b, n)`.

At timestep `t`, find which neighbors of `b_t` appeared earlier, gather their states, and compute a distance-decayed weighted average:

```
 w(n) = exp(-α · dist(b_t, n))
 retrieval_t = Σ_n w(n) · state_hist[last_pos[n]] / Σ_n w(n)
```

### Reserved bytes

Certain tag bytes are excluded from retrieval/transforms to keep structural control tokens atomic.

### Engine-level role

Retrieval is a fast, local mechanism to "pull" earlier compatible structure forward.

In the visual language:

- it is a built-in tendency to snap the red dip back toward a green manifold by reintroducing a previously stable state that matched under a transform.
- it provides an explicit pathway for alignment that does not require rediscovering the relation through long-range recurrence alone.

### Failure modes / caveats

- Retrieval can hallucinate false continuity when the transform relation is not semantically relevant.
- Retrieval is a bias, not a guarantee: it can help stabilize or it can amplify spurious associations depending on the dataset and learned controller.

## Delta/Eraser (Entropy-Driven Overwrite)

## Training-Time: Consistency / "Same Frequency" Stabilization

## Metrics / Telemetry

## Open Questions
