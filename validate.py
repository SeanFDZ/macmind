#!/usr/bin/env python3
"""MacMind Python Validator — ground-truth reference implementation.

Implements the exact single-layer single-head transformer from the architecture
doc using NumPy.  Trains on the bit-reversal permutation and outputs test
vectors for manual comparison in HyperCard's Message Box.

Usage:  python3 test/validate.py
Requires:  numpy

Acceptance criteria:
  - Asserts parameter count == 1,216 at startup
  - Initial loss is ~2.5 (mean cross-entropy per position; with Xavier init
    the logits are not near-zero so probabilities are not perfectly uniform --
    2.302585 = -ln(0.1) would require uniform outputs)
  - Reaches 100% accuracy well before step 3,000
  - Loss decreases on a rolling 100-step average
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB      = 10   # digits 0-9
SEQ_LEN    = 8
D_MODEL    = 16
NUM_HEADS  = 1
LR         = 0.01
TRAIN_STEPS = 3000
SEED       = 42

# Bit-reversal permutation:  output[i] = input[PERM[i]]
PERM = [0, 4, 2, 6, 1, 5, 3, 7]

# ---------------------------------------------------------------------------
# Xavier uniform initialization
# ---------------------------------------------------------------------------

def xavier_init(rng, fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))

# ---------------------------------------------------------------------------
# Softmax (numerically stable, per-row)
# ---------------------------------------------------------------------------

def softmax(x):
    """Row-wise softmax for a 2-D array."""
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)

# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def init_model(rng):
    W = {}
    W["W_embed"] = xavier_init(rng, VOCAB, D_MODEL)     # [10 x 16]
    W["W_pos"]   = xavier_init(rng, SEQ_LEN, D_MODEL)   # [8 x 16]
    W["W_Q"]     = xavier_init(rng, D_MODEL, D_MODEL)    # [16 x 16]
    W["W_K"]     = xavier_init(rng, D_MODEL, D_MODEL)    # [16 x 16]
    W["W_V"]     = xavier_init(rng, D_MODEL, D_MODEL)    # [16 x 16]
    W["W_out"]   = xavier_init(rng, D_MODEL, VOCAB)      # [16 x 10]

    total = sum(w.size for w in W.values())
    assert total == 1216, f"Parameter count {total} != 1216"

    return W

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(W, digits):
    """Run the full forward pass.  Returns dict of all activations."""
    act = {}

    # 1. Token embedding + position embedding -> act_embedded [8 x 16]
    token_emb = W["W_embed"][digits]           # [8 x 16]
    pos_emb   = W["W_pos"]                     # [8 x 16]
    act["act_embedded"] = token_emb + pos_emb  # [8 x 16]

    # 2. Q, K, V projections
    act["act_Q"] = act["act_embedded"] @ W["W_Q"]  # [8 x 16]
    act["act_K"] = act["act_embedded"] @ W["W_K"]  # [8 x 16]
    act["act_V"] = act["act_embedded"] @ W["W_V"]  # [8 x 16]

    # 3. Attention scores = Q x K^T, scale by 1/sqrt(d_model) = 0.25
    scores = act["act_Q"] @ act["act_K"].T          # [8 x 8]
    scores_scaled = scores * 0.25
    act["act_attn"] = softmax(scores_scaled)         # [8 x 8]

    # 4. Context = attn x V
    act["act_context"] = act["act_attn"] @ act["act_V"]  # [8 x 16]

    # 5. Residual = context + embedded
    act["act_residual"] = act["act_context"] + act["act_embedded"]  # [8 x 16]

    # 6. Logits = residual x W_out
    act["act_logits"] = act["act_residual"] @ W["W_out"]  # [8 x 10]

    # 7. Softmax per row -> probabilities
    act["act_probs"] = softmax(act["act_logits"])  # [8 x 10]

    return act

# ---------------------------------------------------------------------------
# Loss and accuracy
# ---------------------------------------------------------------------------

def compute_loss(act, targets):
    """Cross-entropy loss (mean over positions) and per-position accuracy.

    The displayed loss is averaged over SEQ_LEN positions.  This does NOT
    affect gradients -- grad_logits = probs - one_hot(target) is the
    gradient of the SUM, not the mean.  Dividing by 8 here is purely for
    display; the gradient stays at full scale.
    """
    probs = act["act_probs"]
    # Cross-entropy: -sum_t ln(probs[t, target[t]]) / SEQ_LEN
    loss = 0.0
    for t in range(SEQ_LEN):
        loss -= np.log(probs[t, targets[t]])
    loss /= SEQ_LEN

    # Accuracy: fraction of positions where argmax matches target
    preds = np.argmax(probs, axis=1)
    acc = int(np.sum(preds == targets) * 100 // SEQ_LEN)

    return loss, acc

# ---------------------------------------------------------------------------
# Backward pass  (explicit, no autograd)
# ---------------------------------------------------------------------------

def backward(W, act, digits, targets):
    """Compute all gradients per docs/backprop-math.md."""
    grad = {}

    # Step 1 — Output softmax + cross-entropy gradient
    grad_logits = act["act_probs"].copy()  # [8 x 10]
    for t in range(SEQ_LEN):
        grad_logits[t, targets[t]] -= 1.0
    grad["grad_logits"] = grad_logits

    # Step 2 — Output projection W_out
    #   grad_W_out   = act_residual^T x grad_logits     [16 x 10]
    #   grad_residual = grad_logits x W_out^T            [8 x 16]
    grad["grad_W_out"]   = act["act_residual"].T @ grad_logits
    grad["grad_residual"] = grad_logits @ W["W_out"].T

    # Step 3 — Residual connection
    #   grad_embedded starts as a copy of grad_residual
    grad_embedded = grad["grad_residual"].copy()  # [8 x 16]

    # Step 4 — Value projection and attention weights
    #   grad_attn = grad_residual x act_V^T              [8 x 8]
    #   grad_V    = act_attn^T x grad_residual           [8 x 16]
    #   grad_W_V  = act_embedded^T x grad_V              [16 x 16]
    #   grad_embedded += grad_V x W_V^T                  (V path)
    grad["grad_attn"] = grad["grad_residual"] @ act["act_V"].T
    grad["grad_V"]    = act["act_attn"].T @ grad["grad_residual"]
    grad["grad_W_V"]  = act["act_embedded"].T @ grad["grad_V"]
    grad_embedded    += grad["grad_V"] @ W["W_V"].T

    # Step 5 — Attention softmax backward
    #   For each row t:  grad_scores[t,i] = a[i] * (g[i] - dot(a,g)) / 4
    a = act["act_attn"]       # [8 x 8]
    g = grad["grad_attn"]     # [8 x 8]
    dot_ag = np.sum(a * g, axis=1, keepdims=True)  # [8 x 1]
    grad["grad_scores"] = a * (g - dot_ag) / 4.0   # [8 x 8]

    # Step 6 — Q and K projections
    #   grad_Q   = grad_scores x act_K                   [8 x 16]
    #   grad_K   = grad_scores^T x act_Q                 [8 x 16]
    #   grad_W_Q = act_embedded^T x grad_Q               [16 x 16]
    #   grad_W_K = act_embedded^T x grad_K               [16 x 16]
    #   grad_embedded += grad_Q x W_Q^T                  (Q path)
    #   grad_embedded += grad_K x W_K^T                  (K path)
    grad["grad_Q"]   = grad["grad_scores"] @ act["act_K"]
    grad["grad_K"]   = grad["grad_scores"].T @ act["act_Q"]
    grad["grad_W_Q"] = act["act_embedded"].T @ grad["grad_Q"]
    grad["grad_W_K"] = act["act_embedded"].T @ grad["grad_K"]
    grad_embedded   += grad["grad_Q"] @ W["W_Q"].T
    grad_embedded   += grad["grad_K"] @ W["W_K"].T

    grad["grad_embedded"] = grad_embedded

    # Step 7 — Embedding tables (sparse update)
    #   grad_W_pos = grad_embedded  (direct copy, same shape [8 x 16])
    #   grad_W_embed: accumulate by digit
    grad["grad_W_pos"] = grad_embedded.copy()

    grad_W_embed = np.zeros((VOCAB, D_MODEL))
    for t in range(SEQ_LEN):
        grad_W_embed[digits[t]] += grad_embedded[t]
    grad["grad_W_embed"] = grad_W_embed

    return grad

# ---------------------------------------------------------------------------
# SGD weight update
# ---------------------------------------------------------------------------

def update_weights(W, grad):
    W["W_embed"] -= LR * grad["grad_W_embed"]
    W["W_pos"]   -= LR * grad["grad_W_pos"]
    W["W_Q"]     -= LR * grad["grad_W_Q"]
    W["W_K"]     -= LR * grad["grad_W_K"]
    W["W_V"]     -= LR * grad["grad_W_V"]
    W["W_out"]   -= LR * grad["grad_W_out"]

# ---------------------------------------------------------------------------
# Bit-reversal permutation
# ---------------------------------------------------------------------------

def bit_reversal(digits):
    return np.array([digits[PERM[i]] for i in range(SEQ_LEN)])

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def fmt_row(arr):
    """Format a 1-D array as comma-separated 6-decimal-place values."""
    return ",".join(f"{v:.6f}" for v in arr)

def print_test_vectors(W, label, test_inputs):
    """Run inference on test inputs and print formatted results."""
    print(f"\n=== {label} ===")
    for digits in test_inputs:
        targets = bit_reversal(digits)
        act = forward(W, digits)
        probs = act["act_probs"]
        preds = np.argmax(probs, axis=1)
        conf = [probs[t, preds[t]] * 100.0 for t in range(SEQ_LEN)]

        print(f"\nInput: {','.join(str(d) for d in digits)}")
        print(f"Target:    {','.join(str(d) for d in targets)}")
        print(f"Predicted: {','.join(str(d) for d in preds)}")
        print(f"Confidence (%): {','.join(f'{c:.1f}' for c in conf)}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.RandomState(SEED)
    W = init_model(rng)

    print("=== MacMind Python Validator ===")
    print(f"Parameters: {sum(w.size for w in W.values())}")

    # --- Initial test vectors (before training) ---
    test_input_1 = np.array([3, 7, 1, 9, 5, 2, 8, 4])
    act = forward(W, test_input_1)

    print("\n=== INITIAL TEST VECTORS (before training, seed 42) ===")
    print(f"Input: {','.join(str(d) for d in test_input_1)}")

    print(f"\nact_embedded row 0:\n  {fmt_row(act['act_embedded'][0])}")
    print(f"\nact_Q row 0:\n  {fmt_row(act['act_Q'][0])}")
    print(f"\nact_attn row 0:\n  {fmt_row(act['act_attn'][0])}")
    print(f"\nact_probs row 0:\n  {fmt_row(act['act_probs'][0])}")

    loss, acc = compute_loss(act, bit_reversal(test_input_1))
    print(f"\nloss: {loss:.6f}")

    # --- Training ---
    print("\n=== TRAINING ===")

    rolling_losses = []
    prev_rolling_avg = None

    for step in range(1, TRAIN_STEPS + 1):
        # Generate random 8-digit sequence
        digits = rng.randint(0, VOCAB, size=SEQ_LEN)
        targets = bit_reversal(digits)

        # Forward
        act = forward(W, digits)

        # Loss
        loss, acc = compute_loss(act, targets)

        # Backward
        grad = backward(W, act, digits, targets)

        # Update
        update_weights(W, grad)

        # Rolling loss tracking
        rolling_losses.append(loss)
        if len(rolling_losses) > 100:
            rolling_losses.pop(0)

        if step % 100 == 0:
            rolling_avg = np.mean(rolling_losses)
            print(f"Step {step:4d}: loss {loss:.4f}  acc {acc}%")

            # Verify rolling average is decreasing (after first window)
            if prev_rolling_avg is not None:
                if rolling_avg > prev_rolling_avg + 0.1:
                    print(f"  WARNING: rolling avg increased "
                          f"{prev_rolling_avg:.4f} -> {rolling_avg:.4f}")
            prev_rolling_avg = rolling_avg

    # --- Trained test vectors ---
    test_inputs = [
        np.array([3, 7, 1, 9, 5, 2, 8, 4]),
        np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        np.array([9, 8, 7, 6, 5, 4, 3, 2]),
    ]

    print("\n=== TRAINED TEST VECTORS ===")
    print("(Use these to verify the HyperTalk implementation in HyperCard's")
    print(" Message Box after pasting all scripts and running initModel +")
    print(" loading these weights)")

    # Print full attention map for the first test input
    act_first = forward(W, test_inputs[0])
    print(f"\nact_attn (full 8x8, comma-separated rows):")
    for r in range(SEQ_LEN):
        print(f"row {r}: {fmt_row(act_first['act_attn'][r])}")

    print_test_vectors(W, "TRAINED TEST VECTORS", test_inputs)


if __name__ == "__main__":
    main()
