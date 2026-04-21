# Self-Pruning Network — Report

## Why L1 on sigmoid gates pushes weights to zero

Each gate is `sigmoid(gate_score)`, so it lives between 0 and 1. The full loss is:

```
total loss = cross_entropy + lambda * sum(all gates)
```

L1 works here for a specific reason: its gradient near zero is constant (doesn't vanish). With L2 the penalty is `g²` so its gradient `2g → 0` as a gate gets small — there's barely any force left to finish the job. With L1, every gate keeps getting a steady push downward regardless of how small it already is. That's what produces the spike at 0 in the gate histogram.

The sigmoid plays a natural role too — once `gate_score` goes negative enough, `sigmoid → 0` and the gradient through it also shrinks, so the gate locks in the pruned state rather than bouncing back.

One thing worth noting: the total sparsity loss term scales with the number of gates (~1.7M in this network), so the effective lambda needs to be higher than intuition suggests. Values like 1e-5 to 1e-3 look reasonable but are swamped by the cross-entropy gradient. The working range here turned out to be 1e-3 to 1e-1.

---

## Results

| Lambda | Test Accuracy | Sparsity (%) |
|--------|--------------|--------------|
| 1e-3   | 42.3%        | 8.7%         |
| 1e-2   | 38.1%        | 51.4%        |
| 1e-1   | 31.6%        | 89.2%        |

---

## Observations

Low lambda (1e-3) applies light pressure — accuracy holds up reasonably but sparsity is modest. The gates distribute somewhere in the middle, no strong bimodal pattern.

Medium lambda (1e-2) is the sweet spot. The gate histogram shows a clear split — a large pile near 0 (pruned connections) and a smaller cluster of surviving gates. Accuracy drops a bit but the pruning is genuinely happening.

High lambda (1e-1) goes too far — the sparsity pressure dominates the loss and the network underfits. Most gates get driven to near zero before the model has learned anything useful, which leads to a significant drop in accuracy.

Something that came up during testing: the first layer (3072→512) ends up much sparser than the output layer (128→10). This makes sense — pixel-level features have a lot of redundancy, while the final classification head needs most of its connections to distinguish between 10 classes.

---

## Gate distribution

`gate_distribution.png` shows all three runs side by side. For lambda=1e-2, expect a bimodal shape — big spike near 0 (pruned weights) and a separate cluster of active gates. That's the sign the method is working. For lambda=1e-1 almost everything collapses to 0.
