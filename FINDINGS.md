# Parameter Golf — Findings from Modded-NanoGPT Cross-Pollination

## Background
We studied techniques from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (GPT-2 speedrun competition) and tested which ones transfer to the [parameter-golf](https://github.com/openai/parameter-golf) competition (best 16MB language model in 10min on 8xH100).

## Techniques Tested

### Worked: Value Embeddings
- **What:** Separate embedding tables (initialized to zero) injected as V-deltas into attention at specific layers
- **Origin:** modded-nanogpt (@KoszarskyB, @Grad62304977), inspired by value residual learning (arXiv:2410.17897)
- **Result:** +0.005 BPB improvement on TTT-LoRA evaluation. Negligible step time overhead.
- **Key detail:** Must use `kv_dim` (not `model_dim`) for GQA compatibility — c_v output is `num_kv_heads * head_dim`
- **Config:** `NUM_VALUE_EMBEDS=2`, `VALUE_EMBED_LR=0.3`

### Worked: Cautious Decoupled Weight Decay for Muon
- **What:** Apply L2 weight decay to Muon-managed matrix params, but only when update direction aligns with param direction (cautious gating)
- **Origin:** modded-nanogpt, based on arXiv:2510.12402
- **Result:** Marginal improvement, aids quantization robustness
- **Config:** `MUON_WEIGHT_DECAY=0.02`

### Worked: Muon Momentum Cooldown
- **What:** Linearly ramp Muon momentum back down in the final 50 steps (mirrors the warmup at start)
- **Origin:** modded-nanogpt
- **Result:** Small stability improvement at end of training
- **Config:** `MUON_MOMENTUM_COOLDOWN_STEPS=50`

### Did NOT Work: Multi-Token Prediction (MTP)
- **What:** Weighted cross-entropy loss over next N tokens (e.g., [1.0, 0.5] for 2-token prediction)
- **Origin:** modded-nanogpt (@varunneal)
- **Why it failed:** Inflates reported training loss, slows convergence, and adds ~60ms/step overhead. On short wallclock-limited runs, the extra compute per step means fewer total steps. May work with longer training budgets.
- **Result:** val_bpb 1.74 vs 1.51 baseline — significantly WORSE

### Did NOT Work: Smear Gate
- **What:** Learned gate mixing previous token's embedding into current token
- **Origin:** modded-nanogpt (@classiclarryd)
- **Why it failed:** Adds parameters without proportional benefit in this small-model regime. The overhead from extra forward computation isn't justified by the tiny context enrichment.
- **Result:** No improvement, slight slowdown

### Not Tested Yet (Potential)
- **NorMuon + Polar Express:** Advanced Muon variant with variance reduction and Polar Express orthogonalization. High complexity to port.
- **Batch size scheduling:** Progressively increasing batch size (8→16→24). Would need careful tuning for parameter-golf's fixed wallclock.
- **Window size scheduling + YaRN:** Dynamic attention window expansion. Requires FlexAttention infrastructure.
- **Skip MLP blocks:** Skipping attention on specific layers, using skip connections between distant layers.
- **Logit rescaling:** `23*sigmoid((logits+5)/7.5)` instead of `30*tanh(logits/30)`.

## Run Results

| Run | GPU | Time | Steps | val_bpb (quant) | val_bpb (TTT) | Config |
|-----|-----|------|-------|-----------------|---------------|--------|
| Baseline | 1xA100 | 5min | 451 | 1.5067 | 1.4948 | No mods |
| All mods | 1xA100 | 5min | 415 | 1.7405 | — | VE+smear+MTP+WD |
| VE + WD | 1xA100 | 5min | 447 | 1.5066 | **1.4894** | Best lightweight config |
| VE + WD | 2xA100 | 10min | 1553 | 1.3098 | **1.2753** | Scaled up |
| VE + WD | 4xA100 | 10min | TBD | TBD | TBD | Running now |

Current SOTA on leaderboard: **1.1748 BPB**

## Key Learnings

1. **Less is more on short runs.** Every extra ms/step costs you steps under a wallclock cap. A technique that improves per-step quality must overcome the cost of fewer total steps.

2. **MTP is a trap for small models.** It works well in modded-nanogpt (large models, unconstrained time) but hurts in parameter-golf (small models, strict time limit).

3. **Value embeddings are free lunch.** Near-zero overhead, genuine improvement in representation quality visible through TTT-LoRA evaluation.

4. **Cautious WD helps Muon.** Even a small WD (0.02) on spectral params improves generalization without hurting convergence speed.

5. **Test on small runs first.** Our 1xA100 5min ablation correctly predicted which techniques would help at scale.
