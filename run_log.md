# Parameter Golf Run Log

## Run 1: test_mods_v2 (1xA100 SXM 80GB, 5min cap)
- **Date:** 2026-03-20
- **Config:** NUM_VALUE_EMBEDS=2, ENABLE_SMEAR_GATE=1, MTP_N_PREDICT=2, MUON_WEIGHT_DECAY=0.02
- **Data:** 2 train shards (sp1024)
- **Result:** val_bpb=1.7405 (post-quant), 415 steps, 724ms/step
- **Artifact:** 9.98MB int8+zlib
- **Loss curve:** 4.1079 (step 0) → 1.7114 (step 415)
- **Notes:** Steep curve, limited by data shards and wallclock. Slower per step than baseline (724ms vs 666ms) due to extra params.

## Run 2: baseline_v1 (1xA100 SXM 80GB, 5min cap) — NO MODS
- **Date:** 2026-03-20
- **Config:** NUM_VALUE_EMBEDS=0, ENABLE_SMEAR_GATE=0, MTP_N_PREDICT=1, MUON_WEIGHT_DECAY=0.0
- **Data:** 2 train shards (sp1024)
- **Result:** val_bpb=1.5067 (post-quant), val_bpb=1.4948 (TTT LoRA), 451 steps, 666ms/step
- **Artifact:** 9.78MB int8+zlib
- **Loss curve:** 4.1069 (step 0) → 1.4916 (step 451)

## Comparison (1xA100, 5min, 2 shards)
| | Baseline | With Mods | Delta |
|---|---|---|---|
| val_bpb (post-quant) | **1.5067** | 1.7405 | +0.2338 (WORSE) |
| Steps completed | 451 | 415 | -36 |
| ms/step | 666 | 724 | +58ms (slower) |
| Params | 17.06M | 17.58M | +0.52M |

**Analysis:** Full mods (MTP+smear+VE) WORSE. MTP inflates loss and slows convergence.

## Run 3: test_ve_only (1xA100 SXM 80GB, 5min cap) — VE + WD ONLY
- **Date:** 2026-03-20
- **Config:** NUM_VALUE_EMBEDS=2, MUON_WEIGHT_DECAY=0.02, no smear, no MTP
- **Data:** 2 train shards (sp1024)
- **Result:** val_bpb=1.5066 (post-quant), val_bpb=1.4894 (TTT LoRA), 447 steps, 672ms/step
- **TTT LoRA delta vs baseline:** -0.0054 (improvement!)
- **Notes:** Negligible overhead, TTT LoRA improves. VE helps representation quality.

## Run 4: 2xA100_ve_wd_10min (2xA100 PCIe 80GB, 10min cap) — BEST CONFIG
- **Date:** 2026-03-21
- **Config:** NUM_VALUE_EMBEDS=2, MUON_WEIGHT_DECAY=0.02, no smear, no MTP
- **Data:** 10 train shards (sp1024)
- **Result:**
  - val_bpb = **1.3098** (post-quant int8+zlib)
  - val_bpb = **1.2753** (TTT LoRA)
  - 1553 steps completed, 386ms/step
  - Artifact: 14.49MB int8+zlib (under 16MB!)
- **Loss curve:** 4.1069 → 1.3561 (step 1000) → 1.3088 (step 1553)
- **Weights saved:** saved_runs/2xA100_ve_wd_10min/

## Summary
- Current SOTA on leaderboard: 1.1748 BPB
- Our best (2xA100, 10min): 1.2753 BPB (TTT LoRA) — only 0.1 behind on 2 GPUs vs 8
- Extrapolating to 8xH100: could get ~4x more steps, expect ~1.22-1.24 BPB range
- Value embeds + cautious WD are net positive. MTP and smear gate hurt on this setup.
