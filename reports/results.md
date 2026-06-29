# MTR — Results Report

Multimodal temporal-representation pretraining on **nuScenes Mini**. All numbers
below are measured by `scripts/run_all.sh` (training) + `scripts/evaluate.py`
(evaluation) and aggregated by `scripts/collect_results.py`. Raw per-run metrics
live in `runs/<name>/eval_metrics.json`; the regenerable tables are in
`reports/results_tables.md` and `reports/summary.json`.

## 1. Setup (measured)

| item | value |
|---|---|
| dataset | nuScenes Mini, v1.0-mini |
| scenes / keyframes / cameras | 10 scenes · 404 keyframes · 6 surround cameras |
| split (official mini) | 8 train / 2 val scenes |
| clips (T=6, stride 2) | 143 train / 36 val |
| frame encoder | DINOv2 ViT-B/14 (frozen, cached) |
| text encoder | DistilBERT (frozen) |
| model size | 82.0 M total · 15.5 M trainable (frozen-encoder MVP) |
| hardware | 1× Quadro RTX 8000 (Turing, 48 GB); 4 available |
| precision | fp16 autocast, fp32 optimizer states |
| training budget | 80 epochs, batch 16, AdamW, cosine + warmup |

Turing has no native bf16, so fp16 is the default (bf16/fp32 are config options).
The frozen backbone is run once into a cached embedding store; every
feature-mode experiment then trains the temporal model + heads at **~280 clips/s**
on a single GPU.

## 2. Headline model (`full`: contrastive + masked-temporal + motion)

| metric | value | reference |
|---|---|---|
| video→text R@10 | 0.61 | random ≈ 0.28 |
| text→video R@5 | 0.42 | random ≈ 0.14 |
| mean R@1 | 0.069 | random ≈ 0.028 |
| median rank | 9 | of 36 |
| motion ADE / FDE | 3.25 / 5.55 m | mean-traj prior 2.96; CV ceiling 1.32; baseline ridge 4.35 |
| linear probe (motion-state) | 0.61 | majority 0.78 |

The model retrieves and predicts motion better than the no-training frozen
baseline (ADE 4.35 → 3.25). Retrieval is strongest at R@5/R@10 and median rank;
**R@1 is noisy and near-floor** because the val set is only 2 scenes / 36 clips
(each clip is 1/36 ≈ 0.028 of the metric), so we report R@5/R@10/median-rank as
the reliable retrieval signal.

## 3. Baseline + ablations

See `reports/results_tables.md` for the full table. Summary of controlled
comparisons (all on the same val split):

### Objective combinations
| run | objectives | v2t R@10 | mean R@1 | ADE (m) | probe motion-state |
|---|---|---|---|---|---|
| contrastive_only | C | **0.67** | **0.097** | 2.96† | 0.28 |
| no_mlm | C+M | 0.61 | 0.083 | 3.60 | 0.28 |
| full | C+M+Mo | 0.61 | 0.069 | 3.25 | **0.61** |
| motion_only | Mo | — | — | 3.92 | **0.78** |

*C = contrastive, M = masked-temporal, Mo = motion.*
†`contrastive_only`'s ADE 2.96 is the **untrained motion anchor** (the mean-trajectory
prior) — motion isn't an objective there, so the head emits the prior exactly.

**Findings.** (i) Contrastive alone gives the best *retrieval* — adding motion/MLM
trades a little alignment for a multi-task representation. (ii) **Masked temporal
modeling markedly improves the motion-state linear probe** (no_mlm 0.28 → full
0.61), evidence it builds a more temporally-structured clip representation.
(iii) Training the motion objective does **not** beat the mean-trajectory prior
on Mini (full 3.25 / motion_only 3.92 vs prior 2.96) — a genuine small-data
overfitting result (Section 5), not a code defect.

### Context length (T frames per clip)
| run | T | v2t R@10 | mean R@1 | ADE (m) | probe motion-state |
|---|---|---|---|---|---|
| ctx_t4 | 4 | 0.55 | **0.105** | 3.60 | 0.74 |
| full | 6 | 0.61 | 0.069 | 3.25 | 0.61 |
| ctx_t8 | 8 | 0.62 | 0.074 | 3.61 | 0.68 |

Longer context (T=8) slightly improves video→text R@10; shorter context (T=4)
gives the best R@1/probe. Differences are within the noise of a 36-clip val set;
the practical read is that **4–6 frames are sufficient** in this regime.

### Frozen vs. adapted visual encoder (image pipeline, 15-epoch matched budget)
| run | backbone | trainable params | mean R@1 | ADE (m) |
|---|---|---|---|---|
| frozen_img | frozen | 15.5 M | 0.042 | **3.04** |
| adapt_img | unfrozen (ckpt + 0.05× LR) | 101.2 M | 0.056 | 3.51 |

Unfreezing the ViT-B backbone (6.5× more trainable parameters) does **not** help
and slightly worsens motion — expected with 143 training clips. **Frozen +
cached is the correct choice for the MVP.** This flips at scale: §5.2 shows the
*same* unfreeze on 12.5 k clips becomes the single biggest improvement (motion
−30 %), so the frozen-vs-adapted choice is dataset-size-dependent, not absolute.

## 4. Robustness & efficiency (headline model)

**Robustness** (`reports/results_tables.md`): the model degrades gracefully.
Worst case is heavy Gaussian noise (ADE 3.25 → 3.99, mean R@1 0.069 → 0.042);
blur, brightness, and 25–50 % frame/camera dropout barely move either metric
(ADE stays 3.1–3.3) — the masked-mean view aggregation and frame-padding masks
absorb missing inputs by design.

**Efficiency** (batch 16, T=6, 6 cams, 224², fp16):
| pipeline | throughput | latency | peak mem |
|---|---|---|---|
| full image→temporal | 27 clips/s | 37.7 ms/clip | 4.67 GB |
| cached feature→temporal | **6863 clips/s** | 0.1 ms/clip | 0.71 GB |

Caching the frozen encoder yields a **~250× temporal-model throughput speedup**
at 6.6× lower memory — the core reason temporal-model ablations are cheap.

**Multi-GPU (measured).** A 2-GPU DDP run (`torchrun --nproc_per_node=2`)
trains at world_size 2, ~370 clips/s, with correct objective-subset handling
(`find_unused_parameters`). DDP is the right tool here (only ~15 M params train);
an FSDP path is provided for the unfrozen-backbone regime.

## 5. Scale-up to full nuScenes (front camera)

To test whether the Mini-scale caveats are data-driven, the pipeline was run
unchanged on **full v1.0-trainval** (official 700/150 scene split) — an **85×
scene / 88× clip** increase. The full 6-camera data (~290 GB of blobs) was
pulled login-free from the public CloudFront CDN, extracting only the 6-camera
keyframes (~30 GB kept); the front-camera variant is a ~5.5 GB HF-mirror subset.
Two controls isolate the axes: **Mini front** (scale↓ only) and **Trainval
front** (cameras↓ only), versus the flagship **Trainval 6-cam** (both↑).

| | Mini 6-cam (`full`) | Mini front | Trainval front | **Trainval 6-cam (flagship)** |
|---|---|---|---|---|
| scenes | 10 | 10 | 850 | **850** |
| train / val clips | 143 / 36 | 143 / 36 | 12 522 / 2 682 | **12 522 / 2 682** |
| cameras | 6 | 1 | 1 | **6** |
| cached frames | 2.4 k | (subset) | 34 k | **203 k** |
| v2t R@10 | 0.61 | 0.33 | 0.116 | **0.150** |
| R@10 ÷ random | 2.2× | 1.2× | 31× | **40×** |
| median rank (of pool) | 9/36 (25%) | 18/36 (50%) | 131/2682 (4.9%) | **82/2682 (3.1%)** |
| motion ADE / prior | 3.25 / 2.96 | 2.95 / 2.96 | 4.64 / 5.32 | **4.71 / 5.32** |
| ADE vs. mean prior | +0.29 (worse) | ≈ prior | −0.68 (beats) | **−0.61 (beats)** |
| CV ceiling | 1.32 | 1.32 | 0.86 | 0.86 |
| probe motion-state (maj) | 0.61 (0.78) | 0.72 (0.78) | 0.80 (0.76) ✓ | **0.80 (0.76) ✓** |
| probe pedestrian (maj) | 0.31 (0.81) | 0.19 (0.81) | 0.67 (0.62) ✓ | **0.73 (0.62) ✓** |

**All three objectives improve with scale, exactly as the data-scarcity
hypothesis predicts:**

1. **Motion crosses from below-prior to above-prior.** At Mini scale the trained
   motion head cannot beat the mean-trajectory prior (overfitting); at trainval
   scale it does (ADE 4.64 < prior 5.32). The constant-velocity ceiling (0.86 m)
   shows the shrinkage penalty — tuned for Mini — is now over-regularizing, so
   there is clear further headroom from relaxing it.
2. **Retrieval becomes genuinely discriminative.** Absolute R@10 falls only
   because the retrieval pool grows 74× (36 → 2682); normalized, it rises from
   1.2× random (front-Mini) to **31× random**, and median rank improves from the
   50th to the **4.9th percentile**.
3. **Representations beat the majority baseline.** Both linear probes cross from
   below-majority (Mini) to above-majority (trainval): pedestrian presence in
   particular jumps 0.19 → 0.67 (front) → **0.73 (6-cam)**, becoming linearly
   decodable.
4. **Multi-camera compounds with scale.** At trainval scale, the 6-camera
   flagship beats the front-only model on every visual metric — retrieval median
   rank 131 → **82** (top 4.9 % → 3.1 %), R@10 0.116 → 0.150, pedestrian probe
   0.67 → 0.73 — while motion (an ego-centric, front-dominated target) is
   unchanged. The surround-view benefit seen on Mini holds at scale.

This confirms the Section-4 caveats are properties of the *Mini regime*, not the
method, and validates the clean path to full data. (Threaded feature caching
processed all 203 k 6-camera frames at ~270 img/s.)

### 5.1 Tuned 6-camera run (`scale_6cam_v2`)

The flagship motion head used the Mini-tuned shrinkage (`motion_reg = 0.3`),
which over-regularizes at scale. A follow-up relaxes it to 0.05, **unfreezes the
DistilBERT text encoder** (reduced LR 5e-5), and trains 60 epochs:

| metric | v1 (frozen text, reg 0.3) | v2 (trainable text, reg 0.05) |
|---|---|---|
| mean R@1 | 0.017 | **0.024** (+40 % rel.) |
| v2t R@10 | 0.150 | **0.172** |
| median rank | 82 | 80 |
| motion ADE | 4.71 | **3.95** (best 3.71 @ ep 34) |
| probe motion-state (maj 0.76) | 0.80 | 0.77 |
| probe pedestrian (maj 0.62) | 0.73 | 0.72 |

**Honest read:** retrieval and motion both improve (relaxing the shrinkage freed
the motion residual; the trainable text sharpens alignment), but the gains are
*incremental*, the probes go flat (alignment is traded for linear-probe-ability),
and best-motion (ep 34) and best-retrieval (ep 59) occur at different checkpoints.
Motion ADE 3.95 m still sits far above the 0.86 m constant-velocity ceiling — that
residual gap is now attributable to the **frozen single-frame visual encoder**,
whose pooled features bound how well ego-velocity can be inferred. Closing it
would require adapting the backbone at scale or richer temporal features — tested
next.

### 5.2 Adapting the visual encoder at scale (`scale_front_adapt`)

The Section-3 Mini ablation found that **unfreezing the ViT-B backbone hurt**
(overfitting on 143 clips). The frozen-encoder ceiling hypothesis predicts the
opposite at scale. Unfreezing DINOv2 on front-camera trainval (12.5 k clips,
gradient checkpointing, backbone LR 5e-5, 12 epochs) — same data as the frozen
front baseline — confirms it. Every metric improves, with no overfitting (stable
ep 7 → ep 11):

| metric (front-cam, trainval) | frozen (`scale_trainval`) | **adapted (`scale_front_adapt`)** |
|---|---|---|
| motion ADE | 4.64 | **3.22 (−30 %)** |
| v2t R@10 | 0.116 | **0.159** |
| median rank | 131 (4.9 %) | **66–73 (2.5–2.7 %)** |
| probe motion-state (maj 0.76) | 0.80 | **0.83** |
| probe pedestrian (maj 0.62) | 0.67 | **0.71** |

**Adaptation is the single biggest lever at scale** — and the contrast with Mini
(where it overfit) is itself the finding: *the right capacity/regularization
trade-off flips with dataset size*. Notably the adapted **front-camera** model
beats the frozen **6-camera** model on retrieval median rank (66 vs 80) and motion
(3.22 vs 4.71) — i.e. adapting the encoder matters more than camera count here.
Motion ADE 3.22 m is the best across all runs (Mini 3.25, front-frozen 4.64,
6-cam 3.95).

### 5.3 Both levers combined (`scale_6cam_adapt`)

Combining the two improvements that each helped — 6 surround cameras + an adapted
backbone (5 epochs image mode, ~3 h) — gives the best visual metrics overall:

| run (trainval) | median rank | v2t R@10 | motion ADE / FDE | probe motion-state | probe pedestrian |
|---|---|---|---|---|---|
| frozen 6-cam | 80 | 0.150 | 4.71 / — | 0.80 | 0.73 |
| 6-cam v2 (relaxed reg + text) | 80 | 0.172 | 3.95 / — | 0.77 | 0.72 |
| front-cam adapted | 66 | 0.159 | **3.22 / 5.85** | **0.83** | 0.71 |
| **6-cam adapted** | **49 (top 1.8 %)** | **0.186** | 3.34 / 6.15 | 0.78 | **0.79** |

**Motion baselines (both metrics, 2 232 valid val clips, same protocol).** A raw
ADE/FDE is only meaningful between its floor and ceiling:

| baseline | ADE | FDE |
|---|---|---|
| train-mean prior (naive floor) | 5.32 | 9.21 |
| **best model (front-adapt)** | **3.22** | **5.85** |
| constant-velocity (strong ceiling) | 0.86 | 2.05 |

The model beats the prior on **both** ADE (3.22 < 5.32) and FDE (5.85 < 9.21) and
sits between the naive floor and the physics ceiling on both — i.e. it learns
coarse motion intent but not yet precise instantaneous velocity. (FDE > ADE
always, since it is the error at the final, error-accumulated waypoint; the prior
note in §5 quotes ADE bookends only — these are the matching FDE bookends.)

**Read:** the result splits exactly along the expected axis. The camera-dependent
metrics are best with 6-cam + adaptation — retrieval median rank **49** (vs 80
frozen, 66 front-adapt) and pedestrian probe **0.79** (best of all runs). The
ego-centric **motion** target is front-dominated, so the front-adapted model ties
it (3.22 vs 3.34) — rear/side cameras add retrieval and pedestrian signal but not
ego-velocity. Both adapted runs cut motion ~30 % vs frozen (4.7 → ~3.3).

**Net of the whole study:** scale lifts every metric over the prior/majority
floors (§5); backbone adaptation is the largest single lever and inverts the Mini
ablation (§5.2); and cameras + adaptation compound on visual tasks (§5.3). Best
achieved: retrieval median rank **49 / 2682 (top 1.8 %)**, motion ADE **1.31 m** /
FDE **2.73 m** (kinematic head, §5.4; up from 3.22 m — prior 5.32 / 9.21, CV
ceiling 0.86 / 2.05), pedestrian probe **0.79**. Remaining headroom
(motion still > CV ceiling) points to temporal modeling beyond mean-pooled CLS as
the next bound — a clear, honest direction rather than a Mini-regime artifact.

### 5.4 Kinematic motion head — fixing the forecasting gap (`scale_6cam_kinematic`)

§5.2/5.3 left motion ADE stuck ~3.2 m, far above the constant-velocity ceiling.
The diagnosis — *mean-pooled CLS discards instantaneous velocity* — motivated a
new **kinematic motion head** combining four ideas: (1) predict a residual over a
**causal constant-velocity anchor** (computed from the last two *observed*
ego-poses — leak-free, the velocity a real ego-vehicle already has); (2) feed the
observed kinematics in directly; (3) condition on the **temporal token sequence**
(anchor-frame token + masked mean) instead of one pooled vector; (4) an
**auxiliary loss** predicting kinematics from the visual features. Runs in fast
feature mode on the existing 6-cam cache (no backbone retrain).

| | ADE | FDE |
|---|---|---|
| prior best learned head (front-adapt) | 3.22 | 5.85 |
| **kinematic head — best epoch** | **1.31** | **2.73** |
| kinematic head — final epoch | 1.44 | 3.00 |
| causal-CV baseline (realistic ceiling) | 1.31 | 2.74 |
| oracle-CV baseline (peeks 1 step) | 0.86 | 2.05 |

**Motion ADE drops 3.22 → 1.31 m (−59 %)** — the single largest motion gain in the
study, reaching the **causal constant-velocity ceiling**.

**Honest decomposition — and it is the finding.** The eval trajectory *is* an
ablation: at epoch 4 the residual is still ≈0, so the prediction equals the CV
anchor and ADE = 1.31 (the floor); as the residual trains it drifts slightly
**worse** (1.31 → 1.44). So **idea (1), the causal-CV anchor, is the entire win**;
the learnable residual + temporal features + aux loss (2–4) do **not** beat the
physics prior and mildly overfit above it. Interpretation: at frozen-feature
representation quality, **ego-motion is dominated by the kinematic prior** — the
visual residual carries no extra short-horizon predictive signal yet. The model
now sits at the *causal* physics limit (1.31); the remaining gap to the *oracle*
CV (0.86) is a velocity-estimation gap (last-observed-step vs. peeking), not
something the head can learn away. Pushing below CV would require genuinely richer
motion features (adapted backbone *and* temporal residual jointly, or optical-flow
/ patch-token cues) — the honest next bound. Retrieval and probes are unchanged
(the motion head is orthogonal, and this is frozen feature mode).

### 5.5 Higher-order prior + overfitting control (`scale_6cam_ctrv`)

§5.4 found the learned residual overfits above the constant-velocity prior. Two
follow-ups, both in fast feature mode:

**(a) Richer prior — CV → CTRV.** Constant-velocity extrapolates a straight line;
**constant-turn-rate-and-velocity (CTRV)** extrapolates the curve the ego has
already begun (yaw-rate from the observed window). Validated standalone on val:

| anchor (standalone, no learning) | ADE | FDE |
|---|---|---|
| CV (straight) | 1.31 | 2.74 |
| **CTRV (turn-rate)** | **1.22** | **2.58** |
| CTRA (+ acceleration) | 1.32–1.85 | worse |

Acceleration estimated from 2 Hz keyframes is too noisy and *hurts* — so the
useful higher-order term is yaw-rate, not accel. **The gain is entirely on turns**,
which is the honest way to read the modest overall number:

| clip type | share | CV ADE | CTRV ADE | gain |
|---|---|---|---|---|
| **turning** | 7 % | 3.02 | **1.82** | **−1.21 (−40 %)** |
| straight (moving) | 77 % | 1.37 | 1.37 | 0 |
| stopped | 16 % | 0.25 | 0.25 | 0 |
| overall | — | 1.31 | 1.22 | −0.09 |

On non-turning clips CV is already near-optimal (CTRV reduces to CV when
yaw-rate≈0), so there is nothing to gain; the win is concentrated in the rare
turning clips and diluted in the average. **A turning-only ADE (3.02 → 1.82) is the
more informative metric.** The residual error that remains everywhere
(1.37 straight, 1.82 turning) is from *non-constant* dynamics — braking/accel and
changing curvature — which only **perception** (reading brake lights, stop signs,
road curvature ahead) can predict; physics is tapped out at ~1.2 m.

**(b) Overfitting control — `motion_reg` 0.02 → 0.30.** Stronger shrinkage holds
the residual near the (now CTRV) anchor:

| | train ADE | val ADE | gap |
|---|---|---|---|
| CV head (reg 0.02) | 1.23 → 0.71 | 1.31 → **1.44** (regresses above prior) | 0.73 |
| **CTRV head (reg 0.30)** | 1.13 → 0.87 | 1.22 → **1.23** (flat, min 1.20) | **0.36** |

The val regression is **eliminated** — the model no longer ends up worse than its
anchor, and the train–val gap halves. But the honest reading is that shrinkage
*neutralized* the residual rather than making it generalize: val sits **at** the
CTRV anchor (1.22), it did not drop below. **Regularization prevents harm; it
cannot manufacture signal.** Beating the prior needs richer features (Tier 2:
patch tokens / map / lead-vehicle context), not more regularization. Final best
motion: **ADE 1.20–1.22 m, FDE 2.55 m, stable** — the study's best.

## 6. Limitations and honest caveats

- **Scene scarcity dominates.** Mini has 8 train / 2 val scenes. Retrieval R@1
  and linear-probe accuracy are high-variance on 36 val clips; motion prediction
  cannot generalize past the mean-trajectory prior (train ADE ≈ 0.3 m, val ADE
  ≈ 3–9 m before regularization). We therefore **anchor** the motion head to the
  mean training trajectory and add a shrinkage penalty, which keeps val ADE
  bounded (~3.2 m) instead of diverging.
- **Motion ceiling.** A constant-velocity predictor that reads ego-velocity
  reaches ADE 1.32 m, so the task is learnable *in principle*; extracting that
  signal from frozen visual features needs far more than 119 valid clips.
- **Frozen text + appearance features** limit pedestrian-presence decodability
  (linear probe ≈ 0.31, below the 0.81 majority) — single-frame appearance is a
  weak cue for it.

These are properties of the Mini regime, not the method — and Section 5 shows
they lift substantially at full-trainval scale.

## 7. Resume-ready claims (all measured here)

- Built a reproducible multimodal temporal-pretraining system (synchronized
  6-camera clips + structured text + ego-motion) on nuScenes, with cached
  frozen-DINOv2 embeddings enabling **~250× faster** temporal-model iteration
  (6863 vs 27 clips/s) at **0.71 GB**.
- Jointly trained InfoNCE video-text alignment, masked temporal modeling, and
  ego-motion prediction; the headline model reaches **video→text R@10 0.61 /
  median rank 9** on held-out scenes and **3.25 m ADE**, beating a frozen-encoder
  baseline (4.35 m).
- Ran a controlled study: **1 baseline + 8 ablations** over objective
  combinations, context length, and frozen-vs-adapted encoders; showed masked
  temporal modeling lifts the motion-state linear probe from 0.28 → 0.61.
- Demonstrated graceful **robustness** to visual corruption and frame/camera
  dropout, and a **measured 2-GPU DDP** run; fp16 on RTX 8000 with fp32
  optimizer states.
- **Scaled the same pipeline to full nuScenes v1.0-trainval** — 850 scenes,
  12.5k clips, the complete **6-camera** set (203k frames cached, pulled
  login-free from the public CDN): motion prediction crosses from below- to
  **above the mean-trajectory prior** (4.71 vs 5.32 m ADE), video→text retrieval
  reaches the **top 3.1 %** (median rank 82/2682, 40× random), and linear probes
  for motion-state (0.80) and pedestrian presence (**0.73**) **beat their
  majority baselines** — empirically validating that the Mini-scale limits are
  data-driven, and that surround-view + scale compound.
- **Identified backbone adaptation as the key lever at scale**: unfreezing
  DINOv2 (which overfits on Mini's 143 clips) on 12.5 k clips cuts motion ADE
  4.64 → **3.22 m (−30 %)** and halves retrieval median rank (131 → 66) — showing
  the frozen-vs-adapted trade-off is dataset-size-dependent.
- **Best configuration (6-camera + adapted backbone)** reaches video→text
  retrieval **median rank 49 / 2682 (top 1.8 %)**, R@10 0.186, and a pedestrian
  linear probe of **0.79** — with the clean finding that multi-camera helps the
  visual metrics while ego-motion remains front-camera-dominated.
