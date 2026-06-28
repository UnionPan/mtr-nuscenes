# MTR — Results Summary

## Retrieval, linear probe, and motion (clean val)

| run | v2t R@1 | v2t R@5 | t2v R@1 | mean R@1 | ADE (m) | FDE (m) | probe motion-state | probe pedestrian |
|---|---|---|---|---|---|---|---|---|
| baseline (frozen mean-pool) | — | — | — | — | 4.35 | 7.68 | 0.750 | 0.306 |
| full | 0.056 | 0.278 | 0.083 | 0.069 | 3.25 | 5.55 | 0.611 | 0.306 |
| no_mlm | 0.111 | 0.361 | 0.056 | 0.083 | 3.60 | 5.87 | 0.278 | 0.333 |
| contrastive_only | 0.111 | 0.361 | 0.083 | 0.097 | 2.96 | 5.30 | 0.278 | 0.306 |
| motion_only | — | — | — | — | 3.92 | 6.16 | 0.778 | 0.556 |
| ctx_t4 | 0.105 | 0.395 | 0.105 | 0.105 | 3.60 | 5.94 | 0.737 | 0.316 |
| ctx_t8 | 0.059 | 0.206 | 0.088 | 0.074 | 3.61 | 5.85 | 0.676 | 0.324 |
| frozen_img | 0.028 | 0.194 | 0.056 | 0.042 | 3.04 | 5.38 | 0.583 | 0.306 |
| adapt_img | 0.056 | 0.306 | 0.056 | 0.056 | 3.51 | 5.82 | 0.778 | 0.306 |

_Probe majority-class baselines: motion-state 0.778, pedestrian 0.806._


## Robustness (full model, val)

| condition | mean R@1 | ADE (m) |
|---|---|---|
| clean | 0.069 | 3.25 |
| corrupt::gaussian_noise::s0.5 | 0.069 | 3.28 |
| corrupt::gaussian_noise::s1.0 | 0.042 | 3.99 |
| corrupt::blur::s0.5 | 0.056 | 3.10 |
| corrupt::blur::s1.0 | 0.056 | 3.20 |
| corrupt::brightness::s0.5 | 0.069 | 3.23 |
| corrupt::brightness::s1.0 | 0.083 | 3.21 |
| frame_dropout::0.25 | 0.069 | 3.24 |
| frame_dropout::0.5 | 0.069 | 3.26 |
| camera_dropout::0.25 | 0.056 | 3.20 |
| camera_dropout::0.5 | 0.069 | 3.17 |

## Efficiency (full model)

| pipeline | throughput (clips/s) | latency (ms/clip) | peak mem (GB) |
|---|---|---|---|
| full_image_to_temporal | 27 | 37.7 | 4.67 |
| cached_feature_temporal | 6863 | 0.1 | 0.71 |

_Efficiency config: {'batch_size': 16, 'clip_len': 6, 'views': 6, 'image_size': 224, 'precision': 'fp16'}._
