# SmolVLA 0629 Model/Data Diagnosis Findings

This note summarizes the completed offline diagnosis for the 0629 runtime traces.
Raw CSV/PNG outputs are intentionally ignored under
`xai/artifacts/model_data_diagnosis_smolvla_0629_outputs/`.

## Inputs

- Runtime runs:
  - `recorded_obs-0629-01`
  - `recorded_obs-0629-02`
  - `recorded_obs-0629-nonrtc`
  - `recorded_obs-0629-rtc2`
- Old final model revision: `b6f2aafdbdd793046747fad8207459402c33c4b0`
- New final model revision: `f7029d03d69e149cb4b7cea8747d7158d35a8fd0`
- Important HF detail: one model upload is split into three commits. Use the last
  commit in each group as the full snapshot because it contains weights/config,
  preprocessor, and postprocessor.

## Main Conclusion

The safe-pose drift is primarily a raw-policy/new-checkpoint behavior, not an RTC
artifact.

The strongest evidence is same-observation replay:

| Run | Policy | Observations | Safe-pull count | Safe-pull rate |
| --- | --- | ---: | ---: | ---: |
| `recorded_obs-0629-01` | new | 40 | 22 | 55.0% |
| `recorded_obs-0629-01` | old | 40 | 0 | 0.0% |
| `recorded_obs-0629-02` | new | 27 | 6 | 22.2% |
| `recorded_obs-0629-02` | old | 27 | 0 | 0.0% |
| `recorded_obs-0629-nonrtc` | new | 21 | 7 | 33.3% |
| `recorded_obs-0629-nonrtc` | old | 21 | 0 | 0.0% |
| `recorded_obs-0629-rtc2` | new | 18 | 7 | 38.9% |
| `recorded_obs-0629-rtc2` | old | 18 | 0 | 0.0% |

Across all current-image/current-state probes, the new model has `42/106`
safe-pull cases while the old model has `0/106`.

## What This Rules Out

RTC is not the root cause. The non-RTC run still has new-model safe-pull
(`7/21`) and old-model safe-pull (`0/21`) on the same observations.

Client-side execution is not the root cause. Server logs already contained
safe-pulling postprocessed chunks, and offline replay of the new checkpoint
reproduces the same direction.

JPEG/BGR transport is not the root cause. The `server_jpeg_bgr_q90` replay
changes magnitude and sometimes matches the server closer, but the pattern
remains: new has safe-pull, old does not.

## State vs Image

The new model is much more state-conditioned than image-conditioned in this
failure mode.

For the new model across 106 probes:

| Ablation | Safe-pull count | Safe-pull rate |
| --- | ---: | ---: |
| current image + current state | 42 | 39.6% |
| start image + current state | 57 | 53.8% |
| current image + start state | 2 | 1.9% |
| start image + start state | 2 | 1.9% |

Holding the current state while replacing the image with the start image keeps
or increases safe-pull. Holding the start state while using the current image
nearly removes it. That points to a state-conditioned attractor/shortcut.

## State Scan Evidence

Synthetic state scans confirm that the new model has a danger basin near low or
early progress states. The old model mostly does not.

Aggregate state-scan safe-pull:

| Run | Policy | Points | Safe-pull points | Safe-pull rate |
| --- | --- | ---: | ---: | ---: |
| `recorded_obs-0629-nonrtc` | new | 192 | 57 | 29.7% |
| `recorded_obs-0629-nonrtc` | old | 192 | 5 | 2.6% |
| `recorded_obs-0629-rtc2` | new | 192 | 60 | 31.2% |
| `recorded_obs-0629-rtc2` | old | 192 | 0 | 0.0% |

The plots show the new model's predicted chunk end rising above the diagonal for
many low/mid progress inputs. That is consistent with drifting from start pose
toward safe pose in some closed-loop runs.

## Dataset Signal

Dataset audit found suspicious safe-like action segments, but this is weaker
evidence than replay because the projection is noisy:

- Total frames audited: `73,803`
- Episodes: `175`
- Overall safe-like action rate: `7.47%`
- Early/mid safe-like action rate: `7.33%`
- Highest suspect episodes include `0`, `51`, `57`, `90`, `145`, `60`, `120`,
  `59`, `50`, and `58`.

This supports the hypothesis that the larger/newer training run learned a
state-to-safe shortcut from some data distribution or normalization change, but
it does not prove the dataset alone caused the failure.

## Config/Revision Check

The old and new full snapshots were correctly selected. Both snapshots contain
the expected nine files, and the changed files are:

- `README.md`
- `config.json`
- `train_config.json`
- `preprocessor_config/normalizer.safetensors`
- `postprocessor_config/unnormalizer.safetensors`

Important config differences:

- old: `chunk_size=35`, `n_action_steps=35`, `steps=100000`
- new: `chunk_size=30`, `n_action_steps=30`, `steps=50000`
- scheduler warmup/decay and processor stats changed

Processor-stat changes are important because old vs new differ in more than
weights only.

## Practical Diagnosis

The previous 50-episode model looked better because it did not exhibit this
safe-pull attractor on the recorded live observations. The newer 175-200 episode
model learned a different mapping: for some current states, it predicts chunks
whose end moves toward the safe-pose direction even before RTC/client dynamics
matter.

The immediate engineering response should be:

1. Keep the old model as a runtime baseline.
2. Add an offline safe-pull replay metric to every candidate checkpoint.
3. Inspect and optionally remove or relabel the top suspect dataset episodes.
4. Train new candidates only if they pass recorded-observation replay and state
   scan checks.
5. Use the VLA Harness as a runtime guard, because raw policy instability is now
   demonstrated and may recur even after data cleanup.
