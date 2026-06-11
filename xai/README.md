# XAI — SmolVLA pouring deploy diagnosis

Why `di-techinnova/smolvla-pouring-0.1` (trained on `di-techinnova/so-arm-101-pouring-0.2`,
96 episodes) **stands still** when deployed via the RTC stack, even though it overfit the data.

Notebooks run on the **GPU server** (need the checkpoint + dataset cache). Older XVLA/SmolVLA
attention & occlusion explainability scripts have moved to [`attention/`](attention/).

## The 5-step check flow

Each step narrows the cause; run them in order and stop when one localizes the fault.

| # | Notebook | Question it answers | Status |
|---|----------|---------------------|--------|
| 1 | `REPORT-pouring-startpose-and-phases.ipynb` | What does the data say? Start-pose IQR (safe reset region) + grasp/pour phase detection. | done |
| 2 | `openloop_replay_smolvla.ipynb` | Does the **model** reproduce training actions from training frames? (isolates model+norm+preprocessing) | done |
| 3 | `deploy_image_probe_smolvla.ipynb` | Do the **deploy camera images** make the output collapse? Ablations: RGB/BGR, server 256-resize, image diff vs dataset. | done |
| 4 | _deploy (image+state) pairs_ | Image **or** state? Log the real deploy `observation.state`, pair it with the deploy image, swap each input. | todo |
| 5 | _state-calibration + camera-rig audit_ | Calibration drift vs dataset; camera assignment / FOV / mount parity. | todo |

## Findings so far

- **Step 1** — the old "to zero pose" reset (shoulder_pan +20°, gripper 0) is **OOD** vs the start
  manifold (shoulder_pan always negative, gripper ~1–2.4). Fixed in the GUI (reset to start-pose).
  But resetting in-distribution **did not** fix the standstill ⇒ start-pose is not the (only) cause.
- **Step 2** — predicted actions **track ground-truth almost perfectly** across a whole episode
  (approach → pour). Normalization stats in the checkpoint are correct. ⇒ **model + pipeline are
  fine**; the fault is in the **deploy observation pipeline**.
- **Step 3+** — in progress: is the live observation (camera images / state) off-distribution at
  run time?

## Notes

- `record_obs` currently saves images + `metadata.jsonl` but **not** `observation.state`; Step 4
  needs a small change to log the state so deploy (image, state) pairs can be replayed exactly.
- Step 3 expects the deploy `recorded_obs/` folder copied next to the notebook on the GPU server.
