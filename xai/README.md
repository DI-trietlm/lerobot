# XAI — SmolVLA pouring deploy diagnosis

Why `di-techinnova/smolvla-pouring-0.1` (trained on `di-techinnova/so-arm-101-pouring-0.2`,
96 episodes) **stands still** when deployed via the RTC stack, even though it overfit the data.

**A self-contained presentation of the whole diagnosis** is in
[`diagnosis_presentation.html`](diagnosis_presentation.html) (open in a browser, offline).

Notebooks run on the **GPU server** (need the checkpoint + dataset cache). Older XVLA/SmolVLA
attention & occlusion explainability scripts have moved to [`attention/`](attention/).

## TL;DR — root cause (confirmed)

Pouring **returns the orange cup to its original spot**, so at episode end **both the pose AND the
scene match the start**, but with opposite labels (start → *reach*, end → *rest*). Because the scene
is nearly static, the policy **learned to ignore vision** → at the home pose there is **no signal
(state same, image same)** to tell start from end → it collapses to the majority behaviour (*stay*)
→ **standstill**.

**Control:** `so-arm-101-general-0.2` (pick → box) returns to the same pose but the **cup ends in a
box** → end scene ≠ start scene → no collision → it overfits fine. Every joint/action statistic of
the two datasets is **nearly identical** (general is even "worse"); the only difference is the
**image stream**.

**Fix:** after pouring, **place the cup somewhere else** (like the pick task) + trim the ~2 s of
"return home & hold" at episode end. Skip `n_obs_steps` (SmolVLA uses only the latest obs).

## The check flow (done)

| # | Notebook | Question it answers | Result |
|---|----------|---------------------|--------|
| 1 | `REPORT-pouring-startpose-and-phases.ipynb` | Start-pose IQR + grasp/pour phases | start-pose OOD fixed; not the cause |
| 2 | `openloop_replay_smolvla.ipynb` | Does the **model** reproduce training actions? | yes (MAE ~2°); model+norm fine; image-blind |
| 3 | `deploy_image_probe_smolvla.ipynb` | Do **deploy images** make it collapse? | no — RGB/BGR, 256-resize, JPEG all cleared |
| 4 | `deploy_state_replay_smolvla.ipynb` | Image or state? | state-dominated; `Δ≈0` reproduces standstill |
| 5 | _dataset control_ `pouring-0.2` vs `general-0.2` | Why does pick overfit but pouring not? | joint-stats identical → cause is the **scene** (cup returned to same spot) |

## Findings (why the other suspects are NOT the cause)

- **Start-pose / normalization / camera / RGB-BGR / 256-resize / JPEG** — all eliminated with
  numbers (resetting in-distribution, checkpoint stats correct, image swaps move output only a few
  degrees).
- **Model converged** — open-loop replay reproduces GT incl. the pour (MAE ~2°); so it is **not**
  under-trained and **not** "96 episodes too few".
- **Joint-space data conflict** — *refuted by the control*: `general-0.2` (overfit OK) has **equal
  or higher** per-state action conflict, more idle, slower initiation, yet works. So the fault is
  **not** in the proprioception/action data.
- **Real cause = (pose + scene) collision** from returning the cup to its original position, which
  makes vision useless → image-blind policy → cannot disambiguate start vs end at the home pose.

## Notes

- `record_obs` now logs `observation.state` per frame in `metadata.jsonl`
  ([robot_client.py](../src/lerobot/rtc_inference/robot_client.py)) for Step 4's exact replay.
- Steps 3–4 expect the deploy `recorded_obs/` folder copied next to the notebook on the GPU server.
- The original "state-calibration + camera-rig audit" step was **dropped**: the control shows the
  fault is the task's scene structure, not a sensor/calibration mismatch.
