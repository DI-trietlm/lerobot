# Runtime Trace Analysis 2026-06-29

Phân tích này dùng hai run độc lập:

- `recorded_obs-0629-01`
- `recorded_obs-0629-02`

Mỗi run có đủ:

- `server_actions.jsonl`: raw normalized action, postprocessed action, final sent action sau RTC.
- `client_actions.jsonl`: chunk nhận từ server, queue/aggregation, command gửi robot.
- `metadata.jsonl`: current joint state và camera frame path/timestep.
- `images/camera1`, `images/camera2`.
- `runtime_configs/safe_pose.json`.

Joint order:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

## 1. Độ đầy đủ dữ liệu

### Run `recorded_obs-0629-01`

- Server chunks: `40`
- Client chunks: `40`
- Client events: `376`
- Executed actions: `336`
- Metadata frames: `45`
- State dim: `6`
- Camera frames:
  - camera1: `41`, resolution `1280x720`
  - camera2: `41`, resolution `640x360`
- Duration theo metadata: `22.95s`

### Run `recorded_obs-0629-02`

- Server chunks: `27`
- Client chunks: `27`
- Client events: `265`
- Executed actions: `238`
- Metadata frames: `28`
- State dim: `6`
- Camera frames:
  - camera1: `27`, resolution `1280x720`
  - camera2: `27`, resolution `640x360`
- Duration theo metadata: `15.20s`

Kết luận: dữ liệu đủ để phân tích pipeline runtime.

## 2. Safe-pose drift có thật trong current state

Metric chính là projection `p` trên trục:

```text
start_pose -> safe_pose
```

- `p = 0`: start pose lúc bắt đầu run.
- `p = 1`: safe pose.

### Run `0629-01`

- `dist(start, safe) = 144.57 deg`
- Current state:
  - đầu run: `p = 0.000`, `dist_safe = 144.57`
  - cuối run: `p = 0.884`, `dist_safe = 29.34`
- Nghĩa là robot đã đi rất sâu về phía safe pose.

Timeline quan trọng:

| metadata idx | timestep | elapsed | p(current) | dist_safe |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.00s | 0.000 | 144.57 |
| 10 | 49 | 4.08s | 0.096 | 132.16 |
| 20 | 134 | 9.79s | 0.444 | 87.85 |
| 24 | 168 | 12.08s | 0.572 | 81.75 |
| 28 | 202 | 14.42s | 0.214 | 118.73 |
| 36 | 257 | 18.32s | 0.507 | 80.03 |
| 40 | 291 | 20.67s | 0.860 | 33.99 |
| 44 | 325 | 22.95s | 0.884 | 29.34 |

Run 01 có dao động: robot đi về safe, có đoạn quay lại gần start/approach, rồi lại đi mạnh về safe.

### Run `0629-02`

- `dist(start, safe) = 144.29 deg`
- Current state:
  - đầu run: `p = 0.000`, `dist_safe = 144.29`
  - cuối run: `p = 0.886`, `dist_safe = 28.68`

Timeline quan trọng:

| metadata idx | timestep | elapsed | p(current) | dist_safe |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.00s | 0.000 | 144.29 |
| 4 | 33 | 2.45s | 0.044 | 138.84 |
| 6 | 50 | 3.58s | 0.306 | 104.77 |
| 8 | 67 | 4.72s | 0.662 | 71.25 |
| 10 | 84 | 5.86s | 0.825 | 43.70 |
| 16 | 135 | 9.29s | 0.914 | 20.09 |
| 25 | 205 | 14.05s | 0.928 | 19.89 |
| 27 | 222 | 15.20s | 0.886 | 28.68 |

Run 02 xác nhận safe-pose drift rất rõ, sớm hơn run 01.

## 3. Ảnh xác nhận robot thật đi về vùng safe/rest

Các montage đã sinh:

- `recorded_obs-0629-01/analysis_runtime/camera1_timeline_montage.jpg`
- `recorded_obs-0629-01/analysis_runtime/camera2_timeline_montage.jpg`
- `recorded_obs-0629-02/analysis_runtime/camera1_timeline_montage.jpg`
- `recorded_obs-0629-02/analysis_runtime/camera2_timeline_montage.jpg`

Quan sát bằng mắt:

- Camera1 và camera2 đều cho thấy robot/camera rời vùng start/approach và đi về tư thế/rest vùng safe.
- Run 02 đặc biệt rõ: từ khoảng `t=3.6s` đến `t=5.9s`, robot chuyển mạnh về vùng safe/rest.
- Camera2 loại trừ khả năng đây chỉ là ảo giác do wrist camera.

## 4. RTC không tự tạo safe action trong hai run này

Kiểm tra server:

```text
sent_action == postprocessed_action[rtc_real_delay:]
```

Kết quả:

### Run `0629-01`

- `sent - post same index`: mean `8.46`, median `6.69`, max `36.85`
- `sent - post shifted by rtc_real_delay`: mean `0.0`, median `0.0`, max `0.0`
- `rtc_real_delay`: first chunk `7`, các chunk còn lại `3`

### Run `0629-02`

- `sent - post same index`: mean `7.56`, median `5.81`, max `45.94`
- `sent - post shifted by rtc_real_delay`: mean `0.0`, median `0.0`, max `0.0`
- `rtc_real_delay`: first chunk `2`, các chunk còn lại `3`

Kết luận:

- RTC ở đây chủ yếu cắt bỏ vài action đầu theo latency delay.
- Không thấy RTC merge tạo ra action mới khác `postprocessed_action`.
- Nếu action kéo về safe, nó đã có trong `postprocessed_action`, tức output policy sau unnormalizer trên server.

## 5. Client command không lệch khỏi command gửi motor

`performed_action` trong client log bằng command dict gửi xuống robot:

### Run `0629-01`

- `target_vs_performed`: mean `0.0`, max `0.0`

### Run `0629-02`

- `target_vs_performed`: mean `0.0`, max `0.0`

Lưu ý: `performed_action` nhiều khả năng là echo/action command returned từ `send_action`, không phải measured state sau motor settle. Vì vậy tracking thật phải đọc từ `metadata.state`.

## 6. Robot state bám command tương đối tốt

So current state trong `metadata.jsonl` với command gần nhất trước hoặc tại cùng timestep:

### Run `0629-01`

- tracking-ish error:
  - mean `6.74 deg`
  - median `5.94 deg`
  - max `18.24 deg`

### Run `0629-02`

- tracking-ish error:
  - mean `5.35 deg`
  - median `4.69 deg`
  - max `12.54 deg`

Kết luận:

- Không thấy bằng chứng chính rằng robot dynamics/controller tự kéo về safe khác với command.
- Current state nhìn chung đang đi theo command đã được policy/runtime phát ra.

## 7. Client aggregation có xảy ra, nhưng không phải nguồn chính

Client dùng `weighted_average`, nên action trùng timestep giữa chunk cũ/mới bị aggregate.

### Run `0629-01`

- Executed actions: `336`
- Executed action có aggregation meta: `313`
- `old-new action L2`: mean `24.65`, median `16.40`, max `106.03`
- `target-new action L2`: mean `7.40`
- `target-old action L2`: mean `17.26`

### Run `0629-02`

- Executed actions: `238`
- Executed action có aggregation meta: `218`
- `old-new action L2`: mean `14.90`, median `11.44`, max `59.33`
- `target-new action L2`: mean `4.47`
- `target-old action L2`: mean `10.43`

Ý nghĩa:

- Aggregation thật sự làm smoothing giữa chunk cũ và mới.
- Nhưng target sau aggregation gần chunk mới hơn chunk cũ, đúng công thức `0.3 old + 0.7 new`.
- Vì `postprocessed_action` mới đã tự kéo về safe trên recorded observation, aggregation không phải nguồn gốc đầu tiên của safe drift.
- Aggregation vẫn có thể khuếch đại/ổn định hướng đi sai bằng cách giữ queue dài và trộn nhiều chunk không nhất quán.

## 8. Khi nào raw/postprocessed policy bắt đầu kéo về safe?

### Run `0629-02`: pattern rất rõ

Từ `obs_timestep=50`, current đã ở `p=0.306`, policy bắt đầu dự đoán sâu hơn về safe:

| chunk | obs | elapsed | p(current) | p(post first) | p(post end) | p(sent first) | p(sent end) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.00s | 0.000 | -0.014 | 0.013 | -0.005 | 0.013 |
| 1 | 16 | 1.30s | 0.001 | 0.013 | 0.188 | 0.027 | 0.188 |
| 4 | 35 | 2.58s | 0.069 | 0.112 | 0.129 | 0.186 | 0.129 |
| 5 | 50 | 3.58s | 0.306 | 0.425 | 0.661 | 0.399 | 0.661 |
| 6 | 52 | 3.72s | 0.300 | 0.396 | 0.951 | 0.462 | 0.951 |
| 7 | 67 | 4.72s | 0.662 | 0.673 | 0.347 | 0.763 | 0.347 |
| 9 | 84 | 5.86s | 0.825 | 0.766 | 0.823 | 0.783 | 0.823 |
| 16 | 137 | 9.42s | 0.914 | 0.903 | 0.726 | 0.925 | 0.726 |

Kết luận run 02:

- Ban đầu policy rất conservative/near-current.
- Sau khi state hơi rời start, chunk tại obs `50/52` đã chứa action endpoint gần safe.
- Đây là raw/postprocessed policy behavior trên recorded observation, không phải RTC sáng tạo ra.

### Run `0629-01`: chậm hơn, có dao động

Các mốc chính:

| chunk | obs | elapsed | p(current) | p(post first) | p(post end) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.00s | 0.000 | -0.005 | 0.019 |
| 5 | 49 | 4.08s | 0.096 | 0.104 | 0.378 |
| 7 | 66 | 5.22s | 0.119 | 0.094 | 0.810 |
| 15 | 134 | 9.79s | 0.444 | 0.569 | 0.934 |
| 28 | 238 | 17.05s | 0.153 | 0.115 | 0.799 |
| 30 | 255 | 18.19s | 0.476 | 0.611 | 0.813 |
| 38 | 323 | 22.82s | 0.870 | 0.843 | 0.822 |

Kết luận run 01:

- Có nhiều chunk endpoint kéo về safe từ khá sớm (`obs=49/66`).
- Run này dao động mạnh: có lúc policy/robot quay lại gần start/approach rồi lại kéo về safe.

## 9. Under-active / stay-current bias vẫn xuất hiện ở đầu run

Đầu run:

### Run `0629-01`

- First command:
  - `target_dist_current = 8.83 deg`
  - `p(target) = 0.039`
- Nhiều command đầu vẫn quanh `p=0.02-0.10`.

### Run `0629-02`

- First command:
  - `target_dist_current = 3.49 deg`
  - `p(target) = -0.005`
- Đến action timestep `20`, target vẫn gần start:
  - `p(target) ≈ -0.001`
  - `dist_safe ≈ 144.94`

Kết luận:

- Nghi vấn “new model under-active / near-current early action” được runtime log ủng hộ.
- Nhưng lỗi safe drift không chỉ là đứng yên: sau khi hệ closed-loop đi vào một state trung gian, policy chuyển sang dự đoán hướng safe.

## 10. Timing/latency/queue

### Run `0629-01`

- Server inference:
  - mean `167.8ms`
  - median `160.0ms`
  - max `420.3ms` ở first chunk
- Server total:
  - mean `176.0ms`
  - max `504.4ms`
- Server->client latency:
  - mean `292.1ms`
  - median `280.8ms`
- Control loop action interval:
  - mean `0.068s`, gần 15Hz
- Queue before execute:
  - mean `21.5`
  - min `10`
  - max `30`

### Run `0629-02`

- Server inference:
  - mean `161.1ms`
  - median `162.9ms`
  - max `165.6ms`
- Server total:
  - mean `167.3ms`
- Server->client latency:
  - mean `248.4ms`
  - median `243.5ms`
- Control loop action interval:
  - mean `0.067s`, đúng 15Hz
- Queue before execute:
  - mean `21.9`
  - min `14`
  - max `33`

Kết luận:

- Không thấy control loop bị chậm nghiêm trọng.
- Queue khá dài, overlapping/aggregation xảy ra nhiều.
- Latency khiến RTC drop `2-3` action đầu mỗi chunk. Nhưng do `sent = post[delay:]`, đây không phải nguồn tạo safe action.

## 11. Kết luận theo từng nghi vấn trong `VLA-Suspected-Issues.md`

### 11.1 New model under-active / stay-current bias

**Được ủng hộ.**

- Initial commands gần current/start.
- Run 02 đầu run gần như không tiến task mạnh: target projection quanh `0.0`.

### 11.2 Safe-pose attractor trong raw policy

**Được chứng minh trên recorded observations thật, nhưng không phải ngay ở frame đầu.**

- Offline dataset-frame probe trước đó chưa thấy safe attractor.
- Runtime recorded obs cho thấy sau vài bước, `postprocessed_action` tự kéo theo trục safe.
- Mốc rõ nhất là run 02 chunk `obs=50/52`.

### 11.3 Raw multi-mode stochastic policy

**Chưa kết luận từ JSONL runtime.**

- Mỗi observation runtime chỉ có một inference sample.
- Cần offline replay nhiều seed trên các recorded observations đáng ngờ.

### 11.4 Overshoot

**Không phải trọng tâm hai run này.**

- Hành vi nổi bật là safe drift, không phải đi quá cốc.
- Ảnh cho thấy robot không hoàn thành approach/grasp.

### 11.5 Horizon/chunk size

**Không thấy bằng chứng chunk size 35 là lỗi trực tiếp.**

- Raw/post chunk length luôn `35`.
- RTC delay chỉ shift chunk theo latency.

### 11.6 Aggregation / queue / receding horizon

**Có vai trò runtime, nhưng không phải nguồn gốc đầu tiên.**

- Aggregation xảy ra rất nhiều.
- Các chunk cũ/mới đôi khi khác xa nhau.
- Nhưng safe-directed action đã có trong chunk mới sau postprocess.

### 11.7 RTC

**Không phải nguyên nhân trực tiếp trong hai run này.**

- `sent_action == postprocessed_action[rtc_delay:]` chính xác.

### 11.8 Robot dynamics/controller/tracking

**Không phải nghi phạm chính theo dữ liệu hiện có.**

- Current state bám command tương đối tốt.
- Không thấy current state tự về safe khi command không về safe.

## 12. Kết luận ngắn

Hai run đã xác nhận:

1. Safe-pose drift là thật trong current state và ảnh.
2. RTC không tự tạo safe action.
3. Client command không bị lệch khỏi command gửi xuống robot.
4. Robot/current state nhìn chung đi theo command.
5. Action kéo về safe xuất hiện trong `postprocessed_action`, tức raw policy sau unnormalizer trên recorded observation thật.
6. Lỗi có dạng closed-loop:

```text
initial observation -> conservative action gần current
      -> robot rời start một chút
      -> recorded observation/state rơi vào vùng policy dự đoán return/safe
      -> policy chunk kéo mạnh về safe
      -> robot bám command và đi về safe
```

## 13. Cần phân tích model offline tiếp theo

Từ các file runtime, ta đã xác định tầng lỗi nằm ở policy output trên recorded observation thật.

Phần còn thiếu là: **vì sao policy output đó kéo về safe?**

Cần offline model replay/ablation trên chính recorded observations:

1. Re-run checkpoint trên các frame runtime đáng ngờ để xác nhận output tái lập server log.
2. Tách ảnh và state:
   - current image + current state;
   - start image + current state;
   - current image + start state;
   - camera1 swap;
   - camera2 swap.
3. Chạy nhiều seed trên cùng recorded observation để kiểm tra stochastic multi-mode.
4. So old checkpoint vs new checkpoint trên cùng recorded observation.
5. Plot projection `p(predicted chunk)` so với `p(current_state)`.

Notebook đề xuất: `xai/notebooks/offline_replay_recorded_obs_0629.ipynb`.

## 14. Cập nhật sau offline replay notebook

Notebook `xai/notebooks/offline_replay_recorded_obs_0629.ipynb` đã chạy xong và sinh:

- `xai/artifacts/offline_replay_recorded_obs_0629_outputs/ablation_predictions.csv`
- `xai/artifacts/offline_replay_recorded_obs_0629_outputs/ablation_summary.csv`
- `xai/artifacts/offline_replay_recorded_obs_0629_outputs/recorded_obs-0629-01_ablation_p_end.png`
- `xai/artifacts/offline_replay_recorded_obs_0629_outputs/recorded_obs-0629-02_ablation_p_end.png`

Kết quả này từng **supersede** một phần kết luận ở mục 11-13, nhưng sau đó đã bị dữ liệu live non-RTC mới bác bỏ thêm. Xem mục 15.

### 14.1 Notebook chạy hợp lệ

- Cell 1 xác nhận chạy trên `cuda`, cả hai run `recorded_obs-0629-01` và `recorded_obs-0629-02` tồn tại.
- Cell 2 chọn probe timesteps:
  - `0629-01`: 20 điểm từ `0` tới `325`.
  - `0629-02`: 21 điểm từ `0` tới `222`.
- Cell 3 load `new_latest = di-techinnova/smolvla-pouring-0.3-cutted`.
  - Chỉ model mới được chạy; old 50-episode revision chưa được replay trong notebook này.
  - Có warning tokenizer/model config, nhưng inference vẫn hoàn tất.
- Cell 5 sinh `3280` predictions, `0` errors.
- Cell 6 sinh `164` summary rows.
- Cell 7 sinh đủ 2 plot ablation.

### 14.2 Base non-RTC policy không tự kéo về safe pose

Trong `ablation_summary.csv`, với ablation chính `current_image_current_state`:

- Không có row nào thỏa `p_end_mean > current_p + 0.05`.
- Không có row nào thỏa `p_first_mean > current_p + 0.05`.

Tổng hợp theo run:

| run | current_p mean | p_end_mean | end_minus_current_p mean |
|---|---:|---:|---:|
| `recorded_obs-0629-01` | `0.528` | `0.045` | `-0.483` |
| `recorded_obs-0629-02` | `0.779` | `0.132` | `-0.647` |

Nghĩa là khi replay offline theo đường **base non-RTC**, model không dự đoán đi sâu hơn về safe pose. Ngược lại, chunk endpoint thường nằm thấp hơn rất nhiều so với current projection.

### 14.3 Ảnh plot xác nhận điều này

Hai plot:

- `recorded_obs-0629-01_ablation_p_end.png`
- `recorded_obs-0629-02_ablation_p_end.png`

đều có cùng pattern:

- Đường `current_image_current_state` luôn nằm thấp hơn đường `p_end = current_p`.
- Đường `start_image_current_state` cao hơn `current_image_current_state`, nhưng vẫn nằm thấp hơn xa current khi current đã gần safe.
- Hai ablation dùng `start_state` gần như flat quanh `p_end ~= -0.11`.
- Không đường nào tiến tới vùng `p=1`.

Vì vậy hình ảnh không ủng hộ giả thuyết base model thuần đang có safe-pose attractor trên recorded observations.

### 14.4 Runtime server và offline replay khác nhau rất lớn

Đối chiếu cùng timestep giữa runtime `server_actions.jsonl` và offline replay `current_image_current_state`:

| run | obs | current p | runtime post end p | offline non-RTC end p |
|---|---:|---:|---:|---:|
| `0629-01` | `49` | `0.096` | `0.378` | `-0.078` |
| `0629-01` | `66` | `0.119` | `0.810` | `-0.055` |
| `0629-01` | `119` | `0.158` | `0.742` | `-0.043` |
| `0629-01` | `134` | `0.444` | `0.934` | `0.040` |
| `0629-02` | `50` | `0.306` | `0.661` | `-0.010` |
| `0629-02` | `52` | `0.300` | `0.951` | `-0.014` |
| `0629-02` | `101` | `0.829` | `0.913` | `0.147` |

Sai khác trung bình `runtime post end p - offline end p`:

- `0629-01`: `+0.613`
- `0629-02`: `+0.543`

Đây là sai khác quá lớn để coi là noise seed thông thường.

### 14.5 Chẩn đoán tạm thời từ notebook: nghi phạm RTC-conditioned generation

Source server cho thấy khi `rtc_enabled=True`, action không được sinh bởi base call:

```python
policy.predict_action_chunk(observation)
```

mà bởi:

```python
policy.predict_action_chunk(
    observation,
    inference_delay=current_delay,
    prev_chunk_left_over=prev_actions,
)
```

Sau đó server mới log `postprocessed_action`.

Vì vậy kết luận cũ "`postprocessed_action` nghĩa là raw/base policy sau unnormalizer" là chưa chính xác trong RTC mode. Trong hai run này, `postprocessed_action` là action sau **RTC-conditioned model generation**, rồi mới unnormalize.

Điều vẫn đúng:

- `sent_action == postprocessed_action[rtc_real_delay:]`.
- Phần cắt delay không tự tạo vector mới.

Điều cần sửa:

- RTC không chỉ cắt delay ở cuối pipeline.
- RTC còn đi vào chính `predict_action_chunk()` thông qua `prev_chunk_left_over` và `inference_delay`.
- Safe-directed action xuất hiện trước khi gửi client, nhưng nhiều khả năng đã xuất hiện do RTC-conditioned generation, không phải base non-RTC model.

### 14.6 Các giả thuyết sau notebook

**Nghi phạm chính: RTC path dependence / history conditioning.**

- Runtime bật `rtc_enabled=true`.
- Offline replay hiện tại là non-RTC.
- Runtime có safe-pull mạnh.
- Offline non-RTC không có safe-pull.
- Source xác nhận RTC truyền `prev_chunk_left_over` vào model.

**Nghi phạm phụ: queue/aggregation làm khuếch đại sai lệch.**

- Client queue dài và overlap nhiều.
- Aggregation không phải nguồn đầu tiên, nhưng có thể giữ hướng sai đủ lâu để robot đi sâu về safe.

**Yếu đi: underfit/base safe attractor.**

- Nếu base model underfit trực tiếp về safe, offline non-RTC phải tái hiện safe-pull trên recorded obs.
- Kết quả hiện tại không tái hiện.

**Yếu đi: stochastic multimode là nguyên nhân chính.**

- 20 seeds mỗi điểm.
- `p_end_std` thấp, thường khoảng `0.005-0.018`.
- Không có seed-mean nào kéo vượt current về safe.

**Yếu đi: image trigger đơn lẻ.**

- `current_image_start_state` gần flat quanh start-ish output.
- `start_image_current_state` vẫn theo current state nhiều hơn image.
- Image có ảnh hưởng, nhưng không đủ giải thích safe drift.

**Yếu đi: robot/controller tự về safe.**

- Current state bám command tương đối tốt.
- Runtime command/action đã đi theo hướng safe trước khi robot tới đó.

### 14.7 Kết luận tạm thời sau notebook

Mô hình lỗi hiện tại hợp lý nhất là:

```text
base non-RTC SmolVLA trên recorded obs
    -> không kéo safe rõ ràng

live RTC SmolVLA
    -> dùng prev_chunk_left_over + inference_delay
    -> chunk mới bị điều kiện hóa bởi history/queue
    -> một sai lệch nhỏ ban đầu được RTC làm thành trajectory liên tục
    -> postprocessed_action trong server log kéo mạnh về safe
    -> client nhận/aggregate action đó
    -> robot bám command và đi về safe
```

Vì vậy bước kiểm chứng tiếp theo nên ưu tiên:

1. Chạy live hoặc offline replay với `rtc_enabled=false`.
2. Chạy RTC nhưng giảm `rtc_max_guidance_weight` từ `10.0` xuống `1.0-2.0`.
3. Log thêm projection của `prev_chunk_left_over` trước mỗi inference.
4. Log song song base non-RTC chunk và RTC-conditioned chunk trên cùng observation để đo trực tiếp delta.

## 15. Cập nhật sau live probe non-RTC và RTC2

Hai run mới:

- `recorded_obs-0629-nonrtc`
- `recorded_obs-0629-rtc2`

đã bác bỏ kết luận tạm thời ở mục 14 rằng RTC là nguồn chính.

### 15.1 Non-RTC live vẫn đi về safe pose

Run `recorded_obs-0629-nonrtc`:

- `server_actions.jsonl` xác nhận `rtc_enabled = false` cho toàn bộ `21/21` chunks.
- `rtc_real_delay = None` cho toàn bộ chunks.
- Current state:
  - đầu run: `p = 0.000`
  - cuối run: `p = 0.897`
  - max: `p = 0.927`
  - dist-to-safe giảm từ `147.68` xuống `28.63`.
- Server action:
  - `postprocessed_action == sent_action` về mặt horizon; non-RTC không có cắt RTC.
  - `9/21` chunks có `post_end_p > current_p + 0.05`.
  - Các chunk rõ nhất:

| obs | current p | post end p | delta |
|---:|---:|---:|---:|
| `17` | `0.206` | `0.776` | `+0.570` |
| `19` | `0.171` | `0.660` | `+0.489` |
| `38` | `0.224` | `0.829` | `+0.605` |
| `55` | `0.548` | `0.758` | `+0.211` |
| `57` | `0.596` | `0.852` | `+0.256` |

Kết luận: live safe drift **không cần RTC**.

### 15.2 RTC2 cũng đi về safe pose

Run `recorded_obs-0629-rtc2`:

- `server_actions.jsonl` xác nhận `rtc_enabled = true`.
- `rtc_max_guidance_weight = 2.0` theo config probe.
- Current state:
  - đầu run: `p = 0.000`
  - cuối run: `p = 0.886`
  - max: `p = 0.967`
  - dist-to-safe giảm từ `147.79` xuống `29.56`.
- Server action:
  - `10/19` chunks có `post_end_p > current_p + 0.05`.
  - Các chunk rõ nhất:

| obs | current p | post end p | delta |
|---:|---:|---:|---:|
| `15` | `0.292` | `0.829` | `+0.538` |
| `17` | `0.283` | `0.617` | `+0.334` |
| `32` | `0.403` | `0.866` | `+0.463` |
| `34` | `0.385` | `0.923` | `+0.538` |
| `49` | `0.587` | `0.915` | `+0.328` |

Kết luận: giảm RTC guidance từ `10` xuống `2` không loại bỏ safe drift.

### 15.3 Kết luận sửa lại

Kết luận đúng hơn sau live non-RTC:

```text
safe drift xuất hiện trong live base/non-RTC policy output
    -> không phải do RTC là nguồn chính
    -> không phải do client aggregation là nguồn chính
    -> không phải do robot/controller tự trôi về safe
```

RTC và aggregation vẫn có thể làm mượt/khuếch đại trajectory đã sai, nhưng chúng không phải điều kiện cần.

### 15.4 Mâu thuẫn còn lại: offline non-RTC notebook không tái tạo live non-RTC

Notebook offline trước đó chạy non-RTC trên hai run cũ `0629-01/02` nhưng không thấy safe-pull. Live non-RTC mới lại safe-pull rất rõ.

Sau khi kiểm tra lại notebook, có một lỗi pipeline replay cụ thể:

- Server live dùng `raw_observation_to_observation()`.
- Notebook cũ dùng `prepare_raw_observation()` trực tiếp.
- Vì vậy notebook thiếu bước `prepare_image(v).unsqueeze(0)`, tức thiếu scale ảnh về `[0, 1]` và thiếu batch dim theo đúng server path trước policy preprocessor.

Do đó kết quả notebook offline cũ **không còn là bằng chứng hợp lệ** để bác bỏ server JSONL. Notebook đã được sửa để dùng `raw_observation_to_observation()` giống server.

Vì vậy nghi phạm quan trọng hiện tại không còn là RTC, mà là một trong các điểm sau:

1. Base SmolVLA live thật sự dự đoán safe-pull trong một số observation/state.
2. Offline replay trước đó sai pipeline nên không tái tạo server.
3. Cần rerun notebook đã sửa trên chính `recorded_obs-0629-nonrtc` và các run cũ để khớp server JSONL.
4. Sau khi replay đúng pipeline, nếu vẫn lệch server thì mới xét tiếp khác biệt transport/JPEG/state serialization.

Bước tiếp theo nên là replay offline trên chính `recorded_obs-0629-nonrtc`, không phải hai run cũ, rồi so `postprocessed_action` offline với `server_actions.jsonl` cùng timestep.

## 16. Cập nhật sau notebook replay đã sửa pipeline

Notebook `xai/notebooks/offline_replay_recorded_obs_0629.ipynb` đã được sửa để dùng đúng path giống server:

```text
raw_observation_to_observation()
```

thay vì gọi trực tiếp `prepare_raw_observation()`. Sau khi rerun, notebook xuất:

- `ablation_predictions.csv`: `12,160` rows, `0` errors.
- `ablation_summary.csv`: `608` rows.
- `server_replay_compare.csv`: `184` rows.
- 8 plot ablation cho 4 run x 2 image variants.

Các run đã replay:

- `recorded_obs-0629-01`
- `recorded_obs-0629-02`
- `recorded_obs-0629-nonrtc`
- `recorded_obs-0629-rtc2`

Các image variants:

- `saved_rgb`
- `server_jpeg_bgr_q90`

`server_jpeg_bgr_q90` mô phỏng đường online hiện tại: ảnh RGB được encode JPEG quality 90 rồi decode bằng OpenCV, tức mảng trả về có thứ tự BGR.

### 16.1 Offline replay đã tái hiện được hướng safe-pull của server

So sánh `server_actions.jsonl` với offline replay trên cùng observation/timestep:

| run | RTC | image variant | n | server safe-pull | offline safe-pull | mean abs delta p_end |
|---|---:|---|---:|---:|---:|---:|
| `recorded_obs-0629-01` | true | `saved_rgb` | 20 | 10 | 8 | 0.128 |
| `recorded_obs-0629-01` | true | `server_jpeg_bgr_q90` | 20 | 10 | 8 | 0.161 |
| `recorded_obs-0629-02` | true | `saved_rgb` | 21 | 3 | 2 | 0.095 |
| `recorded_obs-0629-02` | true | `server_jpeg_bgr_q90` | 21 | 3 | 2 | 0.089 |
| `recorded_obs-0629-nonrtc` | false | `saved_rgb` | 18 | 7 | 5 | 0.080 |
| `recorded_obs-0629-nonrtc` | false | `server_jpeg_bgr_q90` | 18 | 7 | 5 | 0.088 |
| `recorded_obs-0629-rtc2` | true | `saved_rgb` | 18 | 9 | 7 | 0.093 |
| `recorded_obs-0629-rtc2` | true | `server_jpeg_bgr_q90` | 18 | 9 | 7 | 0.094 |

Kết luận: notebook cũ sai pipeline, nhưng notebook mới đã tái hiện được pattern chính. Lỗi safe-drift không phải artifact riêng của server runtime; offline policy forward trên recorded obs cũng tạo hướng kéo về safe ở nhiều điểm.

Một số mismatch lớn vẫn còn:

| run | obs | current p | server p_end | offline p_end | note |
|---|---:|---:|---:|---:|---|
| `recorded_obs-0629-01` | 134 | 0.444 | 0.934 | 0.276 | server kéo safe, offline không |
| `recorded_obs-0629-01` | 66 | 0.119 | 0.810 | 0.350 | cùng hướng nhưng offline yếu hơn |
| `recorded_obs-0629-02` | 120 | 0.864 | 0.449 | 0.767 | server đi ngược khỏi safe mạnh hơn |
| `recorded_obs-0629-rtc2` | 32 | 0.403 | 0.866 | 0.589 | cùng hướng nhưng offline yếu hơn |

Vì vậy offline replay hiện đủ tốt để chẩn đoán hướng lỗi, nhưng chưa phải bit-exact với server. Các khác biệt còn lại có thể đến từ processor stateful/context, sampling seed, hoặc chi tiết dtype/device, nhưng không đảo ngược kết luận chính.

### 16.2 JPEG/BGR không giải thích safe-drift

Nếu lỗi do màu ảnh/JPEG transport, `server_jpeg_bgr_q90` phải đổi pattern lớn so với `saved_rgb`. Kết quả không như vậy:

- `recorded_obs-0629-nonrtc`: safe-pull offline vẫn `5/18` ở cả hai variants.
- `recorded_obs-0629-rtc2`: safe-pull offline vẫn `7/18` ở cả hai variants.
- Mean abs delta với server chỉ dao động nhỏ, không có dấu hiệu BGR/JPEG là nguyên nhân chính.

Kết luận: nên coi RGB/BGR/JPEG là rủi ro phụ cần clean sau, không phải lời giải thích cho hiện tượng safe pose.

### 16.3 Ablation chỉ ra state là tín hiệu chi phối

Notebook chạy 4 ablations:

- `current_image_current_state`
- `current_image_start_state`
- `start_image_current_state`
- `start_image_start_state`

Kết quả tổng hợp `saved_rgb`:

| run | ablation | n obs | mean p_end-current_p | safe-pull count |
|---|---|---:|---:|---:|
| `recorded_obs-0629-nonrtc` | `current_image_current_state` | 18 | -0.004 | 5 |
| `recorded_obs-0629-nonrtc` | `current_image_start_state` | 18 | -0.648 | 1 |
| `recorded_obs-0629-nonrtc` | `start_image_current_state` | 18 | +0.134 | 6 |
| `recorded_obs-0629-nonrtc` | `start_image_start_state` | 18 | -0.554 | 1 |
| `recorded_obs-0629-rtc2` | `current_image_current_state` | 17 | +0.035 | 6 |
| `recorded_obs-0629-rtc2` | `current_image_start_state` | 17 | -0.654 | 1 |
| `recorded_obs-0629-rtc2` | `start_image_current_state` | 17 | +0.204 | 13 |
| `recorded_obs-0629-rtc2` | `start_image_start_state` | 17 | -0.529 | 1 |

Pattern quan trọng:

```text
giữ current_state, dù dùng start_image
    -> vẫn kéo mạnh về safe

giữ start_state, dù dùng current_image
    -> gần như không kéo safe
```

Nói cách khác, safe-drift hiện tại chủ yếu đi qua nhánh state/closed-loop state distribution, không phải do ảnh nhìn thấy cốc hay không. Ảnh hiện tại có thể làm yếu/mạnh hướng kéo, nhưng state là biến quyết định.

### 16.4 Safe-drift xảy ra sớm rồi tự duy trì

Trong `recorded_obs-0629-nonrtc` với `current_image_current_state`:

| obs | elapsed s | current p | offline p_end | delta |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.140 | +0.140 |
| 17 | 1.849 | 0.206 | 0.702 | +0.495 |
| 19 | 2.005 | 0.171 | 0.511 | +0.340 |
| 38 | 3.362 | 0.224 | 0.846 | +0.622 |
| 57 | 4.638 | 0.596 | 0.685 | +0.089 |

Sau khi robot đã gần safe (`p ~ 0.8-0.9`), nhiều chunk không còn tiếp tục tăng p, nhưng lúc đó robot đã bị đưa vào vùng safe rồi. Đây khớp với quan sát live: nó không cần "bật công tắc safe"; chỉ cần vài chunk đầu kéo sai đủ mạnh, sau đó closed-loop rơi vào basin gần safe/rest.

### 16.5 Kết luận hiện tại

Kết luận cập nhật:

```text
safe drift là hành vi của raw/base SmolVLA policy trên live recorded obs,
đặc biệt khi state hiện tại rơi vào một vùng mà model dự đoán trajectory tiến về safe/rest.

RTC không phải nguyên nhân chính.
Client aggregation không phải nguyên nhân chính.
JPEG/BGR không phải nguyên nhân chính.
Robot/controller tự trôi không phải nguyên nhân chính.
```

Nghi phạm còn lại có trọng lượng cao nhất:

1. Dataset/model học một shortcut state -> safe/rest, có thể do phân phối state trong dataset sau cắt/nối/train.
2. Closed-loop compounding: vài action đầu hơi lệch về safe, state mới lại làm model càng chọn mode safe hơn.
3. Model mới trên 200 eps/150K steps có thể đã học basin này mạnh hơn model 50 eps, nhưng notebook hiện tại **chưa so trực tiếp checkpoint old vs new**, nên chưa kết luận được vì sao old có vẻ ổn hơn.

Việc cần làm tiếp theo:

1. Chạy cùng notebook/replay với old checkpoint `05ac253...` và new checkpoint `420c66a...` trên cùng 4 run để đo khác biệt safe-pull.
2. Tạo counterfactual state scan: giữ ảnh start/live cố định, quét state dọc đoạn start -> safe để tìm ngưỡng p nơi model bắt đầu kéo safe.
3. Kiểm tra dataset train quanh vùng state đó: episode nào có state gần start nhưng action/trajectory đi safe/rest.
4. Nếu train lại: thêm eval offline theo safe-pull metric này, không chỉ train loss.
