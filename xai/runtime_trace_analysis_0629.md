# Runtime Trace Analysis 2026-06-29

Phân tích này dùng hai run độc lập:

- `recorded_obs-0629-01`
- `recorded_obs-0629-02`

Mỗi run có đủ:

- `server_actions.jsonl`: raw normalized action, postprocessed action, final sent action sau RTC.
- `client_actions.jsonl`: chunk nhận từ server, queue/aggregation, command gửi robot.
- `metadata.jsonl`: current joint state và camera frame path/timestep.
- `images/camera1`, `images/camera2`.
- `safe_pose.json`.

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

Notebook đề xuất: `xai/offline_replay_recorded_obs_0629.ipynb`.
