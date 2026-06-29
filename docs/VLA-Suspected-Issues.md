# VLA Suspected Issues

Tài liệu này tổng hợp các nghi vấn hiện tại quanh pipeline record -> cut -> train -> infer cho SO-ARM pouring task, đã cập nhật theo notebook mới nhất `xai/offline_action_mode_probe_smolvla.ipynb`.

Mục tiêu: tách rõ điều đã có bằng chứng, điều notebook đã làm yếu đi, và điều cần log ở runtime để kết luận.

## Bối cảnh ngắn

- Task: `Pour from orange cup to blue cup.`
- Dataset chính: `di-techinnova/so-arm-101-pouring-0.3-cutted`
- Model old: train khoảng 50 episodes / khoảng 18K steps, ngoài đời có vẻ ổn hơn.
- Model new: train khoảng 175-200 episodes / nhiều steps hơn, ngoài đời có run thành công, run đi lệch/quá cốc, và run đi dần về gần safe pose.
- Notebook cũ: `xai/offline_compare_smolvla_old_new.ipynb`
- Notebook mới: `xai/offline_action_mode_probe_smolvla.ipynb`
- Output notebook mới đã đọc: 36 outputs, 10 ảnh, 10 bảng, 0 error.

## Kết luận cập nhật

Notebook mới **không chứng minh raw new policy có safe-pull/overshoot/multi-mode mạnh trên cùng dataset observation**.

Tín hiệu mạnh nhất hiện tại là:

1. New model thường có early action gần `current_state` hơn old.
2. New model có `chunk_mean_speed` thấp hơn old ở các start-frame probe.
3. Offline phase sweep gần như không thấy safe-pull rõ.
4. Offline phase sweep không thấy overshoot theo threshold hiện tại.
5. Lỗi ngoài đời có thể nằm ở hệ closed-loop: raw policy chunk -> aggregation/queue -> RTC -> robot dynamics -> observation tiếp theo.

Vì vậy nghi vấn chính chuyển từ "raw model tự chọn safe mode ngay lập tức" sang:

**new policy under-active trong early action, và lỗi safe-like drift nhiều khả năng xuất hiện trong runtime closed-loop hoặc trên recorded observations thật, không phải trong dataset-frame one-shot probe.**

## 1. New model under-active / stay-current bias

**Mức độ nghi ngờ:** cao.

Triệu chứng:

- Ngoài đời có run robot loanh quanh hoặc không rời vùng thao tác đủ quyết đoán.
- Offline new model thường có first/early action gần current state hơn old.
- `chunk_mean_speed` của new thấp hơn old trong deep probe start frames.

Bằng chứng từ notebook cũ:

- Trung bình trên probe set:
  - `first_action_dist_state`: new khoảng `5.33`, old khoảng `7.87`.

Bằng chứng từ notebook mới, deep probe 100 seed trên start frame:

- ep0:
  - new `first_action_dist_state_mean = 6.996`
  - old `first_action_dist_state_mean = 15.740`
  - new `chunk_mean_speed_mean = 4.061`
  - old `chunk_mean_speed_mean = 5.181`
- ep51:
  - new `first_action_dist_state_mean = 6.187`
  - old `14.696`
  - new `chunk_mean_speed_mean = 3.997`
  - old `5.267`
- ep120:
  - new `chunk_mean_speed_mean = 3.067`
  - old `4.824`
- ep145:
  - new `chunk_mean_speed_mean = 3.431`
  - old `4.800`

Ở 8/8 start-frame probe, `chunk_mean_speed` của new thấp hơn old.

Kết luận:

- Đây là nghi vấn model-level mạnh nhất hiện tại.
- Nhưng nó **chưa đủ** để giải thích toàn bộ hiện tượng đi dần về safe pose ngoài đời.

Bước kiểm chứng:

- Replay `recorded_obs` thật của run fail.
- Plot theo thời gian:
  - `dist(raw_first_action, current_state)`
  - `dist(raw_chunk_end, current_state)`
  - `chunk_mean_speed`
  - `dist(current_state, safe_pose)`
  - `dist(current_state, start_pose)`

## 2. Safe-pose attractor trong raw policy chưa được chứng minh

**Mức độ nghi ngờ:** thấp đến trung bình, cần recorded_obs thật.

Triệu chứng ngoài đời:

- Một số run robot đi dần về gần safe pose.

Notebook mới làm yếu giả thuyết "raw policy có safe attractor ngay trên dataset-frame":

- Phase sweep safe-pull rate gần như toàn 0.
- Heatmap `new safe-pull rate` chỉ có vài điểm nhỏ, max khoảng `0.12`.
- Deep suspicious frame ep10 frame25 có:
  - `safe_pull_rate_new = 0`
  - `overshoot_rate_new = 0`
  - `stay_current_rate_new = 0.92`

Điểm cần phân biệt:

- `start_pose`: pose robot lúc bắt đầu run/eval.
- `safe_pose`: pose an toàn cố định.

Không được gọi một run là safe-pose drift nếu chỉ thấy robot quanh start pose. Cần chứng minh:

- `dist(current_state, safe_pose)` giảm theo thời gian;
- đồng thời `dist(current_state, start_pose)` tăng hoặc không giảm tương ứng.

Bước kiểm chứng:

- Log runtime và plot:
  - `dist(current_state_t, start_pose)`
  - `dist(current_state_t, safe_pose)`
  - `dist(raw_action_t, safe_pose)`
  - `dist(executed_action_t, safe_pose)`

Nếu raw action không kéo về safe nhưng executed/current state vẫn về safe, nghi phạm nằm sau raw policy: aggregation, RTC, queue, controller, hoặc robot dynamics.

## 3. Raw multi-mode stochastic policy chưa được chứng minh mạnh

**Mức độ nghi ngờ:** trung bình thấp sau notebook mới.

Giả thuyết ban đầu:

- Cùng một observation, new model có thể sinh nhiều mode: success, overshoot, safe-like.

Notebook mới kiểm tra trực tiếp:

- Cùng 1 observation, chạy nhiều seed.
- PCA action chunk và k-means cluster.
- Phase sweep nhiều episode/frame.

Kết quả:

- PCA/cluster xuất hiện ở cả old và new.
- `pca_spread` và `cluster_entropy` new/old khá gần nhau trên start frames.
- ep0:
  - new `pca_spread = 3.671`, old `3.668`
  - new cluster counts `[32, 37, 31]`, old `[28, 40, 32]`
- Không thấy 3 cụm tách thành success/overshoot/safe rõ ràng.
- Suspicious frame ep10 frame25 có cluster `[16, 8, 1]`, trong đó cụm `n=1` là outlier, không phải mode lớn.

Kết luận:

- Không nên coi raw stochastic multi-mode là nguyên nhân chính nếu chỉ dựa vào dataset-frame offline.
- Vẫn có thể có multi-mode trên **recorded_obs thật**, khi camera/view/state đã lệch khỏi training distribution.

Bước kiểm chứng:

- Chạy cùng notebook/probe trên `recorded_obs` từ run fail và run success.
- So PCA/cluster trên từng timestep runtime thật.

## 4. Overshoot chưa xuất hiện trong offline threshold hiện tại

**Mức độ nghi ngờ:** thấp trong raw offline, nhưng chưa loại trừ runtime.

Notebook mới:

- Heatmap `new overshoot rate` bằng 0 toàn bộ.
- Top suspicious observations đều có `overshoot_rate_new = 0`.

Ý nghĩa:

- Offline dataset-frame probe không giải thích được run ngoài đời "đi qua/quá cốc".
- Overshoot ngoài đời có thể xuất hiện do:
  - camera/runtime observation khác dataset;
  - closed-loop tích lũy nhiều action chunk;
  - aggregation/queue execute phần không mong muốn;
  - threshold overshoot trong notebook chưa đo đúng workspace/cup-relative progress.

Bước kiểm chứng:

- Cần metric overshoot theo task/world/image space, không chỉ joint-space first action distance.
- Log cup-relative progress hoặc image-space cup position trong runtime.

## 5. Config/action horizon không còn là nghi phạm chính

**Mức độ nghi ngờ:** thấp đến trung bình.

Bằng chứng:

- Old model:
  - `chunk_size = 30`
  - `n_action_steps = 30`
- New model:
  - `chunk_size = 35`
  - `n_action_steps = 35`

Cập nhật theo phân tích mới:

- `chunk_size=35` không nên mặc định bị coi là lỗi.
- Hugging Face/action autocorrelation gợi ý chunk length khoảng `37` steps, nên `35` là hợp lý với phân phối action hiện tại.
- Khác biệt old/new vẫn cần ghi nhớ khi so sánh, nhưng không phải nghi phạm chính.

Bước kiểm chứng:

- Đảm bảo inference client/server không cắt/aggregate chunk theo giả định sai.
- Log:
  - raw chunk length;
  - số action thực sự execute;
  - chunk queue length;
  - action index trong chunk được execute.

## 6. Aggregation / queue / receding horizon là nghi phạm runtime quan trọng

**Mức độ nghi ngờ:** cao.

Lý do:

- Offline one-shot chunk không tái hiện safe drift rõ.
- Ngoài đời là closed-loop nhiều timestep.
- Nếu mỗi vòng inference lấy chunk mới, phần đầu chunk có thể luôn conservative.
- Nếu aggregation trung bình nhiều chunk không thống nhất, action tiến-task có thể bị triệt tiêu.

Triệu chứng có thể tạo ra:

- robot không đi đủ xa khỏi current/start;
- robot drift về vùng trung tính/safe-like;
- run lúc thành công lúc fail dù start/camera gần giống nhau;
- action ngoài đời khác raw chunk offline.

Bước kiểm chứng:

- Log theo timestep:
  - raw action chunk;
  - aggregated action;
  - action sau RTC;
  - action thật gửi motor;
  - current state sau execute.
- Plot:
  - `dist(raw_action, current_state)` vs `dist(executed_action, current_state)`
  - `dist(raw_action, safe_pose)` vs `dist(executed_action, safe_pose)`
  - chunk index/horizon thực sự được dùng.

## 7. RTC chưa phải nguyên nhân trực tiếp, nhưng có thể khuếch đại lỗi

**Mức độ nghi ngờ:** trung bình.

Lý do:

- RTC không tự biết safe pose.
- RTC không nên tự kéo robot về safe pose.
- Nhưng nếu raw/aggregated action đã conservative hoặc không nhất quán, RTC có thể giữ robot gần current state hơn.

Bước kiểm chứng:

- So raw policy action, aggregated action, action sau RTC.
- Nếu raw action mạnh nhưng sau RTC yếu/kéo lại, nghi RTC/harness.
- Nếu raw action đã yếu, RTC chỉ là yếu tố khuếch đại.

## 8. Dataset tổng thể không lệch về safe pose

**Mức độ nghi ngờ:** thấp nếu nói dataset toàn safe pose.

Bằng chứng từ notebook cũ:

- `mean_action_dist_safe` theo bucket vẫn quanh `185-188`.
- First 50 episodes trong dataset mới gần như match old 50 episodes:
  - old 50 frames: `20887`
  - new first50 frames: `20879`
  - action/state mean rất gần nhau.

Kết luận:

- Không thấy bằng chứng dataset tổng thể bị kéo về safe pose.
- Dataset vẫn có thể gây vấn đề theo phase/timing/style, nhưng không phải theo kiểu toàn dataset safe-biased.

## 9. Dataset distribution / timing vẫn có thể làm model trung bình hóa

**Mức độ nghi ngờ:** trung bình.

Quan sát:

- Dataset mở rộng từ 50 episodes lên 175-200 episodes.
- Các bucket mới có distribution và timing khác first50.
- New model có early action/chunk speed thấp hơn old.

Nguy cơ:

- Imitation learning với action tuyệt đối dễ bị trung bình hóa nếu cùng observation/task có nhiều style/timing.
- Trung bình hóa có thể tạo hành vi under-active, đặc biệt ở phase approach.

Bước kiểm chứng:

- Phân tích per-phase:
  - start/approach;
  - grasp/lift;
  - pour;
  - return.
- Train ablation nếu cần:
  - first50 với recipe mới;
  - first50 + file001;
  - first50 + file002;
  - full dataset.

## 10. Gripper collapse là lỗi phụ, không giải thích safe drift

**Mức độ nghi ngờ:** cao cho gripper, thấp cho safe drift.

Bằng chứng:

- New model gripper thấp hơn old rõ ở nhiều episode:
  - ep50: new khoảng `1.63`, old khoảng `3.70`
  - ep120: new khoảng `1.15`, old khoảng `3.33`
  - ep145: new khoảng `1.47`, old khoảng `3.25`

Ý nghĩa:

- Có thể giải thích lỗi "đi tới gần cốc nhưng không gắp được".
- Không giải thích trực tiếp hiện tượng đi dần về safe pose.

## 11. Task string chưa phải nghi phạm chính

**Mức độ nghi ngờ:** thấp.

Notebook cũ so hai task string:

- `Pour from orange cup into blue cup.`
- `Pour from orange cup to blue cup.`

Khác biệt output nhỏ, không đổi bản chất old/new.

## 12. Cần recorded_obs và full runtime action trace

**Mức độ ưu tiên:** rất cao.

Notebook mới đã cho thấy dataset-frame offline chưa đủ để giải thích lỗi ngoài đời. Bước tiếp theo phải lấy runtime log thật.

Cần log mỗi inference step:

- timestamp;
- camera frame hoặc path;
- `current_state`;
- `start_pose`;
- `safe_pose`;
- raw policy chunk;
- aggregated action;
- RTC/action sau safety layer;
- command gửi motor;
- current state sau execute;
- queue/chunk index đang execute.

Plot bắt buộc:

- `dist(current_state, start_pose)`
- `dist(current_state, safe_pose)`
- `dist(raw_action, current_state)`
- `dist(raw_action, safe_pose)`
- `dist(executed_action, current_state)`
- `dist(executed_action, safe_pose)`
- actual-vs-commanded tracking error

Kết luận mong muốn:

- Raw action đã về safe: model/runtime observation gây safe-pose attractor.
- Raw action không về safe nhưng executed action về safe: aggregation/RTC/queue/controller.
- Current state về safe dù command không về safe: robot dynamics/controller/tracking.
- Raw action chỉ gần current liên tục: under-active/receding-horizon issue.

## Kết luận hiện tại

Nghi vấn mạnh nhất sau notebook mới:

1. New model có **under-active / stay-current bias** trong early action.
2. Safe-pose attractor trong raw policy **chưa được chứng minh**.
3. Raw multi-mode stochastic behavior trên dataset-frame **chưa được chứng minh mạnh**.
4. Overshoot trong raw offline **không xuất hiện theo threshold hiện tại**.
5. Horizon `35` của new model **không còn là nghi phạm chính**, vì phù hợp autocorrelation khoảng `37`.
6. Lỗi ngoài đời nhiều khả năng nằm ở **closed-loop runtime**, đặc biệt aggregation/queue/receding horizon/RTC hoặc observation drift.
7. Cần `recorded_obs` + full action trace trước khi quyết định train lại hay sửa runtime.

## Việc nên làm tiếp

1. Thêm full runtime logger.
2. Replay `recorded_obs` của run success và fail bằng notebook/offline analyzer.
3. Log và so:
   - raw policy chunk;
   - aggregated action;
   - action sau RTC;
   - command thật;
   - current state thật.
4. Sau khi xác định tầng lỗi:
   - nếu raw fail: quay lại model/data/retrain;
   - nếu aggregation/RTC fail: sửa runtime harness;
   - nếu command/tracking fail: sửa controller/latency/action queue.
5. Chỉ train lại sau khi có bằng chứng raw model là nguồn lỗi chính.
