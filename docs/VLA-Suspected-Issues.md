# VLA Suspected Issues

Tài liệu này gom các lỗi/nghi vấn hiện tại quanh pipeline record -> cut -> train -> infer cho SO-ARM pouring task. Mục tiêu là tách rõ phần đã có bằng chứng, phần chỉ là giả thuyết, và bước kiểm chứng tiếp theo.

## Bối cảnh ngắn

- Task: `Pour from orange cup to blue cup.`
- Dataset chính: `di-techinnova/so-arm-101-pouring-0.3-cutted`
- Model old: train trên khoảng 50 episodes, khoảng 18K steps, hành vi ngoài đời có vẻ ổn hơn.
- Model new: train trên khoảng 175-200 episodes, khoảng 150K steps, inference thường loanh quanh start pose rồi nhìn giống quay về gần safe pose.
- Notebook phân tích: `xai/offline_compare_smolvla_old_new.ipynb`
- Output đã đọc: các bảng dataframe, camera probe, và trajectory plots old/new trên nhiều episode probe.

## 1. New model bị conservative / under-active

**Mức độ nghi ngờ:** cao.

Triệu chứng:

- Robot thường loanh quanh ở start pose.
- Action đầu của new model thường gần current state hơn old model.
- Nhìn ngoài đời giống "về safe pose", nhưng hiện tại chưa đủ bằng chứng rằng model học một safe-pose attractor thật sự.

Bằng chứng từ notebook:

- Trung bình trên các episode probe:
  - `first_action_dist_state`: new khoảng `5.33`, old khoảng `7.87`.
  - Tức new action gần state hiện tại hơn old.
- Theo episode:
  - ep0: new `5.88`, old `14.52`
  - ep1: new `3.84`, old `4.68`
  - ep2: new `3.63`, old `4.03`
  - ep10: new `3.19`, old `4.16`
  - ep51: new `5.38`, old `13.18`
  - ep120: new `6.99`, old `9.06`
  - ep145: new `6.63`, old `7.82`

Giải thích khả dĩ:

- New model không chủ động đi về safe pose từ mọi trạng thái.
- New model có xu hướng output action gần current/reset pose.
- Vì reset/start pose thường gần safe pose, biểu hiện ngoài đời bị nhìn thành "về safe pose".

Bước kiểm chứng:

- Dùng `recorded_obs` của các lần infer fail.
- Plot theo thời gian:
  - `dist(action, current_state)`
  - `dist(action, safe_pose)`
  - `dist(current_state, safe_pose)`
- Nếu `action` luôn gần `current_state`, lỗi là under-active/stay-put.
- Nếu `action` kéo về safe dù `current_state` xa safe, khi đó mới gọi là safe-pose attractor thật.

## 2. Safe-pose attractor chưa được chứng minh

**Mức độ nghi ngờ:** trung bình thấp ở thời điểm hiện tại.

Triệu chứng:

- Khi infer thật, robot sau một lúc thường quay về gần safe pose.

Bằng chứng chống lại giả thuyết "safe pose attractor toàn cục":

- Trong notebook, ở các episode có start state xa safe, new model không kéo action về safe:
  - ep1 `first_action_dist_safe`: new khoảng `147.39`, old khoảng `146.84`
  - ep2: new khoảng `148.19`, old khoảng `149.50`
  - ep10: new khoảng `149.17`, old khoảng `148.84`
- Nếu là attractor toàn cục về safe, các distance này đáng ra phải thấp hơn rõ rệt.

Bằng chứng vẫn còn đáng chú ý:

- Ở ep0 và ep51, start state gần safe, new model output gần safe hơn old.
- ep0:
  - old `first_action_dist_safe`: khoảng `18.43`
  - new `first_action_dist_safe`: khoảng `10.14`

Kết luận tạm:

- Chưa nên gọi đây là lỗi "model học safe pose".
- Gọi chính xác hơn: "model under-active, và start pose gần safe nên nhìn giống về safe".

## 3. Config/action horizon giữa old và new không khớp

**Mức độ nghi ngờ:** cao.

Bằng chứng từ notebook load model:

- Old model:
  - `chunk_size = 30`
  - `n_action_steps = 30`
- New model:
  - `chunk_size = 35`
  - `n_action_steps = 35`

Vì sao đáng ngại:

- Đây là khác biệt thật giữa checkpoint old và new.
- Nếu client/server đang dùng `actions_per_chunk`, `rtc_execution_horizon`, hoặc aggregation theo giả định khác với config train, hành vi có thể bị mượt quá, chậm phản ứng, hoặc conservative hơn.
- Nó làm phép so sánh old/new kém sạch.

Bước kiểm chứng:

- Retrain một bản new-control với cùng:
  - `--policy.chunk_size=30`
  - `--policy.n_action_steps=30`
- So offline trên cùng notebook:
  - `first_action_dist_state`
  - trajectory mean/std
  - chunk trajectory theo từng joint
- Chỉ thay đổi một biến này trước khi thay đổi nhiều thứ khác.

## 4. Training config có thể bị ghi đè hoặc không đúng như lệnh CLI

**Mức độ nghi ngờ:** trung bình cao.

Triệu chứng đã từng thấy:

- Người dùng chạy lệnh train với scheduler/lr/decay cụ thể.
- Training config sau đó có nhiều giá trị khác kỳ vọng, ví dụ decay steps, lr, hoặc action horizon.

Nguy cơ:

- `policy.path=lerobot/smolvla_base` có thể mang sẵn config.
- Hub checkpoint/config có thể ghi đè một phần CLI.
- Một số field policy/scheduler có thể nằm trong nested config và không override theo cách tưởng tượng.

Bước kiểm chứng:

- Sau khi launch train, luôn lưu và đọc lại config thực tế từ output dir.
- So sánh:
  - command line
  - `train_config.json`
  - `policy_config.json`
  - checkpoint pushed lên Hub
- Fail sớm nếu các field critical không khớp:
  - `chunk_size`
  - `n_action_steps`
  - `optimizer_lr`
  - scheduler type
  - warmup/decay steps
  - `freeze_vision_encoder`
  - expert-only / full fine-tune mode nếu có

## 5. New dataset không lệch tổng thể về safe pose

**Mức độ nghi ngờ:** thấp nếu nói dataset toàn safe pose.

Bằng chứng:

- `mean_action_dist_safe` theo bucket vẫn quanh `185-188`.
- First 50 episodes trong dataset mới gần như match old 50 episodes:
  - old 50 frames: `20887`
  - new first50 frames: `20879`
  - action/state mean rất gần nhau.

Kết luận:

- Không thấy bằng chứng rằng việc fix/cut/upload dataset làm toàn bộ data bị kéo về safe pose.
- Vấn đề hành vi new model nhiều khả năng nằm ở training/model output hơn là dataset tổng thể.

## 6. Distribution mới có thể làm model trung bình hóa hành vi nhiều hơn

**Mức độ nghi ngờ:** trung bình.

Quan sát:

- Dataset mở rộng từ 50 episodes lên 175-200 episodes.
- Các bucket mới có một số khác biệt so với first50:
  - mean speed thấp hơn một phần.
  - action/state distribution thay đổi ở elbow/wrist.
  - episode length và phase timing có thể khác.

Vì sao đáng ngại:

- Với imitation learning, nếu cùng observation/task nhưng có nhiều kiểu thao tác hoặc timing khác nhau, model có thể học trung bình hóa.
- Trung bình hóa trong action space tuyệt đối dễ tạo ra "không đủ lực đi tiếp", đặc biệt ở các bước đầu episode.

Bước kiểm chứng:

- So sánh per-phase, không chỉ toàn episode:
  - 0-2s đầu
  - approach
  - grasp/lift
  - pour
  - return/end
- Tính action speed và distance-to-current-state theo phase.
- Train ablation:
  - chỉ first50 với config new
  - first50 + file001
  - first50 + file002
  - full fixed dataset

## 7. Gripper collapse là tín hiệu phụ, không phải giải thích chính cho safe pose

**Mức độ nghi ngờ:** cao cho lỗi gripper, nhưng thấp nếu dùng nó để giải thích safe pose.

Quan sát:

- New model gripper thấp hơn old rõ ở nhiều episode probe:
  - ep50: new khoảng `1.63`, old khoảng `3.70`
  - ep120: new khoảng `1.15`, old khoảng `3.33`
  - ep145: new khoảng `1.47`, old khoảng `3.25`

Ý nghĩa:

- Có thể giải thích lỗi "đi tới gần cốc nhưng không gắp được".
- Không giải thích trực tiếp hiện tượng quay về gần safe pose.

Bước kiểm chứng:

- Tách riêng metric gripper trong offline eval.
- Không dùng gripper làm bằng chứng chính cho lỗi safe-pose.

## 8. RTC nhiều khả năng không phải nguyên nhân trực tiếp

**Mức độ nghi ngờ:** trung bình.

Lý do:

- RTC giới hạn/điều tiết action để tránh giật xa vị trí hiện tại.
- RTC không tự biết safe pose và không nên tự kéo robot về safe pose.
- Nếu policy output đã gần current state hoặc gần reset pose, RTC có thể làm hiện tượng "ì" rõ hơn, nhưng không phải nguồn gốc duy nhất.

Bước kiểm chứng:

- Chạy cùng recorded observation offline:
  - raw policy action
  - action sau aggregation
  - action sau RTC nếu có log được
- Nếu raw policy đã conservative, lỗi nằm trước RTC.
- Nếu raw policy mạnh nhưng sau RTC bị kéo lại, mới nghi RTC/harness.

## 9. Task string chưa phải nghi phạm chính

**Mức độ nghi ngờ:** thấp.

Quan sát:

- Notebook so cả hai task string:
  - `Pour from orange cup into blue cup.`
  - `Pour from orange cup to blue cup.`
- Khác biệt output nhỏ, không đổi bản chất old/new.

Kết luận:

- Sửa instruction có thể giúp rất ít, nhưng không phải nguyên nhân chính của lỗi hiện tại.

## 10. Cần log inference thật để phân biệt các giả thuyết

**Mức độ ưu tiên:** rất cao.

Notebook hiện dùng frame từ dataset để probe offline. Điều này tốt để so old/new, nhưng chưa thay thế được log ngoài đời.

Cần lưu cho mỗi lần infer fail:

- Observation state theo timestep.
- Raw policy action chunk.
- Action sau aggregation.
- Action sau RTC nếu có.
- Current joint state thực tế sau khi execute.
- Ảnh camera tại các timestep chính.

Các plot cần có:

- `dist(raw_action_t0, current_state)`
- `dist(raw_action_t0, safe_pose)`
- `dist(current_state, safe_pose)`
- joint trajectory actual vs commanded.
- action norm/speed theo timestep.

Kết luận mong muốn:

- Nếu raw action gần current state: model under-active.
- Nếu raw action xa current nhưng command sau RTC gần current: RTC/harness can thiệp quá mạnh.
- Nếu raw action kéo về safe ngay cả khi current xa safe: safe-pose attractor thật.

## Kết luận hiện tại

Nghi vấn mạnh nhất không phải "model học safe pose" mà là:

1. New model bị conservative / under-active so với old.
2. Start/reset pose gần safe pose nên biểu hiện ngoài đời giống "về safe pose".
3. Config old/new không khớp (`chunk_size` và `n_action_steps` khác nhau), cần retrain/so lại với config kiểm soát.
4. Dataset tổng thể không có dấu hiệu bị kéo về safe pose, nhưng distribution/timing của data mới có thể làm model trung bình hóa hành vi.
5. Gripper collapse là lỗi phụ rõ ràng, có thể giải thích fail gắp cốc, nhưng không phải bằng chứng chính cho safe-pose.

## Lệnh/việc nên làm tiếp

1. Tạo offline analyzer cho `recorded_obs` từ các lần infer fail.
2. Retrain một bản control dùng cùng horizon với old:
   - `chunk_size=30`
   - `n_action_steps=30`
3. So lại old/new/control bằng cùng notebook.
4. Thêm check tự động sau train để xác nhận config thực tế không bị ghi đè.
5. Nếu control vẫn under-active, làm ablation dataset theo bucket để tìm nhóm episode làm model co hành vi.
