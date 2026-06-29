## VLA Harness For Reliable SO-ARM Deployment

Tài liệu này cập nhật lại ý tưởng VLA Harness theo đúng các lỗi hiện tại của bài toán pouring:

- cùng checkpoint/model có run thành công, run đi lệch/quá cốc, run đi dần về vùng safe pose;
- offline single-frame probe không tái hiện rõ safe-pull hoặc overshoot;
- raw model và policy loss không đủ để dự đoán hành vi closed-loop ngoài đời;
- cần một lớp runtime supervisor để phát hiện, chặn, recover, và log lỗi.

---

### 1. Bài học chính từ các chẩn đoán hiện tại

**Không nên hiểu lỗi hiện tại là một bug đơn giản.**

Các notebook/offline probe cho thấy:

- Dataset frame probe không chứng minh raw policy luôn kéo về safe pose.
- Safe-pull rate trong phase sweep gần như bằng 0, chỉ có vài điểm nhỏ.
- Overshoot rate theo threshold hiện tại bằng 0 trong offline sweep.
- New model thường có first/early action gần current state hơn, và chunk speed thấp hơn old.
- Ngoài đời vẫn có hành vi ngẫu nhiên: thành công, đi lệch/quá cốc, hoặc đi dần về safe pose.

Vì vậy lỗi nhiều khả năng nằm ở **closed-loop system**, không chỉ nằm ở model một bước:

```
camera/runtime obs
      ↓
policy raw action chunk
      ↓
aggregation / queue / receding horizon
      ↓
RTC / safety constraint
      ↓
robot dynamics
      ↓
new observation
      ↓
...
```

Harness phải quan sát cả vòng lặp này, không chỉ nhìn output model một lần.

---

### 2. Phân biệt start pose và safe pose

Hai khái niệm này phải được tách rõ trong mọi log và metric.

| Khái niệm | Ý nghĩa |
|---|---|
| `start_pose` | Joint state tại thời điểm bắt đầu một run/eval. Do người dùng reset robot trước khi infer. |
| `safe_pose` | Pose an toàn cố định/chuẩn để robot quay về hoặc tránh va chạm. Không nhất thiết trùng start pose. |

Lỗi ngoài đời đang mô tả là: robot **đi dần về gần safe pose trong một số run**.

Để phân biệt với việc chỉ loanh quanh start pose, runtime phải log:

- `dist(current_state_t, start_pose)`
- `dist(current_state_t, safe_pose)`
- `dist(raw_action_t, start_pose)`
- `dist(raw_action_t, safe_pose)`
- `dist(executed_action_t, start_pose)`
- `dist(executed_action_t, safe_pose)`

Nếu `dist(current_state, safe_pose)` giảm trong khi `dist(current_state, start_pose)` tăng, đó mới là safe-pose drift thật.

---

### 3. Vì sao raw VLA chưa đủ để deploy

VLA/policy hiện có thể xử lý happy path, nhưng deployment thật cần thêm guardrail vì:

- một chunk xấu có thể đẩy robot vào state ngoài distribution;
- receding horizon có thể liên tục thay chunk trước khi phần hữu ích được execute;
- aggregation nhiều chunk không thống nhất có thể triệt tiêu action;
- camera/view thay đổi sau vài timestep có thể làm policy đổi mode;
- RTC không tự sinh safe pose, nhưng có thể làm action bị giữ quá gần current nếu raw chunk không đủ nhất quán;
- loss/offline frame probe không phản ánh đầy đủ hệ kín ngoài đời.

Điểm quan trọng: Harness không cần biết chính xác model sai vì đâu mới có ích. Nó cần phát hiện **trajectory/runtime bất thường** đủ sớm để chặn và recover.

---

### 4. Kiến trúc mới: runtime supervisor, không chỉ stuck detector

Phiên bản cũ dùng ý tưởng `2s no-movement`. Với lỗi hiện tại, trigger đúng hơn là:

**no-progress / unsafe-progress / invalid-chunk**

Harness gồm 4 lớp:

1. **Pre-execution validator**
   - Kiểm tra action chunk trước khi gửi xuống robot.
   - Chặn chunk có dấu hiệu kéo về safe pose, đứng quá gần current, hoặc đi ra ngoài vùng thao tác.

2. **Progress monitor**
   - Quan sát robot sau mỗi 0.5-1s.
   - Kiểm tra task có tiến triển thật không, thay vì chỉ kiểm tra robot có di chuyển không.

3. **Minimal recovery primitives**
   - Can thiệp nhỏ, có mục tiêu rõ, rồi trả control lại VLA.
   - Không biến harness thành planner toàn nhiệm vụ.

4. **Full runtime logging**
   - Lưu đủ raw/aggregated/executed action và state để chẩn đoán sau mỗi run.

```
Observation
    ↓
VLA predicts action chunk
    ↓
[Pre-execution validator]
    ├─ valid → execute through aggregator/RTC
    └─ invalid → reject/resample/recover/stop
    ↓
[Progress monitor]
    ├─ task progress OK → continue VLA
    └─ no-progress / safe-drift / bad approach → classify failure
           ↓
     minimal recovery primitive
           ↓
      return control to VLA
```

---

### 5. Signals harness nên theo dõi

#### 5.1 Action-space signals

Các metric từ action chunk:

- `dist(raw_first_action, current_state)`
- `dist(raw_chunk_mean, current_state)`
- `dist(raw_chunk_end, current_state)`
- `dist(raw_first_action, safe_pose)`
- `dist(raw_chunk_end, safe_pose)`
- `chunk_mean_speed`
- direction consistency giữa các chunk liên tiếp
- action jerk / sudden reversal

Các dấu hiệu đáng chặn:

- chunk kéo về safe pose khi task chưa cần;
- nhiều chunk liên tiếp gần current state nhưng task không tiến triển;
- chunk đổi hướng mạnh giữa các inference step;
- endpoint đi ra ngoài workspace hoặc vùng cốc;
- action speed quá thấp trong khi policy đang được kỳ vọng approach.

#### 5.2 State/runtime signals

Các metric từ robot state:

- `dist(current_state, start_pose)`
- `dist(current_state, safe_pose)`
- actual-vs-commanded tracking error
- end-effector movement nếu có forward kinematics
- thời gian ở cùng một vùng state
- có đang đi ra khỏi workspace thao tác không

Các dấu hiệu đáng chặn:

- current state tiến gần safe pose qua nhiều timestep dù task chưa xong;
- current state không rời vùng start/approach sau ngưỡng thời gian;
- robot đi lệch xa hướng tới cốc;
- commanded action và actual state không khớp, gợi ý latency/controller issue.

#### 5.3 Vision/task progress signals

Với pouring task:

- orange cup có nằm trong vùng approach/grasp hợp lý không;
- khoảng cách image-space tới cup có giảm không;
- wrist camera có thấy cốc/object không;
- gripper/object relation có hợp lý không;
- sau khi grasp, cup có lift/tilt đúng phase không.

Ban đầu không cần pose estimation hoàn hảo. Chỉ cần progress proxy đủ tốt để phát hiện “đang đi sai”.

---

### 6. Failure modes hiện tại và cách harness xử lý

| Failure mode | Dấu hiệu | Cách xử lý tối thiểu |
|---|---|---|
| Safe-pose drift | `dist(current_state, safe_pose)` giảm liên tục, task chưa xong | reject chunk, stop hoặc nudge về vùng thao tác |
| Start/current dwell | nhiều timestep gần start/current, progress thấp | start-pose escape / approach nudge |
| Bad approach / đi lệch cốc | EE/camera progress không hướng tới cup | resample chunk hoặc CV-guided pre-grasp |
| Chunk inconsistency | chunk liên tiếp đổi hướng mạnh | lower horizon, resample, hoặc dùng median/guarded aggregation |
| Grasp fail | tới gần cốc nhưng không grasp/lift được | re-grasp primitive |
| Runtime tracking fail | executed state không theo command | pause, reduce speed, reset queue |

---

### 7. MVP cho bài toán hiện tại

MVP không nên bắt đầu bằng full planner. Nên làm 3 phần nhỏ nhưng có giá trị ngay.

#### MVP 1: Runtime logger

Mỗi inference step lưu:

- timestamp;
- camera frame key hoặc path;
- `current_state`;
- `start_pose`;
- `safe_pose`;
- raw policy chunk;
- aggregated action;
- RTC/executed action;
- actual state sau execute;
- các metric distance tới start/safe/current.

Mục tiêu: phân biệt lỗi model raw, aggregation, RTC, hay robot dynamics.

#### MVP 2: Pre-execution chunk validator

Trước khi execute chunk:

- reject nếu chunk endpoint tiến gần safe pose bất thường;
- reject nếu chunk nằm quá gần current state quá nhiều lần liên tiếp mà task chưa progress;
- reject nếu chunk ra ngoài workspace/pose bounds;
- log lý do reject.

Action khi reject:

- resample một lần;
- nếu vẫn fail, dùng recovery primitive hoặc stop.

#### MVP 3: Progress monitor cho pouring

Trong 1-2s đầu:

- robot phải rời start theo hướng hợp lý;
- image/camera phải cho thấy cup/object relation tiến triển;
- không được drift về safe pose nếu task chưa xong.

Nếu fail:

- chạy approach nudge tới pre-grasp vùng cốc;
- trả control lại VLA.

---

### 8. Recovery primitives nên có

Các primitive phải nhỏ, bounded, và dễ rollback.

1. **Stop / hold**
   - Dừng an toàn, clear action queue.

2. **Resample chunk**
   - Gọi policy lại với same/latest observation nếu chunk bị validator reject.

3. **Approach nudge**
   - Dịch robot một đoạn nhỏ về vùng thao tác/cốc.
   - Có thể dựa vào CV đơn giản hoặc waypoint đã calibrate.

4. **Pre-grasp alignment**
   - Đưa wrist/EE vào vùng trước cốc, không tự hoàn thành toàn task.

5. **Re-grasp**
   - Nếu đã tới gần cốc nhưng grasp fail, mở/đóng lại gripper với pose chỉnh nhỏ.

6. **Safe stop**
   - Nếu nhiều lần reject/recovery fail, dừng thay vì cố tiếp.

---

### 9. Relation với model/data work

Harness không thay thế model tốt.

Model vẫn cần:

- kiểm tra config train thực tế;
- checkpoint sweep;
- ablation dataset theo bucket/phase;
- kiểm tra processor/preprocessor đúng giữa train/infer;
- phân tích recorded_obs thật.

Nhưng harness giúp deployment ngay cả khi model chưa hoàn hảo:

- chặn chunk xấu;
- phát hiện drift sớm;
- recover tối thiểu;
- cung cấp log đủ để sửa model/data sau đó.

Nói cách khác:

**model improvement làm happy path tốt hơn; harness làm failure path không phá robot/run.**

---

### 10. Khác biệt với pipeline CV/controller truyền thống

| Pipeline truyền thống | VLA harness |
|---|---|
| CV/controller làm toàn bộ task | VLA vẫn làm happy path |
| Phải hand-code task logic chi tiết | Harness chỉ validate/recover failure |
| Thêm task mới thường rewrite planner | Thêm task mới fine-tune VLA, reuse nhiều guardrail |
| Predictable nhưng cứng | Flexible nhưng cần runtime supervision |

Harness không phải là quay về classical robotics hoàn toàn. Nó là lớp **exception handling** quanh VLA.

---

### 11. Đánh giá khả thi

| Component | Khả thi | Ghi chú |
|---|---|---|
| Runtime logger | Rất cao | Cần làm trước mọi thứ khác |
| Distance-to-start/safe metrics | Rất cao | Joint-space metric đủ cho vòng đầu |
| Pre-execution validator | Cao | Có thể bắt đầu bằng rule-based threshold |
| Progress monitor | Trung bình cao | Cần định nghĩa proxy cho pouring |
| Approach nudge | Trung bình | Cần calibration hoặc CV đơn giản |
| Re-grasp primitive | Trung bình | Có thể làm sau khi approach ổn |
| Full CV fallback target | Trung bình thấp | Dễ vướng occlusion/pose estimation |

Rủi ro lớn nhất:

- false positive làm reject chunk tốt;
- recovery action đưa robot vào state còn tệ hơn;
- latency/logging làm loop chậm;
- threshold tune trên quá ít run.

---

### 12. Roadmap đề xuất

#### Phase 0: Instrumentation

- Log raw chunk, aggregated action, RTC/executed action, current state.
- Plot theo thời gian:
  - distance to start pose;
  - distance to safe pose;
  - raw vs executed action;
  - task progress proxy.

#### Phase 1: Offline replay

- Replay `recorded_obs` từ các run fail.
- Xác định safe-drift nằm ở raw policy hay sau aggregation/RTC.
- So run thành công vs run fail.

#### Phase 2: Guarded inference

- Thêm pre-execution validator.
- Chỉ log/reject, chưa recovery phức tạp.
- Test threshold bằng recorded logs.

#### Phase 3: Minimal recovery

- Add stop/hold.
- Add resample chunk.
- Add approach nudge.

#### Phase 4: Task-level recovery

- Add grasp classifier.
- Add re-grasp.
- Add CV-guided pre-grasp nếu cần.

---

### 13. Contribution tiềm năng

Framing này có thể thành hướng nghiên cứu/thesis/product:

**Reliable VLA Deployment via Runtime Supervision**

Đóng góp không nằm ở việc tạo VLA architecture mới, mà ở cách biến raw VLA thành hệ robot chạy được ngoài đời:

- action chunk validation;
- closed-loop progress monitoring;
- minimal recovery primitives;
- failure logging để cải thiện model/data.

Đây là khoảng trống thực tế: VLA có thể biết task, nhưng hệ deploy cần biết khi nào không nên tin VLA thêm một bước nữa.
