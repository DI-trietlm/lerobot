# VLA Harness For Reliable SO-ARM Deployment

Tài liệu này mô tả hướng triển khai VLA Harness sau các chẩn đoán mới nhất trên
task pouring. Kết luận quan trọng đã thay đổi:

- "Gần safe/folded pose" không luôn là lỗi. Trong demonstration, nhiều episode
  hợp lệ đi qua một waypoint folded/safe-like trước khi approach và grasp.
- Lỗi chính giống mất phase progression hơn là "model homing về safe pose":
  VLA đi vào vùng pre-grasp/folded hợp lệ nhưng đôi khi không thoát tiếp sang
  approach/grasp/pour.
- Vì vậy harness không nên hard-code "cấm về safe pose". Harness nên can thiệp
  nhỏ, data-derived, và giữ VLA làm controller chính.

Mục tiêu: khi VLA rơi vào điểm chết, harness chỉ cần tạo một micro-adjustment
nằm trong phân phối demonstration để VLA thoát basin, sau đó trả quyền lại cho
VLA.

---

## 1. Nguyên tắc thiết kế

### 1.1 Harness không thay VLA

Harness không phải planner cổ điển. Harness chỉ làm ba việc:

1. Phát hiện hành vi runtime bất thường.
2. Chặn các action phá task rõ ràng.
3. Can thiệp tối thiểu rồi yêu cầu VLA infer lại từ observation mới.

Happy path vẫn do VLA xử lý.

### 1.2 Ưu tiên data-derived, không task-hardcode

Task mới không nên phải viết lại rule kiểu `grasp`, `hold`, `release`. Thay vào
đó, từ dataset ta mine:

- mode ổn định;
- transition giữa mode;
- plateau/action invariant;
- envelope tốc độ/action;
- đoạn micro-action giúp thoát vùng kẹt.

Với pouring, các mode có thể tương ứng với prepare/open/hold/pour/release. Với
task khác, tên mode không quan trọng; harness chỉ dùng mode id và invariant đã
mine từ data.

### 1.3 Mọi can thiệp đáng kể phải flush/re-infer

Đây là requirement cứng.

Nếu harness sửa action đang execute nhưng server/policy vẫn tin chunk cũ được
thực thi nguyên vẹn, vòng lặp sẽ bị bất đồng bộ:

```text
policy planned action A
      ↓
harness executes modified action A'
      ↓
next observation comes from A'
      ↓
policy/aggregator/RTC may still carry assumptions from A
```

Vì vậy sau các can thiệp sau, phải clear hoặc invalidate action queue/chunk và
yêu cầu inference lại từ observation mới:

- reject chunk;
- micro-rescue;
- hard clamp gripper/actuator;
- nhiều lần speed clamp liên tiếp;
- stop/hold safety intervention.

Can thiệp nhẹ trong 1-2 frame có thể chỉ log, nhưng nếu state thực tế bị đổi
khác đáng kể so với action VLA định execute thì phải flush.

---

## 2. Bốn harness cần thiết kế

### Harness 1: Dataset-Manifold Micro-Rescue

Mục tiêu: giúp VLA thoát điểm chết bằng action ngắn lấy từ chính phân phối
dataset đã train.

Trigger tổng quát:

- current state không tiến triển trong nhiều step;
- nhiều chunk liên tiếp có endpoint gần current state;
- chunk đổi hướng qua lại nhưng state không ra khỏi vùng cũ;
- robot re-enter một vùng state quen thuộc quá lâu;
- progress proxy không tăng dù action vẫn được gửi.

Offline profile cần build từ dataset:

- index state/action theo episode và frame;
- embedding gần nhất theo joint state, optional image embedding;
- score "future progress" cho mỗi frame, ví dụ sau `0.5-1.5s` state có rời
  vùng hiện tại và đi tới mode kế tiếp không;
- action snippets ngắn từ các neighbor có progress tốt.

Runtime:

1. Khi stuck, tìm `k` neighbor gần current state.
2. Lọc neighbor có future progress tốt.
3. Chọn micro-snippet 3-10 action step.
4. Execute bounded/blended rescue.
5. Flush queue/chunk.
6. Re-infer VLA từ observation mới.

Ràng buộc:

- horizon ngắn, thường `0.3-1.0s`;
- không replay cả episode;
- không dùng snippet nếu state hiện tại OOD so với mọi neighbor;
- log đầy đủ neighbor id, episode/frame source, score, action được execute.

Với pouring, rescue này có thể giúp thoát vùng folded/pre-grasp để tiếp tục
open/approach. Với task mới, chỉ cần dataset mới.

---

### Harness 2: Data-Derived Invariant Guard

Mục tiêu: chặn các action phá task rõ ràng, được chứng minh hiếm/không xuất
hiện trong dataset.

Không hard-code "gripper sau grasp". Ta mine invariant tổng quát:

- **Plateau invariant**: sau khi vào một mode ổn định, rời mode quá sớm là bất
  thường.
- **No-backtrack invariant**: transition `M_i -> M_j` gần như không quay ngược.
- **One-shot event invariant**: một excursion lớn chỉ xảy ra một lần hoặc theo
  thứ tự ổn định.
- **Value envelope invariant**: action/state trong mode nằm trong percentile
  `[p01, p99]`.
- **Velocity envelope invariant**: speed/acceleration không vượt envelope demo.

Offline mining:

1. Với từng action/state dimension, detect plateau, transition, excursion.
2. Cluster recent-history features thành mode `M0, M1, ...`.
3. Học transition graph và duration distribution.
4. Chỉ promote invariant nếu support cao:
   - xuất hiện trong >= 95% episode;
   - violation train <= 1-2%;
   - mode confidence đủ cao;
   - invariant có hậu quả task rõ ràng nếu bị phá.

Runtime:

1. Estimate current mode từ recent state/action history.
2. Kiểm tra chunk sắp execute có vi phạm invariant không.
3. Vi phạm nhẹ: warn/log hoặc soft clamp.
4. Vi phạm nặng/catastrophic: reject/clamp, flush queue, re-infer.

Ví dụ pouring:

- Dataset cho thấy sau khi vào mode giữ cốc, việc mở gripper trước pour/release
  gần như không xảy ra.
- Harness không cần biết tên "grasp"; nó chỉ biết dimension đó đang ở stable
  mode và action sắp rời mode quá sớm.

Giảm rủi ro:

- bật guard cứng chỉ cho invariant confidence cao;
- ban đầu chạy shadow mode để đo false positive trên runtime logs;
- mọi clamp đáng kể phải flush/re-infer.

---

### Harness 3: Action Stability And Speed Envelope Guard

Mục tiêu: chặn spike/action ngoài phân phối, nhưng không biến speed clamp thành
controller chính.

Signals:

- action velocity theo từng dimension;
- action acceleration/jerk;
- endpoint jump so với current state;
- disagreement giữa các chunk liên tiếp;
- tracking error giữa commanded và actual state.

Offline profile:

- percentile speed/acceleration theo toàn dataset;
- percentile theo mode nếu có mode profile;
- normal chunk-to-chunk direction consistency;
- action bounds theo dimension.

Runtime policy:

- clamp spike rõ ràng vượt envelope;
- reject chunk nếu endpoint/action path OOD;
- nếu phải clamp nhiều lần liên tiếp, coi đó là failure, không tiếp tục clamp
  âm thầm;
- sau repeated clamp: flush queue và gọi micro-rescue hoặc re-infer.

Quan điểm ưu tiên:

- Speed guard là safety envelope, không phải behavior shaper.
- Dùng để chặn outlier, không dùng để "lái" robot từng frame.
- Với các task nhạy cảm như cầm cốc, gripper invariant quan trọng hơn speed
  clamp.

---

### Harness 4: Runtime Synchronization, Re-Infer, And Trace Harness

Mục tiêu: biến mọi can thiệp thành một state transition có thể giải thích, không
để policy/queue/RTC chạy tiếp trên giả định cũ.

Thành phần:

1. **Intervention ledger**
   - Log mọi lần reject, clamp, rescue, stop.
   - Lưu reason, metric, action gốc, action thực thi, mode estimate.

2. **Queue/chunk invalidation**
   - Sau can thiệp mạnh, clear action queue.
   - Bỏ các chunk cũ đang pending.
   - Reset aggregation state nếu aggregation dùng lịch sử chunk.

3. **Re-infer coordinator**
   - Chụp observation mới sau can thiệp.
   - Gửi server infer lại.
   - Không dùng tiếp chunk đã bị sửa nhiều.

4. **Runtime trace pack**
   - Lưu đủ dữ liệu để debug offline:
     - timestamp;
     - current state;
     - raw chunk;
     - postprocessed chunk;
     - aggregated/RTC action;
     - executed action;
     - actual state sau execute;
     - intervention flag;
     - mode estimate;
     - dataset neighbor nếu có rescue;
     - camera frame key/path.

Requirement:

- Mỗi episode infer phải tái dựng được timeline "policy muốn gì, harness sửa gì,
  robot thực thi gì, observation sau đó là gì".
- Nếu không tái dựng được, harness đang làm hệ khó debug hơn và phải sửa logging
  trước khi thêm guard mới.

---

## 3. Offline profile cho task mới

Task mới cần chạy một profiler trên dataset:

```text
dataset
  -> state/action arrays
  -> mode discovery
  -> invariant mining
  -> speed/action envelope
  -> nearest-neighbor rescue index
  -> harness_profile.json
```

`harness_profile.json` nên chứa:

- schema state/action dimension;
- fps;
- normalization/scales;
- mode centroids hoặc classifier nhẹ;
- transition graph;
- duration distribution;
- high-confidence invariants;
- speed/action envelopes;
- ANN index metadata cho micro-rescue;
- thresholds đã calibrate từ dataset.

Không nên chứa:

- rule task-specific viết tay nếu có thể mine từ data;
- waypoint cố định nếu không cần;
- tên phase phụ thuộc pouring, trừ khi chỉ để debug/report.

---

## 4. Runtime flow

```text
observation_t
    ↓
VLA predicts chunk
    ↓
mode estimate + progress estimate
    ↓
[Invariant Guard]
    ├─ catastrophic violation -> clamp/reject -> flush -> re-infer
    └─ pass
    ↓
[Speed/Envelope Guard]
    ├─ OOD spike -> clamp/reject
    ├─ repeated clamp -> flush -> micro-rescue/re-infer
    └─ pass
    ↓
execute action through existing stack
    ↓
[Progress/Stuck Monitor]
    ├─ OK -> continue
    └─ stuck -> Dataset-Manifold Micro-Rescue
                  ↓
                flush queue/chunk
                  ↓
                re-infer from fresh observation
```

---

## 4.5 Hybrid server/client architecture

Harness nên chạy theo kiến trúc hybrid:

- **Server quyết định thông minh**: validate chunk, mine/use data-derived
  invariants, chọn micro-rescue, reject/resample, và điều phối re-infer.
- **Client quyết định an toàn cuối cùng**: chặn action trước robot, quản lý
  local queue/RTC/interpolation, theo dõi state thật, và emergency stop.

### Server-side responsibilities

Server phù hợp cho các phần cần nhìn policy/model đầy đủ:

- pre-execution chunk validator;
- data-derived invariant check ở mức raw/postprocessed chunk;
- dataset-manifold micro-rescue proposal;
- reject/resample chunk;
- re-infer coordinator;
- raw chunk, postprocessed chunk, mode estimate, violation reason logging.

Ưu điểm:

- thấy action chunk đầy đủ trước khi execute;
- gần model/preprocessor/postprocessor;
- dễ chạy nearest-neighbor/mode classifier nặng;
- dễ invalidate chunk và infer lại.

Giới hạn:

- không phải safety layer cuối vì không nằm cạnh robot;
- không luôn thấy action cuối cùng sau RTC/client guard;
- can thiệp phụ thuộc network latency.

### Client-side responsibilities

Client phù hợp cho các phần phải phản ứng ngay tại robot:

- final execution guard;
- gripper/actuator hard invariant guard;
- speed/acceleration safety envelope;
- tracking-error monitor;
- local queue/RTC/interpolation clear;
- emergency stop/hold;
- executed-action và actual-state logging.

Ưu điểm:

- thấy state thật và action thật sự được execute;
- phản ứng không phụ thuộc round-trip server;
- là safety fallback cuối cùng.

Giới hạn:

- không nên chạy logic data-manifold nặng;
- nếu tự sửa action mà server không biết sẽ gây bất đồng bộ;
- phải có protocol báo server flush/re-infer.

### Synchronization protocol

Mỗi chunk/action cần có `chunk_id`.

```text
server -> client:
    chunk_id, action_chunk, policy_metadata

client:
    execute normally
    or intervene locally

if client intervenes:
    client clears local queue/RTC pending actions
    client sends intervention_event(chunk_id, reason, executed_action, current_state)
    server invalidates chunk/context
    server requests fresh observation
    server re-infers
```

Requirement:

- Mọi can thiệp đáng kể ở client phải trigger `flush/re-infer` ở server.
- Server không được tiếp tục aggregate/reuse chunk đã bị client sửa mạnh.
- Trace phải lưu được cả action gốc và action đã thực thi.

Kết luận kiến trúc:

**Server-side harness làm reasoning/model-aware guard. Client-side harness làm
safety/execution guard. Hai bên nối với nhau bằng intervention event và
flush/re-infer protocol.**

---

## 5. Áp dụng cho pouring hiện tại

Chẩn đoán mới:

- Nhiều demonstration đi qua vùng folded/safe-like trước khi approach.
- Do đó không được cấm tuyệt đối chuyển động về safe-like pose.
- Failure hiện tại giống kẹt ở pre-grasp/folded waypoint hoặc re-enter waypoint
  đó, thay vì chuyển tiếp sang open/approach/grasp.

Harness ưu tiên:

1. **Micro-rescue** khi robot ở folded/pre-grasp quá lâu mà không mở/approach.
2. **Invariant gripper learned from data**: sau khi mode giữ cốc được detect,
   action mở gripper bất thường phải bị chặn.
3. **Speed/envelope guard** chỉ để chặn spike hoặc action OOD.
4. **Flush/re-infer** sau mọi rescue/clamp gripper/reject chunk.

Điều không nên làm:

- Không hard-code "gần safe pose là lỗi".
- Không xóa folded/safe-like segments khỏi dataset chỉ vì chúng gần safe.
- Không clamp tốc độ liên tục để ép robot đi theo ý harness.
- Không sửa gripper giữa chunk rồi tiếp tục dùng chunk cũ như chưa có gì xảy ra.

---

## 6. MVP đề xuất

### MVP A: Harness Profile Miner

Input: LeRobot dataset.

Output: `harness_profile.json`.

Tối thiểu:

- state/action arrays;
- robust dimension scales;
- speed/action percentile envelopes;
- plateau/transition candidates;
- nearest-neighbor index for state-only micro-rescue;
- dataset frame future-progress score.

### MVP B: Shadow Runtime Monitor

Chạy song song, chưa can thiệp.

Log:

- predicted violation;
- mode estimate;
- stuck score;
- suggested rescue neighbor;
- would-flush reason.

Mục tiêu: đo false positive trên real runs.

### MVP C: Gripper/Actuator Invariant Guard

Bật guard cứng chỉ cho invariant confidence cao và hậu quả nặng.

Với pouring:

- nếu đã vào hold-like mode, không cho mở gripper bất thường;
- nếu guard can thiệp, flush chunk và re-infer.

### MVP D: Micro-Rescue

Khi stuck:

- chọn snippet ngắn từ neighbor có future progress;
- execute bounded;
- flush;
- re-infer.

MVP D phải log đủ để biết rescue có thật sự giúp hay không.

---

## 7. Tiêu chí đánh giá

Offline:

- invariant support/violation trên train dataset;
- rescue neighbor coverage cho các runtime stuck states;
- false-positive rate khi replay recorded runs;
- action OOD rate trước/sau guard.

Online:

- success rate;
- số lần intervention mỗi run;
- intervention nào giúp thoát stuck;
- số lần flush/re-infer;
- số lần guard gripper cứu khỏi làm rơi/đổ cốc;
- latency overhead.

Một harness tốt không phải can thiệp nhiều. Harness tốt là:

- hầu hết run happy path không bị chạm;
- khi có điểm nghẽn, can thiệp nhỏ và đưa VLA về lại manifold;
- mọi can thiệp đều có trace để debug model/data sau đó.

---

## 8. Quan hệ với data/model

Harness không thay thế việc train model tốt hơn.

Nhưng thứ tự hợp lý hiện tại là:

1. Làm harness data-derived để deployment bớt brittle.
2. Dùng trace harness để xác định phase/mode nào fail.
3. Sau đó mới sửa data/model có mục tiêu:
   - thêm demo ở transition fail;
   - lọc outlier;
   - train/eval với safe/stuck metrics;
   - checkpoint selection bằng runtime replay.

Với bài toán hiện tại, không nên sửa data theo cách "xóa mọi đoạn gần safe" vì
đoạn đó là waypoint hợp lệ. Data/model work cần phase-aware hơn: model phải học
đi qua waypoint rồi thoát khỏi nó, không kẹt ở đó.

---

## 9. Contribution tiềm năng

Framing nghiên cứu/product:

**Reliable VLA Deployment via Data-Derived Runtime Harness**

Đóng góp:

- mine invariants từ demonstration thay vì viết task rule thủ công;
- dùng dataset-manifold micro-rescue để thoát điểm chết;
- guard action stability bằng envelope học từ data;
- đảm bảo runtime synchronization bằng flush/re-infer sau can thiệp;
- logging đủ để biến failure online thành data/model improvement.

Đây là lớp còn thiếu giữa "VLA chạy được trong nhiều trường hợp" và "robot
đáng tin trong deployment thật".
