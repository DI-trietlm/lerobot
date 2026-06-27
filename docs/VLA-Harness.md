## Tổng kết: VLA Harness Architecture

---

### 1. Xuất phát điểm

**Observation thực tế:** 300 episodes, overfit còn khó. VLA hype trong robotics vẫn đang dừng ở lab, gap sang real world rất lớn (ngoại trừ xe tự lái với scale data khổng lồ).

**Root cause:** Không phải VLA dốt, mà là thiếu infrastructure bao quanh để handle failure cases.

---

### 2. Analogy dẫn đường

LLM evolution cho thấy hướng đi:

| LLM | VLA |
|---|---|
| Raw model hallucinate | VLA fail silently |
| RAG + grounding | Perception module độc lập |
| Tool use | Classical controller cho low-level |
| Agent harness | **← VLA harness (chỗ còn trống)** |

**Key insight:** LLM chỉ production-ready khi được đặt trong một system. VLA cũng vậy.

---

### 3. Các ideas đã explore

**Idea 1: Hierarchical fallback**
- CV model nhỏ detect cup pose → tính IK target khi VLA fail
- Grasp classifier từ wrist cam + gripper state
- Task decomposition thành phases nhỏ với entry/exit condition rõ ràng

**Idea 2: Explore-then-Execute**
- Probe trajectory trước, evaluate deviation so với training distribution
- Nếu confidence cao → execute, thấp → re-explore
- Tương đương Plan-then-Execute của LLM agent

**Idea 3: Exception handling (converge cuối cùng)**
- VLA đủ thông minh để handle happy path
- Harness chỉ cần catch failure, intervene tối thiểu, trả control lại VLA

---

### 4. Differentiation so với pipeline truyền thống

| Pipeline cũ (CV + dynamics) | VLA-centric harness |
|---|---|
| "Don't trust robot, control everything" | "Trust VLA, only intervene on failures" |
| Hand-craft toàn bộ planning logic | VLA encode task knowledge, harness task-agnostic |
| Thêm task mới → rewrite planning | Thêm task mới → fine-tune VLA, harness giữ nguyên |
| Brittle nhưng predictable | Flexible, failure modes rõ ràng |

---

### 5. Final Architecture

**Key discovery:** VLA behavior binary, không có vùng xám "đang nghĩ chậm". Do đó **2s no-movement = unified trigger** cho mọi failure mode.

```
VLA running (happy path, không can thiệp)
         ↓
  2s no movement detected
         ↓
  [Classify failure type]
  ├─ Gripper open + wrist cam thấy object
  │    → Grasp fail → Re-grasp với CV target
  ├─ Gripper closed + EE position lạ  
  │    → Micro drift → Nudge + Resume
  └─ Object không detect được
       → CV fallback → IK → Resume
         ↓
  Recovery tối thiểu
         ↓
  Trả control lại VLA
```

Harness hoạt động như **exception handler**, không phải replanner.

---

### 6. Đánh giá khả thi

| Component | Khả thi | Ghi chú |
|---|---|---|
| Stuck detector 2s | Rất cao | Trivial với VLA behavior của bạn |
| Grasp classifier | Cao | 300 episodes đủ data |
| CV fallback target | Trung bình | Pose estimation + occlusion là điểm khó |
| FSM orchestrator | Cao | Logic đơn giản, tuning threshold tốn thời gian |

**MVP khả thi:** Stuck detector + Grasp classifier → demo được trong 3-4 tuần làm tập trung.

**Rủi ro lớn nhất:** Không phải technical mà là integration latency và false positive từ recovery action làm state tệ hơn.

---

### 7. Contribution tiềm năng

Framing này có thể thành paper/thesis contribution về **"Reliable VLA Deployment"** - không phải về VLA architecture mà về **how to deploy VLA in real world**. Một gap rất thực tế mà field chưa có answer chuẩn.