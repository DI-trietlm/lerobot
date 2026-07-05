import ast
import contextlib
import json
import os
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk

import draccus
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.gui import start_pose_analysis as spa
from lerobot.robots.so_follower import SO100FollowerConfig, SO101FollowerConfig
from lerobot.rtc_inference.configs import AGGREGATE_FUNCTIONS, RobotClientConfig
from lerobot.rtc_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.vla_harness.config import HarnessConfig


@dataclass
class _RuntimeState:
    connected: bool = False
    start_pose_done: bool = False
    stream_running: bool = False
    busy: bool = False
    action_receiver_thread: threading.Thread | None = None
    control_loop_thread: threading.Thread | None = None


@dataclass
class _FieldSpec:
    key: str
    label: str
    default: str
    field_type: str
    options: tuple[str, ...] = ()


class RTCControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RTC Policy Control Panel")
        self.geometry("1300x820")

        self._state = _RuntimeState()
        self._state_lock = threading.Lock()

        self._client: RobotClient | None = None
        self._client_cfg: RobotClientConfig | None = None

        # Latest analysed start-pose region (median used for reset, IQR kept for reference).
        self._start_pose_stats: spa.StartPoseStats | None = None

        # Safe rest pose (loaded from a capture file; NO defaults) for the auto-reset cycle.
        self._safe_pose: dict[str, float] | None = None
        self._safe_pose_meta: dict | None = None
        self._safe_pose_path: str | None = None

        self._vars: dict[str, tk.StringVar] = {}
        self._widgets: dict[str, ttk.Widget] = {}

        self._log_queue: Queue[str] = Queue()

        self._build_ui()
        self._start_log_pump()
        self._refresh_controls()
        self._try_autoload_safe_pose()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self._build_config_panel(left)
        self._build_control_panel(right)

    def _build_config_panel(self, parent: ttk.Frame):
        title = ttk.Label(parent, text="Configuration", font=("Segoe UI", 12, "bold"))
        title.pack(anchor="w")

        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        form_frame = ttk.Frame(canvas)

        form_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0
        for section_name, specs in self._field_sections().items():
            section_label = ttk.Label(form_frame, text=section_name, font=("Segoe UI", 10, "bold"))
            section_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))
            row += 1

            for spec in specs:
                ttk.Label(form_frame, text=spec.label).grid(
                    row=row, column=0, sticky="w", padx=(0, 8), pady=3
                )
                var = tk.StringVar(value=spec.default)
                self._vars[spec.key] = var

                if spec.field_type in {"bool", "choice"}:
                    values = spec.options if spec.options else ("true", "false")
                    widget = ttk.Combobox(
                        form_frame, textvariable=var, values=values, state="readonly", width=48
                    )
                else:
                    widget = ttk.Entry(form_frame, textvariable=var, width=80)

                widget.grid(row=row, column=1, sticky="we", pady=3)
                self._widgets[spec.key] = widget
                row += 1

        form_frame.columnconfigure(1, weight=1)

    def _build_control_panel(self, parent: ttk.Frame):
        title = ttk.Label(parent, text="Control", font=("Segoe UI", 12, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        self.status_var = tk.StringVar(value="Disconnected")
        status = ttk.Label(parent, textvariable=self.status_var, foreground="#1f4e79")
        status.pack(anchor="w", pady=(0, 10))

        self.btn_connect = ttk.Button(
            parent, text="Connect", command=lambda: self._run_async(self._on_connect)
        )
        self.btn_connect.pack(fill=tk.X, pady=4)

        self.btn_reset = ttk.Button(
            parent,
            text="Reset Start-Pose",
            command=lambda: self._run_async(self._on_reset_start_pose),
        )
        self.btn_reset.pack(fill=tk.X, pady=4)

        self.btn_call = ttk.Button(
            parent, text="Call to Server", command=lambda: self._run_async(self._on_call_server)
        )
        self.btn_call.pack(fill=tk.X, pady=4)

        self.btn_stop_stream = ttk.Button(
            parent,
            text="Stop Stream",
            command=lambda: self._run_async(self._on_stop_stream),
        )
        self.btn_stop_stream.pack(fill=tk.X, pady=4)

        self.btn_disconnect = ttk.Button(
            parent,
            text="Disconnect",
            command=lambda: self._run_async(self._on_disconnect),
        )
        self.btn_disconnect.pack(fill=tk.X, pady=4)

        self.btn_cycle = ttk.Button(
            parent,
            text="Auto Reset Cycle",
            command=lambda: self._run_async(self._on_auto_reset_cycle),
        )
        self.btn_cycle.pack(fill=tk.X, pady=4)

        # Safe pose (loaded from a capture file; required by the Auto Reset Cycle).
        sp_frame = ttk.LabelFrame(parent, text="Safe Pose (pre-disconnect rest)")
        sp_frame.pack(fill=tk.X, pady=(8, 4))
        self.safe_pose_var = tk.StringVar(value="(none loaded)")
        ttk.Label(
            sp_frame, textvariable=self.safe_pose_var, justify="left", wraplength=360, foreground="#555"
        ).pack(anchor="w", padx=6, pady=4)
        self.btn_load_safe = ttk.Button(
            sp_frame, text="Load Safe Pose...", command=self._on_load_safe_pose
        )
        self.btn_load_safe.pack(fill=tk.X, padx=6, pady=(0, 6))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(10, 6))

        self.btn_export = ttk.Button(
            parent,
            text="Export Config (CLI)",
            command=self._on_export_config,
        )
        self.btn_export.pack(fill=tk.X, pady=4)

        self.btn_import = ttk.Button(
            parent,
            text="Import Config",
            command=self._on_import_config,
        )
        self.btn_import.pack(fill=tk.X, pady=4)

        self.btn_analyze = ttk.Button(
            parent,
            text="Analyze Dataset → Start-Pose",
            command=lambda: self._run_bg(self._on_analyze_dataset),
        )
        self.btn_analyze.pack(fill=tk.X, pady=4)

        log_label = ttk.Label(parent, text="Logs", font=("Segoe UI", 10, "bold"))
        log_label.pack(anchor="w", pady=(14, 4))

        self.log_text = tk.Text(parent, width=55, height=36, state="disabled", wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _field_sections(self) -> dict[str, list[_FieldSpec]]:
        return {
            "Connection": [
                _FieldSpec("server_address", "Server Address", "192.168.30.244:4567", "str"),
                _FieldSpec(
                    "task",
                    "Task",
                    "Pour from orange cup to blue cup.",
                    "str",
                ),
            ],
            "Policy": [
                _FieldSpec("policy_type", "Policy Type", "smolvla", "choice", ("smolvla", "xvla")),
                _FieldSpec("pretrained_name_or_path", "Pretrained", "di-techinnova/smolvla-pouring-0.3-cutted", "str"),
                _FieldSpec("policy_device", "Policy Device", "cuda", "str"),
                _FieldSpec("client_device", "Client Device", "cpu", "str"),
                _FieldSpec(
                    "rename_map",
                    "Rename Map (JSON)",
                    '{"observation.images.camera1":"observation.images.camera1","observation.images.camera2":"observation.images.camera2"}',
                    "json_dict",
                ),
            ],
            "Robot": [
                _FieldSpec(
                    "robot_type",
                    "Robot Type",
                    "so100_follower",
                    "choice",
                    ("so101_follower", "so100_follower"),
                ),
                _FieldSpec("robot_port", "Robot Port", "COM6", "str"),
                _FieldSpec("robot_id", "Robot ID", "DI_VLA_FOLLOWER", "str"),
                _FieldSpec("robot_use_degrees", "Robot Use Degrees", "true", "bool", ("true", "false")),
                _FieldSpec(
                    "robot_disable_torque_on_disconnect",
                    "Disable Torque On Disconnect",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec("robot_max_relative_target", "Max Relative Target (empty|float|dict)", "", "str"),
                _FieldSpec(
                    "robot_cameras",
                    "Robot Cameras (JSON)",
                    '{"camera1":{"type":"opencv","index_or_path":1,"width":1280,"height":720,"fps":30},"camera2":{"type":"opencv","index_or_path":0,"width":640,"height":360,"fps":30}}',
                    "json_dict",
                ),
            ],
            "Runtime": [
                _FieldSpec("actions_per_chunk", "Actions Per Chunk", "35", "int"),
                _FieldSpec("chunk_size_threshold", "Chunk Size Threshold", "0.5", "float"),
                _FieldSpec(
                    "aggregate_fn_name",
                    "Aggregate Function",
                    "weighted_average",
                    "choice",
                    tuple(AGGREGATE_FUNCTIONS.keys()),
                ),
                _FieldSpec("fps", "FPS", "15", "int"),
                _FieldSpec(
                    "obs_timestep_independent", "Obs Timestep Independent", "false", "bool", ("true", "false")
                ),
                _FieldSpec(
                    "image_compress_enable", "Image Compress Enable", "true", "bool", ("true", "false")
                ),
                _FieldSpec("image_compress_quality", "Image Compress Quality", "90", "int"),
                _FieldSpec("interpolation_multiplier", "Interpolation Multiplier", "1", "int"),
                _FieldSpec(
                    "debug_visualize_queue_size", "Debug Visualize Queue", "true", "bool", ("true", "false")
                ),
            ],
            "RTC": [
                _FieldSpec("rtc_enabled", "RTC Enabled", "true", "bool", ("true", "false")),
                _FieldSpec("rtc_execution_horizon", "RTC Execution Horizon", "10", "int"),
                _FieldSpec("rtc_max_guidance_weight", "RTC Max Guidance Weight", "10.0", "float"),
                _FieldSpec(
                    "rtc_prefix_attention_schedule",
                    "RTC Prefix Schedule",
                    "EXP",
                    "choice",
                    tuple(s.name for s in RTCAttentionSchedule),
                ),
                _FieldSpec("rtc_debug", "RTC Debug", "false", "bool", ("true", "false")),
                _FieldSpec("rtc_debug_maxlen", "RTC Debug Maxlen", "100", "int"),
                _FieldSpec("inference_delay_steps", "Inference Delay Steps (optional)", "2", "optional_int"),
                _FieldSpec("xvla_domain_id", "XVLA Domain ID (optional, XVLA only)", "15", "optional_int"),
            ],
            "Recording": [
                _FieldSpec(
                    "record_obs_enable", "Record Observations", "false", "bool", ("true", "false")
                ),
                _FieldSpec("record_obs_dir", "Record Output Directory", "recorded_obs", "str"),
                _FieldSpec(
                    "record_action_enable", "Record Actions", "true", "bool", ("true", "false")
                ),
                _FieldSpec("record_action_dir", "Action Trace Directory", "recorded_obs", "str"),
            ],
            "Attention Capture (Server-side, X-VLA only)": [
                _FieldSpec(
                    "capture_attn_enable",
                    "Capture Attention Weights",
                    "false",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec(
                    "capture_attn_dir",
                    "Attention Output Directory",
                    "attention_captures",
                    "str",
                ),
            ],
            "VLA Harness": [
                _FieldSpec("harness_enable", "Harness Enable", "false", "bool", ("true", "false")),
                _FieldSpec("harness_profile_path", "Harness Profile Path", "", "str"),
                _FieldSpec("harness_shadow_mode", "Global Shadow Mode", "true", "bool", ("true", "false")),
                _FieldSpec("harness_fail_closed", "Fail Closed", "false", "bool", ("true", "false")),
                _FieldSpec("harness_log_dir", "Harness Log Directory", "harness_traces", "str"),
                _FieldSpec("harness_server_enable", "Server Enable", "true", "bool", ("true", "false")),
                _FieldSpec("harness_client_enable", "Client Enable", "true", "bool", ("true", "false")),
                _FieldSpec("harness_micro_rescue_enable", "Micro-Rescue Enable", "true", "bool", ("true", "false")),
                _FieldSpec("harness_invariant_guard_enable", "Invariant Guard Enable", "true", "bool", ("true", "false")),
                _FieldSpec("harness_speed_envelope_enable", "Speed Envelope Enable", "true", "bool", ("true", "false")),
                _FieldSpec("harness_sync_enable", "Sync/Flush Enable", "true", "bool", ("true", "false")),
                _FieldSpec("harness_trace_enable", "Trace Enable", "true", "bool", ("true", "false")),
                _FieldSpec(
                    "harness_server_chunk_validator_enable",
                    "Server Chunk Validator",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec(
                    "harness_server_invariant_guard_enable",
                    "Server Invariant Guard",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec(
                    "harness_server_micro_rescue_proposal_enable",
                    "Server Micro-Rescue Proposal",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec(
                    "harness_server_reject_resample_enable",
                    "Server Reject/Fallback",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec("harness_server_max_resample_attempts", "Server Max Resample Attempts", "1", "int"),
                _FieldSpec("harness_server_re_infer_on_intervention", "Server Re-Infer On Intervention", "true", "bool", ("true", "false")),
                _FieldSpec(
                    "harness_client_execution_guard_enable",
                    "Client Execution Guard",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec(
                    "harness_client_hard_invariant_guard_enable",
                    "Client Hard Invariant Guard",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec(
                    "harness_client_speed_envelope_enable",
                    "Client Speed Envelope",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec(
                    "harness_client_tracking_monitor_enable",
                    "Client Tracking Monitor",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec("harness_client_tracking_monitor_window_steps", "Tracking Window Steps", "45", "int"),
                _FieldSpec("harness_client_tracking_monitor_state_radius", "Tracking State Radius", "18.0", "float"),
                _FieldSpec("harness_client_tracking_monitor_min_path_length", "Tracking Min Path Length", "0.0", "float"),
                _FieldSpec("harness_client_tracking_monitor_cooldown_steps", "Tracking Cooldown Steps", "45", "int"),
                _FieldSpec("harness_client_tracking_monitor_dims", "Tracking Dims (JSON list)", "[0,1,2,3,4]", "json_list"),
                _FieldSpec("harness_client_clear_queue_on_intervention", "Client Clear Queue On Intervention", "true", "bool", ("true", "false")),
                _FieldSpec("harness_client_request_reinfer_on_intervention", "Client Request Re-Infer", "true", "bool", ("true", "false")),
                _FieldSpec("harness_micro_rescue_shadow_mode", "Micro-Rescue Shadow", "true", "bool", ("true", "false")),
                _FieldSpec("harness_micro_rescue_state_knn_enable", "Micro-Rescue State KNN", "true", "bool", ("true", "false")),
                _FieldSpec("harness_micro_rescue_image_knn_enable", "Micro-Rescue Image KNN", "false", "bool", ("true", "false")),
                _FieldSpec("harness_micro_rescue_k_neighbors", "Micro-Rescue K Neighbors", "16", "int"),
                _FieldSpec("harness_micro_rescue_snippet_horizon_steps", "Micro-Rescue Horizon Steps", "8", "int"),
                _FieldSpec("harness_micro_rescue_max_duration_s", "Micro-Rescue Max Duration (s)", "1.0", "float"),
                _FieldSpec("harness_micro_rescue_blend_alpha", "Micro-Rescue Blend Alpha", "1.0", "float"),
                _FieldSpec("harness_micro_rescue_ramp_in_steps", "Micro-Rescue Ramp-In Steps", "0", "int"),
                _FieldSpec("harness_micro_rescue_ramp_in_max_joint_delta", "Micro-Rescue Ramp Max Joint Delta", "", "optional_float"),
                _FieldSpec("harness_micro_rescue_min_future_progress_score", "Micro-Rescue Min Future Progress", "0.2", "float"),
                _FieldSpec("harness_micro_rescue_max_state_distance", "Micro-Rescue Max Distance (optional)", "", "optional_float"),
                _FieldSpec("harness_micro_rescue_cooldown_s", "Micro-Rescue Cooldown (s)", "2.0", "float"),
                _FieldSpec("harness_micro_rescue_max_rescues_per_episode", "Micro-Rescue Max Per Episode", "3", "int"),
                _FieldSpec("harness_invariant_guard_shadow_mode", "Invariant Guard Shadow", "true", "bool", ("true", "false")),
                _FieldSpec("harness_invariant_guard_min_support", "Invariant Min Support", "0.95", "float"),
                _FieldSpec("harness_invariant_guard_max_train_violation_rate", "Invariant Max Train Violation", "0.02", "float"),
                _FieldSpec("harness_invariant_guard_min_mode_confidence", "Invariant Min Mode Confidence", "0.7", "float"),
                _FieldSpec("harness_invariant_guard_hard_guard_categories", "Invariant Hard Categories (JSON list)", '["catastrophic_actuator_release"]', "json_list"),
                _FieldSpec("harness_invariant_guard_soft_guard_categories", "Invariant Soft Categories (JSON list)", '["value_envelope","no_backtrack"]', "json_list"),
                _FieldSpec("harness_invariant_guard_flush_on_hard_guard", "Invariant Flush On Hard", "true", "bool", ("true", "false")),
                _FieldSpec("harness_speed_envelope_shadow_mode", "Speed Envelope Shadow", "true", "bool", ("true", "false")),
                _FieldSpec("harness_speed_envelope_percentile_low", "Speed Percentile Low", "0.005", "float"),
                _FieldSpec("harness_speed_envelope_percentile_high", "Speed Percentile High", "0.995", "float"),
                _FieldSpec("harness_speed_envelope_mode_conditioned", "Speed Mode Conditioned", "true", "bool", ("true", "false")),
                _FieldSpec("harness_speed_envelope_max_consecutive_clamps", "Speed Max Consecutive Clamps", "3", "int"),
                _FieldSpec("harness_speed_envelope_flush_after_repeated_clamp", "Speed Flush After Repeated Clamp", "true", "bool", ("true", "false")),
                _FieldSpec("harness_sync_require_chunk_id", "Sync Require Chunk ID", "true", "bool", ("true", "false")),
                _FieldSpec("harness_sync_flush_on_reject", "Sync Flush On Reject", "true", "bool", ("true", "false")),
                _FieldSpec("harness_sync_flush_on_rescue", "Sync Flush On Rescue", "true", "bool", ("true", "false")),
                _FieldSpec("harness_sync_flush_on_hard_clamp", "Sync Flush On Hard Clamp", "true", "bool", ("true", "false")),
                _FieldSpec("harness_sync_flush_on_repeated_speed_clamp", "Sync Flush On Repeated Speed Clamp", "true", "bool", ("true", "false")),
                _FieldSpec("harness_sync_block_execution_until_fresh_chunk", "Sync Block Until Fresh Chunk", "true", "bool", ("true", "false")),
                _FieldSpec("harness_trace_record_images", "Trace Record Images", "false", "bool", ("true", "false")),
                _FieldSpec("harness_trace_record_raw_chunks", "Trace Raw Chunks", "true", "bool", ("true", "false")),
                _FieldSpec("harness_trace_record_postprocessed_chunks", "Trace Postprocessed Chunks", "true", "bool", ("true", "false")),
                _FieldSpec("harness_trace_record_executed_actions", "Trace Executed Actions", "true", "bool", ("true", "false")),
                _FieldSpec("harness_trace_record_mode_estimates", "Trace Mode Estimates", "true", "bool", ("true", "false")),
                _FieldSpec("harness_trace_record_rescue_neighbors", "Trace Rescue Neighbors", "true", "bool", ("true", "false")),
            ],
            # Safe start-pose reset target. Export/import now persists these fields
            # explicitly; defaults are only a fallback before a config is imported or
            # dataset analysis is run.
            "Start Pose (safe reset target, degrees)": [
                _FieldSpec(
                    "dataset_repo_id",
                    "Dataset Repo (for start-pose analysis)",
                    "di-techinnova/so-arm-101-pouring-0.3-cutted",
                    "str",
                ),
                _FieldSpec(
                    "reset_pose_mode",
                    "Reset Mode",
                    "median",
                    "choice",
                    ("median", "random_iqr", "random_minmax"),
                ),
                _FieldSpec("start_pose_shoulder_pan", "shoulder_pan (median)", "-5.5", "float"),
                _FieldSpec("start_pose_shoulder_lift", "shoulder_lift", "-5.5", "float"),
                _FieldSpec("start_pose_elbow_flex", "elbow_flex", "11.7", "float"),
                _FieldSpec("start_pose_wrist_flex", "wrist_flex", "0.6", "float"),
                _FieldSpec("start_pose_wrist_roll", "wrist_roll", "2.5", "float"),
                _FieldSpec("start_pose_gripper", "gripper", "1.4", "float"),
                _FieldSpec("reset_duration_start", "Reset Duration (s)", "8.0", "float"),
                _FieldSpec("reset_duration_after_stop", "Reset Duration After Stop (s)", "8.0", "float"),
            ],
        }

    def _start_log_pump(self):
        def pump():
            try:
                while True:
                    msg = self._log_queue.get_nowait()
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", msg + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
            except Empty:
                pass
            self.after(120, pump)

        self.after(120, pump)

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self._log_queue.put(f"[{timestamp}] {message}")

    def _set_busy(self, busy: bool):
        with self._state_lock:
            self._state.busy = busy
        self.after(0, self._refresh_controls)

    def _set_status(self, text: str):
        self.after(0, lambda: self.status_var.set(text))

    def _refresh_controls(self):
        with self._state_lock:
            connected = self._state.connected
            start_pose_done = self._state.start_pose_done
            stream_running = self._state.stream_running
            busy = self._state.busy

        self.btn_connect.configure(state=("normal" if not connected and not busy else "disabled"))
        self.btn_reset.configure(
            state=("normal" if connected and not stream_running and not busy else "disabled")
        )
        self.btn_call.configure(
            state=(
                "normal" if connected and start_pose_done and not stream_running and not busy else "disabled"
            )
        )
        self.btn_stop_stream.configure(
            state=("normal" if connected and stream_running and not busy else "disabled")
        )
        self.btn_disconnect.configure(state=("normal" if connected and not busy else "disabled"))
        self.btn_cycle.configure(
            state=(
                "normal"
                if connected and not busy and self._safe_pose is not None
                else "disabled"
            )
        )

    def _run_async(self, fn):
        def worker():
            self._set_busy(True)
            try:
                fn()
            except Exception as exc:
                error_text = str(exc)
                self._log(f"[ERROR] {exc}")
                self.after(0, lambda: messagebox.showerror("Error", error_text))
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _run_bg(self, fn, *, show_error: bool = True):
        """Run ``fn`` off the UI thread WITHOUT toggling the robot 'busy' state.

        Used for dataset analysis / Hub checks so they never disable robot controls
        (e.g. Stop Stream) while running.
        """

        def worker():
            try:
                fn()
            except Exception as exc:
                error_text = str(exc)
                self._log(f"[ERROR] {exc}")
                if show_error:
                    self.after(0, lambda: messagebox.showerror("Error", error_text))

        threading.Thread(target=worker, daemon=True).start()

    def _parse_bool(self, key: str) -> bool:
        value = self._vars[key].get().strip().lower()
        if value in {"true", "1", "yes", "y", "on"}:
            return True
        if value in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"{key}: invalid bool '{value}'")

    def _parse_int(self, key: str) -> int:
        return int(self._vars[key].get().strip())

    def _parse_float(self, key: str) -> float:
        return float(self._vars[key].get().strip())

    def _parse_optional_int(self, key: str) -> int | None:
        text = self._vars[key].get().strip()
        if text == "":
            return None
        return int(text)

    def _parse_optional_float(self, key: str) -> float | None:
        text = self._vars[key].get().strip()
        if text == "":
            return None
        return float(text)

    def _parse_dict_like(self, key: str) -> dict:
        text = self._vars[key].get().strip()
        if text == "":
            return {}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception as exc:
                raise ValueError(f"{key}: invalid dict/json format") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"{key}: expected a dict-like object")
        return parsed

    def _parse_list_like(self, key: str) -> list:
        text = self._vars[key].get().strip()
        if text == "":
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception as exc:
                raise ValueError(f"{key}: invalid list/json format") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{key}: expected a list-like object")
        return parsed

    def _parse_optional_float_or_dict(self, key: str):
        text = self._vars[key].get().strip()
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            parsed = self._parse_dict_like(key)
            for name, value in parsed.items():
                parsed[name] = float(value)
            return parsed

    @staticmethod
    def _coerce_bool_value(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")

    def _build_camera_configs(self) -> dict:
        raw = self._parse_dict_like("robot_cameras")
        cameras = {}

        for name, cfg in raw.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"robot_cameras.{name}: expected object")

            camera_type = str(cfg.get("type", "opencv")).lower()
            if camera_type == "opencv":
                if "index_or_path" not in cfg:
                    raise ValueError(f"robot_cameras.{name}: missing index_or_path")

                index_or_path = cfg["index_or_path"]
                if isinstance(index_or_path, str) and index_or_path.strip().isdigit():
                    index_or_path = int(index_or_path.strip())

                cameras[name] = OpenCVCameraConfig(
                    index_or_path=index_or_path,
                    fps=int(cfg["fps"]),
                    width=int(cfg["width"]),
                    height=int(cfg["height"]),
                    color_mode=cfg.get("color_mode", "rgb"),
                    rotation=int(cfg.get("rotation", 0)),
                    warmup_s=int(cfg.get("warmup_s", 1)),
                    fourcc=cfg.get("fourcc"),
                    backend=int(cfg.get("backend", 0)),
                )
                continue

            if camera_type in {"intelrealsense", "realsense"}:
                serial = cfg.get("serial_number_or_name")
                if not serial:
                    raise ValueError(f"robot_cameras.{name}: missing serial_number_or_name for realsense")

                cameras[name] = RealSenseCameraConfig(
                    serial_number_or_name=str(serial),
                    fps=int(cfg["fps"]),
                    width=int(cfg["width"]),
                    height=int(cfg["height"]),
                    color_mode=cfg.get("color_mode", "rgb"),
                    use_depth=self._coerce_bool_value(cfg.get("use_depth", False)),
                    rotation=int(cfg.get("rotation", 0)),
                    warmup_s=int(cfg.get("warmup_s", 1)),
                )
                continue

            raise ValueError(f"robot_cameras.{name}: unsupported camera type '{camera_type}'")

        return cameras

    def _build_robot_config(self):
        robot_type = self._vars["robot_type"].get().strip().lower()
        common_kwargs = {
            "port": self._vars["robot_port"].get().strip(),
            "id": self._vars["robot_id"].get().strip() or None,
            "cameras": self._build_camera_configs(),
            "use_degrees": self._parse_bool("robot_use_degrees"),
            "disable_torque_on_disconnect": self._parse_bool("robot_disable_torque_on_disconnect"),
            "max_relative_target": self._parse_optional_float_or_dict("robot_max_relative_target"),
        }

        if robot_type == "so101_follower":
            return SO101FollowerConfig(**common_kwargs)
        if robot_type == "so100_follower":
            return SO100FollowerConfig(**common_kwargs)

        raise ValueError(
            f"GUI currently supports robot_type in {{so101_follower, so100_follower}}. Received: {robot_type}"
        )

    def _build_harness_config(self) -> HarnessConfig:
        cfg = HarnessConfig(
            enable=self._parse_bool("harness_enable"),
            profile_path=self._vars["harness_profile_path"].get().strip() or None,
            shadow_mode=self._parse_bool("harness_shadow_mode"),
            fail_closed=self._parse_bool("harness_fail_closed"),
            log_dir=self._vars["harness_log_dir"].get().strip() or "harness_traces",
        )
        cfg.server.enable = self._parse_bool("harness_server_enable")
        cfg.server.chunk_validator_enable = self._parse_bool("harness_server_chunk_validator_enable")
        cfg.server.invariant_guard_enable = self._parse_bool("harness_server_invariant_guard_enable")
        cfg.server.micro_rescue_proposal_enable = self._parse_bool(
            "harness_server_micro_rescue_proposal_enable"
        )
        cfg.server.reject_resample_enable = self._parse_bool("harness_server_reject_resample_enable")
        cfg.server.max_resample_attempts = self._parse_int("harness_server_max_resample_attempts")
        cfg.server.re_infer_on_intervention = self._parse_bool("harness_server_re_infer_on_intervention")

        cfg.client.enable = self._parse_bool("harness_client_enable")
        cfg.client.execution_guard_enable = self._parse_bool("harness_client_execution_guard_enable")
        cfg.client.hard_invariant_guard_enable = self._parse_bool(
            "harness_client_hard_invariant_guard_enable"
        )
        cfg.client.speed_envelope_enable = self._parse_bool("harness_client_speed_envelope_enable")
        cfg.client.tracking_monitor_enable = self._parse_bool("harness_client_tracking_monitor_enable")
        cfg.client.tracking_monitor_window_steps = self._parse_int(
            "harness_client_tracking_monitor_window_steps"
        )
        cfg.client.tracking_monitor_state_radius = self._parse_float(
            "harness_client_tracking_monitor_state_radius"
        )
        cfg.client.tracking_monitor_min_path_length = self._parse_float(
            "harness_client_tracking_monitor_min_path_length"
        )
        cfg.client.tracking_monitor_cooldown_steps = self._parse_int(
            "harness_client_tracking_monitor_cooldown_steps"
        )
        cfg.client.tracking_monitor_dims = [
            int(item) for item in self._parse_list_like("harness_client_tracking_monitor_dims")
        ]
        cfg.client.clear_queue_on_intervention = self._parse_bool(
            "harness_client_clear_queue_on_intervention"
        )
        cfg.client.request_reinfer_on_intervention = self._parse_bool(
            "harness_client_request_reinfer_on_intervention"
        )

        cfg.micro_rescue.enable = self._parse_bool("harness_micro_rescue_enable")
        cfg.micro_rescue.shadow_mode = self._parse_bool("harness_micro_rescue_shadow_mode")
        cfg.micro_rescue.state_knn_enable = self._parse_bool("harness_micro_rescue_state_knn_enable")
        cfg.micro_rescue.image_knn_enable = self._parse_bool("harness_micro_rescue_image_knn_enable")
        cfg.micro_rescue.k_neighbors = self._parse_int("harness_micro_rescue_k_neighbors")
        cfg.micro_rescue.snippet_horizon_steps = self._parse_int(
            "harness_micro_rescue_snippet_horizon_steps"
        )
        cfg.micro_rescue.max_duration_s = self._parse_float("harness_micro_rescue_max_duration_s")
        cfg.micro_rescue.blend_alpha = self._parse_float("harness_micro_rescue_blend_alpha")
        cfg.micro_rescue.ramp_in_steps = self._parse_int("harness_micro_rescue_ramp_in_steps")
        cfg.micro_rescue.ramp_in_max_joint_delta = self._parse_optional_float(
            "harness_micro_rescue_ramp_in_max_joint_delta"
        )
        cfg.micro_rescue.min_future_progress_score = self._parse_float(
            "harness_micro_rescue_min_future_progress_score"
        )
        cfg.micro_rescue.max_state_distance = self._parse_optional_float(
            "harness_micro_rescue_max_state_distance"
        )
        cfg.micro_rescue.cooldown_s = self._parse_float("harness_micro_rescue_cooldown_s")
        cfg.micro_rescue.max_rescues_per_episode = self._parse_int(
            "harness_micro_rescue_max_rescues_per_episode"
        )

        cfg.invariant_guard.enable = self._parse_bool("harness_invariant_guard_enable")
        cfg.invariant_guard.shadow_mode = self._parse_bool("harness_invariant_guard_shadow_mode")
        cfg.invariant_guard.min_support = self._parse_float("harness_invariant_guard_min_support")
        cfg.invariant_guard.max_train_violation_rate = self._parse_float(
            "harness_invariant_guard_max_train_violation_rate"
        )
        cfg.invariant_guard.min_mode_confidence = self._parse_float(
            "harness_invariant_guard_min_mode_confidence"
        )
        cfg.invariant_guard.hard_guard_categories = [
            str(item) for item in self._parse_list_like("harness_invariant_guard_hard_guard_categories")
        ]
        cfg.invariant_guard.soft_guard_categories = [
            str(item) for item in self._parse_list_like("harness_invariant_guard_soft_guard_categories")
        ]
        cfg.invariant_guard.flush_on_hard_guard = self._parse_bool(
            "harness_invariant_guard_flush_on_hard_guard"
        )
        cfg.speed_envelope.enable = self._parse_bool("harness_speed_envelope_enable")
        cfg.speed_envelope.shadow_mode = self._parse_bool("harness_speed_envelope_shadow_mode")
        cfg.speed_envelope.percentile_low = self._parse_float("harness_speed_envelope_percentile_low")
        cfg.speed_envelope.percentile_high = self._parse_float("harness_speed_envelope_percentile_high")
        cfg.speed_envelope.mode_conditioned = self._parse_bool("harness_speed_envelope_mode_conditioned")
        cfg.speed_envelope.max_consecutive_clamps = self._parse_int(
            "harness_speed_envelope_max_consecutive_clamps"
        )
        cfg.speed_envelope.flush_after_repeated_clamp = self._parse_bool(
            "harness_speed_envelope_flush_after_repeated_clamp"
        )
        cfg.sync.enable = self._parse_bool("harness_sync_enable")
        cfg.sync.require_chunk_id = self._parse_bool("harness_sync_require_chunk_id")
        cfg.sync.flush_on_reject = self._parse_bool("harness_sync_flush_on_reject")
        cfg.sync.flush_on_rescue = self._parse_bool("harness_sync_flush_on_rescue")
        cfg.sync.flush_on_hard_clamp = self._parse_bool("harness_sync_flush_on_hard_clamp")
        cfg.sync.flush_on_repeated_speed_clamp = self._parse_bool(
            "harness_sync_flush_on_repeated_speed_clamp"
        )
        cfg.sync.block_execution_until_fresh_chunk = self._parse_bool(
            "harness_sync_block_execution_until_fresh_chunk"
        )
        cfg.trace.enable = self._parse_bool("harness_trace_enable")
        cfg.trace.record_images = self._parse_bool("harness_trace_record_images")
        cfg.trace.record_raw_chunks = self._parse_bool("harness_trace_record_raw_chunks")
        cfg.trace.record_postprocessed_chunks = self._parse_bool(
            "harness_trace_record_postprocessed_chunks"
        )
        cfg.trace.record_executed_actions = self._parse_bool("harness_trace_record_executed_actions")
        cfg.trace.record_mode_estimates = self._parse_bool("harness_trace_record_mode_estimates")
        cfg.trace.record_rescue_neighbors = self._parse_bool("harness_trace_record_rescue_neighbors")
        return cfg

    def _build_client_config(self) -> RobotClientConfig:
        schedule_name = self._vars["rtc_prefix_attention_schedule"].get().strip().upper()
        schedule = RTCAttentionSchedule[schedule_name]

        cfg = RobotClientConfig(
            policy_type=self._vars["policy_type"].get().strip(),
            pretrained_name_or_path=self._vars["pretrained_name_or_path"].get().strip(),
            robot=self._build_robot_config(),
            actions_per_chunk=self._parse_int("actions_per_chunk"),
            task=self._vars["task"].get().strip(),
            rename_map=self._parse_dict_like("rename_map"),
            server_address=self._vars["server_address"].get().strip(),
            policy_device=self._vars["policy_device"].get().strip(),
            client_device=self._vars["client_device"].get().strip(),
            chunk_size_threshold=self._parse_float("chunk_size_threshold"),
            fps=self._parse_int("fps"),
            obs_timestep_independent=self._parse_bool("obs_timestep_independent"),
            image_compress_enable=self._parse_bool("image_compress_enable"),
            image_compress_quality=self._parse_int("image_compress_quality"),
            interpolation_multiplier=self._parse_int("interpolation_multiplier"),
            aggregate_fn_name=self._vars["aggregate_fn_name"].get().strip(),
            debug_visualize_queue_size=self._parse_bool("debug_visualize_queue_size"),
            rtc_enabled=self._parse_bool("rtc_enabled"),
            rtc_execution_horizon=self._parse_int("rtc_execution_horizon"),
            rtc_max_guidance_weight=self._parse_float("rtc_max_guidance_weight"),
            rtc_prefix_attention_schedule=schedule,
            rtc_debug=self._parse_bool("rtc_debug"),
            rtc_debug_maxlen=self._parse_int("rtc_debug_maxlen"),
            inference_delay_steps=self._parse_optional_int("inference_delay_steps"),
            xvla_domain_id=self._parse_optional_int("xvla_domain_id"),
            record_obs_enable=self._parse_bool("record_obs_enable"),
            record_obs_dir=self._vars["record_obs_dir"].get().strip() or "recorded_obs",
            record_action_enable=self._parse_bool("record_action_enable"),
            record_action_dir=self._vars["record_action_dir"].get().strip() or "recorded_obs",
            capture_attn_enable=self._parse_bool("capture_attn_enable"),
            capture_attn_dir=self._vars["capture_attn_dir"].get().strip() or "attention_captures",
            harness=self._build_harness_config(),
        )
        return cfg

    def _build_export_payload(self) -> dict:
        """Build a draccus-loadable dict matching the CLI's RTCXVLAClientOnlyConfig.

        Reuses ``_build_client_config`` so the exported file reflects exactly the
        same validated values the GUI would use to connect. The robot sub-config is
        encoded with draccus so its ``type`` discriminator (and per-camera ``type``)
        is included, making the JSON round-trip via ``--config_path``.
        """
        cfg = self._build_client_config()
        schedule = cfg.rtc_prefix_attention_schedule
        return {
            "server_address": cfg.server_address,
            "robot": draccus.encode(cfg.robot),
            "task": cfg.task,
            "policy_type": cfg.policy_type,
            "pretrained_name_or_path": cfg.pretrained_name_or_path,
            "policy_device": cfg.policy_device,
            "client_device": cfg.client_device,
            "rename_map": cfg.rename_map,
            "actions_per_chunk": cfg.actions_per_chunk,
            "chunk_size_threshold": cfg.chunk_size_threshold,
            "aggregate_fn_name": cfg.aggregate_fn_name,
            "rtc_enabled": cfg.rtc_enabled,
            "rtc_execution_horizon": cfg.rtc_execution_horizon,
            "rtc_max_guidance_weight": cfg.rtc_max_guidance_weight,
            "rtc_prefix_attention_schedule": getattr(schedule, "value", str(schedule)),
            "rtc_debug": cfg.rtc_debug,
            "rtc_debug_maxlen": cfg.rtc_debug_maxlen,
            "inference_delay_steps": cfg.inference_delay_steps,
            "xvla_domain_id": cfg.xvla_domain_id,
            "fps": cfg.fps,
            "obs_timestep_independent": cfg.obs_timestep_independent,
            "image_compress_enable": cfg.image_compress_enable,
            "image_compress_quality": cfg.image_compress_quality,
            "interpolation_multiplier": cfg.interpolation_multiplier,
            "debug_visualize_queue_size": cfg.debug_visualize_queue_size,
            "record_obs_enable": cfg.record_obs_enable,
            "record_obs_dir": cfg.record_obs_dir,
            "record_action_enable": cfg.record_action_enable,
            "record_action_dir": cfg.record_action_dir,
            "capture_attn_enable": cfg.capture_attn_enable,
            "capture_attn_dir": cfg.capture_attn_dir,
            "harness": draccus.encode(cfg.harness),
            "dataset_repo_id": self._vars["dataset_repo_id"].get().strip(),
            "reset_pose_mode": self._vars["reset_pose_mode"].get().strip(),
            "start_pose_shoulder_pan": self._parse_float("start_pose_shoulder_pan"),
            "start_pose_shoulder_lift": self._parse_float("start_pose_shoulder_lift"),
            "start_pose_elbow_flex": self._parse_float("start_pose_elbow_flex"),
            "start_pose_wrist_flex": self._parse_float("start_pose_wrist_flex"),
            "start_pose_wrist_roll": self._parse_float("start_pose_wrist_roll"),
            "start_pose_gripper": self._parse_float("start_pose_gripper"),
            "reset_duration_start": self._parse_float("reset_duration_start"),
            "reset_duration_after_stop": self._parse_float("reset_duration_after_stop"),
        }

    def _on_export_config(self):
        # Runs on the Tk main thread (filedialog must not be called off-thread).
        try:
            payload = self._build_export_payload()
        except Exception as exc:
            self._log(f"[ERROR] Export failed: {exc}")
            messagebox.showerror("Export Config", f"Invalid configuration:\n{exc}")
            return

        path = filedialog.asksaveasfilename(
            title="Export RTC config",
            defaultextension=".json",
            initialfile="rtc_config.json",
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self._log(f"[ERROR] Could not write config: {exc}")
            messagebox.showerror("Export Config", f"Could not write file:\n{exc}")
            return

        run_cmd = (
            "uv run python scripts/orchestrator/orchestrator_rtc_client_only.py "
            f'--config_path="{path}"'
        )
        self._log(f"[OK] Config exported to {path}")
        self._log(f"[INFO] Run with: {run_cmd}")
        messagebox.showinfo(
            "Export Config",
            "Configuration exported to:\n"
            f"{path}\n\n"
            "Run it from the repo root with:\n\n"
            f"{run_cmd}\n\n"
            "You can still override any field on the command line, "
            "e.g. append --task=\"...\".",
        )

    @staticmethod
    def _format_var_value(value) -> str:
        """Convert a decoded config value into the string a tk var/widget expects."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _apply_imported_payload(self, payload: dict) -> list[str]:
        """Populate form fields from a draccus-style config dict.

        Returns the list of keys that could not be mapped to a form field.
        """
        # Map robot.<field> -> robot_<var>; "cameras" -> robot_cameras (JSON).
        robot_field_to_var = {
            "type": "robot_type",
            "port": "robot_port",
            "id": "robot_id",
            "use_degrees": "robot_use_degrees",
            "disable_torque_on_disconnect": "robot_disable_torque_on_disconnect",
            "max_relative_target": "robot_max_relative_target",
            "cameras": "robot_cameras",
        }
        harness_field_to_var = {
            "enable": "harness_enable",
            "profile_path": "harness_profile_path",
            "shadow_mode": "harness_shadow_mode",
            "fail_closed": "harness_fail_closed",
            "log_dir": "harness_log_dir",
        }
        harness_nested_to_prefix = {
            "server": "harness_server",
            "client": "harness_client",
            "micro_rescue": "harness_micro_rescue",
            "invariant_guard": "harness_invariant_guard",
            "speed_envelope": "harness_speed_envelope",
            "sync": "harness_sync",
            "trace": "harness_trace",
        }

        skipped: list[str] = []
        for key, value in payload.items():
            if key == "robot":
                if not isinstance(value, dict):
                    skipped.append("robot")
                    continue
                for rkey, rval in value.items():
                    var_key = robot_field_to_var.get(rkey)
                    if var_key is None or var_key not in self._vars:
                        if rkey not in ("calibration_dir",):
                            skipped.append(f"robot.{rkey}")
                        continue
                    self._vars[var_key].set(self._format_var_value(rval))
                continue

            if key == "harness":
                if not isinstance(value, dict):
                    skipped.append("harness")
                    continue
                for hkey, hval in value.items():
                    if hkey in harness_field_to_var:
                        self._vars[harness_field_to_var[hkey]].set(self._format_var_value(hval))
                        continue
                    prefix = harness_nested_to_prefix.get(hkey)
                    if prefix is None:
                        skipped.append(f"harness.{hkey}")
                        continue
                    if not isinstance(hval, dict):
                        skipped.append(f"harness.{hkey}")
                        continue
                    for subkey, subval in hval.items():
                        var_key = f"{prefix}_{subkey}"
                        if var_key in self._vars:
                            self._vars[var_key].set(self._format_var_value(subval))
                continue

            if key in self._vars:
                self._vars[key].set(self._format_var_value(value))
            else:
                skipped.append(key)

        return skipped

    def _on_import_config(self):
        # Runs on the Tk main thread (filedialog must not be called off-thread).
        path = filedialog.askopenfilename(
            title="Import RTC config",
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self._log(f"[ERROR] Could not read config: {exc}")
            messagebox.showerror("Import Config", f"Could not read file:\n{exc}")
            return

        if not isinstance(payload, dict):
            self._log("[ERROR] Import failed: config root must be a JSON object")
            messagebox.showerror("Import Config", "Config root must be a JSON object.")
            return

        skipped = self._apply_imported_payload(payload)

        self._log(f"[OK] Config imported from {path}")
        if skipped:
            self._log(f"[WARN] Ignored unknown keys: {', '.join(skipped)}")
        messagebox.showinfo(
            "Import Config",
            "Configuration loaded into the form."
            + (f"\n\nIgnored unknown keys:\n{', '.join(skipped)}" if skipped else ""),
        )

    # Canonical joint (from start_pose_analysis) -> the GUI start-pose field to fill.
    _CANON_TO_FIELD = {
        "shoulder_pan": "start_pose_shoulder_pan",
        "shoulder_lift": "start_pose_shoulder_lift",
        "elbow_flex": "start_pose_elbow_flex",
        "wrist_flex": "start_pose_wrist_flex",
        "wrist_roll": "start_pose_wrist_roll",
        "gripper": "start_pose_gripper",
    }

    def _on_analyze_dataset(self):
        # Runs off-thread via _run_bg (network + parquet read). Does NOT touch robot state.
        repo = self._vars["dataset_repo_id"].get().strip()
        if not repo:
            raise RuntimeError("Dataset Repo is empty")

        self._log(f"[INFO] Analyzing safe start-pose region for {repo} ...")
        stats = spa.analyze_start_pose(repo)
        spa.save_stats(stats)
        self.after(0, lambda: self._apply_start_pose_stats(stats))

    def _apply_start_pose_stats(self, stats: spa.StartPoseStats):
        # Runs on the Tk main thread (mutates StringVars / shows dialog).
        self._start_pose_stats = stats
        applied = []
        for canon, field_key in self._CANON_TO_FIELD.items():
            if canon in stats.stats and field_key in self._vars:
                self._vars[field_key].set(f"{stats.stats[canon]['median']:.1f}")
                applied.append(canon)

        self._log(
            f"[OK] Analyzed {stats.n_episodes} episodes @ {stats.revision[:8]}. "
            f"Reset target set to per-joint medians."
        )
        for joint in stats.joints:
            s = stats.stats[joint]
            self._log(
                f"    {joint}: median={s['median']:.1f}  "
                f"IQR=[{s['q1']:.1f}, {s['q3']:.1f}]  range=[{s['min']:.1f}, {s['max']:.1f}]"
            )
        missing = [c for c in self._CANON_TO_FIELD if c not in applied]
        if missing:
            self._log(f"[WARN] No analyzed value for: {', '.join(missing)} (kept previous).")
        messagebox.showinfo(
            "Analyze Dataset",
            f"Start-pose region updated from {stats.n_episodes} episodes\n"
            f"{stats.repo_id} @ {stats.revision[:8]}\n\n"
            "Reset target = per-joint median (inside the safe IQR box).",
        )

    def _check_dataset_update(self):
        # Background, non-blocking: compare Hub's latest revision vs the last analysed one.
        repo = self._vars["dataset_repo_id"].get().strip()
        if not repo:
            return
        try:
            latest = spa.resolve_revision(repo)
        except Exception as exc:
            self._log(f"[WARN] Could not check dataset version for {repo}: {exc}")
            return

        cached = spa.load_cached_stats(repo)
        if cached is None:
            self._log(
                f"[INFO] No start-pose analysis cached for {repo}. "
                f"Press 'Analyze Dataset → Start-Pose' to compute the safe region."
            )
        elif cached.revision != latest:
            self._log(
                f"[INFO] New dataset version for {repo}: {latest[:8]} "
                f"(analyzed {cached.revision[:8]}). Press 'Analyze Dataset → Start-Pose' to update."
            )
        else:
            self._log(f"[INFO] Start-pose region up to date for {repo} ({latest[:8]}).")

    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def _extract_current_action_from_observation(
        self,
        current_obs: dict,
        joint_names: list[str],
    ) -> dict[str, float]:
        missing = [k for k in joint_names if k not in current_obs]
        if not missing:
            return {k: self._to_float(current_obs[k]) for k in joint_names}

        state = current_obs.get("observation.state")
        if state is not None:
            if hasattr(state, "detach"):
                state = state.detach()
            if hasattr(state, "cpu"):
                state = state.cpu()
            if hasattr(state, "numpy"):
                state = state.numpy()

            state_array = np.asarray(state, dtype=np.float64).reshape(-1)
            if state_array.size == len(joint_names):
                return {k: float(state_array[i]) for i, k in enumerate(joint_names)}

        raise KeyError(f"Missing joint keys in observation and invalid observation.state fallback: {missing}")

    # Maps each robot joint (matched by substring on its name) to the GUI start-pose field.
    # Tokens are specific enough to disambiguate (shoulder_pan vs shoulder_lift, wrist_flex
    # vs wrist_roll). The gripper value is a position, not an angle (no deg->rad conversion).
    _START_POSE_MAP = (
        (("shoulder_pan",), "start_pose_shoulder_pan", False),
        (("shoulder_lift",), "start_pose_shoulder_lift", False),
        (("elbow",), "start_pose_elbow_flex", False),
        (("wrist_flex",), "start_pose_wrist_flex", False),
        (("wrist_roll",), "start_pose_wrist_roll", False),
        (("gripper", "jaw"), "start_pose_gripper", True),
    )

    def _start_pose_target_value(self, field_key: str) -> float:
        """Per-joint reset target (in degrees) according to the selected Reset Mode.

        - ``median``        -> the (user-editable) median field value.
        - ``random_iqr``    -> uniform sample within [Q1, Q3] of the analyzed region.
        - ``random_minmax`` -> uniform sample within [min, max] of the analyzed region.

        Random modes need a prior dataset analysis; without it they fall back to the
        median field value.
        """
        median_value = self._parse_float(field_key)
        mode = self._vars["reset_pose_mode"].get().strip()
        if mode == "median" or self._start_pose_stats is None:
            return median_value

        canonical = field_key.removeprefix("start_pose_")
        s = self._start_pose_stats.stats.get(canonical)
        if s is None:
            return median_value

        if mode == "random_iqr":
            lo, hi = s["q1"], s["q3"]
        elif mode == "random_minmax":
            lo, hi = s["min"], s["max"]
        else:
            return median_value
        return float(np.random.uniform(lo, hi))

    def _build_start_pose_target(
        self, joint_names: list[str], start_action: dict[str, float], use_degrees: bool
    ) -> tuple[dict[str, float], list[str]]:
        """Build absolute joint targets for the safe start pose.

        Unmatched joints keep their current value (held in place). Returns (target, matched).
        """
        target = dict(start_action)
        matched: list[str] = []
        for name in joint_names:
            lname = name.lower()
            for tokens, field_key, is_gripper in self._START_POSE_MAP:
                if any(tok in lname for tok in tokens):
                    value = self._start_pose_target_value(field_key)
                    if not is_gripper and not use_degrees:
                        value = value * np.pi / 180.0
                    target[name] = value
                    matched.append(name)
                    break
        return target, matched

    def _home_to_pose(
        self,
        robot,
        duration: float,
        start_action: dict[str, float],
        joint_names: list[str],
        target_action: dict[str, float],
    ):
        hz = 50.0
        steps = max(1, int(duration * hz))
        sleep_time = 1.0 / hz

        for i in range(1, steps + 1):
            alpha = i / steps
            smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
            interp_action = {
                k: start_action[k] + smooth_alpha * (target_action[k] - start_action[k]) for k in joint_names
            }
            robot.send_action(interp_action)
            time.sleep(sleep_time)

        robot.send_action(target_action)

    def _reset_robot_to_start_pose(self, duration: float):
        if self._client is None:
            raise RuntimeError("Client is not connected")

        robot = self._client.robot
        current_obs = robot.get_observation()
        joint_names = list(robot.action_features.keys())
        use_degrees = True
        if self._client_cfg is not None and hasattr(self._client_cfg.robot, "use_degrees"):
            use_degrees = bool(self._client_cfg.robot.use_degrees)

        try:
            start_action = self._extract_current_action_from_observation(current_obs, joint_names)
        except KeyError:
            start_action = dict.fromkeys(joint_names, 0.0)

        mode = self._vars["reset_pose_mode"].get().strip()
        if mode != "median" and self._start_pose_stats is None:
            self._log(
                f"[WARN] Reset mode '{mode}' needs a dataset analysis; "
                "falling back to median. Press 'Analyze Dataset → Start-Pose' first."
            )

        target_action, matched = self._build_start_pose_target(joint_names, start_action, use_degrees)
        unmatched = [j for j in joint_names if j not in matched]
        if unmatched:
            self._log(f"[WARN] Start-pose: no mapping for {unmatched}; holding current value.")
        self._log(
            f"[INFO] Reset target (mode={mode}): "
            + ", ".join(f"{k}={target_action[k]:.1f}" for k in joint_names)
        )

        self._home_to_pose(robot, duration, start_action, joint_names, target_action)

    # ----- Safe pose (capture-file driven; no defaults) --------------------------------

    def _try_autoload_safe_pose(self):
        default = "runtime_configs/safe_pose.json"
        if os.path.exists(default):
            try:
                self._load_safe_pose_file(default)
                self._log(f"[OK] Auto-loaded safe pose from {default}")
            except Exception as exc:
                self._log(f"[WARN] Could not auto-load {default}: {exc}")
        else:
            self._log(
                "[INFO] No runtime_configs/safe_pose.json found. Capture one with "
                "'python -m lerobot.gui.capture_safe_pose' and Load it before Auto Reset Cycle."
            )

    def _load_safe_pose_file(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pose = data.get("safe_pose")
        if not isinstance(pose, dict) or not pose:
            raise ValueError("file has no non-empty 'safe_pose' object")

        self._safe_pose = {k: float(v) for k, v in pose.items()}
        self._safe_pose_meta = {k: v for k, v in data.items() if k != "safe_pose"}
        self._safe_pose_path = path

        # Warn if the capture's use_degrees disagrees with the GUI robot setting (scale mismatch).
        file_deg = self._safe_pose_meta.get("use_degrees")
        if file_deg is not None:
            gui_deg = self._vars["robot_use_degrees"].get().strip().lower() in {"true", "1", "yes", "y", "on"}
            if bool(file_deg) != gui_deg:
                self._log(
                    f"[WARN] Safe-pose use_degrees={file_deg} != GUI robot use_degrees={gui_deg}; "
                    "joint values may be wrong-scaled."
                )

        self.after(0, self._render_safe_pose)

    def _render_safe_pose(self):
        if self._safe_pose is None:
            self.safe_pose_var.set("(none loaded)")
            return
        meta = self._safe_pose_meta or {}
        lines = [f"file: {self._safe_pose_path}"]
        if meta.get("captured_at"):
            lines.append(f"captured: {meta['captured_at']}")
        if meta.get("robot_type"):
            lines.append(
                f"robot: {meta.get('robot_type')} (id={meta.get('robot_id')}) "
                f"use_degrees={meta.get('use_degrees')}"
            )
        lines.append(
            ", ".join(f"{k.replace('.pos', '')}={v:.1f}" for k, v in self._safe_pose.items())
        )
        self.safe_pose_var.set("\n".join(lines))

    def _on_load_safe_pose(self):
        # Runs on the Tk main thread (filedialog).
        path = filedialog.askopenfilename(
            title="Load safe pose",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._load_safe_pose_file(path)
        except Exception as exc:
            self._log(f"[ERROR] Could not load safe pose: {exc}")
            messagebox.showerror("Load Safe Pose", f"Could not load file:\n{exc}")
            return
        self._log(f"[OK] Safe pose loaded from {path}")
        self.after(0, self._refresh_controls)

    def _move_to_safe_pose(self, duration: float):
        if self._client is None:
            raise RuntimeError("Client is not connected")
        if not self._safe_pose:
            raise RuntimeError("No safe pose loaded")

        robot = self._client.robot
        current_obs = robot.get_observation()
        joint_names = list(robot.action_features.keys())
        try:
            start_action = self._extract_current_action_from_observation(current_obs, joint_names)
        except KeyError:
            start_action = dict.fromkeys(joint_names, 0.0)

        target = dict(start_action)
        matched: list[str] = []
        for name in joint_names:
            if name in self._safe_pose:  # exact key match (both are "<motor>.pos")
                target[name] = float(self._safe_pose[name])
                matched.append(name)
            else:  # token fallback (e.g. naming differences)
                base = name.lower().replace(".pos", "")
                for key, value in self._safe_pose.items():
                    if key.lower().replace(".pos", "") == base:
                        target[name] = float(value)
                        matched.append(name)
                        break

        unmatched = [j for j in joint_names if j not in matched]
        if unmatched:
            self._log(f"[WARN] Safe pose: no value for {unmatched}; holding current value.")
        self._log(
            "[INFO] Safe-pose target: " + ", ".join(f"{k}={target[k]:.1f}" for k in joint_names)
        )
        self._home_to_pose(robot, duration, start_action, joint_names, target)

    def _stop_client_stream_preserve_robot(self):
        if self._client is None:
            return

        with self._state_lock:
            running = self._state.stream_running
            control_thread = self._state.control_loop_thread
            receiver_thread = self._state.action_receiver_thread

        if not running:
            return

        self._client.shutdown_event.set()
        if control_thread is not None:
            control_thread.join(timeout=2.0)
        if receiver_thread is not None:
            receiver_thread.join(timeout=2.0)

        if receiver_thread is not None and receiver_thread.is_alive():
            with contextlib.suppress(Exception):
                self._client.channel.close()
            receiver_thread.join(timeout=1.0)

        with self._state_lock:
            self._state.stream_running = False

    def _on_connect(self):
        with self._state_lock:
            if self._state.connected:
                self._log("[INFO] Already connected")
                return

        self._log("[INFO] Building config and connecting...")
        cfg = self._build_client_config()
        client = RobotClient(cfg)

        self._client = client
        self._client_cfg = cfg

        with self._state_lock:
            self._state.connected = True
            self._state.start_pose_done = False
            self._state.stream_running = False
            self._state.action_receiver_thread = None
            self._state.control_loop_thread = None

        self._set_status("Connected")
        self._log("[OK] Connected to robot. Press 'Reset Start-Pose' before 'Call to Server'.")
        # Non-blocking: tell the user if a newer dataset version warrants re-analysis.
        self._run_bg(self._check_dataset_update, show_error=False)
        self.after(0, self._refresh_controls)

    def _on_reset_start_pose(self):
        with self._state_lock:
            connected = self._state.connected
            stream_running = self._state.stream_running
        if not connected:
            raise RuntimeError("Please connect first")
        if stream_running:
            raise RuntimeError("Cannot reset while stream is running")

        duration = self._parse_float("reset_duration_start")

        self._log(f"[INFO] Auto-resetting to safe start pose ({duration:.1f}s)...")
        self._reset_robot_to_start_pose(duration=duration)

        with self._state_lock:
            self._state.start_pose_done = True

        self._set_status("Connected | Start Pose Done")
        self._log("[OK] Robot is at safe start pose")
        self.after(0, self._refresh_controls)

    def _on_call_server(self):
        with self._state_lock:
            connected = self._state.connected
            start_pose_done = self._state.start_pose_done
            stream_running = self._state.stream_running

        if not connected:
            raise RuntimeError("Please connect first")
        if not start_pose_done:
            raise RuntimeError(
                "Call to Server requires robot at the safe start pose. Press 'Reset Start-Pose' first"
            )
        if stream_running:
            raise RuntimeError("Stream is already running")
        if self._client is None or self._client_cfg is None:
            raise RuntimeError("Client is not initialized")

        self._log("[INFO] Starting client-server stream...")
        if not self._client.start():
            raise RuntimeError("Failed to connect to policy server")

        receiver = threading.Thread(target=self._client.receive_actions, daemon=True)
        control = threading.Thread(
            target=self._client.control_loop,
            kwargs={"task": self._client_cfg.task},
            daemon=True,
        )
        receiver.start()
        control.start()

        with self._state_lock:
            self._state.stream_running = True
            # Require a fresh start-pose reset before next call cycle.
            self._state.start_pose_done = False
            self._state.action_receiver_thread = receiver
            self._state.control_loop_thread = control

        self._set_status("Connected | Stream Running")
        self._log("[OK] Stream started")
        self.after(0, self._refresh_controls)

    def _on_stop_stream(self):
        with self._state_lock:
            connected = self._state.connected
            stream_running = self._state.stream_running
        if not connected:
            raise RuntimeError("Please connect first")
        if not stream_running:
            raise RuntimeError("Stream is not running")

        self._log("[INFO] Stopping stream (preserve robot connection)...")
        self._stop_client_stream_preserve_robot()

        duration = self._parse_float("reset_duration_after_stop")
        self._log(f"[INFO] Moving to safe pose ({duration:.1f}s)...")
        self._move_to_safe_pose(duration=duration)

        with self._state_lock:
            self._state.start_pose_done = False

        self._set_status("Connected | Safe Pose")
        self._log("[OK] Stream stopped and robot moved to safe pose")
        self.after(0, self._refresh_controls)

    def _on_auto_reset_cycle(self):
        """One-button reset between runs, with a FULL reconnect so no stale stream state.

        Flow: stop stream -> move to SAFE rest pose -> disconnect -> reconnect ->
        reset to start pose (selected mode) -> call to server.
        """
        if self._safe_pose is None:
            raise RuntimeError("No safe pose loaded. Capture & Load a safe-pose file first.")

        with self._state_lock:
            connected = self._state.connected
            stream_running = self._state.stream_running
        if not connected:
            raise RuntimeError("Please connect first")

        self._log("[CYCLE] Auto-reset cycle started.")

        # 1) stop the stream (if running) and move to the safe rest pose
        if stream_running:
            self._log("[CYCLE 1/5] Stopping stream...")
            self._stop_client_stream_preserve_robot()
        else:
            self._log("[CYCLE 1/5] No active stream.")
        self._set_status("Cycle | Moving to safe pose")
        self._move_to_safe_pose(self._parse_float("reset_duration_after_stop"))

        # 2) full disconnect -> drops the old client-server connection
        self._log("[CYCLE 2/5] Disconnecting (closing client-server connection)...")
        self._set_status("Cycle | Disconnecting")
        self._on_disconnect()

        # 3) fresh connect -> brand new RobotClient / channel
        self._log("[CYCLE 3/5] Reconnecting (fresh connection)...")
        self._set_status("Cycle | Reconnecting")
        self._on_connect()
        with self._state_lock:
            if not self._state.connected:
                raise RuntimeError("Reconnect failed; aborting cycle")

        # 4) reset to the in-distribution start pose (median / random_iqr / random_minmax)
        self._log("[CYCLE 4/5] Resetting to start pose...")
        self._set_status("Cycle | Reset start pose")
        self._reset_robot_to_start_pose(self._parse_float("reset_duration_start"))
        with self._state_lock:
            self._state.start_pose_done = True

        # 5) call to server -> start a fresh stream
        self._log("[CYCLE 5/5] Calling to server...")
        self._set_status("Cycle | Calling server")
        self._on_call_server()

        self._log("[CYCLE] Auto-reset cycle complete.")

    def _on_disconnect(self):
        with self._state_lock:
            connected = self._state.connected
        if not connected:
            self._log("[INFO] Already disconnected")
            return
        if self._client is None:
            return

        self._log("[INFO] Disconnecting...")
        self._stop_client_stream_preserve_robot()

        try:
            self._client.stop()
        except Exception:
            with contextlib.suppress(Exception):
                self._client.channel.close()
            with contextlib.suppress(Exception):
                self._client.robot.disconnect()

        self._client = None
        self._client_cfg = None

        with self._state_lock:
            self._state.connected = False
            self._state.start_pose_done = False
            self._state.stream_running = False
            self._state.action_receiver_thread = None
            self._state.control_loop_thread = None

        self._set_status("Disconnected")
        self._log("[OK] Disconnected")
        self.after(0, self._refresh_controls)

    def _on_close(self):
        def close_worker():
            with contextlib.suppress(Exception):
                self._on_disconnect()
            self.after(0, self.destroy)

        threading.Thread(target=close_worker, daemon=True).start()


def run_gui() -> None:
    register_third_party_plugins()
    app = RTCControlGUI()
    app.mainloop()


RTCXVLAControlGUI = RTCControlGUI
