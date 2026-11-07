# SPDX-License-Identifier: Apache-2.0
"""Opt-in server-side timing samples for Metal inference steps."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any


class MetalTimingTrace:
    def __init__(self, capacity: int = 100_000) -> None:
        self.enabled = os.environ.get("VLLM_METAL_TRACE_TIMING") == "1"
        self._steps: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next_step_id = 0
        self._prefill_ms: dict[str, float] = {}

    def begin_step(self) -> tuple[int, int]:
        with self._lock:
            self._next_step_id += 1
            step_id = self._next_step_id
        return step_id, time.perf_counter_ns()

    def record_step(
        self,
        *,
        step_id: int,
        start_ns: int,
        end_ns: int,
        prefill_reqs: list[Any],
        decode_reqs: list[tuple[str, Any]],
        forward_ms: float,
        sampling_ms: float,
        sampled_tokens: int,
        finished_reqs: set[str],
    ) -> None:
        if not self.enabled:
            return
        prefills = []
        with self._lock:
            for req in prefill_reqs:
                self._prefill_ms[req.req_id] = (
                    self._prefill_ms.get(req.req_id, 0.0) + forward_ms
                )
                if req.prompt_len is not None:
                    prefills.append(
                        {
                            "req_id": req.req_id,
                            "ms": self._prefill_ms.pop(req.req_id),
                            "sampling_ms": sampling_ms,
                        }
                    )
        step = {
            "t": "step",
            "step_id": step_id,
            "step_id_first": step_id,
            "step_id_last": step_id,
            "mono_ns_first": start_ns,
            "mono_ns_last": end_ns,
            "ts_first": time.time() - (end_ns - start_ns) / 1e9,
            "ts_last": time.time(),
            "prefills": prefills,
            "decode": {
                "reqs": [req_id for req_id, _ in decode_reqs],
                "tokens": sampled_tokens,
                "forward_ms_samples": [forward_ms] if decode_reqs else [],
                "sampling_ms_samples": [sampling_ms] if decode_reqs else [],
            },
            "finished": sorted(finished_reqs),
        }
        with self._lock:
            self._steps.append(step)

    def snapshot(self, *, after_step_id: int = -1, through_step_id: int | None = None):
        with self._lock:
            steps = list(self._steps)
        return [
            step
            for step in steps
            if step["step_id"] > after_step_id
            and (through_step_id is None or step["step_id"] <= through_step_id)
        ]

    @property
    def total_steps(self) -> int:
        with self._lock:
            return self._next_step_id
