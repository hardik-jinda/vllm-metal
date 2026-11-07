# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP endpoints for collecting Metal model-runner timings."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import os
import threading
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, HTTPException, Request
from starlette.datastructures import State

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient


_request_id: ContextVar[str | None] = ContextVar(
    "metal_timing_request_id", default=None
)
_request_timings: OrderedDict[str, dict] = OrderedDict()
_request_lock = threading.Lock()


def _record_request(request_id: str, key: str, value: float) -> None:
    with _request_lock:
        record = _request_timings.setdefault(
            request_id,
            {
                "t": "req",
                "rid": request_id,
                "request_id": request_id,
                "ts": time.time(),
                "timing": {},
            },
        )
        record["timing"][key] = record["timing"].get(key, 0.0) + value
        _request_timings.move_to_end(request_id)
        while len(_request_timings) > 100_000:
            _request_timings.popitem(last=False)


def _install_request_timing_hooks() -> None:
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer
    from vllm.v1.engine.output_processor import RequestState

    if getattr(OpenAIServingChat, "_metal_timing_installed", False):
        return

    original_create = OpenAIServingChat._create_chat_completion

    @wraps(original_create)
    async def create_chat_completion(self, request, raw_request=None):
        base_id = self._base_request_id(raw_request, request.request_id)
        request_id = f"chatcmpl-{base_id}"
        token = _request_id.set(request_id)
        try:
            return await original_create(self, request, raw_request)
        finally:
            _request_id.reset(token)

    original_render = OpenAIServingChat.render_chat_request

    @wraps(original_render)
    async def render_chat_request(self, request):
        request_id = _request_id.get()
        start = time.perf_counter_ns()
        try:
            return await original_render(self, request)
        finally:
            if request_id is not None:
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                _record_request(request_id, "tokenization_ms", elapsed_ms)

    original_init = RequestState.__init__

    @wraps(original_init)
    def init_request_state(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.detokenizer is not None:
            self.detokenizer._metal_request_id = self.external_req_id

    original_update = BaseIncrementalDetokenizer.update

    @wraps(original_update)
    def update_detokenizer(self, *args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return original_update(self, *args, **kwargs)
        finally:
            request_id = getattr(self, "_metal_request_id", None)
            if request_id is not None:
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                _record_request(request_id, "detokenize_ms", elapsed_ms)

    original_detokenize = BaseIncrementalDetokenizer.get_next_output_text

    @wraps(original_detokenize)
    def get_next_output_text(self, *args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return original_detokenize(self, *args, **kwargs)
        finally:
            request_id = getattr(self, "_metal_request_id", None)
            if request_id is not None:
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                _record_request(request_id, "detokenize_ms", elapsed_ms)

    OpenAIServingChat._create_chat_completion = create_chat_completion
    OpenAIServingChat.render_chat_request = render_chat_request
    OpenAIServingChat._metal_timing_installed = True
    RequestState.__init__ = init_request_state
    BaseIncrementalDetokenizer.update = update_detokenizer
    BaseIncrementalDetokenizer.get_next_output_text = get_next_output_text


class MetalTimingPlugin:
    name = "metal_timing"
    required_tasks = None

    def __init__(self) -> None:
        if os.environ.get("VLLM_METAL_TRACE_TIMING") == "1":
            _install_request_timing_hooks()
        self._active_requests = 0
        self._lock = threading.Lock()

    def attach_router(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def count_inference_requests(request: Request, call_next):
            tracked = request.url.path.startswith("/v1/")
            if tracked:
                with self._lock:
                    self._active_requests += 1
            try:
                return await call_next(request)
            finally:
                if tracked:
                    with self._lock:
                        self._active_requests -= 1

        router = APIRouter()

        @router.get("/debug/trace")
        async def trace_info(request: Request):
            trace = await self._trace(request)
            return {"enabled": trace["enabled"], "step_samples": trace["enabled"]}

        @router.get("/debug/scheduler_metrics")
        async def scheduler_metrics():
            with self._lock:
                active = self._active_requests
            trace = await self._trace_from_app(app)
            return {
                "current": {"running": active},
                "totals": {"total_steps": trace["total_steps"]},
            }

        @router.get("/debug/trace/steps")
        async def trace_steps(
            request: Request,
            after_step_id: int = -1,
            through_step_id: int | None = None,
            order: str = "asc",
            limit: int = 1000,
            cursor: str | None = None,
        ):
            trace = await self._trace(request, after_step_id, through_step_id)
            steps = trace["steps"]
            if order == "desc":
                steps.reverse()
            try:
                offset = int(cursor) if cursor is not None else 0
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid cursor") from exc
            page_size = max(1, min(limit, 1000))
            page = steps[offset : offset + page_size]
            next_cursor = (
                str(offset + page_size) if offset + page_size < len(steps) else None
            )
            return {
                "records": page,
                "next_cursor": next_cursor,
                "truncated": next_cursor is not None,
            }

        @router.get("/debug/trace/requests")
        async def trace_requests(
            since: float | None = None,
            until: float | None = None,
            order: str = "latest",
            limit: int = 100,
            cursor: str | None = None,
        ):
            with _request_lock:
                records = list(_request_timings.values())
            if since is not None:
                records = [record for record in records if record["ts"] >= since]
            if until is not None:
                records = [record for record in records if record["ts"] <= until]
            if order == "latest":
                records.reverse()
            try:
                offset = int(cursor) if cursor is not None else 0
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid cursor") from exc
            page_size = max(1, min(limit, 1000))
            page = records[offset : offset + page_size]
            next_cursor = (
                str(offset + page_size) if offset + page_size < len(records) else None
            )
            return {
                "requests": page,
                "next_cursor": next_cursor,
                "truncated": next_cursor is not None,
            }

        app.include_router(router)

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        state.metal_timing_engine_client = engine_client

    async def _trace_from_app(
        self,
        app: FastAPI,
        after_step_id: int = -1,
        through_step_id: int | None = None,
    ):
        client = getattr(app.state, "metal_timing_engine_client", None)
        if client is None:
            raise HTTPException(status_code=503, detail="Metal engine is unavailable")
        results = await client.collective_rpc(
            "get_metal_timing_trace",
            args=(after_step_id, through_step_id),
        )
        traces = [result for result in results if isinstance(result, dict)]
        if not traces:
            return {"enabled": False, "total_steps": 0, "steps": []}
        return {
            "enabled": any(trace.get("enabled") for trace in traces),
            "total_steps": max(trace.get("total_steps", 0) for trace in traces),
            "steps": sorted(
                (step for trace in traces for step in trace.get("steps", [])),
                key=lambda step: step["step_id"],
            ),
        }

    async def _trace(
        self,
        request: Request,
        after_step_id: int = -1,
        through_step_id: int | None = None,
    ):
        return await self._trace_from_app(
            request.app, after_step_id, through_step_id
        )


def create_plugin() -> MetalTimingPlugin:
    return MetalTimingPlugin()
