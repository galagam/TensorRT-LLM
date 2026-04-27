# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for SpecDecAwareScheduler."""

from unittest.mock import MagicMock

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import SpecDecAwareScheduler
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RequestScheduler, SchedulerOutput


def _make_context_req(req_id: int) -> LlmRequest:
    req = LlmRequest(
        request_id=req_id,
        max_new_tokens=5,
        input_tokens=[req_id],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    req.py_draft_tokens = []
    return req


def _make_extend_req(req_id: int, num_draft_tokens: int = 3) -> LlmRequest:
    req = LlmRequest(
        request_id=req_id,
        max_new_tokens=5,
        input_tokens=[req_id],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    req.py_draft_tokens = list(range(num_draft_tokens))
    return req


def _make_scheduler(output: SchedulerOutput) -> RequestScheduler:
    base = MagicMock(spec=RequestScheduler)
    base.schedule_request.return_value = output
    base.can_schedule.return_value = True
    return base


def _ids(reqs):
    return [r.request_id for r in reqs]


class TestSpecDecAwareScheduler:
    def test_pass_through_no_context(self):
        extend = _make_extend_req(1)
        output = SchedulerOutput([], [extend], [], [], 1)
        sched = SpecDecAwareScheduler(_make_scheduler(output))
        result = sched.schedule_request([], set())
        assert _ids(result.context_requests) == []
        assert _ids(result.generation_requests) == [1]

    def test_pass_through_context_no_spec_dec_extends(self):
        ctx = _make_context_req(1)
        gen = _make_context_req(2)
        gen.py_draft_tokens = []
        output = SchedulerOutput([ctx], [gen], [], [], 2)
        sched = SpecDecAwareScheduler(_make_scheduler(output))
        result = sched.schedule_request([], set())
        assert _ids(result.context_requests) == [1]
        assert _ids(result.generation_requests) == [2]

    def test_defers_context_when_spec_dec_extends_present(self):
        ctx = _make_context_req(1)
        extend = _make_extend_req(2, num_draft_tokens=3)
        output = SchedulerOutput([ctx], [extend], [], [], 2)
        sched = SpecDecAwareScheduler(_make_scheduler(output))
        result = sched.schedule_request([], set())
        assert _ids(result.context_requests) == []
        assert _ids(result.generation_requests) == [2]
        assert sched._defer_counts[1] == 1

    def test_starvation_guard_forces_admission(self):
        ctx = _make_context_req(1)
        extend = _make_extend_req(2, num_draft_tokens=3)
        output = SchedulerOutput([ctx], [extend], [], [], 2)
        base = _make_scheduler(output)
        sched = SpecDecAwareScheduler(base, max_defer_steps=2)

        for _ in range(2):
            result = sched.schedule_request([], set())
            assert _ids(result.context_requests) == []

        result = sched.schedule_request([], set())
        assert _ids(result.context_requests) == [1]
        assert 1 not in sched._defer_counts

    def test_urgent_requests_always_pass_through(self):
        urgent_ctx = _make_context_req(1)
        new_ctx = _make_context_req(2)
        extend = _make_extend_req(3, num_draft_tokens=3)
        output = SchedulerOutput([urgent_ctx, new_ctx], [extend], [], [], 3)
        base = _make_scheduler(output)
        sched = SpecDecAwareScheduler(base, max_defer_steps=1)

        sched._defer_counts[1] = 1
        result = sched.schedule_request([], set())
        assert 1 in _ids(result.context_requests)
        assert 2 not in _ids(result.context_requests)
        assert sched._defer_counts.get(2) == 1

    def test_defer_count_cleared_when_no_spec_dec(self):
        ctx = _make_context_req(1)
        extend = _make_extend_req(2, num_draft_tokens=3)
        base = MagicMock(spec=RequestScheduler)
        sched = SpecDecAwareScheduler(base, max_defer_steps=4)
        sched._defer_counts[1] = 2

        output_no_spec_dec = SchedulerOutput([ctx], [extend], [], [], 2)
        extend.py_draft_tokens = []
        base.schedule_request.return_value = output_no_spec_dec
        result = sched.schedule_request([], set())
        assert _ids(result.context_requests) == [1]
        assert 1 not in sched._defer_counts

    def test_can_schedule_delegates_to_base(self):
        base = MagicMock(spec=RequestScheduler)
        base.can_schedule.return_value = True
        sched = SpecDecAwareScheduler(base)
        reqs = [_make_context_req(1)]
        assert sched.can_schedule(reqs) is True
        base.can_schedule.assert_called_once_with(reqs)
