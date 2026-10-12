"""Adapter artifact lists are cumulative, not proof of action causality."""
import copy

import pytest

from osworld_agent.pipeline import Execution
from test_pipeline_recovery_wait import (
    EventEnv, clock, make_pipeline, metadata, no_viz, state_and_decision,
)


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_unrelated_cumulative_executor_artifact_does_not_end_start_grace(clock, status):
    other = {"download_id": "other", "page_id": "other-page", "page_url": "https://other.test/",
             "status": status, "completed": status == "completed", "url": "https://other.test/file.pdf"}
    key = "artifacts" if status == "completed" else "download_events"
    events = metadata()
    events[key] = [other]
    env = EventEnv([events])
    pipe = make_pipeline(env, wait_config={"download_start_timeout_s": 0.5})
    state, decision = state_and_decision()
    execution = Execution(kind="browser_dom", obs=state.obs,
        info={"status": "executed", "dispatched": True, key: [copy.deepcopy(other)]})
    result = pipe._settle_execution(decision.action, execution, state)
    assert sum(clock.sleeps) == 0.5
    assert result.info["framework_wait"]["status"] == "no_signal"
    assert result.info["status"] == "executed" and not env.canceled
    assert result.info[key] == [other]  # Still observable, merely unrelated to this operation.
