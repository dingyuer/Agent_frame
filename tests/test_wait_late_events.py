"""A fresh screenshot can dispatch events newer than the last cheap poll."""
import copy

from osworld_agent.pipeline import Execution
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import observation
from test_pipeline_recovery_wait import EventEnv, metadata, make_pipeline, state_and_decision, clock


def test_completion_arriving_during_refresh_is_not_overwritten(tmp_path, clock):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"downloaded file")
    url = "https://example.test/paper.pdf"
    artifact = {"download_id": "late", "url": url, "path": str(path), "status": "completed", "completed": True}
    env = EventEnv([metadata()])
    fresh = observation()
    fresh.info.update(metadata(artifacts=[artifact]))
    env.observe = lambda: fresh
    pipe = make_pipeline(env, wait_config={"download_start_timeout_s": .25})
    pipe.completion = CompletionVerifier("task", {"checks": [{"kind": "download", "url": url}]})
    state, decision = state_and_decision()
    result = pipe._settle_execution(decision.action, Execution(kind="browser_dom", obs=state.obs,
        info={"status": "executed", "dispatched": True}), state)
    pipe.completion.observe(result.obs, result.info)
    assert pipe.completion.evaluate()["status"] == "passed"
    assert result.obs.info["artifacts"] == [artifact]


def test_download_started_during_refresh_is_waited_without_new_decision(tmp_path, clock):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"downloaded file")
    artifact = {"download_id": "d1", "url": "https://example.test/paper.pdf", "path": str(path),
                "status": "completed", "completed": True}
    class LateEnv(EventEnv):
        def __init__(self):
            super().__init__([metadata()])
            self.phase = 0

        def browser_event_metadata(self):
            self.event_reads += 1
            if self.phase == 0:
                return metadata()
            self.phase += 1
            return metadata(True) if self.phase == 2 else metadata(artifacts=[artifact])

        def observe(self):
            self.observe_count += 1
            obs = observation()
            if self.phase == 0:
                self.phase = 1
                obs.info.update(metadata(True))
            else:
                obs.info.update(metadata(artifacts=[artifact]))
            return obs

    env = LateEnv()
    pipe = make_pipeline(env, wait_config={"download_start_timeout_s": .25})
    pipe.completion = CompletionVerifier("task")
    state, decision = state_and_decision()
    result = pipe._settle_execution(decision.action, Execution(kind="browser_dom", obs=state.obs,
        info={"status": "executed", "dispatched": True}), state)
    assert result.info["artifacts"] == [artifact]
    assert not result.info["pending_downloads"]
    assert env.observe_count == 2 and not state.waiting_download_ids


def test_download_failure_does_not_erase_original_rejection(clock):
    event = {"download_id": "new", "page_id": "task-page", "status": "failed"}
    env = EventEnv([{**metadata(), "download_events": [event]}])
    pipe = make_pipeline(env)
    state, decision = state_and_decision()
    original = {"status": "rejected", "dispatched": False, "reason": "specific stale target", "error_code": "stale"}
    result = pipe._settle_execution(decision.action,
        Execution(kind="browser_dom", obs=state.obs, info=copy.deepcopy(original)), state)
    assert result.info["reason"] == original["reason"]
    assert result.info["wait_error_code"] == "download_failed"
    assert result.info["wait_reason"] and result.info["dispatched"] is False
