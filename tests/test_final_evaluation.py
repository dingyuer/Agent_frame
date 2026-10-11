"""Final page predicates must use observations made after external evaluation."""
from types import SimpleNamespace

import pytest

from osworld_agent.env import Observation
from osworld_agent.pipeline import AgentResult, Pipeline
from osworld_agent.verifiers import CompletionVerifier


TARGET = "https://example.test/target"
OTHER = "https://example.test/other"


def page(url=TARGET, *nodes):
    return Observation(context=[{"kind": "browser_ax", "url": url, "nodes": list(nodes)}])


class FinalEnv:
    def __init__(self, *, evaluator=False, next_page=None, observation_error=False):
        self.evaluation_available = evaluator
        self.obs = page()
        self.next_page = next_page
        self.observation_error = observation_error
        self.observe_calls = 0
        self.evaluate_calls = 0

    def observe(self):
        self.observe_calls += 1
        if self.observation_error:
            raise RuntimeError("browser disconnected")
        return self.obs

    def evaluate(self):
        self.evaluate_calls += 1
        if self.next_page is not None:
            self.obs = self.next_page
        return 1.0


def pipeline(env, spec):
    instance = Pipeline(model=SimpleNamespace(), grounding=None, env=env, system_prompt="test")
    instance.completion = CompletionVerifier("task", spec)
    instance.completion.observe(page())
    return instance


def prior_success():
    return AgentResult(task="task", success=True, termination_reason="verified")


@pytest.mark.parametrize("evaluator", [False, True])
def test_artifact_only_final_evaluation_does_not_observe(tmp_path, evaluator):
    env = FinalEnv(evaluator=evaluator, observation_error=True)
    instance = pipeline(env, {"checks": [{"kind": "download", "url": "https://example.test/a.bin"}]})
    path = tmp_path / "a.bin"
    path.write_bytes(b"completed artifact")
    instance.completion.observe(None, {"action": "download", "status": "executed",
                                       "url": "https://example.test/a.bin", "path": str(path)})
    result = prior_success()
    assert instance._evaluate(result) == 1.0
    assert result.success and result.evaluation_available
    assert env.observe_calls == 0
    assert env.evaluate_calls == int(evaluator)
    assert instance.events[-1]["event_type"] == "verification"
    assert instance.events[-1]["stage"] == "final_evaluation"


def test_environment_evaluation_page_change_revokes_cached_url_and_archives_evidence():
    env = FinalEnv(evaluator=True, next_page=page(OTHER))
    instance = pipeline(env, {"checks": [{"kind": "url_equals", "url": TARGET}]})
    result = prior_success()
    assert instance._evaluate(result) == 0.0
    assert not result.success and result.evaluation_available
    assert result.verification["status"] == "failed"
    assert result.verification["completion"]["checks"][0]["actual_url"] == OTHER
    assert env.evaluate_calls == 1 and env.observe_calls == 1
    observation_event, verification_event = instance.events
    assert observation_event["event_type"] == "observation"
    assert observation_event["stage"] == "final_evaluation"
    assert observation_event["observation"]["context"][0]["url"] == OTHER
    assert verification_event["report"]["status"] == "failed"


def test_contract_only_page_change_revokes_cached_success():
    env = FinalEnv()
    instance = pipeline(env, {"checks": [{"kind": "url_equals", "url": TARGET}]})
    env.obs = page(OTHER)
    result = prior_success()
    assert instance._evaluate(result) == 0.0
    assert not result.success and result.termination_reason == "verification_changed"
    assert result.verification["status"] == "failed"
    assert env.observe_calls == 1 and env.evaluate_calls == 0


@pytest.mark.parametrize("evaluator", [False, True])
def test_final_page_observation_failure_masks_reward(evaluator):
    env = FinalEnv(evaluator=evaluator, observation_error=True)
    instance = pipeline(env, {"checks": [{"kind": "url_equals", "url": TARGET}]})
    result = prior_success()
    assert instance._evaluate(result) == 0.0
    assert not result.success and not result.evaluation_available
    assert result.reward_source == "unavailable" and result.verification["status"] == "unknown"
    assert env.observe_calls == 1 and env.evaluate_calls == int(evaluator)
    assert instance.events[0]["observation"]["info"]["observation_error"] == "RuntimeError"
    assert instance.events[-1]["report"]["status"] == "unknown"


def test_page_milestone_alone_remains_latched_without_observation():
    env = FinalEnv(evaluator=True, next_page=page(OTHER), observation_error=True)
    instance = pipeline(env, {"checks": [{"kind": "url_equals", "url": TARGET, "scope": "milestone"}]})
    result = prior_success()
    assert instance._evaluate(result) == 1.0
    assert result.success and result.evaluation_available
    assert env.observe_calls == 0 and env.evaluate_calls == 1
    assert result.verification["completion"]["checks"][0]["source"] == "milestone"


def test_final_refresh_preserves_prior_page_milestone():
    env = FinalEnv(evaluator=True, next_page=page(OTHER))
    instance = pipeline(env, {"checks": [
        {"id": "visited", "kind": "url_equals", "url": TARGET, "scope": "milestone"},
        {"id": "final", "kind": "url_equals", "url": OTHER},
    ]})
    result = prior_success()
    assert instance._evaluate(result) == 1.0
    assert result.success and env.observe_calls == 1 and env.evaluate_calls == 1
    checks = result.verification["completion"]["checks"]
    assert checks[0]["source"] == "milestone" and checks[0]["actual_url"] == TARGET
    assert checks[1]["actual_url"] == OTHER


def test_private_final_page_check_is_also_refreshed():
    env = FinalEnv(evaluator=True, next_page=page(OTHER))
    instance = pipeline(env, {"checks": []})
    instance.completion = CompletionVerifier("task", {"checks": []},
        {"checks": [{"kind": "url_equals", "url": TARGET}]})
    instance.completion.observe(page())
    result = prior_success()
    assert instance._evaluate(result) == 0.0
    assert not result.success and result.verification["status"] == "failed"
    assert env.observe_calls == 1 and env.evaluate_calls == 1
