"""Avoid redundant work without weakening evidence or sampled-output provenance."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from osworld_agent.actions import action_from_decision
from osworld_agent.config import ModelConfig
from osworld_agent.model.decision_model import ChatModel
from osworld_agent.model.decision_judge import DecisionJudge
from osworld_agent.pipeline import Pipeline
from test_pipeline_completion import Model, Env, observation, pipeline
from test_decision_judge import CONFIG, transport


def test_explicit_finish_string_needs_one_call_and_keeps_raw_output(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    raw = json.dumps({'next_action': 'request_finish', 'summary': '请求核查下载文件'})
    model = Model([])
    model.responses = [raw]
    pipe = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test')
    decision = pipe.decide([{'role': 'user', 'content': 'task'}])
    assert len(model.calls) == 1 and len(decision.calls) == 1
    assert decision.action.action == 'request_finish'
    assert decision.action.args == {'answer': '请求核查下载文件'}
    assert decision.raw == decision.calls[0]['response'] == raw
    assert decision.calls[0]['parser_normalization'] == 'finish_string_with_existing_explanation'
    assert decision.calls[0]['parsed_action']['type'] == 'request_finish'


@pytest.mark.parametrize('obj', [
    {'next_action': 'click', 'summary': '点击'},
    {'next_action': 'hotkey', 'keys': ['ctrl', 's']},
    {'next_action': 'request_finish'},
    {'next_action': 'request_finish', 'summary': ''},
    {'next_action': 'request_finish', 'summary': '已完成', 'url': 'https://example.test/'},
    {'next_action': 'request_finish', 'summary': '已完成', 'answer': []},
])
def test_ambiguous_shorthand_still_rejected(obj):
    assert action_from_decision(obj) is None


def test_normalized_finish_cannot_pass_failed_contract(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    model = Model([])
    model.responses.append(json.dumps({'next_action': 'request_finish', 'summary': '声称完成'}))
    pipe = pipeline(model, Env(), {'checks': [{'kind': 'url_equals', 'url': 'https://missing.test/'}]}, steps=1)
    result = pipe.run('打开 https://missing.test/')
    assert not result.success
    assert result.trajectory[0]['valid_action']
    assert len(model.calls) == 2  # one planner, one decision; no format repair


def test_normalized_finish_is_not_a_positive_sft_format_example():
    from osworld_agent.trajectory import iter_sft_records
    trajectory = {'task': {'id': 'test'}, 'steps': [{
        'step': 1, 'trainable': True, 'action': {'action': 'request_finish'},
        'verification': {'status': 'passed'}, 'policy_input': [],
        'policy_output': '{"next_action":"request_finish","summary":"done"}',
        'decision_calls': [{'parser_normalization': 'finish_string_with_existing_explanation'}]}]}
    assert not list(iter_sft_records(trajectory, require_success=False))


class HTTPError(Exception):
    def __init__(self, status, text):
        super().__init__(text)
        self.status_code = status


def failing_model(error, monkeypatch):
    calls, waits = [], []
    def create(**request):
        calls.append(request)
        if len(calls) == 1:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{}'), finish_reason='stop')])
    model = ChatModel(ModelConfig())
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr('osworld_agent.model.decision_model.time.sleep', waits.append)
    return model, calls, waits


@pytest.mark.parametrize('status', [400, 401, 403, 404, 422])
def test_permanent_http_error_has_no_identical_retry_or_backoff(status, monkeypatch):
    model, calls, waits = failing_model(HTTPError(status, 'invalid request'), monkeypatch)
    with pytest.raises(RuntimeError):
        model.chat([{'role': 'user', 'content': 'task'}])
    assert len(calls) == 1 and not waits
    assert model.last_call['attempts'][0]['retry_policy'] == 'permanent_http_error'


def test_unsupported_option_retries_immediately_but_rate_limit_backs_off(monkeypatch):
    model, calls, waits = failing_model(HTTPError(400, 'response_format unsupported'), monkeypatch)
    assert model.chat([], json_mode=True) == '{}'
    assert 'response_format' in calls[0] and 'response_format' not in calls[1]
    assert not waits
    model, calls, waits = failing_model(HTTPError(429, 'rate limited'), monkeypatch)
    assert model.chat([]) == '{}'
    assert len(calls) == 2 and waits == [1.5]


@pytest.mark.parametrize('whole_task', [False, True])
def test_review_encodes_each_frame_once_and_hashes_exact_sent_images(monkeypatch, whole_task):
    import osworld_agent.model.decision_judge as module
    original = module.encode_image
    encoded = []
    def encode(image):
        if isinstance(image, Image.Image):
            encoded.append(image)
        return original(image)
    monkeypatch.setattr(module, 'encode_image', encode)
    requests = []
    judge = DecisionJudge(CONFIG, transport(requests))
    before, after = observation(color='red'), observation(color='blue')
    step = {'id': 'whole_task' if whole_task else 's1', 'goal': 'test', 'expected_evidence': 'result'}
    try:
        result = judge.review('task', step, before, after, {}, {})
        assert len(encoded) == (1 if whole_task else 2)
        assert result['evidence'] == ['视觉判断引用截图 SHA256: ' + hashlib.sha256(image.encode()).hexdigest()
                                      for image in requests[0]['images']]
        first_image = requests[0]['images'][-1]
        after.screenshot.putpixel((0, 0), (0, 255, 0))
        judge.review('task', step, before, after, {}, {})
        assert requests[1]['images'][-1] != first_image  # no stale cross-request cache
        assert result['latency_ms'] >= result['model_request_ms']
        assert set(result['judgment']['timing_ms']) == {'prepare', 'request', 'decode'}
    finally:
        judge.close()


def test_official_candidates_fetch_concurrently_but_selection_waits_for_all():
    import threading
    from osworld_agent.paper_resolver import resolve_arxiv_candidates
    from test_paper_resolver import metadata_html
    identifiers = ['2603.00001', '2603.00002', '2603.00003', '2603.00004']
    barrier = threading.Barrier(4)
    seen = []
    def fetch(url, timeout):
        barrier.wait(timeout=3)  # Serial code cannot complete this barrier.
        seen.append(url)
        return {'url': url, 'text': metadata_html(identifier=url.rsplit('/', 1)[-1])}
    def select(query, candidates, **kw):
        assert len(seen) == 4
        assert [c.canonical_id for c in candidates] == identifiers
        return {'status': 'uncertain', 'reason': 'No unique match', 'evidence': []}
    result = resolve_arxiv_candidates('test', ['https://arxiv.org/abs/' + i for i in identifiers],
                                      fetcher=fetch, identity_selector=select)
    assert result.reason_code == 'semantic_uncertain'
