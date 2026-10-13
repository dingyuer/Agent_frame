"""Presentation provenance and real review path tested with a local transport."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from osworld_agent.viz import jev_demo
from osworld_agent.viz.server import app
from osworld_agent.model.decision_judge import DecisionJudge
from test_jev_debug import picture


def payload(scenario='search', mode='live'):
    return {'scenario_id': scenario, 'mode': mode, 'images': [picture(), picture()],
            'config': {'url': 'http://jev.invalid/v1/systemone', 'model': 'vllm-sr/d3-flash', 'api_key': 'demo-secret'}}


def fake_judge(monkeypatch, label='progress', probability=.97, status=200):
    calls = []
    def handler(request):
        data = json.loads(request.content)
        calls.append(data)
        if status != 200:
            return httpx.Response(status, json={'detail': 'fixture unavailable'})
        options = data['questions']['step']['criteria']
        probabilities = {key: probability if key == label else (1-probability)/(len(options)-1) for key in options}
        return httpx.Response(200, json={'model': 'actual-fixture-9b', 'usage': {'input_tokens': 500, 'output_tokens': 0},
            'answers': {'step': {'type': 'choice', 'choice': label, 'probabilities': probabilities, 'confidence': .8}}})
    monkeypatch.setattr(jev_demo, 'DecisionJudge', lambda cfg: DecisionJudge(cfg, httpx.MockTransport(handler)))
    return calls


def test_scenarios_are_explicitly_synthetic_and_not_real_run():
    response = TestClient(app).get('/jev/demo/scenarios')
    assert response.status_code == 200
    assert response.json()['synthetic_observations'] is True
    assert len(response.json()['scenarios']) == 5


def test_replay_never_calls_a_model_or_invents_scores(monkeypatch):
    monkeypatch.setattr(jev_demo, 'DecisionJudge', lambda *_: pytest.fail('no remote requests in replay'))
    client = TestClient(app)
    for scenario in jev_demo.SCENARIOS:
        data = client.post('/jev/demo/evaluate', json={'scenario_id': scenario['id'], 'mode': 'replay'}).json()
        assert data['outcome'] == scenario['replay']
        assert not data['model_called'] and data['model_latency_ms'] is None
        assert data['trainable'] is False and data['synthetic_observations'] is True
        assert 'probabilities' not in json.dumps(data) and 'demo-secret' not in json.dumps(data)


def test_live_uses_real_review_compilation_two_images_one_question(monkeypatch):
    calls = fake_judge(monkeypatch)
    response = TestClient(app).post('/jev/demo/evaluate', json=payload())
    assert response.status_code == 200
    data = response.json()
    assert data['outcome'] == 'progress' and data['report']['status'] == 'unknown'
    assert data['model_called'] and data['model_latency_ms'] >= 0
    assert data['model'] == 'actual-fixture-9b' and data['usage']['output_tokens'] == 0
    assert data['report']['action_effect']['status'] == 'passed'
    assert len(calls) == 1 and len(calls[0]['images']) == 2
    assert list(calls[0]['questions']) == ['step']
    state = json.loads(calls[0]['state'])
    assert 'replay' not in state and 'takeaway' not in state and 'narration' not in state
    assert state['image_order'] == ['before', 'after']
    assert 'demo-secret' not in json.dumps(data)


def test_live_preserves_unexpected_model_output_instead_of_expected_script(monkeypatch):
    fake_judge(monkeypatch, label='passed')
    result = TestClient(app).post('/jev/demo/evaluate', json=payload('wrong_paper')).json()
    assert result['outcome'] == 'passed'  # Not secretly replaced with scripted "failed".
    assert result['source'] == 'live_jev'


def test_low_confidence_stays_unknown_and_http_failure_does_not_replay(monkeypatch):
    fake_judge(monkeypatch, label='passed', probability=.6)
    client = TestClient(app)
    assert client.post('/jev/demo/evaluate', json=payload()).json()['outcome'] == 'unknown'
    fake_judge(monkeypatch, status=503)
    result = client.post('/jev/demo/evaluate', json=payload()).json()
    assert result['outcome'] == 'unknown' and result['source'] == 'live_jev'
    assert result['report']['judgment']['http_status'] == 503


@pytest.mark.parametrize('scenario, expected', [('transferring', 'pending'), ('artifact', 'passed')])
def test_code_scenarios_skip_jev_in_live_mode(monkeypatch, scenario, expected):
    monkeypatch.setattr(jev_demo, 'DecisionJudge', lambda *_: pytest.fail('code evidence must bypass the model'))
    data = TestClient(app).post('/jev/demo/evaluate', json={'scenario_id': scenario, 'mode': 'live'}).json()
    assert data['source'] == 'code_on_fixture' and not data['model_called'] and data['outcome'] == expected


@pytest.mark.parametrize('changes', [dict(scenario_id='missing'), dict(mode='fake'), dict(images=[]),
                                    dict(images=[picture()]), dict(unexpected=1), dict(images=['broken', 'broken'])])
def test_invalid_live_requests_are_rejected(changes):
    assert TestClient(app).post('/jev/demo/evaluate', json=dict(payload(), **changes)).status_code == 422


def test_replay_available_while_live_request_is_in_flight():
    client = TestClient(app)
    jev_demo._live_lock.acquire()
    try:
        assert client.post('/jev/demo/evaluate', json=payload()).status_code == 409
        assert client.post('/jev/demo/evaluate', json={'scenario_id': 'search', 'mode': 'replay'}).status_code == 200
    finally:
        jev_demo._live_lock.release()
