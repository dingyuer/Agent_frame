"""Slow PDF and blank HTML loading stays in bounded framework observation."""
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from osworld_agent.actions import Action
from osworld_agent.env import Observation
from osworld_agent.pipeline import Execution, StepState
from osworld_agent.visual_readiness import content_thumbnail
from test_browser_operational_feedback import pdf_obs, pipe


@pytest.fixture
def loading_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        assert 0 < seconds <= 1.0
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr('osworld_agent.settlement.time.monotonic', lambda: clock.now)
    monkeypatch.setattr('osworld_agent.settlement.time.sleep', sleep)
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    return clock


def judge(clock, calls, label=None, answered=True):
    def readiness(image, **kwargs):
        calls.append((clock.now, kwargs['remaining_s']))
        verdict = label or ('ready' if content_thumbnail(image) is not None else 'waiting')
        return {'status': 'answered' if answered else 'unknown', 'model': 'fixture', 'latency_ms': 10,
                'answers': {'readiness': {'label': verdict, 'confidence': .99}} if answered else {}}
    return SimpleNamespace(readiness=readiness, last_call={})


def test_slow_pdf_exceeds_short_render_slice_without_policy_wait_or_document_switch_interrupt(loading_clock):
    p = pipe(wait_config={'render_timeout_s': .5, 'operation_timeout_s': 10})
    initial, ready = pdf_obs(), pdf_obs('white', True)
    calls = []
    p.decision_judge = judge(loading_clock, calls)

    def observe():
        fresh = pdf_obs() if loading_clock.now < 4 else ready
        if loading_clock.now >= 2:
            fresh.info['document_id'] = 'new-pdf-plugin-document'
        return fresh

    p._observe = observe
    state = StepState(obs=initial, step_id=4)
    execution = p._settle_rendering(Execution(kind='browser_dom', obs=initial), state)
    assert execution.info['render_wait']['status'] == 'completed'
    assert execution.info['render_wait']['elapsed_s'] == 5
    assert execution.info['render_wait']['jev_polls'] == 5
    assert [when for when, _ in calls] == [0, 1, 2, 3, 4]
    assert all(timeout <= 2.5 for _, timeout in calls)
    assert state.step_id == 4 and not p.model.calls and not p.env.actions
    assert execution.obs.info['document_id'] == 'new-pdf-plugin-document'


def test_jev_ready_label_and_pdf_url_cannot_make_a_blank_page_ready(loading_clock):
    p = pipe(wait_config={'render_timeout_s': .5, 'operation_timeout_s': 3})
    blank = pdf_obs()
    p._observe = lambda: blank
    calls = []
    p.decision_judge = judge(loading_clock, calls, label='ready')
    state = StepState(obs=blank)
    result = p._settle_rendering(Execution(obs=blank), state)
    assert result.info['render_wait']['status'] == 'timeout'
    assert result.info['render_wait']['elapsed_s'] == 3
    assert result.obs.info['render_wait']['suppress_policy_wait'] is True
    assert len(calls) == 3
    p._observe = lambda: pytest.fail('Expired render budget must not restart on the next policy turn')
    cached = p._settle_rendering(result, state)
    assert cached.obs.info['render_wait']['suppress_policy_wait'] is True
    assert len(calls) == 3


def test_temporarily_unavailable_jev_does_not_restart_timeouts_or_delegate_loading_to_policy(loading_clock):
    p = pipe(wait_config={'render_timeout_s': .5, 'operation_timeout_s': 5})
    blank, ready = pdf_obs(), pdf_obs('white', True)
    p._observe = lambda: blank if loading_clock.now < 1.25 else ready
    calls = []
    p.decision_judge = judge(loading_clock, calls, answered=False)
    result = p._settle_rendering(Execution(obs=blank), StepState(obs=blank))
    assert result.info['render_wait']['status'] == 'completed'
    assert result.info['render_wait']['elapsed_s'] == 1.5
    assert len(calls) == 1
    assert not p.model.calls and not p.env.actions


def test_pdf_readiness_respects_remaining_task_budget(loading_clock):
    p = pipe(wait_config={'operation_timeout_s': 10})
    p._task_deadline = .5
    blank = pdf_obs()
    p._observe = lambda: blank
    calls = []
    p.decision_judge = judge(loading_clock, calls)
    result = p._settle_rendering(Execution(obs=blank), StepState(obs=blank))
    assert result.info['render_wait']['elapsed_s'] == .5
    assert result.info['render_wait']['status'] == 'timeout'
    assert calls == [(0, .5)]


def test_blank_white_html_also_gets_framework_readiness_observation(loading_clock):
    url = 'https://example.test/new-page'
    blank = Observation(screenshot=Image.new('RGB', (600, 400), 'white'),
        context=[{'kind': 'browser_ax', 'url': url, 'nodes': [{'role': 'RootWebArea', 'name': 'New page'}]}],
        info={'url': url, 'active_surface': 'browser'})
    image = Image.new('RGB', (600, 400), 'white')
    ImageDraw.Draw(image).rectangle((120, 100, 300, 220), fill='blue')
    ready = Observation(screenshot=image, context=[{'kind': 'browser_ax', 'url': url,
        'nodes': [{'role': 'heading', 'name': 'Loaded content'}]}], info={'url': url, 'active_surface': 'browser'})
    p = pipe()
    p._observe = lambda: ready
    calls = []
    p.decision_judge = judge(loading_clock, calls)
    result = p._settle_page_readiness(Action('goto', {'url': url}), Execution(obs=blank))
    assert result.obs.screenshot is ready.screenshot and result.obs.context == ready.context
    assert result.info['page_readiness_wait']['status'] == 'ready_for_policy'
    assert len(calls) == 1 and not p.model.calls and not p.env.actions


def loading_html():
    return Observation(screenshot=Image.new('RGB', (600, 400), 'white'),
        context=[{'kind': 'browser_ax', 'url': 'https://example.test/new-page',
                  'nodes': [{'role': 'RootWebArea', 'name': 'New page'}]}],
        info={'url': 'https://example.test/new-page', 'active_surface': 'browser'})


def test_html_loading_timeout_is_visible_to_next_policy_and_does_not_restart(loading_clock):
    p = pipe(wait_config={'operation_timeout_s': 3})
    blank = loading_html()
    p._observe = lambda: blank
    calls = []
    # A model's ready answer alone cannot release a blank/unusable page.
    p.decision_judge = judge(loading_clock, calls, label='ready')
    result = p._settle_page_readiness(Action('goto', {'url': blank.info['url']}), Execution(obs=blank))
    wait = result.info['page_readiness_wait']
    assert wait['status'] == 'timeout' and wait['suppress_policy_wait'] is True
    assert wait['elapsed_s'] == 3 and wait['polls'] == 3
    assert result.obs.info['page_readiness_wait'] == wait
    assert 'page_readiness_wait' not in blank.info
    p._observe = lambda: pytest.fail('The same timed-out observation cannot restart a loading wait')
    again = p._settle_page_readiness(Action('click', {'target': 'loading page'}), result)
    assert again.info['page_readiness_wait'] == wait and len(calls) == 3
    assert not p.model.calls and not p.env.actions


def test_html_readiness_polls_once_per_second_including_jev_request_time(loading_clock):
    p = pipe(wait_config={'operation_timeout_s': 3})
    blank = loading_html()
    p._observe = lambda: blank
    calls = []

    def delayed_request(image, **kwargs):
        calls.append(loading_clock.now)
        loading_clock.now += .2
        return {'status': 'answered', 'answers': {'readiness': {'label': 'waiting'}}}

    p.decision_judge = SimpleNamespace(readiness=delayed_request, last_call={})
    result = p._settle_page_readiness(Action('goto', {'url': blank.info['url']}), Execution(obs=blank))
    assert calls == [0, 1, 2]
    assert result.obs.info['page_readiness_wait']['elapsed_s'] == 3
    assert not p.model.calls and not p.env.actions


def test_html_readiness_stops_retrying_unavailable_jev_and_respects_task_budget(loading_clock):
    p = pipe(wait_config={'operation_timeout_s': 10})
    p._task_deadline = 2.5
    blank = loading_html()
    p._observe = lambda: blank
    calls = []
    p.decision_judge = judge(loading_clock, calls, answered=False)
    result = p._settle_page_readiness(Action('goto', {'url': blank.info['url']}), Execution(obs=blank))
    wait = result.obs.info['page_readiness_wait']
    assert calls == [(0, 2.5)]
    assert wait['status'] == 'timeout' and wait['elapsed_s'] == 2.5
    assert wait['suppress_policy_wait'] is True
    assert not p.model.calls and not p.env.actions
