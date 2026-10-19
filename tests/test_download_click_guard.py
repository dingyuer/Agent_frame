"""A failed browser download click cannot be repeated on unchanged evidence."""
import copy

from PIL import Image
import pytest

from osworld_agent.actions import Action
from osworld_agent.download_click_guard import remember_unconfirmed_download_click, repeated_download_click_error
from osworld_agent.env import StepResult
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import AgentResult, Decision, Execution, Pipeline, StepState
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model, observation


def browser_observation():
    obs = observation('https://example.test/paper.pdf')
    obs.info.update(document_id=10, scroll_offset=[0, 0], snapshot_id='first')
    obs.context[0]['nodes'] = [{'ref': '1', 'role': 'button', 'name': 'Download PDF',
                              'states': {'disabled': False}, 'direct': True}]
    return obs


def record(target='Download PDF', *, before=None, after=None, info=None):
    before = before or browser_observation()
    state = StepState(obs=before, memory=StructuredMemory())
    original = Action('click', {'target': target})
    resolved = Action('click', {'x': 50, 'y': 30})
    receipt = {'status': 'uncertain', 'dispatched': True, 'coord': [50, 30],
               'coordinate_space': 'css_viewport', 'framework_wait': {'status': 'no_signal'}}
    receipt.update(info or {})
    remember_unconfirmed_download_click(state, original, resolved,
                                        Execution(info=receipt, obs=after or before))
    return state


@pytest.mark.parametrize('target', ['Download PDF', 'Save file', '点击下载图标', '保存文件', '另存为'])
def test_original_intent_survives_grounding_and_small_coordinate_variation(target):
    state = record(target)
    fresh = copy.deepcopy(state.obs)
    fresh.info.update(snapshot_id='new', captured_at=999)
    fresh.context[0]['snapshot_id'] = 'new'
    fresh.context[0]['nodes'][0].update(ref='97', backend_id=200)
    fresh.context[0]['nodes'][0]['states']['focused'] = True
    state.obs = fresh
    error = repeated_download_click_error(Action('click', {'x': 55, 'y': 35}), state)
    assert error['error_code'] == 'repeated_unconfirmed_download_click'
    assert error['dispatched'] is False and '未执行' in error['reason']


@pytest.mark.parametrize('change', ['page', 'document', 'url', 'screen', 'ax_text', 'ax_value', 'enabled', 'scroll'])
def test_new_observed_content_or_identity_releases_retry(change):
    state = record()
    fresh = copy.deepcopy(state.obs)
    if change == 'page': fresh.info['page_id'] = 'new-tab'
    elif change == 'document': fresh.info['document_id'] = 11
    elif change == 'url': fresh.context[0]['url'] = 'https://example.test/new.pdf'
    elif change == 'screen': fresh.screenshot = Image.new('RGB', (100, 80), 'black')
    elif change == 'ax_text': fresh.context[0]['nodes'][0]['name'] = 'Download ready'
    elif change == 'ax_value': fresh.context[0]['nodes'][0]['value'] = '100'
    elif change == 'enabled': fresh.context[0]['nodes'][0]['states']['disabled'] = True
    elif change == 'scroll': fresh.info['scroll_offset'] = [0, 100]
    state.obs = fresh
    assert repeated_download_click_error(Action('click', {'x': 50, 'y': 30}), state) is None
    assert not state.unconfirmed_download_click


@pytest.mark.parametrize('signal', ['download_events', 'artifacts', 'pending_downloads', 'artifact_events'])
def test_new_download_evidence_releases_guard(signal):
    state = record()
    state.obs.info[signal] = [{'download_id': 'observed'}]
    assert repeated_download_click_error(Action('click', {'x': 50, 'y': 30}), state) is None
    assert not state.unconfirmed_download_click
    assert not record(info={signal: [{'download_id': 'observed'}]}).unconfirmed_download_click


@pytest.mark.parametrize('info', [{'dispatched': False}, {'framework_wait': {'status': 'completed'}},
                                {'framework_wait': {'status': 'timeout'}}])
def test_guard_requires_actual_dispatch_and_explicit_no_signal(info):
    assert not record(info=info).unconfirmed_download_click


def test_navigation_click_and_missing_browser_identity_are_not_guarded():
    assert not record('View PDF').unconfirmed_download_click
    missing = browser_observation()
    missing.info.pop('document_id')
    assert not record(before=missing).unconfirmed_download_click
    stale = browser_observation()
    stale.info['screenshot_stale'] = True
    assert not record(before=stale).unconfirmed_download_click


def test_click_that_opens_a_menu_is_not_treated_as_no_effect():
    before = browser_observation()
    after = copy.deepcopy(before)
    after.context[0]['nodes'].append({'ref': '2', 'role': 'menuitem', 'name': 'Save original'})
    assert not record(before=before, after=after).unconfirmed_download_click


@pytest.mark.parametrize('action', [Action('click', {'x': 59, 'y': 30}), Action('type', {'x': 50, 'y': 30, 'text': 'a'}),
                                  Action('hotkey', {'keys': ['cmd', 's']}), Action('observe_browser', {})])
def test_guard_does_not_block_other_points_or_recovery_primitives(action):
    assert repeated_download_click_error(action, record()) is None


def test_pipeline_records_after_wait_and_refuses_second_real_input(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    class DownloadEnv(Env):
        def step(self, action):
            self.actions.append(action)
            return StepResult(self.obs, info={'status': 'uncertain', 'dispatched': True,
                'coord': [action.args['x'], action.args['y']], 'coordinate_space': 'css_viewport',
                'framework_wait': {'status': 'no_signal'}, 'download_events': [], 'artifacts': []})

    env = DownloadEnv()
    env.obs = browser_observation()
    pipe = Pipeline(model=Model([]), env=env, grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier('Download a PDF')
    state = StepState(obs=env.obs, memory=StructuredMemory())
    original = Action('click', {'target': '下载图标'})
    resolved = Action('click', {'x': 50, 'y': 30})
    state.resolved_action = resolved
    state.screenshot_history = [env.obs.screenshot]
    first = pipe.execute(resolved, state)
    state, _ = pipe.postprocess(1, Decision(action=original), first, state, AgentResult(task='Download a PDF'))
    second = pipe.execute(Action('click', {'x': 53, 'y': 30}), state)
    assert len(env.actions) == 1
    assert second.info['error_code'] == 'repeated_unconfirmed_download_click'
    assert second.info['dispatched'] is False
    # A visible content change restores execution without any automatic click.
    state.obs = copy.deepcopy(state.obs)
    state.obs.screenshot = Image.new('RGB', (100, 80), 'gray')
    pipe.execute(resolved, state)
    assert len(env.actions) == 2
