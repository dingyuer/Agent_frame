"""DOM/identity guards, PDF readiness, and compact event feedback regressions."""
import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from osworld_agent.actions import Action
from osworld_agent.browser_feedback import active_paper, operational_facts, paper_action_error, same_paper
from osworld_agent.browser_session import BrowserOptions, BrowserSession
from osworld_agent.env import Observation
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import Execution, Pipeline, StepState, WaitConfig
from osworld_agent.verifiers import CompletionVerifier
from osworld_agent.visual_readiness import content_thumbnail, is_pdf_observation
from test_pipeline_completion import Env, Model, observation
from test_pipeline_recovery_wait import clock  # deterministic wall clock fixture
from test_paper_resolver import resolve, TITLE, IDENTIFIER


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)


def bound():
    verifier = CompletionVerifier('搜索论文roboclaw并下载，然后改名为data')
    verifier.bind_paper('check_1', resolve().binding)
    return verifier


def pipe(env=None, **kw):
    return Pipeline(model=Model([]), env=env or Env(), grounding=None, system_prompt='test', **kw)


def pdf_obs(color='black', content=False):
    obs = observation('https://arxiv.org/pdf/' + IDENTIFIER, color=color)
    obs.info.update(url=obs.context[0]['url'], document_id='pdf-doc', document_kind='pdf')
    obs.context[0]['nodes'] = []
    if content:
        draw = ImageDraw.Draw(obs.screenshot)
        for y in range(25, 65, 6):
            draw.line((20, y, 80, y), fill='black')
    return obs


def test_compact_prompt_preserves_binding_and_completed_download():
    p = pipe()
    p.completion = bound()
    obs = observation('https://arxiv.org/pdf/2610.03476')
    obs.info['artifacts'] = [{'completed': True, 'path': '/downloads/2610.03476v1.pdf',
                              'bytes': 12345, 'url': 'https://arxiv.org/pdf/2610.03476v1'}]
    state = StepState(task=p.completion.task if hasattr(p.completion, 'task') else '下载论文', obs=obs,
                      memory=StructuredMemory())
    p.build_input(state)
    text = state.prompt_text
    assert TITLE in text and IDENTIFIER in text and 'data.pdf' in text
    assert 'https://arxiv.org/pdf/' + IDENTIFIER in text
    assert '下载传输已完成' in text and '12345' in text and '来源属于其他论文' in text
    assert '任务计划与系统验收条件' not in text
    assert p.completion.evaluate()['status'] != 'passed'


def test_wrong_link_rejected_before_dispatch_and_sample_kept():
    p = pipe()
    p.completion = bound()
    obs = observation()
    obs.context[0]['nodes'] = [{'ref': 'bad', 'role': 'link', 'direct': True,
                              'href': 'https://arxiv.org/pdf/2610.03476'}]
    action = Action('click', {'target': 'PDF link', 'target_ref': 'bad'})
    state = StepState(obs=obs)
    resolved = p.ground(action, state)
    result = p.execute(resolved, state)
    assert result.info['dispatched'] is False
    assert state.route['error_code'] == 'paper_identity_mismatch'
    assert not p.env.actions and action.args == {'target': 'PDF link', 'target_ref': 'bad'}
    good = Action('goto', {'url': 'https://arxiv.org/pdf/' + IDENTIFIER})
    assert not paper_action_error(good, obs, active_paper(p.completion))


def test_wrong_viewer_save_guard_and_versions():
    p = bound()
    assert paper_action_error(Action('hotkey', {'keys': ['ctrl', 's']}),
                              observation('https://arxiv.org/pdf/2610.03476'), active_paper(p))
    assert same_paper('2603.11558v1', '2603.11558')
    assert not same_paper('2603.11558v2', '2603.11558v1')
    multiple = SimpleNamespace(paper_goals=lambda: [p.paper_goals()[0], dict(p.paper_goals()[0], check_id='other')])
    assert active_paper(multiple) is None
    assert active_paper(multiple, {'check_ids': ['other']})['check_id'] == 'other'


def test_download_event_does_not_override_failed_verification():
    obs = observation()
    obs.info['artifacts'] = [{'completed': True, 'path': '/downloads/original.pdf'}]
    report = {'checks': [{'kind': 'paper_download', 'status': 'failed',
                         'reason': 'downloaded filename does not match'}]}
    text = '\n'.join(operational_facts(obs, bound(), report))
    assert '文件名仍不符合要求' in text and '不等于任务验收通过' in text


def test_missing_dom_ref_gets_one_repair_without_dispatch():
    env = Env()
    env.route_action = lambda a, o: None
    p = pipe(env, max_retries=3)
    obs = observation()
    obs.context[0]['nodes'][0]['direct'] = True
    first = {'next_action': {'type': 'type', 'target': 'Search input', 'text': 'roboclaw'}}
    second = {'next_action': dict(first['next_action'], target_ref='1')}
    p.model.responses = [json.dumps(first), json.dumps(second)]
    result = p.decide([{'role': 'user', 'content': 'AX ref 1'}], StepState(obs=obs))
    assert result.action.args['target_ref'] == '1'
    assert len(result.calls) == 2 and result.calls[0]['validation_error_type'] == 'missing_dom_reference'
    assert not env.actions and result.raw == json.dumps(second)


@pytest.mark.parametrize('surface', ['pdf', 'desktop', 'no_exposed_nodes'])
def test_dom_repair_does_not_intercept_visual_only_surfaces(surface):
    p = pipe()
    p.env.route_action = lambda a, o: None
    obs = pdf_obs() if surface == 'pdf' else observation()
    if surface == 'desktop':
        obs.info['active_surface'] = 'desktop'
    elif surface == 'no_exposed_nodes':
        obs.context[0]['nodes'][0]['direct'] = True
        obs.info['exposed_refs'] = []
    assert p._dom_reference_feedback(Action('click', {'target': 'Download toolbar'}), StepState(obs=obs)) is None


def test_exact_name_and_role_resolve_dom_but_duplicate_pdf_does_not():
    s = BrowserSession(options=BrowserOptions())
    s.snapshot = {'snapshot_id': 's'}
    s.nodes = {str(i): dict(ref=str(i), role=role, name=name, direct=True) for i, (role, name) in enumerate([
        ('textbox', 'Search'), ('button', 'Search'), ('link', 'PDF'), ('link', 'PDF')])}
    assert s.resolve_target('Search button') == '1'
    assert s.resolve_target('Search input') == '0'
    assert s.resolve_target('PDF link') is None and s.last_resolution_error == 'ambiguous'
    assert s.resolve_target('Search button', allowed=set()) is None


@pytest.mark.parametrize('color', ['black', 'white', '#333333'])
def test_uniform_viewer_is_not_ready(color):
    assert content_thumbnail(pdf_obs(color).screenshot) is None
    assert content_thumbnail(pdf_obs('white', True).screenshot) is not None


def test_render_wait_polls_black_frames_without_spending_decisions(clock):
    p = pipe(wait_config={'render_timeout_s': 2})
    first = pdf_obs()
    ready = pdf_obs('white', True)
    frames = iter([pdf_obs(), pdf_obs(), ready, ready])
    p.env.observe = lambda: next(frames)
    state = StepState(obs=first, step_id=4)
    result = p._settle_rendering(Execution(kind='env', obs=first), state)
    assert result.info['render_wait']['status'] == 'completed'
    assert result.info['render_wait']['elapsed_s'] == 1
    assert state.step_id == 4 and not p.model.calls and not p.env.actions
    assert 'render_wait' not in first.info
    assert content_thumbnail(result.obs.screenshot) is not None
    # No repeated wait for this now-ready document.
    assert p._settle_rendering(result, state).obs.info['render_wait']['status'] == 'completed'


def test_stable_black_times_out_once_and_is_visible_to_policy(clock):
    p = pipe(wait_config={'render_timeout_s': .5})
    p.env.obs = pdf_obs()
    state = StepState(obs=p.env.obs)
    result = p._settle_rendering(Execution(obs=p.env.obs), state)
    assert result.info['render_wait']['status'] == 'timeout'
    count = p.env.observe_count
    result = p._settle_rendering(result, state)
    assert p.env.observe_count == count
    assert 'PDF 渲染等待已超时' in '\n'.join(operational_facts(result.obs, None, {}))


def test_render_wait_honors_task_deadline_and_native_dialog(clock):
    p = pipe(wait_config={'render_timeout_s': 10})
    p.env.obs = pdf_obs()
    p._task_deadline = clock.now + .25
    result = p._settle_rendering(Execution(obs=p.env.obs), StepState())
    assert result.info['render_wait']['elapsed_s'] == .25
    p.env.obs.info['dialog'] = {'type': 'alert'}
    count = p.env.observe_count
    p._settle_rendering(Execution(obs=p.env.obs), StepState())
    assert p.env.observe_count == count


def test_navigation_during_render_wait_yields_new_page(clock):
    p = pipe()
    result = p._settle_rendering(Execution(obs=pdf_obs()), StepState())
    assert result.obs.context[0]['url'] == 'https://example.test/'
    assert result.info['render_wait']['status'] == 'interrupted'


def test_invalid_render_timeout_rejected():
    for value in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            WaitConfig(render_timeout_s=value)


@pytest.mark.parametrize('adapter', ['windows', 'vmware'])
def test_adapters_reject_ambiguous_dom_instead_of_grounding(adapter):
    from osworld_agent.adapters.windows_env import WindowsEnvironment
    from osworld_agent.adapters.vmware_env import VMwareEnvironment
    cls = WindowsEnvironment if adapter == 'windows' else VMwareEnvironment
    env = object.__new__(cls)
    session = BrowserSession(options=BrowserOptions())
    session.snapshot = {'snapshot_id': 'initial'}
    session.nodes = {str(i): {'ref': str(i), 'role': 'link', 'name': 'PDF', 'direct': True} for i in (1, 2)}
    env._get_browser_session = lambda: session
    obs = observation()
    obs.info['scene'] = {'browser_use': 1, 'structure_available': True}
    route = env.route_action(Action('click', {'target': 'PDF link'}), obs)
    assert route['channel'] == 'reobserve' and route['error_code'] == 'ambiguous'
    assert env.route_action(Action('click', {'target': 'PDF link', 'target_ref': '2'}), obs)['target_ref'] == '2'


def test_timeout_document_can_become_ready_without_stale_timeout_feedback(clock):
    p = pipe(wait_config={'render_timeout_s': .5})
    p.env.obs = pdf_obs()
    state = StepState()
    p._settle_rendering(Execution(obs=p.env.obs), state)
    p.env.obs = pdf_obs('white', True)
    result = p._settle_rendering(Execution(obs=p.env.obs), state)
    assert result.obs.info['render_wait']['status'] == 'completed'


def test_private_verifier_details_never_enter_operational_brief():
    p = pipe()
    p.completion = CompletionVerifier('task', private_spec={'checks': [{'kind': 'url_equals', 'url': 'https://private.invalid/secret'}]})
    state = StepState(task='task', obs=observation(), memory=StructuredMemory())
    p.completion.observe(state.obs)
    state.verification = p.completion.evaluate()
    p.build_input(state)
    assert 'private.invalid' not in state.prompt_text


@pytest.mark.parametrize('phase', ['identify', 'download'])
def test_code_identity_conflict_blocks_queue_before_visual_review(phase):
    from osworld_agent.execution_queue import ExecutionQueue
    p = pipe()
    p.completion = bound()
    state = StepState(task='下载roboclaw', obs=observation('https://arxiv.org/pdf/2610.03476'),
                      memory=StructuredMemory(), queue=ExecutionQueue({'steps': [
                          {'idx': 1, 'goal': '核对目标论文', 'phase': phase, 'verification': [{'kind': 'visual'}]}]}))
    state.verification = p.completion.evaluate()
    p._review_visual = lambda *a, **kw: pytest.fail('Known identity mismatch must not be delegated to a model')
    p._update_plan_progress(state, Execution(kind='browser_dom', info={'status': 'executed'}))
    assert state.step_review['status'] == 'failed' and state.step_review['method'] == 'code'
    assert '2603.11558' in state.step_review['reason'] and state.queue.current is not None
