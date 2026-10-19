"""The actual Jev choice drives prompts and targeting, with fresh goal bindings."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from osworld_agent.actions import Action
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.browser_feedback import operational_facts
from osworld_agent.decision_modes import decision_mode
from osworld_agent.execution_queue import ExecutionQueue
from osworld_agent.model.judge_selection import RuntimeDecisionJudge
from osworld_agent.pipeline import Pipeline, StepState
from test_mode_prompts import browser_observation, desktop_observation
from test_pipeline_completion import Env, Model


class Judge:
    last_call = {}
    def __init__(self, *labels):
        self.labels = list(labels)
        self.calls = []
    def select_mode(self, task, step, observation, **kwargs):
        self.calls.append({'step': copy.deepcopy(step), 'snapshot': observation.info.get('snapshot_id')})
        label = self.labels.pop(0)
        return {'status': 'answered', 'answers': {'next_mode': {'label': label}}, 'latency_ms': .2}


def setup(*labels, scene='browser_dom'):
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe.decision_judge = Judge(*labels)
    state = StepState(task='打开指定方案', obs=browser_observation(mode=scene), step_id=2)
    return pipe, state


@pytest.mark.parametrize('label,scene,expected', [
    ('gui','browser_dom','gui'), ('dom','browser_gui','dom')])
def test_jev_overrides_density_both_ways_and_drives_real_prompt_menu(label,scene,expected):
    pipe,state = setup(label,scene=scene)
    messages = pipe.build_input(state)
    profile = state.prompt_metadata['prompt_profile']
    assert (profile['mode'],profile['source']) == (expected,'jev')
    assert profile['judgment']['label'] == label
    assert expected in state.prompt_metadata['prompt_rules']
    assert ('本轮优先模式：' + expected.upper()) in messages[0]['content']
    assert f'"mode": "browser_{expected}"' in messages[1]['content']
    assert '19' in state.prompt_metadata['exposed_refs']
    assert len(pipe.decision_judge.calls) == 1
    assert [e for e in pipe.events if e['event_type']=='decision_mode'][-1]['profile']['source']=='jev'
    # Rebuilding the exact same prompt does not call Jev again just because
    # exposed_refs/decision_mode were written into observation metadata.
    pipe.build_input(state)
    assert len(pipe.decision_judge.calls) == 1


@pytest.mark.parametrize('change', ['screenshot','snapshot','node','page','goal','queue_advance'])
def test_old_mode_is_not_reused_after_a_new_frame_page_ax_or_goal(change):
    pipe,state = setup('gui','dom')
    state.queue = ExecutionQueue({'steps': [
        {'id':'one','goal':'打开指定方案','expected_evidence':'详情页'},
        {'id':'two','goal':'点击下载按钮','expected_evidence':'下载开始'}]})
    first_id, second_id = [step['id'] for step in state.queue.plan['steps']]
    pipe.build_input(state)
    if change=='screenshot':
        state.obs.screenshot.putpixel((0,0),(255,0,0))
    elif change=='snapshot':
        state.obs.info['snapshot_id']='new'
        state.obs.info['scene']['snapshot_id']='new'
        state.obs.context[0]['snapshot_id']='new'
    elif change=='node':
        state.obs.context[0]['nodes'][0]['name']='Download attachment'
    elif change=='page':
        state.obs.info['page_id']='new-tab'
    elif change=='goal':
        state.queue.current['goal']='处理新出现的下载按钮'
    else:
        state.queue.accept_review({'plan_step_id':first_id,'status':'passed'})
        assert state.queue.current['id']==second_id
    pipe.build_input(state)
    assert len(pipe.decision_judge.calls)==2
    assert state.prompt_metadata['prompt_profile']['mode']=='dom'
    if change=='queue_advance':
        assert pipe.decision_judge.calls[-1]['step']['id']==second_id


@pytest.mark.parametrize('missing', ['unavailable','snapshot','disabled','static'])
def test_jev_dom_cannot_override_missing_or_invalid_current_ax(missing):
    pipe,state = setup('dom')
    if missing=='unavailable':
        state.obs.info['scene']['structure_available']=False
    elif missing=='snapshot':
        state.obs.info['snapshot_id']='missing'
    elif missing=='disabled':
        state.obs.context[0]['nodes'][0]['states']={'disabled':True}
    else:
        state.obs.context[0]['nodes']=[{'ref':'1','role':'StaticText','name':'Card title'}]
    pipe.build_input(state)
    profile = state.prompt_metadata['prompt_profile']
    assert profile['mode']=='gui' and profile['source']=='capability'
    assert profile['fallback_reason']=='jev_dom_without_current_ax'


def test_offscreen_target_remains_usable_for_jev_dom():
    pipe,state = setup('dom',scene='browser_gui')
    state.obs.context[0]['nodes'][0]['in_viewport']=False
    pipe.build_input(state)
    assert state.prompt_metadata['prompt_profile']['source']=='jev'
    assert state.prompt_metadata['prompt_profile']['mode']=='dom'


@pytest.mark.parametrize('kind', ['desktop','dialog','missing_image','stale','drag'])
def test_hard_surface_constraints_do_not_call_jev_or_adopt_a_stale_hint(kind):
    pipe,state = setup('dom')
    if kind=='desktop':
        state.obs=desktop_observation()
    elif kind=='dialog':
        state.obs.info['dialog']={'type':'alert','message':'Blocked'}
        state.obs.screenshot=None
    elif kind=='missing_image':
        state.obs.screenshot=None
    elif kind=='stale':
        state.obs.info['screenshot_stale']=True
    else:
        state.queue=ExecutionQueue({'steps':[{'goal':'拖动滑块','expected_evidence':'位置变化'}]})
    profile=pipe._select_decision_mode(state)
    assert profile.source=='capability'
    assert profile.mode==({'dialog':'dialog','missing_image':'observe','stale':'observe'}.get(kind,'gui'))
    assert not pipe.decision_judge.calls


def test_unknown_and_transport_failure_use_explicit_statistical_fallback():
    pipe,state=setup('unknown',scene='browser_gui')
    pipe.build_input(state)
    profile=state.prompt_metadata['prompt_profile']
    assert (profile['mode'],profile['source'],profile['fallback_reason'])==('gui','fallback','jev_unknown')
    pipe.decision_judge.select_mode=Mock(side_effect=TimeoutError)
    state.obs.context[0]['nodes'][0]['name']='New target'
    pipe.build_input(state)
    assert state.prompt_metadata['prompt_profile']['fallback_reason']=='mode_judge_TimeoutError'


def test_jev_gui_skips_both_ref_repair_and_hidden_semantic_dom_rerouting():
    pipe,state=setup('gui')
    pipe.build_input(state)
    action=Action('click',{'target':'Open item'})
    pipe.env.route_action=Mock(return_value={'channel':'browser_dom','target_ref':'19'})
    pipe._resolve=Mock(return_value=Action('click',{'x':10,'y':20}))
    assert pipe._dom_reference_feedback(action,state) is None
    resolved=pipe.ground(action,state)
    pipe.env.route_action.assert_not_called()
    pipe._resolve.assert_called_once()
    assert state.route is None and resolved.args['coordinate_space']=='css_viewport'
    # Explicit AX references remain independently validated for mixed pages.
    pipe.ground(Action('click',{'target':'Open item','target_ref':'19'}),state)
    pipe.env.route_action.assert_called_once()


def test_windows_semantic_router_honors_jev_dom_and_gui_over_scene_hint():
    session=Mock()
    session.route.return_value=None
    session.resolve_target.return_value='19'
    env=WindowsEnvironment()
    env._get_browser_session=Mock(return_value=session)
    gui=browser_observation()
    gui.info['decision_mode']={'mode':'gui','source':'jev'}
    assert env.route_action(Action('click',{'target':'Open item'}),gui) is None
    session.resolve_target.assert_not_called()
    dom=browser_observation(mode='browser_gui')
    dom.info['decision_mode']={'mode':'dom','source':'jev'}
    env.route_action(Action('click',{'target':'Open item'}),dom)
    session.resolve_target.assert_called_once()


def test_count_based_feedback_does_not_contradict_jev_target_judgment():
    obs=browser_observation(mode='browser_gui')
    obs.info['scene']['reason']='content_without_actionable_ax'
    obs.info['decision_mode']={'mode':'dom','source':'jev'}
    facts=operational_facts(obs,None,{})
    assert not any('目标内容卡片请' in text for text in facts)


def test_runtime_sends_every_current_control_including_offscreen_without_count_mode_hint():
    captured=[]
    def handle(request):
        data=json.loads(request.content)
        captured.append(data)
        return httpx.Response(200,json={'answers':{'next_mode':{'type':'choice',
            'probabilities':{'dom':.7,'gui':.2,'unknown':.1}}}})
    judge=RuntimeDecisionJudge({'url':'http://judge.test'},httpx.MockTransport(handle),
                               selection_policy='maximum_probability')
    obs=browser_observation(mode='browser_gui')
    obs.context[0]['nodes']=[{'ref':str(i),'role':'link','name':f'Paper {i}',
                            'direct':True,'in_viewport':False} for i in range(336)]
    try:
        result=judge.select_mode('打开 Paper 335',{'id':'current','goal':'打开 Paper 335'},obs)
        assert result['answers']['next_mode']['label']=='dom'
        state=json.loads(captured[0]['state'])
        assert len(state['ax_pages'][0]['controls'])==336
        assert state['ax_pages'][0]['controls'][-1][0]=='335'
        assert state['ax_pages'][0]['controls_complete'] is True
        assert state['current_goal']['id']=='current'
        assert 'browser_gui' not in captured[0]['state']
        assert len(captured[0]['images'])==1
    finally:
        judge.close()


def test_oversized_jev_context_is_explicit_unknown_not_silent_control_truncation():
    called=[]
    judge=RuntimeDecisionJudge({'url':'http://judge.test'},httpx.MockTransport(lambda req:called.append(req)))
    obs=browser_observation()
    obs.context[0]['nodes']=[{'ref':str(i),'role':'link','name':'x'*1000,'direct':True} for i in range(100)]
    try:
        result=judge.select_mode('打开目标',{},obs)
        assert result['status']=='unknown' and result['reason']=='mode_ax_context_too_large'
        assert result['state_chars']>32000 and not called and not judge.last_call
    finally:
        judge.close()
