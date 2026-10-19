"""Cross-action stalls, partial progress and pending operations stay distinct."""
import json

import pytest

from osworld_agent.actions import Action
from osworld_agent.pipeline import Pipeline
from osworld_agent.step_progress import ProgressConfig, StepProgressMonitor
from osworld_agent.trajectory import build_trajectory
from osworld_agent.env import StepResult
from test_pipeline_completion import Model, Env, observation


STEP = {'id': 'p0_1', 'goal': '搜索目标论文并显示结果'}
REVIEW = {'plan_step_id': 'p0_1', 'status': 'failed'}
EXECUTED = {'status': 'executed'}


def observe(monitor, action, before=None, after=None, review=None, info=None, step=None):
    before = before or observation()
    return monitor.observe(step or STEP, review or REVIEW, before, after or before,
                           action, EXECUTED if info is None else info)


def test_alternating_actions_and_cosmetic_screen_changes_still_stall():
    monitor = StepProgressMonitor()
    actions = [Action('press', {'key': 'ENTER'}), Action('click', {'target': 'Search'}),
               Action('hotkey', {'keys': ['CTRL', 'L']}), Action('click', {'target': 'Different button'})]
    for index, action in enumerate(actions):
        result = observe(monitor, action, after=observation(color='blue' if index % 2 else 'red'))
    assert result['status'] == 'stalled' and result['idle_decisions'] == 4
    assert len(result['recent_attempts']) == 4


def test_real_distinct_inputs_can_exceed_recovery_threshold_without_stalling():
    monitor = StepProgressMonitor()
    previous = observation()
    for index in range(12):
        value = str(index)
        current = observation(value=value)
        result = observe(monitor, Action('type', {'target': 'Email', 'text': value}), previous, current)
        assert result['idle_decisions'] == 0 and result['status'] == 'watching'
        previous = current


def test_repeating_input_values_is_not_endless_progress():
    monitor = StepProgressMonitor()
    previous = observation(value='a')
    for value in ['b', 'a', 'b', 'a', 'b']:
        current = observation(value=value)
        result = observe(monitor, Action('type', {'text': value}), previous, current)
        previous = current
    assert result['status'] == 'stalled'


def test_pending_downloads_suspend_counter_and_completion_is_progress():
    monitor = StepProgressMonitor()
    action = Action('press', {'key': 'ENTER'})
    for _ in range(2):
        observe(monitor, action)
    for _ in range(30):
        result = observe(monitor, action, info={'status': 'in_progress'})
        assert result['status'] == 'waiting' and result['idle_decisions'] == 2
    result = observe(monitor, action, info={'status': 'executed', 'artifacts': [
        {'artifact_id': 'd1', 'completed': True}]})
    assert result['idle_decisions'] == 0
    assert result['status'] != 'advanced'  # Completed transfer is not task completion.


def test_model_wait_without_pending_operation_cannot_suspend_monitor():
    monitor = StepProgressMonitor()
    for _ in range(4):
        result = observe(monitor, Action('wait', {'seconds': 1}))
    assert result['status'] == 'stalled'


def test_unchanged_passing_subcheck_cannot_reset_counter_with_new_evidence_text():
    monitor = StepProgressMonitor()
    for i in range(5):
        review = {**REVIEW, 'code_checks': [{'kind': 'url_host', 'status': 'passed', 'evidence': [str(i)]}]}
        result = observe(monitor, Action('press', {'key': 'ENTER'}), review=review)
    assert result['status'] == 'stalled'


def test_navigation_exploration_is_bounded_even_with_unique_urls():
    monitor = StepProgressMonitor()
    previous = observation()
    for i in range(6):
        current = observation(url=f'https://example.test/{i}')
        result = observe(monitor, Action('goto', {'url': current.context[0]['url']}), previous, current)
        previous = current
    assert result['status'] == 'stalled'


def test_new_queue_step_resets_monitor_and_old_pending_state_does_not_leak():
    monitor = StepProgressMonitor()
    action = Action('press', {'key': 'ENTER'})
    for _ in range(3):
        observe(monitor, action)
    result = observe(monitor, action, review={**REVIEW, 'status': 'passed'})
    assert result['status'] == 'advanced'
    result = observe(monitor, action, step={**STEP, 'id': 'p1_1'}, review={**REVIEW, 'plan_step_id': 'p1_1'})
    assert result['idle_decisions'] == 1 and result['step_decisions'] == 1


@pytest.mark.parametrize('kwargs', [{'no_progress_limit': 1}, {'no_progress_limit': True},
                                   {'exploration_allowance': -1}, {'exploration_allowance': 5}])
def test_configuration_is_bounded(kwargs):
    with pytest.raises(ValueError):
        ProgressConfig(**kwargs)


def test_full_episode_recovers_before_last_step_and_exports_reason(monkeypatch):
    logs = []
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', logs.append)
    env = Env()
    env.route_action = lambda action, obs: None if action.action in {'fail', 'request_finish'} else {'channel': 'browser_dom'}
    # Flashing screen and alternating keys previously evaded the action tracker.
    def execute(action, route):
        env.actions.append(action)
        env.obs = observation(color='blue' if len(env.actions) % 2 else 'red')
        return StepResult(env.obs, info={'status': 'executed', 'dispatched': True})
    env.execute_routed = execute
    initial = {'steps': [{'goal': '搜索并显示结果', 'expected_evidence': '结果已显示'}]}
    replanned = {'steps': [{'goal': '更换入口搜索', 'expected_evidence': '看到结果'}],
                 'strategy_change': '此前反复按键未产生结果，改用新的搜索入口并核查结果列表。'}
    actions = [{'type': 'press', 'key': key} for key in ['ENTER', 'TAB', 'ENTER', 'TAB']]
    model = Model(actions, initial)
    model.responses += [json.dumps(replanned), json.dumps({'next_action': {'type': 'fail', 'reason': 'fixture ended'}})]
    result = Pipeline(model=model, env=env, grounding=None, system_prompt='test', max_steps=15).run('搜索目标信息')
    recovery = [e for e in result.events if e['event_type'] == 'framework_recovery']
    assert recovery[0]['remaining_decisions'] == 11
    assert recovery[0]['failure_class'] == 'step_no_progress'
    assert recovery[0]['step_progress']['idle_decisions'] == 4
    assert recovery[1]['status'] == 'replanned'
    assert result.steps == 5 and len(env.actions) == 4 and env.reset_count == 1
    assert any(s['event'] == 'step_progress' and s['report']['status'] == 'stalled' for s in logs)
    assert any(s['event'] == 'framework_recovery' for s in logs)
    exported = build_trajectory('progress', 'browser', result.task, result)
    assert exported['steps'][3]['step_progress']['status'] == 'stalled'
    assert exported['run_manifest']['runtime_config']['progress_config']['no_progress_limit'] == 4
    assert not result.evaluation_available


def test_collection_can_progress_beyond_exploration_allowance_but_not_repeat_content():
    from osworld_agent.env import Observation
    monitor=StepProgressMonitor()
    before=Observation(context=[{'kind':'browser_ax','url':'https://test/index','nodes':[]}],info={})
    review={'plan_step_id':'p0_1','status':'unknown','step_outcome':'progress','progress_hint':False}
    for i in range(6):
        url=f'https://test/course/{i}'
        text=f'课程 {i} 的报名时间：每周三下午 18:00'
        after=Observation(context=[{'kind':'browser_ax','url':url,'nodes':[{'role':'StaticText','name':text}]}],info={})
        result=monitor.observe(STEP,review,before,after,Action('click',{}),EXECUTED,collected_sources=[{'url':url,'text':text}])
        assert result['status']=='watching' and result['idle_decisions']==0
        before=after
    for _ in range(4):
        result=monitor.observe(STEP,review,before,after,Action('click',{}),EXECUTED,collected_sources=[{'url':url,'text':text}])
    assert result['status']=='stalled'
