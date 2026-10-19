"""Measured scrolling reaches review and policy without proving task success."""
from types import SimpleNamespace
from osworld_agent.actions import Action
from osworld_agent.decision_context import current_brief
from osworld_agent.judge_evidence import effect_review
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import Execution, Pipeline, StepState
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model, observation


def feedback(delta=120, *, moved=True, bottom=False):
    return {'available': True, 'requested_delta': 120, 'actual_delta': delta,
            'moved': moved, 'noop': not moved, 'at_top': False, 'at_bottom': bottom,
            'target_container': {'kind': 'element', 'id': 'results'},
            'containers': [{'diagnostic': 'full local chain'}]}


def test_measured_scroll_effect_is_local_and_noop_is_not_a_task_pass():
    obs = observation()
    report = effect_review(obs, obs, {'status': 'executed', 'action': 'scroll',
        'scroll_feedback': feedback()})
    assert report['status'] == 'passed' and '目标内容仍须核查' in report['reason']
    report = effect_review(obs, obs, {'status': 'executed', 'action': 'scroll',
        'scroll_feedback': feedback(0, moved=False, bottom=True)})
    assert report['status'] == 'unknown' and '底部' in report['reason']
    report = effect_review(obs, obs, {'status': 'executed', 'action': 'scroll',
        'scroll_feedback': {'available': False, 'moved': None, 'noop': None}})
    assert report['status'] == 'unknown'


def test_next_policy_gets_actual_movement_and_boundary_from_copied_feedback():
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    state = StepState()
    info = {'status': 'executed', 'scroll_feedback': feedback(0, moved=False, bottom=True)}
    pipe._update_program_info(Action('scroll', {'direction': 'down', 'amount': 1}),
                              Execution(kind='browser_dom', info=info), state)
    info['scroll_feedback']['at_bottom'] = False
    prompt = current_brief(state)
    assert 'results' in prompt and '未测得位移' in prompt and '已到底部' in prompt
    assert state.execution_feedback['scroll_feedback']['at_bottom'] is True


def test_opposite_boundary_does_not_explain_a_noop_in_requested_direction():
    scroll = feedback(0, moved=False)
    scroll['at_top'] = True  # A downward request can still move away from top.
    obs = observation()
    report = effect_review(obs, obs, {'status': 'executed', 'action': 'scroll', 'scroll_feedback': scroll})
    prompt = current_brief(StepState(execution_feedback={'status': 'executed', 'scroll_feedback': scroll}))
    assert '已到达顶部' not in report['reason'] and '已到顶部' not in prompt
    scroll['requested_delta'] = -120
    scroll['at_top'], scroll['at_bottom'] = False, True
    report = effect_review(obs, obs, {'status': 'executed', 'action': 'scroll', 'scroll_feedback': scroll})
    prompt = current_brief(StepState(execution_feedback={'status': 'executed', 'scroll_feedback': scroll}))
    assert '已到达底部' not in report['reason'] and '已到底部' not in prompt


def test_jev_receives_scroll_summary_without_unbounded_container_diagnostics(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    class Judge:
        config = SimpleNamespace(transition_diagnostics=True)
        last_call = {}
        def assess_transition(self, task, step, before, after, execution, code, **kwargs):
            self.execution = execution
            self.effect = code
            return {'status': 'answered', 'answers': {}}
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier('test')
    judge = pipe.decision_judge = Judge()
    state = StepState(task='test', obs=observation(), memory=StructuredMemory(),
                      verification={'status': 'passed'}, step_id=1)
    execution = Execution(kind='browser_dom', obs=state.obs,
                          info={'status': 'executed', 'dispatched': True, 'scroll_feedback': feedback()})
    pipe._update_plan_progress(state, execution, before_observation=observation(),
                              action=Action('scroll', {'direction': 'down', 'amount': 1}))
    assert judge.execution['scroll_feedback']['actual_delta'] == 120
    assert 'containers' not in judge.execution['scroll_feedback']
    assert judge.effect['status'] == 'passed'
