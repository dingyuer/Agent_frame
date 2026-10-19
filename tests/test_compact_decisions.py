"""Policy briefs stay small without weakening framework checks or raw rollouts."""
import copy
import json

import pytest
from PIL import Image

from osworld_agent.config import ContextConfig
from osworld_agent.context import PromptCompiler
from osworld_agent.decision_context import current_brief, selected_memory
from osworld_agent.execution_queue import ExecutionQueue, MAX_PLAN_STEPS
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.actions import Action
from osworld_agent.trajectory import build_trajectory
from test_pipeline_completion import Model, Env, observation


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)


def test_summary_only_decision_executes_and_next_input_remembers_it():
    model = Model([])
    first = {'next_action': {'type': 'type', 'target': 'Search input', 'text': 'roboclaw'},
             'summary': '搜索框已显示，输入roboclaw以准备搜索。'}
    model.responses.extend([json.dumps(first), json.dumps({'next_action': {'type': 'press', 'key': 'Enter'},
                                                          'summary': '关键词已填好，按回车显示结果。'})])
    result = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test', max_steps=2).run('搜索roboclaw')
    assert result.steps == 2 and all(t['valid_action'] for t in result.trajectory)
    second_input = model.calls[2][-1]['content']
    assert first['summary'] in second_input and '近期行动摘要' in second_input
    assert '结构化记忆' not in second_input and '任务计划与系统验收条件' not in second_input
    assert result.trajectory[0]['policy_output'] == json.dumps(first)
    assert build_trajectory('compact', 'browser', result.task, result)['steps'][0]['reasoning']['summary'] == first['summary']


def test_framework_evidence_is_preserved_but_not_repeated_in_operator_brief():
    state = StepState(task='保留完整原始任务', obs=observation())
    state.queue = ExecutionQueue({'steps': [
        {'goal': '当前目标', 'expected_evidence': '当前结果', 'verification': [{'kind': 'visual'}], 'check_ids': ['trusted-id']},
        {'goal': '后续不可见目标', 'expected_evidence': '另一结果'}]})
    state.step_review = {'status': 'failed', 'reason': '输入框仍为空', 'review_id': 'secret-large-id',
                         'evidence': ['LONG_EVIDENCE' * 1000]}
    state.execution_feedback = {'status': 'executed', 'reason': json.dumps({'latency_ms': 123, 'artifacts': []})}
    before = copy.deepcopy(state.step_review)
    text = current_brief(state)
    assert all(s in text for s in ('当前目标', '当前结果', '输入框仍为空'))
    assert all(s not in text for s in ('后续不可见目标', 'secret-large-id', 'LONG_EVIDENCE', 'latency_ms', 'trusted-id'))
    assert state.step_review == before
    state.finish_feedback = '仍缺少目标文件'
    assert '仍缺少目标文件' in current_brief(state)


def test_memory_keeps_latest_sentences_and_attaches_failed_dispatch():
    memory = StructuredMemory()
    memory.known_facts['unverified_fact'] = 'must not leak'
    for i in range(12):
        memory.add_decision_note(i, f'观察{i}，准备执行。', 'click target', 'rejected')
    text = selected_memory(memory)
    assert len(text.splitlines()) == 8 and '11.' in text and '0.' not in text.splitlines()[0]
    assert '被拒绝，未执行' in text and 'unverified_fact' not in text
    clipped = PromptCompiler().recent_lines(text, 160)
    assert memory.decision_notes[-1] in clipped and PromptCompiler().count(clipped) <= 160


def test_invalid_auxiliary_patch_cannot_block_valid_primitive():
    model = Model([])
    model.responses = [json.dumps({'next_action': {'type': 'press', 'key': 'Enter'}, 'state_update': 'obsolete'})]
    pipe = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test')
    decision = pipe.decide([{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'task'}])
    assert decision.action.action == 'press' and decision.state_update == {}
    assert len(decision.calls) == 1 and decision.calls[0]['ignored_fields'] == ['state_update']


def test_format_retries_are_bounded_diagnosable_and_do_not_mutate_base_messages():
    model = Model([])
    invalid = json.dumps({'next_action': {'type': 'type', 'text': 'query'}, 'summary': '长' * 10000})
    model.responses = [invalid, 'not-json', json.dumps({'next_action': {'type': 'press', 'key': 'Enter'}})]
    pipe = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test', max_retries=3)
    base = [{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'task'}]
    before = copy.deepcopy(base)
    result = pipe.decide(base)
    assert result.action.action == 'press' and base == before
    assert result.calls[0]['response'] == invalid and 'target' in result.calls[0]['validation_error']
    assert result.calls[1]['validation_error_type'] == 'invalid_json'
    assert len(model.calls[1]) == len(model.calls[2]) == len(base) + 2
    for call in model.calls[1:]:
        assert sum(pipe.compiler.count(m['content']) for m in call[len(base):]) + 128 <= 1280
        assert '尚未执行' in call[-1]['content'] and 'observed_effect' not in call[-1]['content']
    assert all(e['dispatched'] is False for e in pipe.events if e['event_type'] == 'decision_format_error')


@pytest.mark.parametrize('count,accepted', [(1, True), (13, True), (MAX_PLAN_STEPS + 1, False)])
def test_milestone_count_is_not_twelve_but_has_a_resource_guard(count, accepted):
    plan = {'steps': [{'goal': f'目标{i}', 'expected_evidence': f'结果{i}'} for i in range(count)]}
    model = Model([{'type': 'press', 'key': 'Enter'}], plan)
    if not accepted:
        model.responses = [json.dumps(plan)] * 2
    env = Env()
    result = Pipeline(model=model, env=env, grounding=None, system_prompt='test', max_steps=1).run('依次操作多个对象')
    assert (result.termination_reason != 'planning_failed') is accepted
    if not accepted:
        assert not env.actions


def test_actual_image_sizes_are_recorded_without_resizing():
    compiler = PromptCompiler(ContextConfig())
    shots = [('older', Image.new('RGB', (1920, 1080))), ('current', Image.new('RGB', (2560, 1440)))]
    _, images, metadata = compiler.compile(system='s', task='t', memory='', auxiliary='',
        blocks=[{'kind': 'browser_ax', 'nodes': [{'ref': '1', 'role': 'textbox', 'name': 'Search'}]}],
        images=shots, require_image=True)
    assert len(images) == 1 and images[0][1] is shots[-1][1]
    assert metadata['image_sizes'] == [[2560, 1440]] and metadata['image_resize'] == 'none_client_side'


def test_compact_browser_observation_keeps_native_select_options():
    compiler = PromptCompiler(ContextConfig())
    block = {'kind': 'browser_ax', 'snapshot_id': 's', 'nodes': [
        {'ref': '1', 'role': 'combobox', 'name': 'Sort results by', 'tag': 'select', 'value': 'Newest', 'direct': True},
        {'ref': '2', 'parent': '1', 'role': 'MenuListPopup'},
        {'ref': '3', 'parent': '2', 'role': 'option', 'name': 'Newest'},
        {'ref': '4', 'parent': '2', 'role': 'option', 'name': 'Relevance'},
    ]}
    rendered, exposed, _ = compiler.structure([block], 'sort results', 1000)
    assert 'options=Newest;Relevance' in rendered
    assert '1' in exposed


def test_policy_receives_every_pdf_candidate_without_repeated_snapshot_fields():
    obs = observation()
    obs.info['download_candidates'] = [
        {'url': f'https://arxiv.org/pdf/2601.{index:05d}', 'page_id': 'same-page',
         'document_id': 5, 'snapshot_id': 'same-snapshot', 'source': 'anchor', 'label': 'pdf'}
        for index in range(50)]
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    messages = pipe.build_input(StepState(task='搜索论文并下载', obs=obs))
    text = messages[-1]['content']
    assert '2601.00049' in text and '2601.00000' in text
    assert 'download_candidates_omitted' not in text
    assert 'same-snapshot' not in text and 'same-page' not in text


def test_operation_mode_uses_route_before_jev_review():
    mode = Pipeline._operation_mode
    assert mode(Action('click', {'target': '12'}), {'channel': 'browser_dom'}) == 'dom'
    assert mode(Action('click', {'target': 'PDF toolbar'}), None) == 'gui'
    assert mode(Action('goto', {'url': 'https://arxiv.org/abs/1706.03762'}), {'channel': 'reobserve'}) == 'blocked'
    assert mode(Action('request_finish', {'answer': 'done'}), None) == 'framework'
