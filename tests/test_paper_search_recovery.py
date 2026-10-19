"""Recover submitted searches without rewriting filled fields or guessing IDs."""
import json
from urllib.parse import urlencode

import pytest
from PIL import Image

from osworld_agent.actions import Action
from osworld_agent.browser_feedback import operational_facts
from osworld_agent.env import Observation
from osworld_agent.paper_search import PaperSearchRecovery
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model


def obs(submitted='WorldGUI', value=None, *, engine='scholar', match=False, empty=False, filters=None):
    value = submitted if value is None else value
    url = ('https://scholar.google.com/scholar?' if engine == 'scholar' else 'https://arxiv.org/search/?')
    params = {'q' if engine == 'scholar' else 'query': submitted, 'hl': 'en'}
    params.update(filters or {})
    nodes = [{'ref': '1', 'role': 'textbox', 'id': 'gs_hdr_tsi' if engine == 'scholar' else 'query',
              'parent': 'form', 'name': 'Suchen', 'value': value, 'direct': True, 'states': {'focused': True}},
             {'ref': 'form', 'role': 'form'},
             {'ref': 'submit', 'parent': 'form', 'role': 'button', 'name': 'Search',
              'id': 'gs_hdr_tsb' if engine == 'scholar' else 'submit', 'direct': True}]
    if empty:
        nodes.append({'ref': 'empty', 'role': 'heading', 'name': 'Sorry, no results.'})
    else:
        nodes.extend([{'ref': 'heading', 'role': 'heading', 'tag': 'h3'},
                      {'ref': 'result', 'parent': 'heading', 'role': 'link', 'direct': True,
                       'name': 'WorldGUI: A Desktop Benchmark' if match else 'Another GUI Agent',
                       'href': 'https://publisher.example/paper'}])
    page = url + urlencode(params)
    return Observation(screenshot=Image.new('RGB', (640, 480), 'white'),
        context=[{'kind': 'browser_ax', 'url': page, 'nodes': nodes}],
        info={'url': page, 'active_surface': 'browser', 'screenshot_space': 'css_viewport'})


def type_query(text='WorldGUI', overwrite=True):
    return Action('type', {'target': 'Suchen', 'target_ref': '1', 'text': text, 'overwrite': overwrite})


def test_literal_observed_id_or_ref_cannot_bypass_query_or_submit_guards():
    page = obs('Attention is all you need', engine='arxiv')
    recovery = PaperSearchRecovery()
    for target in ['#query', 'query', '1']:
        assert recovery.action_error(Action('type', {'target': target,
            'text': 'Attention is all you need Vaswani', 'overwrite': True}), page, 'Attention is all you need')
    corrupt = obs(value='WorldGUIWorldGUI')
    for target in ['#gs_hdr_tsb', 'gs_hdr_tsb', 'submit', 'Search']:
        assert recovery.action_error(Action('click', {'target': target}), corrupt, 'WorldGUI')


def test_forward_pagination_is_read_before_expanding_query():
    page = obs(engine='arxiv', empty=True, filters={'order': ''})
    page.context[0]['nodes'].append({'ref': 'next', 'role': 'link', 'name': 'Next', 'direct': True,
        'href': '/search/?query=WorldGUI&order=&start=50'})
    advice = ' '.join(PaperSearchRecovery().facts(page, 'WorldGUI'))
    assert 'Next' in advice and 'ref=next' in advice and 'start=50' in advice
    assert 'WorldGUI paper' not in advice
    page.context[0]['nodes'][-1]['href'] = '/search/?query=WorldGUI&start=0'
    assert 'WorldGUI paper' in ' '.join(PaperSearchRecovery().facts(page, 'WorldGUI'))


def test_arxiv_no_match_prefers_observed_relevance_sort_before_mutating_name():
    page = obs(engine='arxiv', empty=True)
    page.context[0]['nodes'].append({'ref': 'sort', 'id': 'order', 'role': 'combobox',
        'value': 'Announcement date (newest first)', 'options': ['Relevance', 'Announcement date (newest first)']})
    recovery = PaperSearchRecovery()
    advice = ' '.join(recovery.facts(page, 'WorldGUI'))
    assert 'ref=sort' in advice and 'Relevance' in advice and 'Go' in advice
    assert 'WorldGUI paper' not in advice
    page.context[0]['nodes'][-1]['value'] = 'Relevance'
    assert '排序控件已选择' in ' '.join(recovery.facts(page, 'WorldGUI'))
    page.info['url'] += '&order='
    page.context[0]['url'] = page.info['url']
    assert 'WorldGUI paper' in ' '.join(recovery.facts(page, 'WorldGUI'))
    assert '排序控件已选择' not in ' '.join(recovery.facts(page, 'WorldGUI'))


def test_relevance_options_are_read_from_raw_ax_descendants():
    page = obs(engine='arxiv', empty=True)
    page.context[0]['nodes'].extend([
        {'ref': 'sort', 'id': 'order', 'role': 'combobox', 'value': 'Announcement date (newest first)'},
        {'ref': 'menu', 'parent': 'sort', 'role': 'MenuListPopup'},
        {'ref': 'option', 'parent': 'menu', 'role': 'option', 'name': 'Relevance'}])
    advice = ' '.join(PaperSearchRecovery().facts(page, 'WorldGUI'))
    assert 'ref=sort' in advice and 'Relevance' in advice and 'Go' in advice


def test_name_search_cannot_guess_author_but_can_use_observed_words_or_neutral_expansion():
    page = obs('Attention is all you need', engine='arxiv')
    recovery = PaperSearchRecovery()
    query = 'Attention is all you need'
    action = type_query(query + ' Vaswani')
    assert 'vaswani' in recovery.action_error(action, page, query)
    assert not recovery.action_error(type_query(query + ' paper'), page, query)
    page.context[0]['nodes'].append({'ref': 'author', 'role': 'StaticText', 'name': 'Vaswani'})
    assert not recovery.action_error(action, page, query)


def test_short_name_does_not_default_to_quotes_or_require_exact_full_title():
    recovery = PaperSearchRecovery()
    facts = ' '.join(recovery.facts(obs(), 'WorldGUI'))
    assert 'WorldGUI paper' in facts
    assert '\\"WorldGUI\\"' not in facts
    assert '不能证明论文不存在' in facts
    matched = ' '.join(recovery.facts(obs(match=True), 'WorldGUI'))
    assert '名称字面命中' in matched and 'ref=result' in matched
    assert '其他实际结果也可打开详情' in matched
    assert '下载前仍须核实身份' in matched


def test_no_lexical_hit_does_not_claim_candidates_are_unrelated():
    recovery = PaperSearchRecovery()
    page = obs()
    page.context[0]['nodes'][-1]['name'] = 'Interactive Desktop Agents from Any Starting Point'
    facts = ' '.join(recovery.facts(page, 'WorldGUI'))
    assert '这不表示候选不相关' in facts and '语义判断' in facts
    assert not recovery.action_error(Action('click', {'target_ref': 'result'}), page, 'WorldGUI')


def test_literal_hit_does_not_forbid_query_refinement_after_semantic_inspection():
    page = obs(match=True)
    assert not PaperSearchRecovery().action_error(type_query('WorldGUI paper'), page, 'WorldGUI')


def test_filled_unsubmitted_query_gets_enter_without_retyping():
    recovery = PaperSearchRecovery()
    page = obs(value='WorldGUI paper')
    facts = ' '.join(recovery.facts(page, 'WorldGUI'))
    assert '已填写' in facts and 'press Enter' in facts
    assert recovery.action_error(type_query('WorldGUI paper'), page, 'WorldGUI')
    assert not recovery.action_error(Action('press', {'key': 'Enter'}), page, 'WorldGUI')


@pytest.mark.parametrize('overwrite', [False, None, 'true'])
def test_populated_paper_search_requires_boolean_overwrite(overwrite):
    error = PaperSearchRecovery().action_error(type_query('WorldGUI paper', overwrite), obs(), 'WorldGUI')
    assert '追加' in error and 'overwrite:true' in error


def test_submission_of_finished_no_match_search_is_not_new_progress():
    recovery = PaperSearchRecovery()
    page = obs()
    recovery.facts(page, 'WorldGUI')
    assert recovery.action_error(Action('press', {'key': 'Enter'}), page, 'WorldGUI')
    assert recovery.action_error(Action('click', {'target': 'Search', 'target_ref': 'submit'}), page, 'WorldGUI')
    # Enter on a result link can open it, and must remain available.
    page.context[0]['nodes'][0]['states']['focused'] = False
    assert not recovery.action_error(Action('press', {'key': 'Enter'}), page, 'WorldGUI')


def test_same_query_different_input_is_not_global_duplicate_typing_rule():
    action = Action('type', {'target': 'Note', 'target_ref': 'another-input', 'text': 'WorldGUI'})
    assert not PaperSearchRecovery().action_error(action, obs(), 'WorldGUI')
    page = obs()
    page.info['active_surface'] = 'desktop'
    assert not PaperSearchRecovery().action_error(type_query(), page, 'WorldGUI')


def test_no_match_variants_are_remembered_across_observations():
    recovery = PaperSearchRecovery()
    assert 'WorldGUI paper' in ' '.join(recovery.facts(obs(), 'WorldGUI'))
    facts = ' '.join(recovery.facts(obs('WorldGUI paper'), 'WorldGUI'))
    assert '换搜索入口' in facts and 'https://arxiv.org/search/' in facts
    assert recovery.action_error(type_query('WorldGUI'), obs('WorldGUI paper'), 'WorldGUI')
    # Changed filters are a distinct search, rather than a permanent ban.
    assert not recovery.action_error(type_query('WorldGUI'), obs('WorldGUI paper', filters={'as_ylo': '2025'}), 'WorldGUI')


def test_known_empty_results_recover_but_a_loading_form_does_not_mean_no_match():
    recovery = PaperSearchRecovery()
    assert 'WorldGUI paper' in ' '.join(recovery.facts(obs(empty=True), 'WorldGUI'))
    loading = obs()
    loading.context[0]['nodes'] = loading.context[0]['nodes'][:3]
    new = PaperSearchRecovery()
    new.facts(loading, 'WorldGUI')
    assert not new.attempts
    assert not new.action_error(Action('press', {'key': 'Enter'}), loading, 'WorldGUI')


def test_title_field_change_can_still_be_submitted_without_retyping():
    page = obs(engine='arxiv', filters={'searchtype': 'all'})
    page.context[0]['nodes'].append({'ref': 'field', 'id': 'searchtype', 'role': 'combobox', 'value': 'Title'})
    recovery = PaperSearchRecovery()
    assert '检索字段已改变' in ' '.join(recovery.facts(page, 'WorldGUI'))
    assert not recovery.action_error(Action('press', {'key': 'Enter'}), page, 'WorldGUI')
    assert not recovery.action_error(Action('click', {'target': 'Search', 'target_ref': 'submit'}), page, 'WorldGUI')


@pytest.mark.parametrize('repeated', ['WorldGUIWorldGUI', 'WorldGUIWorldGUIWorldGUI', '"WorldGUI""WorldGUI"'])
def test_existing_repeated_append_is_repaired_not_submitted(repeated):
    facts = ' '.join(PaperSearchRecovery().facts(obs(value=repeated), 'WorldGUI'))
    assert '重复追加' in facts and 'overwrite:true' in facts


def test_pipeline_repairs_duplicate_action_before_dispatch():
    model = Model([])
    model.responses = [json.dumps({'next_action': vars(type_query('WorldGUI paper'))['args'] | {'type': 'type'}}),
                       json.dumps({'next_action': {'type': 'press', 'key': 'Enter'}})]
    env = Env()
    pipe = Pipeline(model=model, env=env, grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier('下载一篇叫WorldGUI的论文')
    state = StepState(task='下载一篇叫WorldGUI的论文', obs=obs(value='WorldGUI paper'))
    decision = pipe.decide([{'role': 'user', 'content': '当前查询框'}], state)
    assert decision.action.action == 'press'
    assert decision.calls[0]['validation_error_type'] == 'redundant_search_action'
    assert not env.actions


def test_pipeline_ground_guard_also_rejects_duplicate_without_dispatch():
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier('下载一篇叫WorldGUI的论文')
    state = StepState(task='下载一篇叫WorldGUI的论文', obs=obs())
    pipe.ground(type_query(), state)
    assert state.route['channel'] == 'reobserve'
    assert state.route['error_code'] == 'redundant_search_action'
    assert not pipe.env.actions


def test_feedback_does_not_keep_requesting_input_after_it_is_filled():
    verifier = CompletionVerifier('下载一篇叫WorldGUI的论文')
    facts = ' '.join(operational_facts(obs(value='WorldGUI paper'), verifier, {}, {'phase': 'search'}))
    assert '不要再次输入' in facts and '直接 press Enter' in facts
    assert '输入带英文双引号' not in facts


def test_arxiv_broadening_title_to_all_fields_is_a_valid_submission():
    page = obs(engine='arxiv', filters={'searchtype': 'title'})
    page.context[0]['nodes'].append({'ref': 'field', 'id': 'searchtype', 'role': 'combobox', 'value': 'All fields'})
    assert not PaperSearchRecovery().action_error(Action('press', {'key': 'Enter'}), page, 'WorldGUI')


def test_two_exhausted_sources_do_not_suggest_switching_back_forever():
    recovery = PaperSearchRecovery()
    recovery.facts(obs(), 'WorldGUI')
    recovery.facts(obs('WorldGUI paper'), 'WorldGUI')
    # Record an arXiv no-result response under the same unfiltered scope.
    recovery.facts(obs(engine='arxiv', empty=True), 'WorldGUI')
    facts = ' '.join(recovery.facts(obs('WorldGUI paper', engine='arxiv', empty=True), 'WorldGUI'))
    assert '普通网页搜索' in facts and '两个耗尽入口' in facts
    returned = obs('', value='WorldGUI')
    assert '不要重复提交' in ' '.join(recovery.facts(returned, 'WorldGUI'))
    assert recovery.action_error(Action('press', {'key': 'Enter'}), returned, 'WorldGUI')
