"""AX result grouping keeps every observed paper link within a bounded prompt."""
from osworld_agent.browser_results import paper_result_cards
from osworld_agent.config import ContextConfig
from osworld_agent.context import ObservationIncompleteError, PromptCompiler
from osworld_agent.env import Observation
from osworld_agent.judge_evidence import select_frame
from osworld_agent.actions import Action
from osworld_agent.browser_feedback import (unbound_paper_candidate_error,
                                            redundant_sort_selection_error,
                                            operational_facts)
import pytest


class PendingPaper:
    def paper_goals(self):
        return [{'check_id': 'check_1', 'status': 'pending', 'query': 'Attention Is All You Need',
                 'binding': None}]


def result_block(count=50, exact=True):
    nodes = [{'ref': '0', 'role': 'RootWebArea', 'name': 'Search'}]
    for index in range(count):
        base = 10 + index * 10
        ident = f'2601.{index:05d}'
        title = 'Attention Is All You Need' if exact and index == count - 1 else f'Other Paper {index}'
        nodes.extend([
            {'ref': str(base), 'parent': '0', 'role': 'listitem', 'tag': 'li'},
            {'ref': str(base + 1), 'parent': str(base), 'role': 'paragraph', 'tag': 'p'},
            {'ref': str(base + 2), 'parent': str(base + 1), 'role': 'link', 'name': f'arXiv:{ident}',
             'href': f'https://arxiv.org/abs/{ident}'},
            {'ref': str(base + 3), 'parent': str(base + 1), 'role': 'link', 'name': 'pdf',
             'href': f'https://arxiv.org/pdf/{ident}'},
            {'ref': str(base + 4), 'parent': str(base), 'role': 'paragraph', 'tag': 'p'},
            {'ref': str(base + 5), 'parent': str(base + 4), 'role': 'StaticText', 'name': title[:12]},
            {'ref': str(base + 6), 'parent': str(base + 4), 'role': 'StaticText', 'name': title[12:]},
        ])
    return {'kind': 'browser_ax', 'snapshot_id': 'current', 'url': 'https://arxiv.org/search/?query=Attention',
            'title': 'Search', 'nodes': nodes}


def test_all_fifty_results_keep_titles_and_actionable_links():
    block = result_block()
    cards = paper_result_cards(block)
    assert len(cards) == 50 and cards[-1]['title'] == 'Attention Is All You Need'
    compiler = PromptCompiler(ContextConfig())
    rendered, exposed, selection = compiler.structure([block], '按标题搜索', 18000,
                                                       paper_query='Attention Is All You Need')
    assert '"columns":["rank","title","abs_ref","abs_href","pdf_ref"]' in rendered
    assert rendered.count('"/abs/') == 50
    assert '"exact_title_positions":[50]' in rendered
    assert selection[0]['paper_results_shown'] == 50
    assert cards[0]['abs_ref'] in exposed and cards[-1]['abs_ref'] in exposed
    assert cards[-1]['pdf_ref'] in exposed


def test_full_interactive_index_includes_offscreen_links():
    block = result_block()
    for index in range(400):
        block['nodes'].append({'ref': str(1000 + index), 'parent': '0',
            'role': 'link', 'name': f'Author {index}', 'direct': True,
            'in_viewport': index < 3,
            'href': f'https://arxiv.org/search/?searchtype=author&query=Author+{index}'})
    rendered, exposed, selection = PromptCompiler(ContextConfig()).structure(
        [block], '搜索 Attention Is All You Need', 23000,
        paper_query='Attention Is All You Need')
    assert len([ref for ref in exposed if str(ref).isdigit() and int(ref) >= 1000]) == 400
    assert selection[0]['interactive_omitted'] == 0
    assert selection[0]['paper_results_shown'] == 50
    assert 'Author 399' in rendered


def test_dense_results_keep_all_controls_before_optional_memory_and_prose():
    block = result_block()
    for index in range(400):
        block['nodes'].append({'ref': str(1000 + index), 'parent': '0',
            'role': 'link', 'name': f'Author {index}', 'direct': True,
            'href': f'https://arxiv.org/search/?searchtype=author&query=Author+{index}'})
    _, _, metadata = PromptCompiler(ContextConfig(max_input_tokens=15000)).compile(
        system='system' * 250, task='搜索论文 Attention Is All You Need 并下载',
        memory='old fact ' * 500, blocks=[block], auxiliary='tool ' * 250,
        images=[], required='当前目标' * 100)
    selected = metadata['selection'][0]
    assert selected['interactive_total'] == selected['interactive_shown'] == 500
    assert selected['paper_results_shown'] == 50
    assert selected['interactive_omitted'] == 0


def test_insufficient_budget_rejects_incomplete_interaction_index():
    block = result_block(0)
    block['nodes'].extend({'ref': str(1000 + index), 'parent': '0',
        'role': 'link', 'name': f'Action {index}', 'direct': True}
        for index in range(40))
    with pytest.raises(ObservationIncompleteError, match='不隐藏节点'):
        PromptCompiler(ContextConfig()).structure([block], 'task', 150)


def test_very_dense_search_page_compacts_all_author_refs_in_small_budget():
    # A real arXiv search exposed 767 controls, mostly author links. The
    # previous 32-character minimum aborted the next decision after search.
    block = result_block()
    for index in range(667):
        block['nodes'].append({'ref': str(1000 + index), 'parent': '0',
            'role': 'link', 'name': f'Author {index}', 'direct': True,
            'href': f'https://arxiv.org/search/?searchtype=author&query=Author+{index}'})
    rendered, exposed, selection = PromptCompiler(ContextConfig()).structure(
        [block], '搜索论文', 16000, paper_query='Attention Is All You Need')
    assert selection[0]['interactive_total'] == selection[0]['interactive_shown'] == 767
    assert selection[0]['paper_results_shown'] == 50
    assert '"/abs/' in rendered
    assert '1666' in exposed


def test_visible_focused_search_overlay_survives_result_compaction():
    block = result_block()
    block['nodes'].append({'ref': '900', 'parent': '0', 'role': 'textbox',
                           'name': 'Search arXiv', 'id': 'arxiv-search-input',
                           'states': {'focused': True}, 'direct': True, 'in_viewport': True})
    rendered, exposed, selection = PromptCompiler(ContextConfig()).structure(
        [block], '核对标题', 18000, paper_query='Attention Is All You Need')
    assert 'arxiv-search-input' in rendered and '900' in exposed
    assert selection[0]['paper_results_shown'] == 50


def test_judge_sees_offscreen_exact_match_even_with_four_ax_nodes():
    obs = Observation(context=[result_block()])
    frame = select_frame(obs, {'goal': '核对标题“Attention Is All You Need”的论文身份'}, max_nodes=4)
    assert frame['selection']['nodes_selected'] == 4
    assert frame['paper_results']['current_page_count'] == 50
    assert frame['paper_results']['exact_title_matches'][0]['position'] == 50


def test_lexical_title_differences_do_not_block_candidate_inspection():
    obs = Observation(context=[result_block()])
    wrong = Action('click', {'target': '11'})
    correct = Action('click', {'target': str(10 + 49 * 10 + 2)})
    assert not unbound_paper_candidate_error(wrong, obs, PendingPaper(), {'target_ref': '12'})
    assert not unbound_paper_candidate_error(correct, obs, PendingPaper(), {'target_ref': '502'})
    facts = ' '.join(operational_facts(Observation(context=[result_block(49, exact=False)]),
                                      PendingPaper(), {}, None))
    assert '语义判断' in facts and '可直接打开详情' in facts
    assert '这不表示候选不相关' in facts


def test_repeated_sort_selection_requires_submission():
    block = result_block(1)
    block['nodes'].append({'ref': '900', 'role': 'combobox', 'id': 'order',
                           'value': 'Relevance', 'options': ['Relevance']})
    obs = Observation(context=[block])
    same = Action('select', {'target': '900', 'option': 'Relevance'})
    other = Action('select', {'target': '900', 'option': 'Submission date (newest first)'})
    assert 'Go' in redundant_sort_selection_error(same, obs, {'target_ref': '900'})
    assert not redundant_sort_selection_error(other, obs, {'target_ref': '900'})


def test_title_filter_selection_requires_search_submission():
    block = result_block(1, exact=False)
    block['url'] = 'https://arxiv.org/search/?query=Attention&searchtype=all'
    block['nodes'].append({'ref': '900', 'role': 'combobox', 'tag': 'select',
                           'id': 'searchtype', 'value': 'Title', 'direct': True})
    obs = Observation(context=[block])
    assert 'Search 按钮' in redundant_sort_selection_error(
        Action('select', {'target': '900', 'option': 'Title'}), obs, {'target_ref': '900'})
    assert 'Search 按钮' in redundant_sort_selection_error(
        Action('click', {'target': '900'}), obs, {'target_ref': '900'})
    assert any('结果 URL 仍不是标题检索' in fact and 'Search 按钮' in fact
               for fact in operational_facts(obs, PendingPaper(), {}, None))
    block['url'] = 'https://arxiv.org/search/?query=Attention&searchtype=title'
    assert not redundant_sort_selection_error(
        Action('click', {'target': '900'}), obs, {'target_ref': '900'})
