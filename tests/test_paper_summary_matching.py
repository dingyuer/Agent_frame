"""Policy and judge summaries share the official title/complete-alias matcher."""
import copy
import json

import pytest

from osworld_agent.context import PromptCompiler
from osworld_agent.env import Observation
from osworld_agent.judge_evidence import select_frame
from test_browser_result_cards import result_block


FULL_TITLE = 'WorldGUI: An Interactive Benchmark for Desktop GUI Automation from Any Starting Point'


def arxiv_results(title, *, count=1, snapshot='results'):
    block = result_block(count, exact=False)
    block['snapshot_id'] = snapshot
    for node in block['nodes']:
        node['in_viewport'] = False
        if node.get('role') == 'link':
            node['direct'] = True
    base = 10 + (count - 1) * 10
    for node in block['nodes']:
        if node['ref'] == str(base + 5):
            node['name'] = title[:12]
        elif node['ref'] == str(base + 6):
            node['name'] = title[12:]
    return block


def summaries(blocks, query):
    rendered, refs, selection = PromptCompiler().structure(blocks, '核对候选', 23000, paper_query=query)
    policy = [item for line in rendered.splitlines() if line.startswith('{')
              for item in [json.loads(line)] if item.get('kind') == 'paper_results']
    judge = select_frame(Observation(context=blocks),
                         {'verification': [{'kind': 'search_results', 'query': query}]}, max_nodes=4)
    return policy, judge['paper_results'], refs, selection


@pytest.mark.parametrize('query,title,kind', [
    ('WorldGUI', FULL_TITLE, 'leading_alias'),
    (FULL_TITLE, FULL_TITLE, 'full_title'),
    ('ＷｏｒｌｄＧＵＩ', 'ＷｏｒｌｄＧＵＩ： An Interactive Benchmark', 'leading_alias'),
    ('WORLDGUI', 'worldgui: An Interactive Benchmark', 'leading_alias'),
    ('UIBench', 'UIBench: A Different Paper', 'leading_alias'),
])
def test_complete_title_and_alias_match_in_policy_and_judge(query, title, kind):
    block = arxiv_results(title)
    before = copy.deepcopy(block)
    policy, judge, refs, selection = summaries([block], query)
    assert block == before
    assert policy[0]['exact_title_positions'] == [1]
    assert policy[0]['title_matches'] == [{'position': 1, 'match_kind': kind}]
    assert judge['exact_title_matches'][0]['match_kind'] == kind
    assert judge['exact_title_matches'][0]['title'] == title
    assert judge['matched_count'] == 1
    assert policy[0]['identity_verified'] is judge['identity_verified'] is False
    assert {'12', '13'} <= set(refs)
    assert selection[0]['interactive_omitted'] == 0


@pytest.mark.parametrize('title', [
    'A Comparison of WorldGUI Methods', 'WorldGUI-Based Automation',
    'WorldGUI, An Interactive Benchmark', 'WorldGUI， An Interactive Benchmark',
    'WorldGUI (Extended): An Interactive Benchmark', 'WorldGUIs: An Interactive Benchmark',
])
def test_related_titles_and_other_delimiters_do_not_match_short_name(title):
    policy, judge, refs, _ = summaries([arxiv_results(title)], 'WorldGUI')
    assert policy[0]['exact_title_positions'] == []
    assert policy[0]['title_matches'] == []
    assert judge['exact_title_matches'] == [] and judge['matched_count'] == 0
    assert {'12', '13'} <= set(refs)  # Nonmatches remain available for inspection.


def test_offscreen_alias_is_kept_without_expanding_or_truncating_interactive_nodes():
    block = arxiv_results(FULL_TITLE, count=50)
    for index in range(400):
        block['nodes'].append({'ref': str(1000 + index), 'role': 'link', 'name': f'Author {index}',
                               'href': f'https://arxiv.org/search/?searchtype=author&query=Author+{index}',
                               'direct': True, 'in_viewport': False})
    policy, judge, refs, selection = summaries([block], 'WorldGUI')
    assert policy[0]['exact_title_positions'] == [50]
    assert judge['exact_title_matches'][0]['position'] == 50
    assert selection[0]['interactive_total'] == selection[0]['interactive_shown'] == 500
    assert selection[0]['paper_results_shown'] == 50
    assert '1399' in refs and '502' in refs and '503' in refs


def test_judge_counts_all_observed_result_snapshots_and_preserves_match_origin():
    first = arxiv_results('Other Paper', count=2, snapshot='first-results')
    second = arxiv_results(FULL_TITLE, count=3, snapshot='second-results')
    second['url'] = 'https://arxiv.org/search/?query=WorldGUI&start=50'
    policy, judge, _, selection = summaries([first, second], 'WorldGUI')
    assert [item['total'] for item in policy] == [2, 3]
    assert [item['exact_title_positions'] for item in policy] == [[], [3]]
    assert sum(item['paper_results_shown'] for item in selection) == 5
    assert judge['current_page_count'] == 5 and len(judge['pages']) == 2
    assert judge['exact_title_matches'][0]['snapshot_id'] == 'second-results'
    assert judge['exact_title_matches'][0]['page_url'] == second['url']


def scholar_results(title):
    return {'kind': 'browser_ax', 'snapshot_id': 'scholar-results',
            'url': 'https://scholar.google.com/scholar?q=WorldGUI', 'nodes': [
                {'ref': '0', 'role': 'RootWebArea'},
                {'ref': '10', 'parent': '0', 'role': 'generic'},
                {'ref': '11', 'parent': '10', 'role': 'heading', 'tag': 'h3'},
                {'ref': '12', 'parent': '11', 'role': 'link', 'name': title,
                 'href': 'https://publisher.example/paper', 'direct': True, 'in_viewport': False},
                {'ref': '13', 'parent': '10', 'role': 'link', 'name': '[PDF]',
                 'href': 'https://publisher.example/paper.pdf', 'direct': True, 'in_viewport': False}]}


def test_scholar_alias_summary_uses_real_result_and_associated_pdf_refs():
    block = scholar_results(FULL_TITLE)
    policy, judge, refs, _ = summaries([block], 'WorldGUI')
    assert policy[0]['source'] == 'google_scholar'
    assert policy[0]['exact_title_positions'] == [1]
    assert {'12', '13'} <= set(refs)
    matched = judge['exact_title_matches'][0]
    assert matched['source'] == 'google_scholar'
    assert matched['abs_url'] == 'https://publisher.example/paper'
    assert matched['result_ref'] == '12'
    assert matched['pdf_links'] == [{'ref': '13', 'href': 'https://publisher.example/paper.pdf', 'label': '[PDF]'}]
    assert judge['identity_verified'] is False


def test_match_totals_report_bounded_judge_summary_without_claiming_omitted_candidates_absent():
    blocks = [arxiv_results(FULL_TITLE, snapshot=f'results-{i}') for i in range(6)]
    _, judge, _, _ = summaries(blocks, 'WorldGUI')
    assert judge['current_page_count'] == judge['matched_count'] == 6
    assert judge['matches_shown'] == 5 and judge['matches_omitted'] == 1
