from osworld_agent.env import Observation
from osworld_agent.step_verifier import code_review
from osworld_agent.paper_resolver import arxiv_id_from_url
from osworld_agent.actions import Action
from osworld_agent.browser_feedback import search_navigation_error
from osworld_agent.execution_queue import compile_download_milestones


def test_arxiv_query_echo_does_not_complete_search_step():
    step = {'goal': '搜索论文并进入搜索结果列表页', 'expected_evidence': '显示搜索结果',
            'verification': [{'kind': 'url_host', 'host': 'arxiv.org'},
                             {'kind': 'text_present', 'text': 'Attention Is All You Need'}]}
    nodes = [{'ref': '1', 'role': 'textbox', 'name': 'Search arXiv'},
             {'ref': '2', 'parent': '1', 'role': 'StaticText', 'name': 'Attention Is All You Need'}]
    homepage = Observation(context=[{'kind': 'browser_ax', 'url': 'https://arxiv.org/', 'nodes': nodes}])
    assert code_review(step, homepage, {'status': 'executed'}, {})['status'] == 'unknown'
    results = Observation(context=[{'kind': 'browser_ax', 'url': 'https://arxiv.org/search/?query=Attention', 'nodes': nodes}])
    assert code_review(step, results, {'status': 'executed'}, {})['status'] == 'passed'


def test_official_export_mirror_keeps_arxiv_identity():
    assert arxiv_id_from_url('https://export.arxiv.org/abs/1706.03762') == '1706.03762'
    assert arxiv_id_from_url('https://export.arxiv.org/pdf/1706.03762') == '1706.03762'
    assert arxiv_id_from_url('https://export.arxiv.org.attacker.test/pdf/1706.03762') is None


def test_title_search_cannot_be_replaced_by_guessed_paper_url():
    task = '在 arXiv 按名称搜索 Attention Is All You Need，然后下载论文。'
    target = 'https://export.arxiv.org/abs/1706.03762'
    goto = Action('goto', {'url': target})
    search_step = {'phase': 'search'}
    homepage = Observation(context=[{'kind': 'browser_ax', 'url': 'https://arxiv.org/', 'nodes': []}])
    assert search_navigation_error(goto, task, search_step, homepage)

    result = Observation(context=[{'kind': 'browser_ax', 'url': 'https://arxiv.org/search/?query=Attention',
                                   'nodes': [{'role': 'link', 'href': target, 'name': 'Attention Is All You Need'}]}])
    assert not search_navigation_error(goto, task, search_step, result)
    assert not search_navigation_error(goto, task + ' ' + target, search_step, homepage)


def test_search_milestone_checks_submitted_query_and_correct_official_result():
    steps = [{'phase': 'search', 'goal': '提交标题搜索并查看结果',
              'expected_evidence': '显示搜索结果', 'check_ids': ['paper'],
              'verification': [{'kind': 'url_equals', 'url': 'http://127.0.0.1:8090/demo/paper-search'},
                               {'kind': 'visual'}]}]
    goal = {'check_id': 'paper', 'query': 'Attention Is All You Need'}
    step = compile_download_milestones(steps, [goal])[0]
    assert step['verification'] == [{'kind': 'search_results', 'query': goal['query'], 'host': '127.0.0.1'}]

    homepage = Observation(context=[{'kind': 'browser_ax',
        'url': 'http://127.0.0.1:8090/demo/paper-search',
        'nodes': [{'role': 'textbox', 'name': 'Attention Is All You Need'}]}])
    assert code_review(step, homepage, {'status': 'executed'}, {})['status'] == 'unknown'

    result = Observation(context=[{'kind': 'browser_ax',
        'url': 'http://127.0.0.1:8090/demo/paper-search?query=Attention+Is+All+You+Need',
        'nodes': [{'role': 'link', 'name': 'A Survey of Attention Mechanisms',
                   'href': 'https://demo.invalid/pdf/survey'},
                  {'role': 'link', 'name': 'Attention Is All You Need',
                   'href': 'https://export.arxiv.org/abs/1706.03762'}]}])
    assert code_review(step, result, {'status': 'executed'}, {})['status'] == 'passed'


def test_search_milestone_does_not_require_target_in_first_page():
    step = {'phase': 'search', 'goal': '按标题搜索', 'expected_evidence': '显示结果列表',
            'verification': [{'kind': 'search_results', 'query': 'Attention is all you need'}]}
    result = Observation(context=[{'kind': 'browser_ax',
        'url': 'https://arxiv.org/search/?query=Attention+is+all+you+need&searchtype=all',
        'nodes': [{'role': 'link', 'name': 'arXiv:2610.10952',
                   'href': 'https://arxiv.org/abs/2610.10952'}]}])
    review = code_review(step, result, {'status': 'executed'}, {})
    assert review['status'] == 'passed'
    assert '身份待单独核实' in review['checks'][0]['reason']
