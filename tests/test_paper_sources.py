"""Search-entry preferences never replace user source or paper identity."""
import pytest

from osworld_agent.paper_sources import (ARXIV_SEARCH_URL, default_paper_start_url,
    paper_search_instruction, paper_search_policy)
from osworld_agent.paper_search import PaperSearchRecovery
from osworld_agent.actions import Action
from osworld_agent.browser_feedback import search_navigation_error
from test_paper_search_recovery import obs


@pytest.mark.parametrize('task', [
    '搜索论文WorldGUI并下载。', '下载Attention is all you need论文',
    'Search and download the WorldGUI paper',
    '下载《Learning from Google Scholar》论文',
    '下载Google发布的论文WorldGUI', '搜索WorldGUI论文并下载，不要用谷歌学术',
])
def test_unspecified_source_gets_only_a_search_entry(task):
    policy = paper_search_policy(task)
    assert not policy['explicit'] and policy['preferred_source'] == 'arxiv'
    assert policy['search_url'] == ARXIV_SEARCH_URL
    assert default_paper_start_url(task, is_paper_download=True) == ARXIV_SEARCH_URL
    assert '/abs/' not in paper_search_instruction(task)
    assert '/pdf/' not in paper_search_instruction(task)


@pytest.mark.parametrize('task,source', [
    ('用谷歌学术搜索WorldGUI论文并下载', 'scholar'),
    ('在Google Scholar搜索WorldGUI并下载论文', 'scholar'),
    ('谷歌学术搜索WorldGUI论文并下载', 'scholar'),
    ('Search Google Scholar for the WorldGUI paper and download it', 'scholar'),
    ('从Nature官网下载论文WorldGUI', 'nature'),
    ('在OpenReview查找论文WorldGUI并下载', 'openreview'),
    ('从Semantic Scholar下载WorldGUI论文', 'semantic_scholar'),
    ('使用谷歌搜索WorldGUI论文并下载', 'google'),
    ('只在arXiv查找WorldGUI论文并下载', 'arxiv'),
    ('不要用arxiv而使用谷歌学术搜索WorldGUI论文并下载', 'scholar'),
])
def test_named_sources_are_not_overridden(task, source):
    policy = paper_search_policy(task)
    assert policy['explicit'] and policy['preferred_source'] == source
    assert default_paper_start_url(task, is_paper_download=True) is None


@pytest.mark.parametrize('task,url', [
    ('在 https://openreview.net/ 搜索WorldGUI论文并下载', 'https://openreview.net/'),
    ('从https://example.org/papers下载WorldGUI论文。', 'https://example.org/papers'),
    ('在 https://arxiv.org/abs/2502.08047 找到WorldGUI并下载论文', 'https://arxiv.org/abs/2502.08047'),
])
def test_original_urls_remain_user_supplied_destinations(task, url):
    policy = paper_search_policy(task)
    assert policy['explicit'] and policy['search_url'] == url
    assert default_paper_start_url(task, is_paper_download=True) is None


@pytest.mark.parametrize('task', [
    '从某大学论文数据库搜索WorldGUI论文并下载',
    '在NeurIPS搜索WorldGUI论文并下载',
    '在该论文官方项目页查找并下载论文',
    '搜索WorldGUI论文并下载，不要用arxiv',
])
def test_unknown_source_requirements_are_left_to_planner(task):
    assert paper_search_policy(task)['explicit']
    assert default_paper_start_url(task, is_paper_download=True) is None


def test_explicit_initial_environment_and_nonpaper_tasks_are_retained():
    for initial in ['http://127.0.0.1:8090/demo/paper-search', 'https://scholar.google.com/', 'about:blank']:
        assert default_paper_start_url('下载WorldGUI论文', is_paper_download=True, browser_start_url=initial) == initial
    assert default_paper_start_url('下载一份PPT', is_paper_download=False) is None
    assert default_paper_start_url('阅读WorldGUI论文', is_paper_download=False) is None


def test_source_restrictions_survive_recovery_after_no_candidates():
    task = '只在谷歌学术搜索WorldGUI论文并下载'
    recovery = PaperSearchRecovery(task=task)
    recovery.facts(obs(empty=True), 'WorldGUI')
    advice = ' '.join(recovery.facts(obs('WorldGUI paper', empty=True), 'WorldGUI'))
    assert '不得' in advice and '换搜索入口' not in advice
    assert ARXIV_SEARCH_URL not in advice
    assert paper_search_policy(task)['exclusive']


def test_login_recovery_does_not_override_explicit_scholar():
    class Completion:
        def paper_goals(self):
            return [{'query': 'WorldGUI', 'binding': None}]
    task = '用谷歌学术搜索WorldGUI论文并下载'
    feedback = search_navigation_error(Action('goto', {'url': 'https://arxiv.org/submit'}),
        task, {'phase': 'search'}, obs(), Completion())
    assert 'scholar.google.com' in feedback and ARXIV_SEARCH_URL not in feedback
