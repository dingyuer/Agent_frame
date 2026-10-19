"""Windows Scholar regressions from a German, broadly ranked result page."""
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

from osworld_agent.actions import Action
from osworld_agent.browser_feedback import operational_facts, unbound_paper_candidate_error
from osworld_agent.browser_session import BrowserSession
from osworld_agent.env import Observation
from osworld_agent.scholar_search import english_scholar_url, scholar_title_links
from osworld_agent.step_verifier import code_review
from test_browser_lifecycle import EventBrowser, session


TITLE = 'Attention Is All You Need'


class PendingPaper:
    def paper_goals(self):
        return [{'check_id': 'paper', 'status': 'pending', 'query': TITLE, 'binding': None}]


def scholar_obs(query='Attention+Is+All+You+Need', *, exact=False):
    nodes = [
        {'ref': '1', 'role': 'textbox', 'id': 'gs_hdr_tsi', 'name': 'Suchen',
         'value': TITLE, 'direct': True},
        {'ref': '2', 'role': 'button', 'id': 'gs_hdr_tsb', 'name': 'Suchen', 'direct': True},
        {'ref': '10', 'role': 'heading', 'tag': 'h3'},
        {'ref': '11', 'parent': '10', 'role': 'link', 'name': 'How to do nothing: Resisting the attention economy',
         'href': 'https://books.google.com/book', 'direct': True},
        {'ref': '20', 'role': 'heading', 'tag': 'h3'},
        {'ref': '21', 'parent': '20', 'role': 'link',
         'name': TITLE if exact else TITLE + ' for llm-based code vulnerability localization',
         'href': 'https://publisher.example/paper', 'direct': True},
    ]
    return Observation(context=[{'kind': 'browser_ax',
        'url': 'https://scholar.google.com/scholar?hl=de&q=' + query,
        'nodes': nodes}])


def test_scholar_url_changes_only_locale_and_not_other_hosts():
    original = 'https://scholar.google.com/scholar?hl=de&as_sdt=0%2C5&q=Attention+Is+All+You+Need&btnG='
    normalized = english_scholar_url(original)
    parts = parse_qs(urlsplit(normalized).query, keep_blank_values=True)
    assert parts == {'as_sdt': ['0,5'], 'q': [TITLE], 'btnG': [''], 'hl': ['en']}
    assert english_scholar_url('https://scholar.google.com.attacker.test/scholar?hl=de') == \
        'https://scholar.google.com.attacker.test/scholar?hl=de'


def test_browser_goto_rewrites_german_scholar_locale(session, monkeypatch):
    browser = EventBrowser()
    session._connect = Mock(return_value=(None, browser))
    monkeypatch.setattr(BrowserSession, '_settle', staticmethod(lambda page: None))
    session.begin_episode('scholar-test')
    result = session.goto('https://scholar.google.com/scholar?hl=de&q=Attention+Is+All+You+Need')
    assert result['status'] == 'executed'
    assert parse_qs(urlsplit(result['url']).query)['hl'] == ['en']
    assert parse_qs(urlsplit(result['url']).query)['q'] == [TITLE]


def test_german_page_uses_ax_ref_and_improves_query_without_forced_quotes():
    obs = scholar_obs()
    facts = '\n'.join(operational_facts(obs, PendingPaper(), {}, {'phase': 'search'}))
    assert 'hl=en' in facts
    assert 'ref=1' in facts and TITLE + ' paper' in facts
    assert '输入带英文双引号' not in facts
    assert 'press Enter' in facts
    assert len(scholar_title_links(obs.context[0])) == 2


def test_search_results_complete_search_but_the_model_must_assess_candidate_identity():
    obs = scholar_obs()
    step = {'verification': [{'kind': 'search_results', 'query': TITLE}]}
    review = code_review(step, obs, {'status': 'executed'}, {})
    assert review['status'] == 'passed'
    assert '由模型' in review['checks'][0]['reason']
    wrong = Action('click', {'target': 'first result', 'target_ref': '11'})
    assert not unbound_paper_candidate_error(
        wrong, obs, PendingPaper(), {'target_ref': '11'})


def test_exact_title_result_can_advance_search_and_be_clicked():
    obs = scholar_obs('%22Attention+Is+All+You+Need%22', exact=True)
    step = {'verification': [{'kind': 'search_results', 'query': TITLE}]}
    assert code_review(step, obs, {'status': 'executed'}, {})['status'] == 'passed'
    good = Action('click', {'target': TITLE, 'target_ref': '21'})
    assert not unbound_paper_candidate_error(good, obs, PendingPaper(), {'target_ref': '21'})
