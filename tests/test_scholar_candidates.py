"""Scholar candidates retain title/PDF association before identity resolution."""
from osworld_agent.scholar_search import scholar_result_cards, scholar_title_links
from osworld_agent.paper_workflow import observed_paper_urls, PaperWorkflow
from osworld_agent.paper_resolver import resolve_arxiv_candidates
from osworld_agent.env import Observation
from osworld_agent.verifiers import CompletionVerifier
from functools import partial
from types import SimpleNamespace


def results():
    nodes = [{'ref': '0', 'role': 'RootWebArea'}]
    for index, title in enumerate(('WorldGUI: A GUI Benchmark', 'A Study Using WorldGUI', 'WorldGUI++: Another Benchmark')):
        row, heading, link, pdf = [str(index * 10 + i) for i in range(1, 5)]
        identifier = '2610.' + str(index + 1).zfill(5)
        nodes.extend([{'ref': row, 'parent': '0', 'role': 'listitem'},
                      {'ref': heading, 'parent': row, 'role': 'heading', 'tag': 'h3', 'name': title},
                      {'ref': link, 'parent': heading, 'role': 'link', 'name': title,
                       'href': 'https://arxiv.org/abs/' + identifier},
                      {'ref': pdf, 'parent': row, 'role': 'link', 'name': '[PDF]',
                       'href': 'https://arxiv.org/pdf/' + identifier}])
    nodes.append({'ref': '99', 'parent': '0', 'role': 'link', 'name': '[PDF]',
                  'href': 'https://arxiv.org/pdf/2610.99999'})
    return {'kind': 'browser_ax', 'url': 'https://scholar.google.com/scholar?q=WorldGUI', 'nodes': nodes}


def test_scholar_titles_and_pdfs_are_associated_per_result_not_by_global_rank():
    block = results()
    cards = scholar_result_cards(block)
    assert len(scholar_title_links(block)) == len(cards) == 3
    for index, card in enumerate(cards):
        assert card['container_ref'] == str(index * 10 + 1)
        assert [pdf['href'] for pdf in card['pdf_links']] == ['https://arxiv.org/pdf/2610.' + str(index + 1).zfill(5)]
    obs = Observation(context=[block], info={'url': block['url']})
    urls = observed_paper_urls(obs, 'WorldGUI')
    assert len(urls) == 6
    assert urls[:2] == ['https://arxiv.org/abs/2610.00001', 'https://arxiv.org/pdf/2610.00001']
    assert observed_paper_urls(obs, 'World') == urls  # All candidates reach semantic selection.


def test_wrong_pdf_in_same_container_and_orphan_pdf_do_not_become_candidates():
    block = results()
    block['nodes'][4]['href'] = 'https://arxiv.org/pdf/2610.00002'
    card = scholar_result_cards(block)[0]
    assert card['pdf_links'] == []
    obs = Observation(context=[block], info={'url': block['url']})
    urls = observed_paper_urls(obs, 'WorldGUI')
    assert urls[0] == 'https://arxiv.org/abs/2610.00001' and len(urls) == 5
    assert 'https://arxiv.org/pdf/2610.99999' not in urls


def test_scholar_missing_result_wrappers_does_not_pair_a_page_pdf_with_a_title():
    block = results()
    block['nodes'] = [node for node in block['nodes'] if node['ref'] not in {'1', '11', '21'}]
    for node in block['nodes']:
        if node.get('parent') in {'1', '11', '21'}:
            node['parent'] = '0'
    assert all(not card['pdf_links'] for card in scholar_result_cards(block))


def test_matching_scholar_candidate_still_requires_verified_metadata_and_never_downloads():
    block = results()
    obs = Observation(context=[block], info={'url': block['url']})
    session = SimpleNamespace(calls=[])
    env = SimpleNamespace(_browser_session=session)
    verifier = CompletionVerifier('请下载一篇叫做WorldGUI的论文')
    requested = []
    def fetch(url, **kwargs):
        requested.append(url)
        return {'url': url, 'text': '<h1>WorldGUI: A GUI Benchmark</h1>'}
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=fetch))
    events = []
    workflow.advance(obs, verifier, env, lambda name, **data: events.append((name, data)))
    assert requested == ['https://arxiv.org/abs/2610.00001', 'https://arxiv.org/abs/2610.00002',
                         'https://arxiv.org/abs/2610.00003']
    assert not verifier.bindings and not session.calls
    assert events[0][1]['status'] == 'unavailable'
    assert events[0][1]['reason_code'] == 'metadata_invalid'
