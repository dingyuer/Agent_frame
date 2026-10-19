"""Real local Chromium search recovery, official-host fixtures and PDF evidence.

Every request is fulfilled or aborted locally. The synthetic future ID is only
an observed fixture identity, never a planner hint or an actual research paper.
"""
import json
from html import escape
from urllib.parse import parse_qs, urlsplit

from osworld_agent.actions import Action
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.browser_session import browser_thread
from osworld_agent.paper_workflow import PaperWorkflow, observed_paper_urls
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.scholar_search import scholar_result_cards
from osworld_agent.verifiers import CompletionVerifier
from test_browser_live import live_browser, wait_for
from test_paper_completion import pdf_file


IDENTIFIER = '9912.00000'  # Synthetic future record; never requested from arXiv.
TITLE = 'WorldGUI: A Synthetic Desktop Benchmark'
DETAIL = 'https://arxiv.org/abs/' + IDENTIFIER
PDF = 'https://arxiv.org/pdf/' + IDENTIFIER


def evaluate(session, expression):
    return browser_thread(lambda owner: owner._pick_page(owner._connect()[1]).evaluate(expression))(session)


class ScriptedModel:
    vision = False

    def __init__(self, actions):
        self.actions, self.calls = list(actions), []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return json.dumps({'next_action': self.actions.pop(0), 'summary': 'Fixture decision'})


def execute(pipe, action, observation):
    state = StepState(task='下载一篇叫WorldGUI的论文', obs=observation)
    resolved = pipe.ground(action, state)
    execution = pipe.execute(resolved, state)
    assert execution.info.get('status') == 'executed', execution.info
    return execution


def test_live_short_name_search_repairs_retyping_then_binds_and_downloads(live_browser, tmp_path, monkeypatch):
    session, _ = live_browser
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    body = pdf_file(tmp_path / 'source.pdf', TITLE, 'Fixture Author', 'Abstract',
                    'This paper and its identifier are synthetic.').read_bytes()
    requests = []

    async def respond(route):
        url = route.request.url
        parsed = urlsplit(url)
        requests.append(url)
        if parsed.hostname == 'scholar.google.com' and parsed.path == '/scholar':
            query = parse_qs(parsed.query).get('q', [''])[0]
            results = ''
            if query:
                results = ('<ol><li><h3><a id=paper href="' + DETAIL + '">' + TITLE + '</a></h3>'
                           '<a id=paper_pdf href="' + PDF + '">[PDF]</a></li>'
                           '<li><h3><a href="https://arxiv.org/abs/9912.99999">A Study Using WorldGUI</a></h3>'
                           '<a href="https://arxiv.org/pdf/9912.99999">[PDF]</a></li></ol>')
            html = ('<meta charset=utf-8><title>Local Scholar fixture</title>'
                    '<form action=/scholar method=get><label for=gs_hdr_tsi>Suchen</label>'
                    '<input id=gs_hdr_tsi name=q value="' + escape(query, quote=True) + '">'
                    '<input type=hidden name=hl value=en><button id=gs_hdr_tsb>Search</button></form>'
                    '<script>window.inputEvents=0;document.getElementById("gs_hdr_tsi")'
                    '.addEventListener("input",()=>inputEvents++)</script>' + results)
            await route.fulfill(status=200, content_type='text/html', body=html)
        elif url == DETAIL:
            html = ('<meta charset=utf-8><title>Local official-record fixture</title>'
                    '<meta name=citation_title content="' + TITLE + '">'
                    '<meta name=citation_author content="Fixture Author">'
                    '<meta name=citation_arxiv_id content="' + IDENTIFIER + '">'
                    '<meta name=citation_pdf_url content="' + PDF + '">'
                    '<h1>' + TITLE + '</h1><a id=download href="' + PDF + '">Download PDF</a>')
            await route.fulfill(status=200, content_type='text/html', body=html)
        elif url == PDF:
            await route.fulfill(status=200, content_type='application/pdf', body=body,
                                headers={'Content-Disposition': 'attachment; filename="WorldGUI.pdf"'})
        else:
            await route.abort()  # No public network, distractor or favicon traffic.

    browser_thread(lambda owner: owner._owned_context.route('**/*', respond))(session)
    assert session.goto('https://scholar.google.com/scholar?hl=en')['status'] == 'executed'
    env = WindowsEnvironment()
    env._browser_session = session
    env._runtime_state = lambda: {}
    verifier = CompletionVerifier('下载一篇叫WorldGUI的论文')
    pipe = Pipeline(model=ScriptedModel([]), env=env, grounding=None, system_prompt='fixture', max_retries=3)
    pipe.completion = verifier

    first = env.observe()
    search = next(n for n in first.context[0]['nodes'] if n.get('id') == 'gs_hdr_tsi')
    typed = execute(pipe, Action('type', {'target': 'Suchen', 'target_ref': search['ref'],
                                        'text': 'WorldGUI', 'overwrite': True}), first)
    assert evaluate(session, 'document.getElementById("gs_hdr_tsi").value') == 'WorldGUI'
    input_count = evaluate(session, 'window.inputEvents')
    assert input_count == 1
    current = typed.obs
    search = next(n for n in current.context[0]['nodes'] if n.get('id') == 'gs_hdr_tsi')
    pipe.model = ScriptedModel([
        {'type': 'type', 'target': 'Suchen', 'target_ref': search['ref'], 'text': 'WorldGUI'},
        {'type': 'type', 'target': 'Suchen', 'target_ref': search['ref'], 'text': 'WorldGUI', 'overwrite': True},
        {'type': 'press', 'key': 'Enter'}])
    state = StepState(task='下载一篇叫WorldGUI的论文', obs=current)
    decision = pipe.decide([{'role': 'user', 'content': 'Submit the filled search field'}], state)
    assert decision.action.action == 'press'
    assert [call.get('validation_error_type') for call in decision.calls[:2]] == ['redundant_search_action'] * 2
    assert evaluate(session, 'window.inputEvents') == input_count
    assert evaluate(session, 'document.getElementById("gs_hdr_tsi").value') == 'WorldGUI'
    submitted = execute(pipe, decision.action, current).obs
    assert parse_qs(urlsplit(submitted.info['url']).query)['q'] == ['WorldGUI']
    assert len([url for url in requests if urlsplit(url).hostname == 'scholar.google.com'
                and parse_qs(urlsplit(url).query).get('q') == ['WorldGUI']]) == 1
    cards = scholar_result_cards(submitted.context[0])
    target = next(card for card in cards if card['title'] == TITLE)
    assert [link['href'] for link in target['pdf_links']] == [PDF]
    assert observed_paper_urls(submitted, 'WorldGUI') == [DETAIL, PDF]
    assert not verifier.bindings

    opened = execute(pipe, Action('click', {'target': TITLE, 'target_ref': target['ref']}), submitted).obs
    assert opened.info['url'] == DETAIL and opened.info.get('paper_metadata')
    workflow = PaperWorkflow(resolver=None)
    # Trusted metadata must remove the need for the resolver's external fetch.
    monkeypatch.setattr('osworld_agent.paper_resolver._fetch_official',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('No external identity request')))
    workflow.advance(opened, verifier, env, lambda *a, **k: None)
    assert verifier.bindings and not session.event_metadata()['artifacts']
    download = next(n for n in opened.context[0]['nodes'] if n.get('id') == 'download')
    execute(pipe, Action('click', {'target': 'Download PDF', 'target_ref': download['ref']}), opened)
    try:
        finished = wait_for(session, lambda obs: bool(obs.info.get('artifacts')))
    except AssertionError as error:
        raise AssertionError({'requests': requests, 'events': session.event_metadata()}) from error
    verifier.observe(finished)
    assert verifier.evaluate()['status'] == 'passed'
    artifact = finished.info['artifacts'][0]
    assert artifact['url'] == PDF and artifact['bytes'] == len(body) and artifact['sha256']
    assert len([url for url in requests if url == PDF]) == 1
