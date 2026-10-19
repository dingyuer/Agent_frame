"""Paper semantic selection is auditable without inflating Jev call counts."""
import copy
import json

from fastapi.testclient import TestClient

from osworld_agent.viz.server import _local_traj_to_frontend, app
from test_jev_call_ui_mapping import systemone_call
from test_review_status_ui import review_page


def identity_call(call_id='identity-2', step=2, status='matched'):
    report = {'status': status, 'selected_id': '2502.08047' if status == 'matched' else None,
              'reason': '官方摘要说明这是用户所指的桌面自动化基准。',
              'evidence': [{'candidate_id': '2502.08047', 'field': 'abstract',
                            'quote': 'desktop GUI automation benchmark'}]}
    return {'call_id': call_id, 'call_type': 'paper_identity', 'decision_step': step,
            'model': 'configured-decision-model', 'selection_result': report,
            'messages': [{'role': 'user', 'content': json.dumps({'original_query': 'WorldGUI',
                'candidates': [{'canonical_id': '2502.08047', 'abstract': 'desktop GUI automation benchmark'}]})}],
            'response': json.dumps(report, ensure_ascii=False), 'latency_ms': 750}


def test_history_assigns_selection_to_its_decision_dedupes_and_does_not_count_it_as_jev():
    call = identity_call()
    jev = systemone_call('actual-jev')
    trajectory = {'steps': [{'step': 1}, {'step': 2, 'review_calls': [copy.deepcopy(call)]}],
                  'review_calls': [copy.deepcopy(call)],
                  'events': [{'event_type': 'model_call', 'parent_decision_id': 'a_2', 'call': call},
                             {'event_type': 'paper_identity_review', 'parent_decision_id': 'a_2',
                              'query': 'WorldGUI', 'report': copy.deepcopy(call['selection_result'])},
                             {'event_type': 'model_call', 'parent_decision_id': 'a_2', 'call': jev}]}
    original = copy.deepcopy(trajectory)
    rows = _local_traj_to_frontend(trajectory)
    assert rows[0]['paper_identity_reviews'] == []
    assert len(rows[1]['paper_identity_reviews']) == 1
    entry = rows[1]['paper_identity_reviews'][0]
    assert entry['query'] == 'WorldGUI'
    assert entry['report']['status'] == 'matched'
    assert entry['call']['messages'] == call['messages']
    assert entry['call']['response'] == call['response']
    assert [item['call_id'] for item in rows[1]['jev_calls']] == ['actual-jev']
    assert trajectory == original


def test_no_model_call_diagnostic_still_shows_uncertain_and_does_not_guess_an_orphan_step():
    report = {'status': 'uncertain', 'reason': '论文语义核查模型不可用', 'evidence': []}
    trajectory = {'steps': [{'step': 1}], 'events': [
        {'event_type': 'paper_identity_review', 'query': 'WorldGUI', 'report': report},
        {'event_type': 'paper_identity_review', 'parent_decision_id': 'a_1',
         'query': 'WorldGUI', 'report': report}]}
    row = _local_traj_to_frontend(trajectory)[0]
    assert row['paper_identity_reviews'] == [{'query': 'WorldGUI', 'report': report, 'call': None}]
    assert not row['jev_calls']


def test_live_and_history_selection_redact_credentials_but_keep_citations(monkeypatch):
    from osworld_agent.viz import server
    call = identity_call()
    call['messages'][0]['content'] = json.dumps({'original_query': 'WorldGUI',
                                               'api_key': 'nested-private-secret'})
    call['selection_result']['api_key'] = 'report-private-secret'
    payload = {'kind': 'status', 'event': 'paper_identity_review', 'step': 2,
               'query': 'WorldGUI', 'report': call['selection_result'], 'calls': [call]}
    monkeypatch.setattr(server, 'logs', [])
    with TestClient(app) as client:
        assert client.post('/log', json=payload).json()['ok'] is True
    assert 'private-secret' not in json.dumps(server.logs)
    assert server.logs[0]['report']['evidence'][0]['field'] == 'abstract'
    row = _local_traj_to_frontend({'steps': [{'step': 2}], 'review_calls': [call]})[0]
    assert 'private-secret' not in json.dumps(row)
    assert row['paper_identity_reviews'][0]['query'] == 'WorldGUI'


def test_live_selection_displays_reason_and_folded_evidence_separately_from_jev(review_page):
    from playwright.sync_api import expect
    page = review_page
    call = identity_call(step=1)
    page.evaluate('''call => {
      addCard({kind:'status',event:'task_start',instruction:'论文语义选择UI',max_steps:5,ts:1});
      addCard({kind:'status',event:'step',step:1,ts:1});
      addCard({kind:'status',event:'paper_identity_review',step:1,
        query:'WorldGUI',report:call.selection_result,calls:[call],ts:1});
    }''', call)
    card = page.locator('.compact-step[data-step="1"]').last
    expect(card.locator('.paper-identity-label')).to_have_text('论文选择：符合')
    expect(card.locator('.paper-identity-reason')).to_contain_text('官方摘要')
    assert not card.locator('.paper-identity-details').evaluate('element => element.open')
    assert not card.locator('.paper-identity-model-io').evaluate('element => element.open')
    assert card.locator('.jev-io').is_hidden()
    assert card.locator('.compact-step-effect').is_hidden()
    card.locator('.paper-identity-details').evaluate('element => element.open = true')
    expect(card.locator('.paper-identity-evidence')).to_contain_text('desktop GUI automation benchmark')
    card.locator('.paper-identity-model-io').evaluate('element => element.open = true')
    expect(card.locator('.paper-identity-model-io .prompt')).to_contain_text('original_query')


def test_history_keeps_proposed_action_and_unconfirmed_reason_beside_identity_status(review_page):
    page = review_page
    call = identity_call(status='uncertain')
    report = call['selection_result']
    report['reason'] = '简称可能对应多个项目，需要进一步观察。'
    row = {'step_num': 2, 'action': json.dumps({'action': 'click', 'args': {'target_ref': '113'}}),
           'operation_mode': 'blocked',
           'execution_info': {'dispatched': False, 'status': 'rejected',
                              'error_code': 'paper_identity_unconfirmed', 'reason': '论文身份尚未确认，暂不下载。'},
           'paper_identity_reviews': [{'query': 'WorldGUI', 'report': report, 'call': call}]}
    html = page.evaluate('row => localStepCard("fixture", row)', row)
    assert '论文选择：尚未确认' in html
    assert '拟执行动作' in html and '未派发' in html
    assert '论文身份尚未确认，暂不下载。' in html
    assert 'Jev 输入 / 输出' not in html
    assert '<details class="paper-identity-details">' in html
    assert '<details class="paper-identity-model-io">' in html
    assert '简称可能对应多个项目' in html
    malicious = copy.deepcopy(row)
    malicious['paper_identity_reviews'][0]['report']['reason'] = '<img src=x onerror=alert(1)>'
    safe = page.evaluate('row => localStepCard("fixture", row)', malicious)
    assert '<img src=x onerror=alert(1)>' not in safe
    assert '&lt;img' in safe
