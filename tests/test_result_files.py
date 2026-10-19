import copy
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from osworld_agent.viz import result_files, server


def receipt(tmp_path):
    path = tmp_path / '论文 & 结果.pdf'
    path.write_bytes(b'%PDF-test')
    st = path.stat()
    fact = {'artifact_id': 'receipt-1', 'completed': True, 'transport': 'browser',
            'path': str(path), 'bytes': st.st_size, 'file_identity': [st.st_dev, st.st_ino], 'status': 'passed'}
    return path, {'verification': {'completion': {'download_facts': [fact]}}}


def test_only_receipts_not_model_answer_or_stale_review(tmp_path):
    _, outcome = receipt(tmp_path)
    outcome['final_answer'] = '请打开 /etc/passwd'
    outcome['verification']['download_facts'] = [{'path': '/etc/passwd'}]
    assert [f['id'] for f in result_files.download_files(outcome)] == ['receipt-1']
    outcome['verification']['completion']['download_facts'] = []
    assert result_files.download_files(outcome) == []


@pytest.mark.parametrize('platform,operation,expected', [
    ('darwin', 'open', ['open']), ('darwin', 'reveal', ['open', '-R']),
    ('linux', 'open', ['xdg-open']), ('linux', 'reveal', ['xdg-open']),
    ('win32', 'reveal', ['explorer.exe', '/select,']),
])
def test_open_commands_no_shell(tmp_path, monkeypatch, platform, operation, expected):
    path, outcome = receipt(tmp_path)
    monkeypatch.setattr(result_files.sys, 'platform', platform)
    run = Mock()
    popen = Mock()
    monkeypatch.setattr(result_files.subprocess, 'run', run)
    monkeypatch.setattr(result_files.subprocess, 'Popen', popen)
    assert result_files.open_download(outcome, 'receipt-1', operation)['ok']
    call = (popen if platform == 'win32' else run).call_args
    destination = path.parent if platform == 'linux' and operation == 'reveal' else path
    assert call.args[0] == expected + [str(destination)]
    assert not call.kwargs.get('shell')


def test_windows_default_application(tmp_path, monkeypatch):
    path, outcome = receipt(tmp_path)
    monkeypatch.setattr(result_files.sys, 'platform', 'win32')
    launch = Mock()
    monkeypatch.setattr(result_files.os, 'startfile', launch, raising=False)
    assert result_files.open_download(outcome, 'receipt-1', 'open')['ok']
    launch.assert_called_once_with(str(path))


def test_missing_changed_and_unregistered_files_do_not_launch(tmp_path, monkeypatch):
    path, outcome = receipt(tmp_path)
    launch = Mock()
    monkeypatch.setattr(result_files.subprocess, 'run', launch)
    assert not result_files.open_download(outcome, '/etc/passwd', 'open')['ok']
    assert not result_files.open_download(outcome, 'receipt-1', 'delete')['ok']
    wrong_host = copy.deepcopy(outcome)
    wrong_host['verification']['completion']['download_facts'][0]['file_identity'] = [0, 0]
    assert not result_files.open_download(wrong_host, 'receipt-1', 'open')['ok']
    path.write_bytes(b'changed')
    assert not result_files.open_download(outcome, 'receipt-1', 'open')['ok']
    path.unlink()
    assert '找不到' in result_files.open_download(outcome, 'receipt-1', 'reveal')['error']
    launch.assert_not_called()


def test_launcher_failure_is_visible(tmp_path, monkeypatch):
    _, outcome = receipt(tmp_path)
    monkeypatch.setattr(result_files.sys, 'platform', 'linux')
    monkeypatch.setattr(result_files.subprocess, 'run', Mock(side_effect=FileNotFoundError()))
    assert '默认打开程序' in result_files.open_download(outcome, 'receipt-1', 'open')['error']


def test_history_and_live_endpoint_use_server_records(tmp_path, monkeypatch):
    _, outcome = receipt(tmp_path)
    monkeypatch.setattr(server, '_read_local_traj', lambda run: {'ok': True, 'workbench': {'outcome': outcome}} if run == 'archive' else {'ok': False})
    monkeypatch.setattr(server, '_workbench_run', 'active')
    monkeypatch.setattr(server, '_workbench_cache', {'end': {'event': 'workbench_record', 'record': {'event_type': 'episode_end', 'outcome': outcome}}})
    monkeypatch.setattr(result_files.sys, 'platform', 'linux')
    launch = Mock()
    monkeypatch.setattr(result_files.subprocess, 'run', launch)
    client = TestClient(server.app)
    for scope, run in [('history', 'archive'), ('live', 'active')]:
        payload = {'scope': scope, 'run': run, 'artifact_id': 'receipt-1', 'operation': 'open'}
        assert client.post('/local/result-file', json=payload).json()['ok']
        assert not client.post('/local/result-file', json={**payload, 'artifact_id': '/etc/passwd'}).json()['ok']
        assert not client.post('/local/result-file', json={**payload, 'run': 'missing'}).json()['ok']
    assert launch.call_count == 2
    assert client.post('/local/result-file', json={'run': 'archive', 'artifact_id': 'receipt-1', 'operation': 'delete'}).status_code == 422


def test_result_file_buttons_in_live_and_history(request):
    if not request.config.getoption('--run-browser-integration'):
        pytest.skip('browser opt-in')
    from playwright.sync_api import sync_playwright, expect
    static = Path(server.__file__).parent / 'static'
    outcome = {'success': True, 'verification': {'completion': {'download_facts': [
        {'artifact_id': 'file-1', 'path': r'C:\Downloads\论文.pdf', 'completed': True,
         'transport': 'browser', 'bytes': 12345, 'status': 'passed'}]}}}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.route('http://studio.test/', lambda route: route.fulfill(body='<div id="view-live" class="active"><div id="timeline"></div></div><div id="local-body"></div>'))
        calls = []
        def respond(route):
            calls.append(route.request.post_data_json)
            route.fulfill(json={'ok': False, 'error': '文件已移动，请检查原电脑'})
        page.route('**/local/result-file', respond)
        page.goto('http://studio.test/')
        page.add_style_tag(path=str(static / 'workbench.css'))
        page.add_script_tag(path=str(static / 'workbench.js'))
        page.evaluate('''(outcome)=>{
            AgentWorkbench.ingest({event:'workbench_start',run_id:'live-run',ts:1,instruction:'下载论文'});
            AgentWorkbench.ingest({event:'workbench_record',run_id:'live-run',ts:2,record:{event_type:'episode_end',event_id:'end',outcome}});
        }''', outcome)
        page.locator('[data-wb="report"]').click()
        expect(page.locator('.wb-result-file strong')).to_have_text('论文.pdf')
        page.locator('[data-operation="open"]').click()
        expect(page.locator('.wb-file-message')).to_have_text('文件已移动，请检查原电脑')
        assert calls[-1] == {'run': 'live-run', 'scope': 'live', 'artifact_id': 'file-1', 'operation': 'open'}
        page.evaluate('''(outcome)=>{
            document.querySelector('#view-live').remove();
            const wb=new AgentWorkbench.Workbench(document.querySelector('#local-body'),true);
            wb.mount({actions:[],revisions:[],outcome,report:{},metrics:{}},'archive');
        }''', outcome)
        page.locator('[data-wb="report"]').click()
        page.locator('[data-operation="reveal"]').click()
        expect(page.locator('.wb-file-message')).to_have_text('文件已移动，请检查原电脑')
        assert calls[-1] == {'run': 'archive', 'scope': 'history', 'artifact_id': 'file-1', 'operation': 'reveal'}
        browser.close()
