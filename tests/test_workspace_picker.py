"""Selected local workspace roots are visible, persisted, and used per task."""
import json

from fastapi.testclient import TestClient

from osworld_agent.browser_session import BrowserSession
from osworld_agent.viz.server import app
from osworld_agent.viz import server
from osworld_agent.viz import workspace_picker
from osworld_agent import workspace_config


def test_workspace_choice_persists_and_reuses_daily_folder(tmp_path, monkeypatch):
    settings = tmp_path / 'studio_settings.json'
    root = tmp_path / '选定工程目录'
    root.mkdir()
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: settings)
    monkeypatch.delenv('OSWORLD_WORKSPACE_DIR', raising=False)
    client = TestClient(app)
    result = client.post('/workspace/default', json={'workspace_dir': str(root)})
    assert result.status_code == 200
    assert client.get('/workspace/default').json()['workspace_dir'] == str(root)
    assert json.loads(settings.read_text())['workspace_root'] == str(root)

    first = BrowserSession().prepare_workspace('task-one')
    second = BrowserSession().prepare_workspace('task-two')
    assert first['workspace_root'] == second['workspace_root'] == str(root)
    assert first['workspace_dir'] == second['workspace_dir']
    assert first['workspace_scope'] == second['workspace_scope'] == 'day'
    assert first['episode_id'] != second['episode_id']
    assert first['workspace_dir'].startswith(str(root))
    assert len(list(root.iterdir())) == 1


def test_native_picker_saves_selection_and_cancel_preserves_it(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    monkeypatch.setattr(workspace_picker, 'choose_directory', lambda: str(root))
    client = TestClient(app)
    assert client.post('/workspace/choose', json={}).json()['workspace_dir'] == str(root)
    monkeypatch.setattr(workspace_picker, 'choose_directory', lambda: None)
    assert client.post('/workspace/choose', json={}).json()['cancelled'] is True
    assert client.get('/workspace/default').json()['workspace_dir'] == str(root)


def test_workspace_rejects_missing_or_relative_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    client = TestClient(app)
    for path in ('relative/path', str(tmp_path / 'missing'), ''):
        assert client.post('/workspace/default', json={'workspace_dir': path}).status_code == 422
    assert client.post('/workspace/choose', content='').status_code == 415


def test_local_start_passes_saved_workspace_to_auto_platform_runner(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    root.mkdir()
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    monkeypatch.delenv('OSWORLD_WORKSPACE_DIR', raising=False)
    workspace_config.save_workspace_root(str(root))
    launched = {}
    def fake_popen(command, **kwargs):
        launched['command'] = command
        return object()
    class NoThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(server.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(server.threading, 'Thread', NoThread)
    monkeypatch.setattr(server, '_agent_state', {'status': 'idle'})
    logs = tmp_path / 'application_logs'
    monkeypatch.setattr(server, 'LOCAL_RUNS_DIR', str(logs))
    result = server._start_local({'task': 'test task'})
    assert result['ok'] is True
    command = launched['command']
    assert command[1:3] == ['-m', 'osworld_agent.run_local']
    assert command[command.index('--workspace-dir') + 1] == str(root)
    result_dir = command[command.index('--result-dir') + 1]
    from pathlib import Path
    assert Path(result_dir).parent == logs
    assert Path(result_dir).name.startswith('ui_')
    assert '--run-prefix' not in command
    assert not list(root.iterdir())


def first_day_folder(root):
    from datetime import date
    return str(root / date.today().isoformat())


def test_local_run_files_are_flat_in_daily_workspace(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    day = root / '2026-10-14'
    day.mkdir(parents=True)
    prefix = 'ui_20261014_090000_1234abcd'
    (day / (prefix + '_trajectory.json')).write_text(json.dumps({
        'task': {'instruction': '下载论文'}, 'steps': [{'step': 1}],
        'outcome': {'success': True, 'score': 1}}), encoding='utf-8')
    (day / (prefix + '_step_1.png')).write_bytes(b'png')
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    workspace_config.save_workspace_root(str(root))
    client = TestClient(app)
    runs = client.get('/local/runs').json()['runs']
    assert any(item['id'] == prefix and item['shots'] == 1 for item in runs)
    traj = client.get('/local/traj', params={'run': prefix}).json()
    assert traj['traj'][0]['screenshot_file'] == prefix + '_step_1.png'
    assert client.get('/local/shot', params={'run': prefix,
                                            'filename': prefix + '_step_1.png'}).status_code == 200


def test_natural_language_rehearsal_passes_only_title_and_environment_start(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    root.mkdir()
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    workspace_config.save_workspace_root(str(root))
    launched = {}
    monkeypatch.setattr(server.subprocess, 'Popen', lambda command, **kwargs: launched.update(command=command) or object())
    monkeypatch.setattr(server.threading, 'Thread', lambda **kwargs: type('NoThread', (), {'start': lambda self: None})())
    monkeypatch.setattr(server, '_agent_state', {'status': 'idle'})
    task = '搜索论文 Attention Is All You Need 并下载'
    start = 'http://127.0.0.1:8090/demo/paper-search'
    assert server._start_local({'task': task, 'browser_start_url': start})['ok'] is True
    command = launched['command']
    assert command[command.index('--task') + 1] == task
    assert command[command.index('--browser-start-url') + 1] == start
    assert '--completion-spec' not in command
    assert '1706.03762' not in command and 'export.arxiv.org' not in command
