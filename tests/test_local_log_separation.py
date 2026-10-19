"""Task files share a dated workspace; runtime logs have independent run folders."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from osworld_agent import workspace_config
from osworld_agent.viz import server


def test_log_root_is_independent_of_workspace_choice(tmp_path, monkeypatch):
    logs = tmp_path / 'program' / 'local_runs'
    monkeypatch.setenv('OSWORLD_LOCAL_RUNS_DIR', str(logs))
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    initial = workspace_config.default_local_runs_root()
    first_workspace, second_workspace = tmp_path / 'workspace-one', tmp_path / 'workspace-two'
    first_workspace.mkdir()
    second_workspace.mkdir()
    workspace_config.save_workspace_root(str(first_workspace))
    first = Path(workspace_config.allocate_local_result_dir())
    workspace_config.save_workspace_root(str(second_workspace))
    second = Path(workspace_config.allocate_local_result_dir())
    assert workspace_config.default_local_runs_root() == initial == str(logs)
    assert first.parent == second.parent == logs
    assert first != second and first.is_dir() and second.is_dir()
    assert not list(first_workspace.iterdir()) and not list(second_workspace.iterdir())


@pytest.mark.parametrize('explicit', [False, True])
def test_cli_default_logs_are_separate_and_explicit_result_dir_still_wins(tmp_path, monkeypatch, explicit):
    from osworld_agent import run_windows_local as runner
    from test_incognito_lifecycle import _stub_main
    workspace = tmp_path / 'task-files'
    workspace.mkdir()
    previous = workspace / 'existing.txt'
    previous.write_text('keep this file')
    logs = tmp_path / 'program_logs'
    monkeypatch.setenv('OSWORLD_LOCAL_RUNS_DIR', str(logs))
    _stub_main(monkeypatch, str(tmp_path / 'owned-temp-profile'))
    sys.argv += ['--workspace-dir', str(workspace)]
    selected = tmp_path / 'explicit-results'
    if explicit:
        sys.argv += ['--result-dir', str(selected)]
    monkeypatch.setattr(runner.time, 'sleep', lambda _:None)
    written = []
    monkeypatch.setattr(runner, 'set_results_dir', lambda directory, prefix:written.append(Path(directory)))
    assert runner.main() == 0
    actual = written[0]
    assert actual == selected if explicit else actual.parent == logs
    assert (actual / 'trajectory.json').is_file()
    assert (actual / 'traj_raw.jsonl').is_file()
    assert list(workspace.iterdir()) == [previous]
    assert previous.read_text() == 'keep this file'
    if explicit:
        assert not logs.exists()


def test_studio_new_run_logs_remain_in_history_after_workspace_change(tmp_path, monkeypatch):
    logs = tmp_path / 'local_runs'
    workspace = tmp_path / 'files'
    workspace.mkdir()
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    workspace_config.save_workspace_root(str(workspace))
    monkeypatch.setattr(server, 'LOCAL_RUNS_DIR', str(logs))
    monkeypatch.setattr(server, '_agent_state', {'status':'idle'})
    real_thread = server.threading.Thread
    monkeypatch.setattr(server.threading, 'Thread', lambda **kwargs:SimpleNamespace(start=lambda:None))
    directories = []
    def start(command, **kwargs):
        directory = Path(command[command.index('--result-dir') + 1])
        directories.append(directory)
        assert '--run-prefix' not in command
        (directory / 'trajectory.json').write_text(json.dumps({
            'task':{'instruction':'download a file'}, 'steps':[{'step':1}],
            'outcome':{'success':True, 'score':1}}))
        (directory / 'step_1.png').write_bytes(b'png')
        return object()
    monkeypatch.setattr(server.subprocess, 'Popen', start)
    assert server._start_local({'task':'下载文件'})['ok']
    assert server._start_local({'task':'下载另一个文件'})['ok']
    assert directories[0] != directories[1]
    assert all(directory.parent == logs for directory in directories)
    assert not list(workspace.iterdir())
    new_workspace = tmp_path / 'other-files'
    new_workspace.mkdir()
    workspace_config.save_workspace_root(str(new_workspace))
    monkeypatch.setattr(server.threading, 'Thread', real_thread)
    client = TestClient(server.app)
    run_ids = {run['id'] for run in client.get('/local/runs').json()['runs']}
    for directory in directories:
        assert directory.name in run_ids
        response = client.get('/local/traj', params={'run':directory.name}).json()
        assert response['ok'] and response['traj'][0]['screenshot_file'] == 'step_1.png'
        assert client.get('/local/shot', params={'run':directory.name, 'filename':'step_1.png'}).status_code == 200


def test_studio_preserves_explicit_result_directory(tmp_path, monkeypatch):
    workspace = tmp_path / 'files'
    workspace.mkdir()
    selected = tmp_path / 'chosen-logs'
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'settings.json')
    workspace_config.save_workspace_root(str(workspace))
    logs = tmp_path / 'default_logs'
    monkeypatch.setattr(server, 'LOCAL_RUNS_DIR', str(logs))
    monkeypatch.setattr(server, '_agent_state', {'status':'idle'})
    monkeypatch.setattr(server.threading, 'Thread', lambda **kwargs:SimpleNamespace(start=lambda:None))
    command = []
    monkeypatch.setattr(server.subprocess, 'Popen', lambda args, **kwargs:command.extend(args) or object())
    assert server._start_local({'task':'下载文件', 'result_dir':str(selected)})['ok']
    assert command[command.index('--result-dir') + 1] == str(selected)
    assert not logs.exists()


def test_new_log_directory_can_use_an_explicit_filename_prefix(tmp_path, monkeypatch):
    logs = tmp_path / 'local_runs'
    directory = Path(workspace_config.allocate_local_result_dir(logs))
    prefix = 'named_run_'
    (directory / (prefix + 'trajectory.json')).write_text(json.dumps({
        'task':{'instruction':'download'}, 'steps':[{'step':1}], 'outcome':{}}))
    (directory / (prefix + 'step_1.png')).write_bytes(b'png')
    monkeypatch.setattr(server, 'LOCAL_RUNS_DIR', str(logs))
    client = TestClient(server.app)
    assert any(run['id'] == directory.name for run in client.get('/local/runs').json()['runs'])
    result = client.get('/local/traj', params={'run':directory.name}).json()
    assert result['traj'][0]['screenshot_file'] == prefix + 'step_1.png'
    assert client.get('/local/shot', params={'run':directory.name, 'filename':prefix + 'step_1.png'}).status_code == 200
