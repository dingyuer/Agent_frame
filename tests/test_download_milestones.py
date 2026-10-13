"""Download progress is deterministic, distinct from final naming acceptance."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from osworld_agent.actions import Action
from osworld_agent.browser_session import BrowserSession, BrowserOptions
from osworld_agent.execution_queue import compile_download_milestones, ExecutionQueue, normalize_verification
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import Execution, StepState
from osworld_agent.step_verifier import code_review
from osworld_agent.verifiers import CompletionVerifier
from test_browser_operational_feedback import pipe, bound
from test_browser_lifecycle import session
from test_paper_goal_binding import official_event
from test_paper_completion import pdf_file, event
from test_paper_resolver import TITLE, IDENTIFIER
from test_pipeline_completion import observation


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)


def downloads(tmp_path):
    verifier = bound()
    wrong = pdf_file(tmp_path / '978-book.pdf', 'Unrelated book', 'Someone', 'Abstract')
    right = pdf_file(tmp_path / (IDENTIFIER + 'v3.pdf'), TITLE, 'Example Author', 'Abstract')
    events = [event(wrong, ident='wrong', url='https://publisher.test/book.pdf'), official_event(right, ident='right')]
    obs = observation('https://arxiv.org/pdf/' + IDENTIFIER)
    obs.info['artifacts'] = events
    verifier.observe(obs)
    return verifier, obs, right


def plan(verifier):
    return compile_download_milestones([
        {'idx': 1, 'phase': 'download', 'goal': 'Download then rename', 'expected_evidence': 'saved',
         'verification': [{'kind': 'visual'}], 'check_ids': ['check_1']},
        {'idx': 2, 'phase': 'download', 'goal': 'Click viewer save', 'verification': [{'kind': 'visual'}]},
        {'idx': 3, 'phase': 'verify', 'goal': 'Final acceptance', 'verification': [], 'check_ids': ['check_1']}], verifier.paper_goals())


def test_correct_download_beats_old_wrong_artifact_and_ignores_only_filename(tmp_path):
    verifier, obs, right = downloads(tmp_path)
    original = copy.deepcopy(verifier.public_spec)
    final = verifier.public_report(verifier.evaluate())
    assert final['status'] == 'failed'
    assert final['checks'][0]['reason'] == 'downloaded filename does not match'
    assert final['checks'][0]['path'] == str(right)
    progress = verifier.download_progress()
    assert progress[0]['status'] == 'passed' and progress[0]['path'] == str(right)
    assert verifier.public_spec == original  # milestone never weakens final contract
    steps = plan(verifier)
    assert [s['phase'] for s in steps] == ['download', 'rename', 'verify']
    assert steps[0]['check_ids'] == [] and steps[0]['verification'] == [{'kind': 'download_complete', 'check_id': 'check_1'}]
    assert 'data.pdf' in steps[1]['goal'] and steps[1]['check_ids'] == ['check_1']


def test_download_completion_advances_queue_without_visual_model(tmp_path):
    verifier, obs, right = downloads(tmp_path)
    p = pipe()
    p.completion = verifier
    p._review_visual = lambda *a, **kw: pytest.fail('Download must not use screenshot review')
    state = StepState(task='下载roboclaw并改名data', obs=obs, memory=StructuredMemory(),
                      queue=ExecutionQueue({'steps': plan(verifier)}), verification=verifier.evaluate())
    p._update_plan_progress(state, Execution(kind='env', obs=obs, info={'status': 'executed'}))
    assert state.step_review['status'] == 'passed' and state.step_review['source'] == 'download_artifact_milestone'
    assert state.queue.current['phase'] == 'rename'
    assert state.verification['status'] == 'failed' and not p.model.calls
    # Rename itself remains an action through the UI, not automatically executed.
    assert right.exists() and not (tmp_path / 'data.pdf').exists()


def test_late_download_event_advances_before_next_policy_call(tmp_path):
    verifier, obs, _ = downloads(tmp_path)
    p = pipe()
    p.completion = verifier
    state = StepState(task='下载roboclaw并改名data', obs=obs, memory=StructuredMemory(),
                      queue=ExecutionQueue({'steps': plan(verifier)}), step_id=3)
    p.preprocess(state)
    assert state.queue.current['phase'] == 'rename'
    assert state.step_id == 3 and not p.model.calls and not p.env.actions
    p.build_input(state)
    assert '将已下载论文改名为 data.pdf' in state.prompt_text


def test_wrong_or_deleted_artifact_never_completes_download(tmp_path):
    verifier, obs, right = downloads(tmp_path)
    right.unlink()
    assert verifier.download_progress()[0]['status'] != 'passed'
    c = {'download_progress': verifier.download_progress()}
    review = code_review(plan(verifier)[0], obs, {}, c)
    assert review['status'] != 'passed'


def test_nonexistent_artifact_claim_is_not_download_evidence(tmp_path):
    verifier = bound()
    obs = observation()
    obs.info['artifacts'] = [{'path': str(tmp_path / 'fiction.pdf'), 'completed': True, 'bytes': 4000,
                             'url': 'https://arxiv.org/pdf/' + IDENTIFIER}]
    verifier.observe(obs)
    assert verifier.download_progress()[0]['status'] != 'passed'


def test_download_predicate_normalization():
    assert normalize_verification([{'kind': 'download_complete', 'check_id': 'check_1'}])[0]['check_id'] == 'check_1'
    with pytest.raises(ValueError):
        normalize_verification([{'kind': 'download_complete'}])


def artifact(session, name='original.pdf'):
    path = Path(session.downloads_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'%PDF fixture body')
    url = 'https://arxiv.org/pdf/' + IDENTIFIER
    return session._record_artifact(path=str(path), url=url, final_url=url, download_id='one')


def test_twenty_sequential_downloads_do_not_create_copies(session):
    one = artifact(session)
    for _ in range(20):
        duplicate = SimpleNamespace(url=one['url'], suggested_filename='copy.pdf', cancel=Mock(), save_as=Mock())
        session._queue_download(SimpleNamespace(url=one['url']), duplicate)
        duplicate.cancel.assert_called_once()
        duplicate.save_as.assert_not_called()
    assert len(session._artifacts) == 1 and not session._pending_downloads
    assert len(list(Path(session.downloads_dir).glob('*.pdf'))) == 1
    assert all(e['status'] == 'deduplicated' for e in session._download_events)


def test_rename_preserves_dedup_but_deleted_or_changed_file_allows_retry(session):
    one = artifact(session)
    Path(one['path']).rename(Path(one['path']).with_name('data.pdf'))
    obs = observation(one['url'])
    action = Action('click', {'target': 'PDF viewer download button', 'x': 2452, 'y': 26})
    result = session.prevent_duplicate_download(action, obs)
    assert result['error_code'] == 'already_downloaded' and result['path'].endswith('data.pdf')
    Path(result['path']).write_bytes(b'changed')
    assert session.prevent_duplicate_download(action, obs) is None
    Path(result['path']).unlink()
    assert session.prevent_duplicate_download(action, obs) is None


def test_duplicate_guard_stops_dispatch_and_not_file_manager_rename(session):
    one = artifact(session)
    p = pipe()
    p.env.prevent_duplicate_download = session.prevent_duplicate_download
    p.env.step = lambda a: pytest.fail('Duplicate dispatched')
    obs = observation(one['url'])
    action = Action('click', {'target': 'Download PDF', 'x': 2452, 'y': 26})
    result = p.execute(action, StepState(obs=obs))
    assert not result.info['dispatched'] and result.info['error_code'] == 'already_downloaded'
    obs.info.update(active_surface='desktop', tabs=[{'url': one['url'], 'active': True}],
                    runtime_state={'applications': [{'foreground': True, 'process_name': 'explorer.exe'}]})
    assert session.prevent_duplicate_download(Action('click', {'target': 'Save filename'}), obs) is None


def test_size_cap_from_headers_cancels_before_save(session):
    session.options.download_max_bytes = 100
    url = 'https://publisher.test/book.pdf'
    session._record_download_headers({'url': url, 'headers': {'Content-Length': '200000000'}})
    download = SimpleNamespace(url=url, suggested_filename='book.pdf', cancel=Mock(), save_as=Mock())
    session._queue_download(SimpleNamespace(url=url), download)
    download.cancel.assert_called_once()
    download.save_as.assert_not_called()
    assert session._download_events[-1]['error_code'] == 'download_size_limit'
    assert not session._active_downloads


def test_saved_size_limit_removes_untracked_file_and_disabled_limit_allows_it(session):
    session.options.download_max_bytes = 5
    def transfer(name):
        download = SimpleNamespace(url='https://publisher.test/' + name, suggested_filename=name,
                                   cancel=Mock(), save_as=lambda path: Path(path).write_bytes(b'%PDF test body'))
        session._queue_download(SimpleNamespace(url=download.url), download)
        session._drain_downloads()
        return download
    too_large = transfer('large.pdf')
    too_large.cancel.assert_called_once()
    assert session._download_events[-1]['error_code'] == 'download_size_limit'
    assert not list(Path(session.downloads_dir).glob('*.pdf')) and not session._artifacts
    session.options.download_max_bytes = 0
    transfer('allowed.pdf')
    assert len(session._artifacts) == 1


@pytest.mark.parametrize('value', [-1, True, 1.5, '50'])
def test_invalid_download_resource_limits_rejected(value):
    from osworld_agent.pipeline import WaitConfig
    from osworld_agent.viz.server import _wait_cli_args
    with pytest.raises(ValueError):
        BrowserOptions(download_max_bytes=value)
    with pytest.raises(ValueError):
        WaitConfig(download_max_bytes=value)
    with pytest.raises(ValueError):
        _wait_cli_args({'download_max_mib': value})


def test_zero_download_limit_reaches_cli():
    from osworld_agent.viz.server import _wait_cli_args
    args = _wait_cli_args({'download_max_mib': 0})
    assert args[args.index('--download-max-mib') + 1] == '0'


def test_multi_paper_download_uses_current_milestone_identity():
    from osworld_agent.browser_feedback import active_paper
    first = {'check_id': 'first', 'binding': {'canonical_id': '2603.11558'}}
    second = {'check_id': 'second', 'binding': {'canonical_id': '2603.11111'}}
    completion = SimpleNamespace(paper_goals=lambda: [first, second])
    assert active_paper(completion, {'check_ids': [], 'verification': [
        {'kind': 'download_complete', 'check_id': 'second'}]}) == second
