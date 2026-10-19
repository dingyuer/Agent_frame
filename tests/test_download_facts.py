"""Durable download facts survive page changes without becoming task success."""
import copy
import hashlib
import zipfile

import pytest

from osworld_agent.verifiers import CompletionVerifier


def receipt(path, ident='download-1', **changes):
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    event = {'artifact_id': ident, 'download_id': ident, 'path': str(path),
        'url': 'https://example.test/files/get?id=123',
        'page_url': 'https://example.test/library', 'transport': 'browser',
        'status': 'completed', 'completed': True, 'bytes': stat.st_size,
        'sha256': digest, 'initial_sha256': digest,
        'file_identity': [stat.st_dev, stat.st_ino], 'version': 1, 'rename_history': []}
    event.update(changes)
    return event


def completed(tmp_path):
    path = tmp_path / 'download'
    with zipfile.ZipFile(path, 'w') as package:
        package.writestr('[Content_Types].xml', '<Types/>')
        package.writestr('ppt/presentation.xml', '<presentation/>')
    verifier = CompletionVerifier('去指定门户下载一个 PPT')
    event = receipt(path)
    verifier.observe({'info': {'artifacts': [event]}})
    return verifier, path, event


def test_download_facts_survive_popup_close_later_observations_and_replanning(tmp_path):
    verifier, path, event = completed(tmp_path)
    first = verifier.download_facts()
    assert first == [{'artifact_id': 'download-1', 'path': str(path), 'version': 1,
        'transport': 'browser', 'completed': True,
        'urls': [event['url']], 'source_urls': [event['page_url']],
        'bytes': event['bytes'], 'sha256': event['sha256'], 'status': 'passed',
        'file_identity': event['file_identity'],
        'reason': 'browser download completed; current local file matches its original receipt',
        'format': 'pptx'}]
    verifier.observe({'context': [{'kind': 'browser_ax', 'url': event['page_url'],
        'active_modals': [], 'nodes': []}], 'info': {}}, {'action': 'click'})
    verifier.observe({'info': {'artifacts': []}})
    assert verifier.download_facts() == first
    # Calling this read-only API cannot replace a missing completion contract.
    assert verifier.evaluate()['status'] == 'unknown'
    assert verifier.public_spec == {'checks': []}


@pytest.mark.parametrize('change', ['bytes', 'replace', 'delete', 'empty'])
def test_facts_freshly_recheck_file_instead_of_latching_success(tmp_path, change):
    verifier, path, event = completed(tmp_path)
    assert verifier.download_facts()[0]['status'] == 'passed'
    data = path.read_bytes()
    if change == 'bytes':
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif change == 'replace':
        path.rename(tmp_path / 'old')
        path.write_bytes(data)
    elif change == 'delete':
        path.unlink()
    else:
        path.write_bytes(b'')
    fact = verifier.download_facts()[0]
    assert fact['status'] != 'passed'
    assert fact['artifact_id'] == event['artifact_id']
    assert 'format' not in fact


def test_cumulative_completion_cannot_rebaseline_changed_bytes(tmp_path):
    verifier, path, event = completed(tmp_path)
    path.write_bytes(path.read_bytes() + b'tampered')
    new_claim = receipt(path)
    verifier.observe({'info': {'artifacts': [new_claim]}}, {'artifacts': [event]})
    fact = verifier.download_facts()[0]
    assert fact['status'] == 'failed'
    assert 'changed since' in fact['reason']
    assert fact['sha256'] != event['sha256']


@pytest.mark.parametrize('change', [
    {'transport': 'http'}, {'transport': None}, {'completed': False}, {'completed': None},
    {'bytes': 1}, {'bytes': True}, {'sha256': '0' * 64},
    {'file_identity': [0, 0]}, {'file_identity': None},
    {'artifact_id': None, 'download_id': None},
])
def test_only_completed_browser_receipts_with_valid_baselines_are_facts(tmp_path, change):
    path = tmp_path / 'payload'
    path.write_bytes(b'actual downloaded bytes')
    verifier = CompletionVerifier('download something from a portal')
    verifier.observe(None, {'artifacts': [receipt(path, **change)]})
    assert verifier.download_facts() == []


def test_missing_receipt_fields_do_not_upgrade_via_duplicate_event(tmp_path):
    path = tmp_path / 'payload'
    path.write_bytes(b'actual downloaded bytes')
    event = receipt(path)
    incomplete = copy.deepcopy(event)
    del incomplete['file_identity']
    verifier = CompletionVerifier('download something from a portal')
    verifier.observe(None, {'artifacts': [incomplete]})
    verifier.observe(None, {'artifacts': [event]})
    assert verifier.download_facts() == []


def test_download_facts_preserve_existing_legacy_contract_behavior(tmp_path):
    path = tmp_path / 'legacy.bin'
    path.write_bytes(b'legacy executor output')
    event = {'path': str(path), 'url': 'https://example.test/data', 'status': 'executed'}
    verifier = CompletionVerifier('下载 https://example.test/data')
    verifier.observe(None, {'artifacts': [event]})
    assert verifier.evaluate()['status'] == 'passed'
    assert verifier.download_facts() == []


def renamed(event, before, after):
    newer = copy.deepcopy(event)
    newer['version'] += 1
    newer['path'] = str(after)
    newer['rename_history'].append({'from_path': str(before), 'to_path': str(after),
        'from_version': event['version'], 'to_version': newer['version'],
        'sha256': event['initial_sha256']})
    return newer


def test_valid_rename_keeps_id_receipt_and_original_baseline(tmp_path):
    verifier, path, event = completed(tmp_path)
    destination = tmp_path / 'renamed.pptx'
    path.rename(destination)
    updated = renamed(event, path, destination)
    verifier.observe(None, {'artifacts': [updated]})
    verifier.observe(None, {'artifacts': [event]})
    fact = verifier.download_facts()[0]
    assert fact['status'] == 'passed'
    assert fact['artifact_id'] == event['artifact_id']
    assert fact['path'] == str(destination)
    assert fact['sha256'] == event['sha256']
    assert fact['version'] == 2


def test_invalid_migration_does_not_rebase_even_if_original_file_still_exists(tmp_path):
    verifier, path, event = completed(tmp_path)
    destination = tmp_path / 'untrusted.pptx'
    destination.write_bytes(path.read_bytes())
    verifier.observe(None, {'artifacts': [{**event, 'version': 2, 'path': str(destination)}]})
    fact = verifier.download_facts()[0]
    assert fact['status'] == 'failed'
    assert 'rename chain' in fact['reason']
    assert fact['path'] == str(path)


def test_download_facts_do_not_expose_mutable_internal_receipts(tmp_path):
    verifier, _, _ = completed(tmp_path)
    fact = verifier.download_facts()[0]
    fact['urls'].clear()
    fact['source_urls'].append('https://untrusted.test/')
    fact['artifact_id'] = 'invented'
    current = verifier.download_facts()[0]
    assert current['artifact_id'] == 'download-1'
    assert current['urls'] == ['https://example.test/files/get?id=123']
    assert current['source_urls'] == ['https://example.test/library']


def test_pdf_fact_format_uses_saved_magic_not_filename(tmp_path):
    path = tmp_path / 'not-a-pdf.ppt'
    path.write_bytes(b'%PDF-1.7\nbody\n%%EOF\n')
    verifier = CompletionVerifier('download a document')
    verifier.observe(None, {'artifacts': [receipt(path)]})
    assert verifier.download_facts()[0]['format'] == 'pdf'
