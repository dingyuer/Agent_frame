"""A browser delivery ZIP may contain real PPT/PDF bytes without extracting it."""
import hashlib
import io
import struct
import zipfile

import pytest

from osworld_agent import artifact_download
from osworld_agent.artifact_download import inspect_download_format, verified_download
from osworld_agent.verifiers import CompletionVerifier
from test_download_facts import receipt


def archive_bytes(members, compression=zipfile.ZIP_STORED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=compression) as package:
        for name, data in members:
            package.writestr(name, data)
    return stream.getvalue()


def pptx_bytes():
    return archive_bytes([('[Content_Types].xml', b'<Types/>'),
                          ('ppt/presentation.xml', b'<presentation>valid slides</presentation>')])


def browser_file(tmp_path, payload, name='download-all.zip', **changes):
    path = tmp_path / name
    path.write_bytes(payload)
    return path, receipt(path, **changes)


def verify(path, event, format='ppt'):
    return verified_download([event], format=format)


def test_real_delivery_bundle_accepts_pptx_and_preserves_outer_receipt(tmp_path, monkeypatch):
    payload = archive_bytes([('Readme.txt', b'How to use'), ('attachments/solution.pptx', pptx_bytes())])
    path, event = browser_file(tmp_path, payload)
    monkeypatch.setattr(zipfile.ZipFile, 'extract', lambda *a, **k: pytest.fail('No extraction is allowed'))
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', lambda *a, **k: pytest.fail('No extraction is allowed'))
    assert verify(path, event) is event
    assert set(tmp_path.iterdir()) == {path}
    verifier = CompletionVerifier('从方案库下载一个 PPT')
    verifier.observe({'info': {'artifacts': [event]}})
    fact = verifier.download_facts()[0]
    assert fact['status'] == 'passed'
    assert fact['format'] == fact['container'] == 'zip'
    assert fact['contained_formats'] == ['pptx']
    assert fact['contents_status'] == 'verified'
    assert fact['verified_members'] == [{'name': 'attachments/solution.pptx', 'format': 'pptx',
                                        'bytes': len(pptx_bytes())}]
    assert fact['path'] == str(path) and fact['sha256'] == hashlib.sha256(payload).hexdigest()
    assert fact['artifact_id'] == event['artifact_id']
    assert fact['urls'] == [event['url']] and fact['source_urls'] == [event['page_url']]
    assert verifier.evaluate()['status'] == 'unknown'


@pytest.mark.parametrize(('name', 'payload', 'format', 'reported'), [
    ('slides.pptx', pptx_bytes(), 'ppt', 'pptx'),
    ('slides.ppt', bytes.fromhex('D0CF11E0A1B11AE1') + b'legacy', 'ppt', 'ppt'),
    ('article.pdf', b'%PDF-1.7\nbody\n%%EOF\n', 'pdf', 'pdf'),
    # Suffixes select candidates; actual saved bytes determine their format.
    ('misleading.pptx', b'%PDF-1.7\nbody', 'pdf', 'pdf'),
])
def test_real_document_members_are_validated_by_content(tmp_path, name, payload, format, reported):
    path, event = browser_file(tmp_path, archive_bytes([(name, payload)]), name='download.bin')
    assert verify(path, event, format) is event
    assert inspect_download_format(path)['contained_formats'] == [reported]


@pytest.mark.parametrize('name', ['fake.pptx', 'fake.ppt', 'fake.pdf'])
def test_html_renamed_to_document_in_zip_is_not_downloaded_document(tmp_path, name):
    path, event = browser_file(tmp_path, archive_bytes([(name, b'<html>Please sign in</html>')]))
    assert verify(path, event, 'ppt') is None
    assert verify(path, event, 'pdf') is None
    details = inspect_download_format(path)
    assert details['container'] == 'zip'
    assert details['contained_formats'] == []


def test_zip_named_pptx_without_required_inner_members_is_not_presentation(tmp_path):
    false_pptx = archive_bytes([('[Content_Types].xml', b'<Types/>'),
                               ('word/document.xml', b'<document/>')])
    path, event = browser_file(tmp_path, archive_bytes([('slides.pptx', false_pptx)]))
    assert verify(path, event) is None


def test_pdf_and_presentation_requests_cannot_use_the_other_format(tmp_path):
    path, event = browser_file(tmp_path, archive_bytes([('slides.pptx', pptx_bytes())]))
    assert verify(path, event, 'pdf') is None
    path, event = browser_file(tmp_path, archive_bytes([('article.pdf', b'%PDF-1.7\nbody')]))
    assert verify(path, event, 'ppt') is None


def flip_saved_member(payload, content):
    value = bytearray(payload)
    offset = value.index(content)
    value[offset] ^= 1
    return bytes(value)


def test_invalid_outer_member_crc_is_rejected_even_with_valid_browser_hash(tmp_path):
    valid = archive_bytes([('slides.pptx', pptx_bytes())])
    damaged = flip_saved_member(valid, b'<presentation>valid slides</presentation>')
    path, event = browser_file(tmp_path, damaged)
    assert event['sha256'] == hashlib.sha256(damaged).hexdigest()
    assert verify(path, event) is None
    assert inspect_download_format(path)['contents_status'] == 'unavailable'


def test_invalid_inner_pptx_crc_is_rejected_even_when_outer_crc_is_valid(tmp_path):
    damaged = flip_saved_member(pptx_bytes(), b'<presentation>valid slides</presentation>')
    path, event = browser_file(tmp_path, archive_bytes([('slides.pptx', damaged)]))
    with zipfile.ZipFile(path) as package:
        assert package.testzip() is None
    assert verify(path, event) is None
    assert inspect_download_format(path)['contained_formats'] == []


@pytest.mark.parametrize('which', ['outer', 'inner'])
def test_truncated_archives_are_rejected(tmp_path, which):
    inner = pptx_bytes()
    payload = archive_bytes([('slides.pptx', inner[:-32] if which == 'inner' else inner)])
    if which == 'outer':
        payload = payload[:-32]
    path, event = browser_file(tmp_path, payload)
    assert verify(path, event) is None


def encrypted_flag(payload):
    value = bytearray(payload)
    for signature, delta in ((b'PK\x03\x04', 6), (b'PK\x01\x02', 8)):
        offset = value.index(signature) + delta
        flag = struct.unpack_from('<H', value, offset)[0]
        struct.pack_into('<H', value, offset, flag | 1)
    return bytes(value)


@pytest.mark.parametrize('which', ['outer', 'inner'])
def test_encrypted_members_are_not_claimed_as_verified(tmp_path, which):
    if which == 'outer':
        payload = encrypted_flag(archive_bytes([('article.pdf', b'%PDF-1.7\nbody')]))
    else:
        payload = archive_bytes([('slides.pptx', encrypted_flag(pptx_bytes()))])
    path, event = browser_file(tmp_path, payload)
    assert verify(path, event, 'ppt' if which == 'inner' else 'pdf') is None
    assert inspect_download_format(path)['contained_formats'] == []


def test_unsupported_zip_decoder_is_not_run(tmp_path):
    payload = bytearray(archive_bytes([('article.pdf', b'%PDF-1.7\nbody')]))
    # An unknown decoder must be rejected from metadata before decompression.
    struct.pack_into('<H', payload, payload.index(b'PK\x03\x04') + 8, 99)
    struct.pack_into('<H', payload, payload.index(b'PK\x01\x02') + 10, 99)
    path, event = browser_file(tmp_path, bytes(payload))
    assert verify(path, event, 'pdf') is None
    assert 'compression method' in inspect_download_format(path)['contents_reason']


@pytest.mark.parametrize('which', ['outer', 'inner'])
def test_member_count_limit_applies_to_outer_and_inner_archives(tmp_path, monkeypatch, which):
    monkeypatch.setattr(artifact_download, 'ZIP_MAX_MEMBERS', 1)
    if which == 'outer':
        payload = archive_bytes([('one.pdf', b'%PDF-1.7\none'), ('two.pdf', b'%PDF-1.7\ntwo')])
    else:
        payload = archive_bytes([('slides.pptx', pptx_bytes())])
    path, event = browser_file(tmp_path, payload)
    assert verify(path, event, 'pdf' if which == 'outer' else 'ppt') is None


def change_eocd_counts(payload, count):
    value = bytearray(payload)
    offset = value.rindex(b'PK\x05\x06')
    struct.pack_into('<HH', value, offset + 8, count, count)
    return bytes(value)


@pytest.mark.parametrize('forged_count', [False, True])
def test_outer_member_bound_precedes_zipfile_allocation(tmp_path, monkeypatch, forged_count):
    payload = archive_bytes([(str(index) + '.pdf', b'%PDF-1.7\nbody') for index in range(3)])
    if forged_count:
        # ZipFile parses by directory byte size, not the EOCD entry count.
        payload = change_eocd_counts(payload, 1)
    path, _ = browser_file(tmp_path, payload)
    monkeypatch.setattr(artifact_download, 'ZIP_MAX_MEMBERS', 2)
    monkeypatch.setattr(artifact_download.zipfile, 'ZipFile',
                        lambda *a, **k: pytest.fail('Directory limit must precede ZipFile allocation'))
    details = inspect_download_format(path)
    assert details['contents_status'] == 'unavailable'
    assert 'member count' in details['contents_reason']


def test_inner_member_bound_precedes_inner_zipfile_allocation(tmp_path, monkeypatch):
    inner = change_eocd_counts(pptx_bytes(), 3)
    path, _ = browser_file(tmp_path, archive_bytes([('slides.pptx', inner)]))
    monkeypatch.setattr(artifact_download, 'ZIP_MAX_MEMBERS', 2)
    original = zipfile.ZipFile
    calls = []
    def guarded_constructor(stream, *args, **kwargs):
        calls.append(type(stream).__name__)
        assert not isinstance(stream, io.BytesIO), 'Inner preflight must reject before allocation'
        return original(stream, *args, **kwargs)
    monkeypatch.setattr(artifact_download.zipfile, 'ZipFile', guarded_constructor)
    details = inspect_download_format(path)
    assert len(calls) == 1  # Only the small valid outer directory was allocated.
    assert details['contained_formats'] == []


def test_zip64_directory_sentinel_is_rejected_before_zipfile_allocation(tmp_path, monkeypatch):
    payload = change_eocd_counts(archive_bytes([('article.pdf', b'%PDF-1.7\nbody')]), 0xffff)
    path, _ = browser_file(tmp_path, payload)
    monkeypatch.setattr(artifact_download.zipfile, 'ZipFile',
                        lambda *a, **k: pytest.fail('ZIP64 must be rejected before allocation'))
    details = inspect_download_format(path)
    assert details['contents_status'] == 'unavailable'
    assert 'ZIP64' in details['contents_reason']


def test_normal_zip_comment_is_preserved_by_bounded_preflight(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as package:
        package.writestr('slides.pptx', pptx_bytes())
        package.comment = b'Normal delivery archive metadata'
    path, event = browser_file(tmp_path, stream.getvalue())
    assert verify(path, event) is event
    assert inspect_download_format(path)['contained_formats'] == ['pptx']


def test_member_decompressed_size_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_download, 'ZIP_MAX_MEMBER_BYTES', 24)
    payload = archive_bytes([('article.pdf', b'%PDF-1.7\n' + b'X' * 2000)], zipfile.ZIP_DEFLATED)
    path, event = browser_file(tmp_path, payload)
    assert verify(path, event, 'pdf') is None
    assert 'size limit' in inspect_download_format(path)['contents_reason']


def test_total_decompressed_budget_counts_outer_members(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_download, 'ZIP_MAX_TOTAL_BYTES', 1800)
    payload = archive_bytes([(str(index) + '.pdf', b'%PDF-1.7\n' + b'X' * 1000)
                             for index in range(3)], zipfile.ZIP_DEFLATED)
    path, event = browser_file(tmp_path, payload)
    assert len(payload) < 1800
    assert verify(path, event, 'pdf') is None
    assert 'budget' in inspect_download_format(path)['contents_reason']


def test_total_budget_includes_inner_pptx_decompression(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_download, 'ZIP_MAX_TOTAL_BYTES', 1200)
    inner = archive_bytes([('[Content_Types].xml', b'<Types/>'),
                           ('ppt/presentation.xml', b'<presentation/>' + b'X' * 4000)],
                          zipfile.ZIP_DEFLATED)
    payload = archive_bytes([('slides.pptx', inner)], zipfile.ZIP_DEFLATED)
    path, event = browser_file(tmp_path, payload)
    assert len(payload) < 1200 and len(inner) < 1200
    assert verify(path, event) is None


def test_does_not_descend_through_arbitrarily_nested_delivery_archives(tmp_path):
    nested = archive_bytes([('real.pptx', pptx_bytes())])
    path, event = browser_file(tmp_path, archive_bytes([('pretend.pptx', nested)]))
    assert verify(path, event) is None


def test_bundle_member_evidence_is_bounded_and_reports_omissions(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_download, 'ZIP_REPORTED_MEMBERS', 2)
    monkeypatch.setattr(artifact_download, 'ZIP_REPORTED_NAME_CHARS', 10)
    payload = archive_bytes([(f'attachments/long-name-{index}.pdf', b'%PDF-1.7\nbody')
                             for index in range(3)])
    path, event = browser_file(tmp_path, payload)
    assert verify(path, event, 'pdf') is event
    details = inspect_download_format(path)
    assert details['verified_member_count'] == 3
    assert details['verified_members_omitted'] == 1
    assert len(details['verified_members']) == 2
    assert all(len(member['name']) <= 10 and member['name_truncated']
               for member in details['verified_members'])


@pytest.mark.parametrize('change', ['http', 'incomplete', 'hash', 'identity', 'size', 'consumed',
                                  'deleted', 'changed', 'replaced'])
def test_valid_bundle_still_requires_unchanged_completed_browser_receipt(tmp_path, change):
    payload = archive_bytes([('slides.pptx', pptx_bytes())])
    path, event = browser_file(tmp_path, payload)
    assert verify(path, event) is event
    consumed = ()
    if change == 'http':
        event['transport'] = 'http'
    elif change == 'incomplete':
        event['completed'] = False
    elif change == 'hash':
        event['sha256'] = '0' * 64
    elif change == 'identity':
        event['file_identity'] = [0, 0]
    elif change == 'size':
        event['bytes'] += 1
    elif change == 'consumed':
        consumed = [event['artifact_id']]
    elif change == 'deleted':
        path.unlink()
    elif change == 'changed':
        path.write_bytes(payload + b'changed')
    else:
        path.rename(tmp_path / 'original.zip')
        path.write_bytes(payload)
    assert verified_download([event], format='ppt', consumed=consumed) is None


def test_direct_formats_keep_existing_contract(tmp_path):
    path, event = browser_file(tmp_path, pptx_bytes(), name='direct.bin')
    assert verify(path, event) is event
    assert inspect_download_format(path) == {'format': 'pptx'}
    path, event = browser_file(tmp_path, b'%PDF-1.7\nbody', name='direct.bin')
    assert verify(path, event, 'pdf') is event
    assert inspect_download_format(path) == {'format': 'pdf'}
    path, event = browser_file(tmp_path, bytes.fromhex('D0CF11E0A1B11AE1') + b'legacy', name='direct.bin')
    assert verify(path, event) is event
    assert inspect_download_format(path) == {'format': 'ole'}
