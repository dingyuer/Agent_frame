"""Compound user goals, trusted bindings and content-preserving renames."""
import copy
import hashlib
import json

import pytest

from osworld_agent.paper_resolver import PaperBinding
from osworld_agent.task_specs import infer_paper_spec, task_profile
from osworld_agent.verifiers import CompletionVerifier
from test_paper_completion import event, pdf_file
from test_paper_resolver import TITLE, IDENTIFIER, resolve


@pytest.mark.parametrize('task', [
    '请下载一篇叫做Roboclaw的论文，并重命名为Roboclaw',
    '下载Roboclaw论文并重命名为Roboclaw',
    '下载论文Roboclaw，然后保存为Roboclaw.pdf',
])
def test_compound_download_and_filename_are_one_pending_goal(task):
    verifier = CompletionVerifier(task)
    goal = verifier.paper_goals()[0]
    assert goal['query'] == 'Roboclaw' and goal['filename'] == 'Roboclaw.pdf'
    assert goal['status'] == 'pending' and verifier.evaluate()['status'] == 'unknown'
    assert task_profile(task, verifier.public_spec)['kind'] == 'paper_download'
    assert len(verifier.public_spec['checks']) == 1


def test_explicit_filename_extension_is_not_silently_rewritten():
    spec = infer_paper_spec('下载Roboclaw论文并重命名为archive.bin')
    assert spec['checks'][0]['filename'] == 'archive.bin'
    assert spec['checks'][0]['format'] == 'pdf'


def test_unsupported_tail_is_preserved_and_prevents_false_success(tmp_path):
    task = '请下载一篇叫做Roboclaw的论文，并重命名为Roboclaw，然后发邮件给老师'
    verifier = CompletionVerifier(task)
    checks = verifier.public_spec['checks']
    assert checks[0]['filename'] == 'Roboclaw.pdf'
    assert checks[1]['kind'] == 'unresolved_requirement' and '发邮件' in checks[1]['text']
    verifier.bind_paper('check_1', resolve().binding)
    path = pdf_file(tmp_path / 'Roboclaw.pdf', TITLE, 'Example Author', 'Abstract')
    verifier.observe(None, {'artifacts': [official_event(path)]})
    report = verifier.evaluate()
    assert report['checks'][0]['status'] == 'passed' and report['status'] == 'unknown'


def official_event(path, **kwargs):
    url = 'https://arxiv.org/pdf/' + IDENTIFIER
    return event(path, url=url, final_url=url, source_url='https://arxiv.org/abs/' + IDENTIFIER, **kwargs)


def test_query_needs_binding_and_filename_applies_to_same_real_pdf(tmp_path):
    verifier = CompletionVerifier('下载Roboclaw论文并改名为Roboclaw')
    path = pdf_file(tmp_path / 'Roboclaw.pdf', TITLE, 'Example Author', 'Abstract')
    verifier.observe(None, {'artifacts': [official_event(path)]})
    assert verifier.evaluate()['status'] == 'unknown'
    before = copy.deepcopy(verifier.public_spec)
    assert verifier.bind_paper('check_1', resolve().binding)
    assert verifier.public_spec == before
    assert not verifier.pending_paper_goals()
    assert verifier.paper_goals()[0]['binding']['full_title'] == TITLE
    assert verifier.evaluate()['status'] == 'passed'
    assert verifier.bind_paper('check_1', resolve().binding) is False


def test_binding_cannot_be_model_dict_relax_explicit_title_or_switch_identity():
    binding = resolve().binding
    verifier = CompletionVerifier('下载Roboclaw论文')
    with pytest.raises(ValueError, match='adapter'):
        verifier.bind_paper('check_1', binding.to_dict())
    forged = PaperBinding(**{**binding.to_dict(), 'authors': tuple(binding.authors)})
    with pytest.raises(ValueError, match='adapter'):
        verifier.bind_paper('check_1', forged)
    explicit = CompletionVerifier('task', {'checks': [{'kind': 'paper_download', 'title': 'RoboClaw'}]})
    with pytest.raises(ValueError, match='explicit'):
        explicit.bind_paper('check_1', binding)
    verifier.bind_paper('check_1', binding)
    with pytest.raises(ValueError, match='cannot be changed'):
        verifier.bind_paper('check_1', resolve(identifier='2603.99999').binding)


@pytest.mark.parametrize('qualifier', [
    {'author': 'Another Author'}, {'author': 'Authorx'}, {'year': 2025}, {'doi': '10.1234/wrong'},
])
def test_binding_rejects_official_qualifier_conflict_without_freezing_candidate(qualifier):
    binding = resolve(extra_metadata='<meta name="citation_date" content="2026/03/12">'
                      '<meta name="citation_doi" content="10.1234/paper">').binding
    verifier = CompletionVerifier('task', {'checks': [{'kind': 'paper_download', 'query': 'Roboclaw', **qualifier}]})
    with pytest.raises(ValueError):
        verifier.bind_paper('check_1', binding)
    assert verifier.bindings == {} and verifier.pending_paper_goals()


def test_binding_keeps_matching_qualifiers_and_missing_metadata_does_not_waive_pdf_checks(tmp_path):
    verifier = CompletionVerifier('task', {'checks': [{'kind': 'paper_download', 'query': 'Roboclaw',
        'author': 'Example Author', 'year': 2026, 'doi': '10.1234/paper'}]})
    original = verifier.public_spec
    assert verifier.bind_paper('check_1', resolve().binding)
    assert verifier.public_spec == original
    path = pdf_file(tmp_path / 'paper.pdf', TITLE, 'Example Author', 'Abstract')
    verifier.observe(None, {'artifacts': [official_event(path)]})
    assert verifier.evaluate()['status'] == 'unknown'
    pdf_file(path, TITLE, 'Example Author 2026', 'doi:10.1234/paper', 'Abstract')
    verifier.observe(None, {'artifacts': [official_event(path, ident='replacement')]})
    assert verifier.evaluate()['status'] == 'passed'


def margin_pdf(path, identifier=IDENTIFIER):
    """Actual rotated margin stamp, deliberately after first-page body text."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject
    pdf_file(path, TITLE, 'Example Author', 'Abstract', 'Body text')
    reader = PdfReader(path)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    stream = DecodedStreamObject()
    stream.set_data(writer.pages[0].get_contents().get_data() +
        ('\nq 0 1 -1 0 0 0 cm BT /F1 12 Tf 226 -32 Td '
         f'(arXiv:{identifier}v3 [cs.RO] 1 Apr 2026) Tj ET Q').encode('ascii'))
    writer.pages[0][NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as handle:
        writer.write(handle)
    return path


def test_actual_margin_stamp_is_identity_but_body_reference_is_not(tmp_path):
    path = margin_pdf(tmp_path / 'stamp.pdf')
    verifier = CompletionVerifier('task', {'checks': [{'kind': 'paper_download', 'title': TITLE,
                                                     'arxiv_id': IDENTIFIER}]})
    verifier.observe(None, {'artifacts': [event(path)]})
    assert verifier.evaluate()['status'] == 'passed'
    body = pdf_file(tmp_path / 'body.pdf', TITLE, 'Example Author', 'Abstract',
                    f'arXiv:{IDENTIFIER}v3 [cs.RO] 1 Apr 2026')
    other = CompletionVerifier('task', verifier.public_spec)
    other.observe(None, {'artifacts': [event(body)]})
    assert other.evaluate()['status'] == 'unknown'


def test_wrong_margin_identity_overrides_official_source_binding(tmp_path):
    path = margin_pdf(tmp_path / 'wrong.pdf', '2603.99999')
    verifier = CompletionVerifier('下载Roboclaw论文')
    verifier.bind_paper('check_1', resolve().binding)
    verifier.observe(None, {'artifacts': [official_event(path)]})
    assert verifier.evaluate()['status'] == 'failed'


def test_bound_title_with_wrong_origin_or_wrong_filename_is_not_success(tmp_path):
    verifier = CompletionVerifier('下载Roboclaw论文并改名为Roboclaw')
    verifier.bind_paper('check_1', resolve().binding)
    path = pdf_file(tmp_path / 'Roboclaw.pdf', TITLE, 'Abstract')
    verifier.observe(None, {'artifacts': [event(path)]})
    assert verifier.evaluate()['status'] == 'failed'
    other = pdf_file(tmp_path / 'other.pdf', TITLE, 'Abstract')
    verifier.observe(None, {'artifacts': [official_event(other, ident='d2')]})
    assert verifier.evaluate()['status'] == 'failed'


def registered(path, ident='stable-download'):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return event(path, ident=ident, artifact_id=ident, initial_sha256=digest, sha256=digest,
                 version=1, rename_history=[])


def renamed(artifact, before, after):
    result = copy.deepcopy(artifact)
    result['rename_history'].append({'from_path': str(before), 'to_path': str(after),
        'from_version': result['version'], 'to_version': result['version'] + 1,
        'sha256': result['initial_sha256'], 'renamed_at': 123})
    result.update(path=str(after), version=result['version'] + 1)
    return result


def test_tracked_rename_preserves_initial_baseline_and_ignores_stale_cumulative_event(tmp_path):
    old = pdf_file(tmp_path / 'source.pdf', 'AAA', 'Abstract')
    artifact = registered(old)
    verifier = CompletionVerifier('task', {'checks': [{'kind': 'paper_download', 'title': 'AAA',
                                                     'filename': 'Roboclaw.pdf'}]})
    verifier.observe(None, {'artifacts': [artifact]})
    assert verifier.evaluate()['status'] == 'failed'
    new = tmp_path / 'Roboclaw.pdf'
    old.rename(new)
    changed = renamed(artifact, old, new)
    verifier.observe({'info': {'artifacts': [changed]}}, {'artifacts': [artifact]})
    report = verifier.evaluate()
    assert report['status'] == 'passed'
    assert report['checks'][0]['artifact_id'] == artifact['artifact_id']
    assert report['checks'][0]['version'] == 2
    pdf_file(new, 'AAA', 'Altered content', 'Abstract')
    verifier.observe(None, {'artifacts': [changed]})
    assert verifier.evaluate()['status'] == 'failed'


@pytest.mark.parametrize('corruption', ['missing_chain', 'changed_bytes', 'forged_initial_hash', 'version_gap'])
def test_rename_cannot_create_new_hash_baseline(tmp_path, corruption):
    old = pdf_file(tmp_path / 'source.pdf', 'AAA', 'Abstract')
    artifact = registered(old)
    verifier = CompletionVerifier('task', {'checks': [{'kind': 'paper_download', 'title': 'AAA'}]})
    verifier.observe(None, {'artifacts': [artifact]})
    new = tmp_path / 'named.pdf'
    old.rename(new)
    changed = renamed(artifact, old, new)
    if corruption == 'missing_chain':
        changed['rename_history'] = []
    elif corruption == 'changed_bytes':
        pdf_file(new, 'AAA', 'Changed payload', 'Abstract')
    elif corruption == 'forged_initial_hash':
        changed['initial_sha256'] = '0' * 64
    else:
        changed['version'] = 9
    verifier.observe(None, {'artifacts': [changed]})
    assert verifier.evaluate()['status'] == 'failed'


def test_first_observation_can_follow_trusted_rename_and_later_stale_id_cannot_rebaseline(tmp_path):
    old = pdf_file(tmp_path / 'old.pdf', 'AAA', 'Abstract')
    initial = registered(old)
    new = tmp_path / 'new.pdf'
    old.rename(new)
    changed = renamed(initial, old, new)
    verifier = CompletionVerifier('task', {'checks': [{'kind': 'paper_download', 'title': 'AAA'}]})
    verifier.observe(None, {'artifacts': [changed]})
    assert verifier.evaluate()['status'] == 'passed'
    replacement = registered(new, ident='new-transfer')
    verifier.observe(None, {'artifacts': [replacement]})
    pdf_file(new, 'AAA', 'Tampered', 'Abstract')
    verifier.observe(None, {'artifacts': [changed]})
    assert verifier.evaluate()['status'] == 'failed'
