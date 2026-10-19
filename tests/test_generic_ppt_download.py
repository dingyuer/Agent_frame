"""Generic PPT downloads need a real current-run browser artifact."""
import hashlib
import json
import os
import zipfile

import pytest

from osworld_agent.artifact_download import verified_download
from osworld_agent.env import Observation
from osworld_agent.execution_queue import normalize_verification
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.step_verifier import code_review
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model


def artifact(path):
    stat = path.stat()
    return {'path': str(path), 'artifact_id': 'dl-1', 'download_id': 'dl-1',
            'completed': True, 'transport': 'browser', 'bytes': stat.st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'file_identity': [stat.st_dev, stat.st_ino]}


def pptx(path):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('[Content_Types].xml', '<Types/>')
        z.writestr('ppt/presentation.xml', '<presentation/>')


def test_generic_ppt_requires_download_event_and_matching_file(tmp_path):
    path = tmp_path / 'slides.pptx'
    pptx(path)
    event = artifact(path)
    predicate = {'kind': 'artifact_download', 'format': 'ppt'}
    step = {'goal': '下载一个 PPT', 'verification': [predicate]}
    obs = Observation(info={'artifacts': [event]})
    assert code_review(step, obs, {}, {'checks': []})['status'] == 'passed'
    assert code_review(step, Observation(info={}), {}, {'checks': []})['status'] == 'unknown'
    assert code_review(step, obs, {}, {'checks': [], 'consumed_artifact_ids': ['dl-1']})['status'] == 'unknown'
    path.write_bytes(b'not a pptx')
    assert verified_download([event], format='ppt') is None


def test_generic_ppt_rejects_extension_only_and_accepts_legacy_signature(tmp_path):
    fake = tmp_path / 'fake.pptx'
    fake.write_bytes(b'not a presentation')
    assert verified_download([artifact(fake)], format='ppt') is None
    legacy = tmp_path / 'legacy.ppt'
    legacy.write_bytes(bytes.fromhex('D0CF11E0A1B11AE1') + b'content')
    assert verified_download([artifact(legacy)], format='ppt') is not None
    assert normalize_verification([{'kind': 'artifact_download', 'format': 'ppt'}]) == [
        {'kind': 'artifact_download', 'format': 'ppt'}]


@pytest.mark.parametrize('filename', ['download', 'download.bin', 'slides.pdf'])
def test_pptx_download_is_identified_by_content_without_correct_extension(tmp_path, filename):
    path = tmp_path / filename
    pptx(path)
    assert verified_download([artifact(path)], format='ppt') is not None


@pytest.mark.parametrize('filename', ['download', 'download.bin', 'slides.pptx'])
def test_legacy_ppt_download_is_identified_by_signature(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(bytes.fromhex('D0CF11E0A1B11AE1') + b'content')
    assert verified_download([artifact(path)], format='ppt') is not None


@pytest.mark.parametrize('filename', ['download', 'download.bin', 'paper.ppt'])
def test_pdf_download_is_identified_by_magic_without_correct_extension(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(b'%PDF-1.7\ncontent\n%%EOF\n')
    assert verified_download([artifact(path)], format='pdf') is not None


@pytest.mark.parametrize(('filename', 'format'), [
    ('fake.pptx', 'ppt'), ('fake.ppt', 'ppt'), ('fake.pdf', 'pdf'),
])
def test_download_rejects_html_with_requested_extension(tmp_path, filename, format):
    path = tmp_path / filename
    path.write_bytes(b'<!doctype html><html>Sign in to download</html>')
    assert verified_download([artifact(path)], format=format) is None


def test_ppt_download_rejects_zip_without_presentation_members(tmp_path):
    path = tmp_path / 'fake.pptx'
    with zipfile.ZipFile(path, 'w') as package:
        package.writestr('[Content_Types].xml', '<Types/>')
        package.writestr('word/document.xml', '<document/>')
    assert verified_download([artifact(path)], format='ppt') is None


def test_extensionless_download_still_requires_unchanged_file_and_browser_event(tmp_path):
    path = tmp_path / 'download'
    pptx(path)
    event = artifact(path)
    assert verified_download([event], format='ppt', consumed=['dl-1']) is None
    assert verified_download([{**event, 'transport': 'http'}], format='ppt') is None
    assert verified_download([{**event, 'completed': False}], format='ppt') is None
    assert verified_download([{**event, 'file_identity': [0, 0]}], format='ppt') is None
    path.write_bytes(path.read_bytes() + b'changed')
    assert verified_download([event], format='ppt') is None


def test_no_contract_plan_converts_old_download_predicate(tmp_path):
    task = '打开内部门户，去方案库随便下载一个 PPT'
    proposal = {'steps': [
        {'goal': '进入方案库', 'expected_evidence': '方案卡片出现', 'check_ids': [],
         'verification': [{'kind': 'visual'}], 'actions': ['click']},
        {'goal': '下载一个 PPT', 'expected_evidence': '文件已下载', 'check_ids': ['fabricated'],
         'verification': [{'kind': 'download_complete', 'check_id': 'fabricated'}], 'actions': ['click']}],
        'keywords': [], 'final_check': '有 PPT 文件'}
    model, env = Model([], proposal), Env()
    pipe = Pipeline(model=model, env=env, grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier(task)
    assert pipe.completion.public_spec['checks'] == []
    state = StepState(task=task, memory=StructuredMemory())
    assert pipe._plan(task, state)
    assert state.memory.plan['steps'][1]['verification'] == [{'kind': 'artifact_download', 'format': 'ppt'}]
    assert state.memory.plan['steps'][1]['check_ids'] == []
    assert 'artifact_download' in json.dumps(model.calls, ensure_ascii=False)


def test_no_contract_download_cannot_use_visual_check_alone():
    task = '在方案库下载一个 PPT'
    proposal = {'steps': [{'goal': '下载一个 PPT', 'expected_evidence': '文件落盘',
                           'verification': [{'kind': 'visual'}]}]}
    model = Model([], proposal)
    pipe = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier(task)
    state = StepState(task=task, memory=StructuredMemory())
    assert pipe._plan(task, state)
    assert state.memory.plan['steps'][0]['verification'] == [{'kind': 'artifact_download', 'format': 'ppt'}]
