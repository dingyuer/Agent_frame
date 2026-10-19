"""Real local Chromium downloads; no public website or model is used."""
import base64
import io
import json
import time
import zipfile

import pytest

from osworld_agent.artifact_download import verified_download
from osworld_agent.browser_session import browser_thread
from osworld_agent.step_verifier import code_review
from test_browser_live import live_browser, click, observe


@pytest.mark.parametrize('filename', ['download', 'attachment.bin', 'slides.pptx'])
def test_blob_download_and_recheck_use_file_content(live_browser, filename):
    session, _ = live_browser
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as package:
        package.writestr('[Content_Types].xml', '<Types/>')
        package.writestr('ppt/presentation.xml', '<presentation/>')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    # Exercise the portal pattern: a JavaScript-generated attachment has no
    # static href or useful extension. Still use the real download/save event.
    document = '''<title>Portal fixture</title><button id="export">Export PPT</button>
    <script>document.getElementById('export').onclick = () => {
      const bytes = Uint8Array.from(atob(DATA), x => x.charCodeAt(0));
      const link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob([bytes], {type: 'application/octet-stream'}));
      link.download = FILENAME;
      document.body.appendChild(link); link.click(); link.remove();
    };</script>'''.replace('DATA', json.dumps(encoded)).replace('FILENAME', json.dumps(filename))
    set_document = browser_thread(lambda self: self._pick_page(self._connect()[1]).set_content(document))
    set_document(session)
    assert click(session, 'export')['status'] == 'executed'
    deadline = time.monotonic() + 5
    observation = observe(session)
    while not observation.info.get('artifacts') and time.monotonic() < deadline:
        time.sleep(.02)
        observation = observe(session)
    artifact = verified_download(observation.info.get('artifacts'), format='ppt')
    assert artifact is not None, observation.info.get('download_events')
    step = {'goal': '下载一个 PPT', 'verification': [{'kind': 'artifact_download', 'format': 'ppt'}]}
    assert code_review(step, observation, {}, {'checks': []})['status'] == 'passed'
    recheck = {'goal': '检查 PPT 下载成功', 'verification': [
        {'kind': 'artifact_download', 'format': 'ppt', 'purpose': 'recheck'}]}
    consumed = [artifact['artifact_id']]
    assert code_review(recheck, observation, {}, {'checks': [], 'consumed_artifact_ids': consumed})['status'] == 'passed'
    # An actual second download still requires a second registered artifact.
    assert code_review(step, observation, {}, {'checks': [], 'consumed_artifact_ids': consumed})['status'] == 'unknown'
