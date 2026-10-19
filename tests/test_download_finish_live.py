"""Opt-in real browser + Jev regression with a scripted operator sequence.

No public/private website or decision-model endpoint is used. The operator
sequence deliberately reproduces download -> close popup -> request_finish.
"""
import base64
import io
import json
import os
import zipfile

import pytest

from osworld_agent.browser_session import browser_thread
from osworld_agent.env import StepResult
from osworld_agent.pipeline import Pipeline
from test_browser_live import live_browser, observe
from test_pipeline_completion import Model


@pytest.mark.parametrize('bundled', [False, True])
def test_real_browser_download_popup_close_and_jev_finish(live_browser, bundled):
    endpoint = os.environ.get('AGENTFRAME_JEV_TEST_URL')
    if not endpoint:
        pytest.skip('Set AGENTFRAME_JEV_TEST_URL explicitly to exercise a real Jev service')
    session, _ = live_browser
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as package:
        package.writestr('[Content_Types].xml', '<Types/>')
        package.writestr('ppt/presentation.xml', '<presentation/>')
    if bundled:
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, 'w') as package:
            package.writestr('Report.pptx', buffer.getvalue())
            package.writestr('README.txt', 'Delivered by download-all')
        buffer = outer
    data = base64.b64encode(buffer.getvalue()).decode('ascii')
    document = '''<meta charset=utf-8><title>Solution Library</title>
      <h1>Solution Library</h1><p>Presentation attachments</p>
      <button id="export">Download PPT</button>
      <div id="notice" role="dialog" aria-label="Notice" hidden
           style="position:fixed;inset:20%;background:#fff;border:1px solid #ccc;padding:30px">
        <p>Your request was submitted.</p>
        <button id="close" onclick="document.getElementById('notice').hidden=true">Close</button>
    </div>
      <script>document.getElementById('export').onclick = () => {
        const bytes = Uint8Array.from(atob(DATA), x => x.charCodeAt(0));
        const link = document.createElement('a');
        link.href = URL.createObjectURL(new Blob([bytes], {type:'application/vnd.openxmlformats-officedocument.presentationml.presentation'}));
        link.download = 'attachment'; document.body.appendChild(link); link.click(); link.remove();
        document.getElementById('notice').hidden = false;
    };</script>'''.replace('DATA', json.dumps(data))
    set_document = browser_thread(lambda self: self._pick_page(self._connect()[1]).set_content(document))
    set_document(session)

    class BrowserEnv:
        evaluation_available = False
        browser_only = True
        os_name = 'mac'
        initial_state = session.initial_state
        _browser_session = session
        def reset(self, task):
            return observe(session)
        def observe(self):
            return observe(session)
        def route_action(self, action, observation):
            if action.action in {'request_finish', 'fail'}:
                return None
            return session.route(action, observation)
        def execute_routed(self, action, route):
            result = session.execute(action, route)
            return StepResult(observe(session), info=result)
        def browser_event_metadata(self):
            return session.event_metadata()

    plan = {'steps': [
        {'goal': '下载一个 PPT', 'expected_evidence': '真实 PPT 文件已经完整保存',
         'verification': [{'kind': 'artifact_download', 'format': 'ppt'}]},
        {'goal': '关闭请求提示窗口', 'expected_evidence': '提示窗口已关闭，回到方案列表',
         'verification': [{'kind': 'visual'}]}]}
    model = Model([{'type': 'click', 'element_id': 'export', 'target': 'Download PPT'},
                   {'type': 'click', 'element_id': 'close', 'target': 'Close'},
                   {'type': 'request_finish', 'answer': 'PPT 下载完成，提示窗口已关闭'}], plan)
    pipe = Pipeline(model=model, env=BrowserEnv(), grounding=None, system_prompt='test',
        max_steps=3, max_replans=0,
        decision_judge={'url': endpoint, 'review_mode': 'exclusive', 'recheck_unknown': False,
                        'selection_policy': 'maximum_probability',
                        'timeout': 5}, wait_config={'operation_timeout_s': 12, 'task_timeout_s': 45})
    result = pipe.run('在当前方案库随便下载一个 PPT，并关闭提示窗口。')
    assert result.success, {'reason': result.termination_reason, 'verification': result.verification,
        'reviews': [{'questions': list(c.get('request', {}).get('questions', {})),
                     'answers': c.get('response', {}).get('answers', {})} for c in result.review_calls]}
    assert result.steps <= 3 and len(result.planning_calls) == 1
    final = result.verification['completion']['download_facts']
    assert len(final) == 1 and final[0]['status'] == 'passed'
    if bundled:
        assert final[0]['format'] == 'zip' and 'pptx' in final[0]['contained_formats']
    else:
        assert final[0]['format'] == 'pptx'
    assert not result.verification['completion']['checks']  # No invented contract.
    assert len(session.event_metadata()['artifacts']) == 1
