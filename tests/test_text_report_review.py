"""Text delivery is reviewed against recorded sources, never a dynamic viewport."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from osworld_agent.report_delivery import wants_report
from osworld_agent.report_review import delivery_digest, review_packet, text_only_task
from osworld_agent.model.judge_selection import RuntimeDecisionJudge
from osworld_agent.pipeline_types import StepState
from osworld_agent.review_service import ReviewPorts, ReviewService
from osworld_agent.step_verifier import StepReviewer
from test_decision_judge import answer

TASK = '去w3网站看一下在3MS可以做什么事情'


def delivery():
    return {'status': 'complete', 'text': '3MS 可以查阅文档、参与社区及学习课程 [S1]。',
            'requirements': [TASK], 'sources': [{'id': 'S1', 'url': 'https://example.test/3ms',
            'title': '3MS', 'text': '3MS 功能入口\n文档：查阅资料\n社区：交流讨论\n课程：在线学习'}]}


@pytest.mark.parametrize('task', [TASK, '看看这个平台能做什么', '介绍这个网站的用途',
    'What can I do on this website?', '总结此页面提供的功能'])
def test_site_capabilities_are_text_deliverables(task):
    assert wants_report(task)
    assert text_only_task(task, {})


@pytest.mark.parametrize('task', ['打开网站并下载附件', '总结网站功能并保存到txt文件',
    '列出名单并发送邮件', '查看费用并提交采购审批', '汇总信息然后写进报告.txt'])
def test_mixed_actions_cannot_use_frozen_text_as_system_proof(task):
    assert not text_only_task(task, {})


def test_snapshot_digest_ignores_display_metadata_but_not_content():
    original = delivery();digest = delivery_digest(TASK, original)
    original['verification_status'] = 'passed';original['sources'][0]['observed_at'] = 123
    assert delivery_digest(TASK, original) == digest
    original['text'] += '不支持的功能'
    assert delivery_digest(TASK, original) != digest
    original = delivery();original['sources'][0]['text'] += '新内容'
    assert delivery_digest(TASK, original) != digest


def test_nonempty_answer_and_keywords_are_not_a_delivery_contract():
    d = delivery();d['sources'] = []
    packet, error = review_packet(TASK, d)
    assert packet is None and '来源' in error
    d = delivery();d['text'] = '3MS 功能齐全 [S99]'
    assert review_packet(TASK, d)[0] is None
    d = delivery();d['text'] = '3MS 支持课程。'
    assert review_packet(TASK, d)[0] is None


def test_large_sources_are_bounded_and_marked_not_silently_claimed_complete():
    d = delivery()
    for i in range(20):
        d['sources'].append({'id':f'S{i+2}', 'url':f'https://example.test/{i}', 'title':'详情',
                            'text':('无关新闻\n'*5000)+'课程：在线学习'})
    packet, error = review_packet(TASK, d)
    assert not error and len(json.dumps(packet, ensure_ascii=False)) <= 28000
    assert packet['deliverable'] == d['text']
    assert packet['observed_sources'][1]['excerpt_truncated']
    assert '课程：在线学习' in packet['observed_sources'][1]['text']


def judge_with(requests, label='passed', tie=False):
    def handle(request):
        payload = json.loads(request.content);requests.append(payload)
        preferred = {'task_type':'browser','next_mode':'dom','settled':'complete','correct':'yes','replan':'no'}
        values = {k:answer(q['criteria'], label if label in q['criteria'] else
                          preferred.get(k, next(iter(q['criteria'])))) for k,q in payload['questions'].items()}
        if tie:
            for v in values.values():
                v.update(choice='passed', probabilities={'passed': .416, 'unknown': .414, 'failed': .16, 'pending': .01})
        return httpx.Response(200, json={'model':'d3-flash', 'answers': values})
    return RuntimeDecisionJudge({'url':'http://judge.test', 'review_mode':'exclusive'},
        httpx.MockTransport(handle), selection_policy='maximum_probability')


@pytest.mark.parametrize('tie,expected', [(False,'passed'), (True,'unknown')])
def test_jev_text_review_needs_no_screenshot_and_ties_are_not_acceptance(tie, expected):
    requests=[];judge=judge_with(requests,tie=tie)
    packet,_=review_packet(TASK,delivery())
    result=judge.review(TASK,{'id':'whole_task'},None,None,{}, {'report_review':packet})
    assert result['status']==expected and result['review_scope']=='text_report'
    assert len(requests)==1 and 'images' not in requests[0]
    state=json.loads(requests[0]['state'])
    assert state['deliverable']==delivery()['text'] and state['observed_sources'][0]['text']
    assert '浏览器页面显示' in requests[0]['questions']['step']['instructions']
    assert judge.last_call['purpose']=='report_review'
    judge.close()


class TextReviewer:
    vision=False
    last_call={}
    def __init__(self,status='passed'):self.calls=[];self.status=status
    def chat(self,messages,**kwargs):
        self.calls.append(copy.deepcopy(messages))
        packet=json.loads(messages[1]['content'])
        assert packet['deliverable'] and packet['observed_sources']
        assert 'images' not in packet
        return json.dumps({'status':self.status,'reason':'按来源逐项核对功能说明',
                           'evidence':['S1 的课程、文档及社区入口支持回答']})


@pytest.mark.parametrize('jev_label,tie,fallback_status,expected,fallback_calls', [
    ('passed',False,'passed','passed',0),
    ('passed',True,'passed','passed',1),
    ('unknown',False,'unknown','unknown',1),
    ('failed',False,'passed','failed',0),
])
def test_unknown_gets_one_text_review_failed_is_not_overridden(jev_label,tie,fallback_status,expected,fallback_calls):
    requests=[];judge=judge_with(requests,jev_label,tie)
    model=TextReviewer(fallback_status);reviews=[];calls=[]
    state=StepState(task=TASK);state.report_delivery=delivery();state.report_sources=delivery()['sources']
    service=ReviewService(ReviewPorts(completion=None,decision_judge=judge,reviewer=StepReviewer(model),
        remaining=lambda:60,record_typed_judge=lambda:None,
        recheck_jev=lambda *a:pytest.fail('Do not recapture dynamic pages for text uncertainty'),
        publish_review=reviews.append,record_call=lambda *a:calls.append(a)))
    report=service._review_visual(state,{'id':'whole_task'}, {}, {'status':'unknown','checks':[]}, {'plan_step_id':'whole_task'})
    assert report['status']==expected and len(model.calls)==fallback_calls
    assert len(requests)==1
    if fallback_calls:
        assert report['decision_judgment']['status']=='unknown'
        assert calls[0][0]['purpose']=='report_review_fallback'
    judge.close()


@pytest.mark.parametrize('mutate', ['', 'answer', 'source', 'final_url', 'uncertain'])
def test_site_description_episode_revalidates_text_not_dynamic_pixels(monkeypatch,mutate):
    from PIL import Image
    from osworld_agent.pipeline import Pipeline
    from test_pipeline_completion import Model,Env
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status',None)
    env=Env();original=env.execute_routed
    def execute(action,route):
        r=original(action,route)
        env.obs.context[0]['nodes'].append({'ref':'2','role':'StaticText','name':'文档：查阅资料；社区：交流讨论；课程：在线学习'})
        return r
    env.execute_routed=execute
    plan={'steps':[
        {'goal':'打开入口','expected_evidence':'入口页面打开','verification':[{'kind':'url_equals','url':'https://example.test/'}]},
        {'goal':'进入3MS查看用途','expected_evidence':'3MS 功能页面可读','verification':[{'kind':'url_equals','url':'https://example.test/3ms'}]},
        {'goal':'说明3MS用途','expected_evidence':'基于来源给出功能说明','phase':'report','verification':[{'kind':'visual'}]}]}
    model=Model([{'type':'goto','url':'https://example.test/'},{'type':'goto','url':'https://example.test/3ms'}],plan)
    def stream(messages,on_delta,timeout):
        text='3MS 支持查阅文档、社区交流和课程学习 [S2]。'
        on_delta(text);return text
    model.chat_stream=stream
    requests=[]
    judge=judge_with(requests, label='unknown' if mutate=='uncertain' else 'passed')
    class TestedPipeline(Pipeline):
        def _evaluate(self,result):
            if mutate=='uncertain':
                assert self._model_final_report is None
                return super()._evaluate(result)
            assert self._model_final_report['review_scope']=='text_report'
            env.obs.screenshot=Image.new('RGB',(100,80),'red')
            if mutate:
                if mutate=='answer':self._reviewed_state.report_delivery['text']='新增无来源功能 [S99]'
                elif mutate=='source':self._reviewed_state.report_delivery['sources']=[]
                else:env.obs.context[0]['url']='https://example.test/wrong'
            if mutate!='final_url':
                self._observe=lambda:pytest.fail('Dynamic screenshot alone must not re-open whole-task review')
            return super()._evaluate(result)
    spec={'checks':[{'kind':'url_equals','url':'https://example.test/3ms','scope':'final'}]} if mutate=='final_url' else None
    pipe=TestedPipeline(model=model,env=env,grounding=None,system_prompt='test',max_steps=5,completion_spec=spec)
    pipe.decision_judge=judge
    if mutate=='uncertain':pipe.reviewer=StepReviewer(TextReviewer('unknown'))
    result=pipe.run(TASK)
    assert result.success is (not mutate)
    assert len(env.actions)==2
    text_requests=[p for p in requests if 'deliverable' in json.loads(p['state'])]
    assert len(text_requests)==1  # invalid new content blocked before another model call
    if not mutate:
        assert any(e['event_type']=='final_review_reused' and e.get('review_scope')=='text_report' for e in result.events)
        assert result.verification['review_scope']=='text_report'
    elif mutate=='uncertain':
        assert result.termination_reason=='completion_inconclusive'
        assert result.run_manifest['report_delivery']['verification_status']=='unknown'
        assert len(pipe.reviewer.model.calls)==1
    else:assert result.termination_reason=='verification_changed'
    judge.close()


def test_text_review_purposes_and_timings_are_visible(request):
    if not request.config.getoption('--run-browser-integration'):
        pytest.skip('browser opt-in')
    from pathlib import Path
    from playwright.sync_api import sync_playwright, expect
    from osworld_agent.viz import server
    static=Path(server.__file__).parent/'static'
    data={'instruction':TASK,'actions':[{'step':4,'action':{'action':'request_finish'},
        'calls':[{'purpose':'report_review','model':'d3-flash','latency_ms':165,'request':{'questions':{'step':{'type':'choice'}}},
                  'response':{'answers':{'step':{'label':'unknown','probabilities':{'passed':.416,'unknown':.414}}}}},
                 {'purpose':'report_review_fallback','model':'qwen-test','latency_ms':1230,'messages':[{'role':'user','content':'原问题、正文与来源快照'}],
                  'response':'{"status":"passed"}'}]}], 'revisions':[], 'outcome':{'success':True},'report':{},'metrics':{}}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True);page=browser.new_page()
        page.set_content('<div id="local-body"></div>')
        page.add_style_tag(path=str(static/'workbench.css'));page.add_script_tag(path=str(static/'workbench.js'))
        page.evaluate('(data)=>{const w=new AgentWorkbench.Workbench(document.querySelector("#local-body"),true);w.mount(data,"archive")}',data)
        expect(page.locator('.wb-call').nth(0)).to_contain_text('Jev · 文字结果与来源核查')
        expect(page.locator('.wb-call').nth(0)).to_contain_text('165 ms')
        expect(page.locator('.wb-call').nth(1)).to_contain_text('模型 · 独立文字语义复核')
        expect(page.locator('.wb-call').nth(1)).to_contain_text('1.2 s')
        browser.close()
