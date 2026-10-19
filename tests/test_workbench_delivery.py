import copy
from types import SimpleNamespace
import pytest

from osworld_agent.report_delivery import collect_sources, generate_report
from osworld_agent.pipeline_types import StepState
from osworld_agent.env import Observation
from osworld_agent.memory.structured_memory import StructuredMemory
from osworld_agent.execution_queue import ExecutionQueue
from osworld_agent.viz.workbench import project_workbench
from osworld_agent.model.decision_model import ChatModel
from osworld_agent.config import ModelConfig


def state():
    return SimpleNamespace(task='比较两门课程的费用并总结', step_id=3, report_sources=[], report_delivery={},
        obs=Observation(context=[{'kind':'browser_ax','url':'https://example.org/course','title':'课程',
            'nodes':[{'role':'StaticText','name':'课程 A 费用 480 元'}]}],info={}))


def test_source_collection_stream_and_fingerprint():
    s=state();collect_sources(s);collect_sources(s)
    assert len(s.report_sources)==1
    events=[];calls=[]
    class Model:
        last_call={'usage':{'input_tokens':20,'output_tokens':10}}
        def chat_stream(self,messages,on_delta,timeout):
            assert '480' in messages[-1]['content']
            # Force multiple real stream flushes without artificial sleeps.
            parts=['课程 A 费用为 480 元 [S1]。'+(' '*200),'证据不足以比较第二门课程 [S1]。']
            for part in parts:on_delta(part)
            return ''.join(parts)
    r=generate_report(s,Model(),lambda e,**p:events.append((e,copy.deepcopy(p))),lambda c,role:calls.append((c,role)),lambda:60)
    assert r['status']=='complete' and r['verification_status']=='pending'
    chunks=[p for e,p in events if e=='report_delta']
    assert len(chunks)==2 and ''.join(p['delta'] for p in chunks)==r['text']
    assert chunks[-1]['text']==r['text'] and calls[0][1]=='reporter'
    size=len(events)
    assert generate_report(s,Model(),lambda e,**p:events.append((e,p)),lambda *a:None,lambda:60)==r
    assert len(events)==size


def test_heading_list_context_survives_navigation_without_old_refs():
    from osworld_agent.report_delivery import observed_navigation_hints
    s = state()
    s.task = '查找实验团队并输出名单'
    s.obs.context[0]['nodes'] = [
        {'ref':'0','role':'RootWebArea'},
        {'ref':'1','parent':'0','role':'heading','name':'实验团队'},
        {'ref':'2','parent':'0','role':'list'},
        {'ref':'3','parent':'0','role':'heading','name':'其他信息'},
        {'ref':'4','parent':'0','role':'list'},
        {'ref':'5','parent':'2','role':'listitem'},
        {'ref':'6','parent':'4','role':'link','name':'不相关链接'},
        {'ref':'7','parent':'5','role':'link','name':'实际团队甲'},
        {'ref':'8','parent':'7','role':'StaticText','name':'实际团队甲'},
    ]
    collect_sources(s)
    s.obs.context[0]['url'] = 'https://example.org/empty'
    s.obs.context[0]['nodes'] = [{'role':'heading','name':'空栏目'}]
    collect_sources(s)
    hints = observed_navigation_hints(s)
    assert hints[0]['url'] == 'https://example.org/course'
    assert hints[0]['sections'][0]['items'] == ['实际团队甲']
    assert 'ref' not in str(hints) and '不相关链接' not in str(hints)
    assert '不证明' in hints[0]['sections'][0]['completeness']


def test_section_does_not_cross_another_heading_or_invent_list_membership():
    from osworld_agent.report_delivery import _observed_sections
    block = {'nodes': [
        {'ref':'1','parent':'0','role':'heading','name':'实验团队'},
        {'ref':'2','parent':'0','role':'heading','name':'新闻'},
        {'ref':'3','parent':'0','role':'list'},
        {'ref':'4','parent':'3','role':'link','name':'新闻条目'},
    ]}
    assert _observed_sections(block, '实验团队') == []


def test_report_receives_real_file_receipts_and_invalidates_after_file_changes():
    import json
    from osworld_agent.report_delivery import report_fingerprint
    s=state();collect_sources(s)
    before=report_fingerprint(s)
    s.obs.info={'cli_capability':{'workspace':'/workspace'},
        'cli_result':{'operation':'open','path':'/workspace/list.txt','status':'executed','effect_verified':False},
        'cli_checks':[{'predicate':{'kind':'file_exists','path':'/workspace/list.txt'},
                       'result':{'operation':'stat','file':{'path':'/workspace/list.txt','exists':True,'bytes':80}}}]}
    assert report_fingerprint(s)!=before
    class Model:
        last_call={}
        def chat(self,messages,retries):
            info=json.loads(messages[-1]['content'])['system_evidence']
            assert info['cli_result']['path']=='/workspace/list.txt'
            assert info['cli_result']['effect_verified'] is False
            assert info['file_checks'][0]['result']['file']['bytes']==80
            return '已保存到 /workspace/list.txt；打开请求已派发。内容来源 [S1]。'
    result=generate_report(s,Model(),lambda *a,**k:None,lambda *a:None,lambda:60)
    assert result['status']=='complete'


@pytest.mark.parametrize('mode',['bad_reference','missing_reference','disconnected'])
def test_report_does_not_claim_complete_with_bad_sources_or_transport(mode):
    s=state();collect_sources(s);events=[]
    class Model:
        last_call={}
        def chat_stream(self,messages,on_delta,timeout):
            text='已经完整交付' if mode=='missing_reference' else '未知内容 [S99]'
            on_delta(text)
            if mode=='disconnected':raise ConnectionError('private upstream details')
            return text
    r=generate_report(s,Model(),lambda e,**p:events.append((e,p)),lambda *a:None,lambda:60)
    assert r['status']=='failed' and r['text'] and r['verification_status']=='pending'
    assert 'private upstream details' not in r['error']
    assert not any(e=='report_complete' for e,p in events)


def test_stream_transport_preserves_order_usage_and_does_not_retry_partial():
    seen=[];requests=[]
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __iter__(self):
            for text in ['第一段','第二段']:
                yield SimpleNamespace(id='stream1',usage=None,choices=[SimpleNamespace(finish_reason=None,delta=SimpleNamespace(content=text))])
            yield SimpleNamespace(id='stream1',usage=SimpleNamespace(model_dump=lambda:{'prompt_tokens':33,'completion_tokens':7}),choices=[])
    def create(**kw):requests.append(kw);return Stream()
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options=lambda **kwargs:client
    model=ChatModel(ModelConfig());model._client=client
    assert model.chat_stream([{'role':'user','content':'总结'}],seen.append,timeout=30)=='第一段第二段'
    assert seen==['第一段','第二段'] and len(requests)==1
    assert requests[0]['stream'] and model.last_call['usage']['prompt_tokens']==33
    assert model.last_call['first_token_ms'] is not None


def test_replan_retains_completed_ids_and_projection_uses_final_revision_states():
    first={'steps':[{'goal':'打开网站'},{'goal':'搜索'}]};q=ExecutionQueue(first)
    q.accept_review({'plan_step_id':'p0_1','status':'passed'})
    old=copy.deepcopy(q.snapshot())
    next_plan={'steps':[{'goal':'换入口搜索'},{'goal':'总结'}]};q2=ExecutionQueue(next_plan,revision=1,previous=q)
    assert q2.current['id']=='p1_1' and q2.cursor==1
    assert q2.plan['steps'][0]['id']=='p0_1' and q2.plan['steps'][0]['status']=='done'
    assert q.snapshot()==old
    t={'steps':[{'step':1,'plan_revision':0,'plan_step_id':'p0_1'},{'step':2,'plan_revision':1,'plan_step_id':'p1_1'}],
       'events':[{'event_type':'plan_revision','revision':0,'queue':old},
                 {'event_type':'plan_revision','revision':1,'queue':q2.snapshot()},
                 {'event_type':'execution_queue','queue':{**q2.snapshot(),'cursor':3}}],
       'run_manifest':{'private_evaluation':{'secret':'must not leak'}}}
    v=project_workbench(t)
    assert v['revisions'][-1]['queue']['cursor']==3
    assert v['actions'][0]['plan_revision']==0
    assert 'private_evaluation' not in str(v)


def test_legacy_association_is_not_invented():
    v=project_workbench({'steps':[{'step':4}], 'plan':{'steps':[{'goal':'a'}]}})
    assert v['actions'][0]['plan_step_id'] is None
    assert v['revisions'][0]['legacy']


def test_page_contract_cannot_complete_an_undelivered_report(monkeypatch):
    from test_pipeline_completion import Model,Env
    from osworld_agent.pipeline import Pipeline
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status',None)
    plan={'steps':[{'goal':'打开页面','expected_evidence':'结果页面已打开','verification':[{'kind':'url_equals','url':'https://example.org/result'}]}]}
    model=Model([{'type':'goto','url':'https://example.org/result'}],plan)
    result=Pipeline(model=model,env=Env(),grounding=None,system_prompt='test',max_steps=1,
        completion_spec={'checks':[{'kind':'url_equals','url':'https://example.org/result'}]}).run('查看页面并总结信息')
    assert not result.success
    assert result.verification['status']!='passed'
    assert result.run_manifest['report_delivery']['status']=='failed'


def test_live_reconnect_state_survives_debug_log_limit_and_is_incremental(monkeypatch):
    from collections import OrderedDict
    from osworld_agent.viz import server
    import asyncio
    monkeypatch.setattr(server,'_workbench_cache',OrderedDict())
    monkeypatch.setattr(server,'_workbench_sequence',0)
    monkeypatch.setattr(server,'_workbench_run',None)
    server._cache_workbench({'event':'workbench_start','run_id':'r','instruction':'long task'})
    server._cache_workbench({'event':'workbench_record','run_id':'r','record':{'event_type':'action_context','step':1,'event_id':'e1'}})
    for n in range(500):server._cache_workbench({'event':'live_frame','run_id':'r','screenshot':'frame'+str(n)})
    full=asyncio.run(server.history())
    assert len(full['logs'])==3 and full['logs'][0]['event']=='workbench_start'
    assert full['logs'][-1]['screenshot']=='frame499'
    assert asyncio.run(server.history(since=full['cursor']))['logs']==[]
    server._cache_workbench({'event':'report_delta','run_id':'r','text':'累计文本','sequence':5})
    delta=asyncio.run(server.history(since=full['cursor']))
    assert len(delta['logs'])==1 and delta['logs'][0]['text']=='累计文本'
    server._cache_workbench({'event':'workbench_start','run_id':'new','instruction':'next task'})
    assert len(asyncio.run(server.history())['logs'])==1


@pytest.mark.parametrize('coverage_label,expected',[('failed','failed'),('unknown','unknown'),('passed','passed')])
def test_report_requirement_questions_can_veto_overall_pass(coverage_label,expected):
    import json,httpx
    from osworld_agent.model.decision_judge import DecisionJudge
    from osworld_agent.report_delivery import report_requirements
    from test_pipeline_completion import observation
    from test_decision_judge import answer
    task='比较每小时费用，列出计算过程和来源，说明哪一门总费用更低。'
    requirements=report_requirements(task)
    assert len(requirements)==3 and '总费用' in requirements[-1]
    requests=[]
    def handle(request):
        payload=json.loads(request.content);requests.append(payload)
        answers={k:answer(q['criteria'],coverage_label if k=='coverage_3' else 'passed') for k,q in payload['questions'].items()}
        return httpx.Response(200,json={'model':'d3-flash','answers':answers})
    judge=DecisionJudge({'url':'http://judge.test/v1','review_mode':'exclusive'},httpx.MockTransport(handle))
    result=judge.review(task,{'id':'whole_task','goal':task},None,observation(),{},
        {'report_delivery':{'text':'单位价 40 和 37.5','requirements':requirements},'observed_sources':[]})
    assert len(requests)==1 and len(requests[0]['questions'])==4
    assert result['status']==expected and result['coverage'][-1]['status']==coverage_label
    judge.close()


def test_incomplete_event_log_retains_sampled_output_and_partial_report():
    from osworld_agent.viz.workbench import project_event_workbench
    events=[{'event_type':'action_context','step':2,'plan_revision':0,'plan_step_id':'p0_1','observation':{'screenshot':'assets/'+('a'*64)+'.png'}},
            {'event_type':'model_call','event_id':'e3','parent_decision_id':'a_2','call':{'call_id':'c3','actor_role':'decision','messages':[{'role':'user','content':'input'}],'response':'{"next_action":{"type":"click"}}'}},
            {'event_type':'report_start','version':1,'sources':[{'id':'S1'}]},
            {'event_type':'report_delta','version':1,'sequence':1,'text':'已生成的部分','delta':'已生成的部分'}]
    p=project_event_workbench(events,'任务',{'input_tokens':25})
    assert p['incomplete'] and 'success' not in p['outcome']
    assert p['actions'][0]['policy_output']=='{"next_action":{"type":"click"}}'
    assert p['actions'][0]['observation']['screenshot'].startswith('assets/')
    assert p['report']['text']=='已生成的部分' and p['report']['status']=='streaming'


def test_coverage_repair_is_separate_from_original_output_and_records_new_version():
    s=state();collect_sources(s)
    s.report_delivery={'status':'needs_revision','version':1,'text':'只有单位价',
        'revision_feedback':[{'requirement':'说明总费用更低的课程','status':'failed'}]}
    events=[]
    class Model:
        last_call={}
        def chat_stream(self,messages,on_delta,timeout):
            assert '说明总费用更低的课程' in messages[-1]['content']
            text='课程 A 总费用 480 元 [S1]。另一课程尚无来源，无法完整比较。'
            on_delta(text);return text
    report=generate_report(s,Model(),lambda e,**p:events.append((e,p)),lambda *args:None,lambda:60)
    assert report['version']==2 and report['verification_status']=='pending'
    assert events[0][0]=='report_start' and events[0][1]['version']==2


@pytest.mark.parametrize('second_passes',[True,False])
def test_pipeline_repairs_text_once_without_an_extra_browser_action(monkeypatch,second_passes):
    import json,httpx
    from osworld_agent.pipeline import Pipeline
    from osworld_agent.model.decision_judge import DecisionJudge
    from test_pipeline_completion import Model,Env
    from test_decision_judge import answer
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status',None)
    env=Env();original=env.execute_routed
    def execute(a,r):
        result=original(a,r)
        env.obs.context[0]['nodes'].append({'role':'StaticText','name':'课程 A：480 元，12 小时；课程 B：600 元，16 小时。'})
        return result
    env.execute_routed=execute
    plan={'steps':[{'goal':'读取课程页面','expected_evidence':'课程页面已打开','phase':'collect',
                   'verification':[{'kind':'url_equals','url':'https://example.test/courses'}]},
                  {'goal':'比较课程费用','expected_evidence':'回答单位费用和总费用问题','phase':'report','verification':[{'kind':'visual'}]}]}
    model=Model([{'type':'goto','url':'https://example.test/courses'}],plan)
    reports=[]
    def stream(messages,on_delta,timeout):
        text=('A 每小时 40 元、B 每小时 37.5 元 [S1]。' if not reports else
              'A 每小时 40 元、B 每小时 37.5 元；B 单位费用更低。A 总费用 480 元小于 B 的 600 元，总费用 A 更低 [S1]。')
        reports.append(text);on_delta(text);return text
    model.chat_stream=stream
    whole_calls=[]
    def handle(request):
        payload=json.loads(request.content)
        if 'coverage_2' in payload['questions']:whole_calls.append(payload)
        selected={'next_mode':'dom','settled':'complete','correct':'yes','replan':'no'}
        values={}
        for k,q in payload['questions'].items():
            chosen=selected.get(k,'passed' if 'passed' in q['criteria'] else next(iter(q['criteria'])))
            if k=='coverage_2' and (len(whole_calls)==1 or not second_passes):chosen='failed'
            values[k]=answer(q['criteria'],chosen)
        return httpx.Response(200,json={'model':'d3-flash','answers':values})
    judge=DecisionJudge({'url':'http://judge.test/v1','review_mode':'exclusive','recheck_unknown':False},httpx.MockTransport(handle))
    pipe=Pipeline(model=model,env=env,grounding=None,system_prompt='test',max_steps=1)
    pipe.decision_judge=judge
    result=pipe.run('比较每小时费用，说明哪一门总费用更低。')
    assert result.success is second_passes and result.steps==1 and len(env.actions)==1
    assert len(reports)==2 and len(whole_calls)==2
    assert result.run_manifest['report_delivery']['version']==2
    assert result.run_manifest['report_delivery']['verification_status']==('passed' if second_passes else 'not_passed')
    if not second_passes:assert result.termination_reason=='report_review_failed'
    assert len([e for e in result.events if e['event_type']=='report_start'])==2
    judge.close()


def test_explicit_tables_and_team_lists_use_report_delivery():
    from osworld_agent.report_delivery import wants_report, observed_navigation_hints
    assert wants_report('读取排名前10的模型名称和分数，最终用中文表格展示排名、模型、分数')
    assert wants_report('最终在界面中展示团队名单与文件保存位置')
    assert not wants_report('打开浏览器并下载附件')
    s=state();s.task='查找研究团队并写入txt'
    s.obs.context[0]['nodes'].append({'role':'heading','name':'研究团队','ref':'88'})
    collect_sources(s)
    s.obs=Observation(context=[{'kind':'browser_ax','url':'https://example.org/empty','title':'空页','nodes':[]}])
    collect_sources(s)
    hints=observed_navigation_hints(s)
    assert hints==[{'url':'https://example.org/course','headings':['研究团队']}]
    assert 'ref' not in str(hints)


def test_verified_file_text_survives_later_open_for_review():
    from osworld_agent.report_delivery import report_system_evidence
    s = state()
    s.written_files = [{'operation': 'write', 'effect_verified': True, 'path': '/work/a.txt',
                        'content_preview': '名单共2个：甲、乙', 'content_truncated': False}]
    s.obs.info['cli_result'] = {'operation': 'open', 'path': '/work/a.txt', 'status': 'executed'}
    evidence = report_system_evidence(s)
    assert evidence['cli_result']['operation'] == 'open'
    assert evidence['written_files'][0]['content_preview'] == '名单共2个：甲、乙'
    s.written_files[0]['content_preview'] = 'changed'
    assert evidence['written_files'][0]['content_preview'] != 'changed'
