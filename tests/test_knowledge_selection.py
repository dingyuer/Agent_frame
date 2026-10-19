import copy
import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from osworld_agent.knowledge_base import (load_knowledge, save_knowledge, entry_markdown,
    parse_entry_markdown, document_markdown, parse_markdown_document)
from osworld_agent.knowledge_selection import select_knowledge, selection_cli_args
from test_knowledge_base import entry, snapshot


def test_markdown_roundtrip_and_conflict(tmp_path):
    doc = snapshot(entry(content='# 校园\n\n- 公告入口\n\n```text\n正文\n```'),entry('other'))['document']
    text = document_markdown(doc)
    assert parse_markdown_document(text) == doc
    assert parse_entry_markdown(entry_markdown(doc['entries'][0])) == doc['entries'][0]
    path = tmp_path/'knowledge.md'
    first = load_knowledge(path)
    saved = save_knowledge(doc, first['revision'], path)
    assert path.read_text().startswith('---\n') and saved['document'] == doc
    assert load_knowledge(path)['entry_markdown'] == [entry_markdown(e) for e in doc['entries']]
    with pytest.raises(FileExistsError):save_knowledge(doc,first['revision'],path)
    with pytest.raises(ValueError):parse_markdown_document(text+text)
    with pytest.raises(ValueError):parse_entry_markdown('# no metadata')


def test_all_is_default_no_model_calls_and_keeps_full_enabled_entries():
    doc = snapshot(entry(),entry('off',enabled=False),entry('erp',title='财务系统'))
    original=copy.deepcopy(doc)
    out=select_knowledge('校园公告',doc)
    assert out['method']=='all' and out['selected_ids']==['campus','erp'] and out['calls']==[]
    assert out['entries'][0]['content']==doc['document']['entries'][0]['content']
    assert doc==original and out['latency_ms']>=0


class Model:
    last_call={'model':'fake','usage':{'prompt_tokens':10,'completion_tokens':4}}
    def __init__(self,ids):self.ids=ids
    def chat(self,messages,**kw):
        self.messages=messages
        return json.dumps({'selected_ids':self.ids,'reason':'任务相关'})


def test_decision_receives_all_candidates_returns_originals_and_rejects_hallucinated_ids():
    doc=snapshot(entry(),entry('erp',title='财务系统'))
    m=Model(['campus','campus']);out=select_knowledge('查询校园公告',doc,method='decision',model=m)
    assert out['selected_ids']==['campus']
    assert len(json.loads(m.messages[1]['content'])['knowledge'])==2
    assert out['calls'][0]['actor_role']=='knowledge_selector' and not out['calls'][0]['policy_trainable']
    with pytest.raises(ValueError):select_knowledge('查询',doc,method='decision',model=Model(['invented']))
    assert select_knowledge('查询',doc,method='decision',model=Model([]))['entries']==[]


class Judge:
    def __init__(self):self.requests=[];self.last_call={}
    def decide(self,state,questions,**kw):
        assert len(questions)<=32 and len(json.dumps(state,ensure_ascii=False))<=32000
        self.requests.append((state,questions));self.last_call={'model':'d3-flash','request':state}
        return {'status':'answered','answers':{k:{'top_label':('unknown' if i%2 else 'relevant'),'probabilities':{}} for i,k in enumerate(questions)}}


def test_jev_batches_without_losing_entries_keeps_uncertain_and_empty_skips_models():
    doc=snapshot(*(entry(str(i),content='资料'*1000) for i in range(35)))
    judge=Judge();out=select_knowledge('查询校园公告',doc,method='jev',judge=judge)
    assert len(out['entries'])==35 and len(judge.requests)>1
    assert len(out['calls'])==len(judge.requests)
    assert select_knowledge('查询',snapshot(),method='jev')['entries']==[]
    with pytest.raises(ValueError):select_knowledge('查询',doc,method='unknown')


def test_markdown_api_preview_has_no_write_and_methods_share_result_shape(monkeypatch,tmp_path):
    monkeypatch.setenv('AGENT_KNOWLEDGE_PATH',str(tmp_path/'knowledge.md'))
    from osworld_agent.viz.server import app
    from osworld_agent.model.decision_model import ChatModel
    monkeypatch.setattr(ChatModel,'chat',lambda self,*a,**kw:json.dumps({'selected_ids':['campus']}))
    client=TestClient(app);loaded=client.get('/knowledge').json();draft=[entry_markdown(entry())]
    for method in ('all','decision'):
        result=client.post('/knowledge/preview',json={'task':'校园公告','method':method,'markdown_entries':draft}).json()
        assert result['ok'] and result['selected_ids']==['campus'] and result['latency_ms']>=0
    assert not (tmp_path/'knowledge.md').exists()
    saved=client.put('/knowledge',json={'revision':loaded['revision'],'markdown_entries':draft})
    assert saved.status_code==200 and saved.json()['document']['entries'][0]['id']=='campus'
    assert client.post('/knowledge/preview',json={'task':'校园','method':'bad'}).status_code==422
    assert client.put('/knowledge',json={'revision':saved.json()['revision'],'markdown_entries':['# bad']}).status_code==422


def test_runtime_configuration_cli_is_independent_of_action_verifier():
    assert selection_cli_args({})==['--knowledge-method','all']
    assert '--knowledge-jev-url' in selection_cli_args({'knowledge_method':'jev','knowledge_jev_url':'http://localhost:8007'})
    with pytest.raises(ValueError):selection_cli_args({'knowledge_method':'jev'})


@pytest.mark.parametrize('method, broken, expected', [('all',False,['campus','erp']),('decision',False,['campus']),('decision',True,['campus','erp'])])
def test_pipeline_passes_selection_to_planner_and_reuses_snapshot_on_replan(monkeypatch,method,broken,expected):
    from osworld_agent import pipeline as module
    from osworld_agent import knowledge_base
    from osworld_agent.pipeline import Pipeline
    doc=snapshot(entry(),entry('erp',title='财务系统'))
    monkeypatch.setattr(knowledge_base,'load_knowledge',lambda:doc)
    model=Model(['invented'] if broken else ['campus']);model.cfg=SimpleNamespace(max_tokens=0,timeout=1);model.close=lambda:None
    monkeypatch.setattr(module,'make_review_model',lambda _:model)
    received=[]
    monkeypatch.setattr(module.PlanningService,'plan',lambda self,*a,**kw:received.append(copy.deepcopy(kw['knowledge'])) or True)
    pipe=Pipeline(model=SimpleNamespace(),grounding=None,env=SimpleNamespace(),system_prompt='',knowledge_method=method)
    pipe.completion=SimpleNamespace(public_spec={},paper_goals=lambda:[],public_report=lambda _: {})
    pipe._publish_diagnostic=lambda *a,**kw:None
    pipe._record_call=lambda *a,**kw:None
    assert pipe._plan('查校园公告',SimpleNamespace())
    assert [e['id'] for e in received[0]['entries']]==expected
    if broken:assert received[0]['status']=='fallback' and received[0]['requested_method']=='decision'
    monkeypatch.setattr(knowledge_base,'load_knowledge',lambda:pytest.fail('replan must reuse original knowledge snapshot'))
    assert pipe._plan('查校园公告',SimpleNamespace(verification={}),repair=True)
    assert received[1]==received[0]


@pytest.mark.parametrize('repair', [False, True])
def test_selected_knowledge_reaches_actual_planner_prompt(repair):
    from test_planning_service import Model as Planner, VALID, state, service, accepting
    proposal = {**VALID, 'strategy_change': '旧公告入口失效，改从当前首页核实新的公告链接'} if repair else VALID
    model=Planner([proposal]);events=[];planner,record=service(model,events)
    selection=select_knowledge('查询',snapshot(entry(),entry('off',enabled=False)))
    current = state()
    current.memory.failure_memory = ['旧公告入口返回 404']
    assert planner.plan('查询',current,{'checks':[]},[],[],{},accepting(events,record,[]),knowledge=selection,repair=repair)
    prompt=model.calls[0][0][0]['content']
    assert 'https://campus.example.test' in prompt and '通知位于公告栏目' in prompt
    assert '"id": "off"' not in prompt
    if repair:
        assert '旧公告入口返回 404' in prompt and 'strategy_change' in prompt


def test_failed_jev_batch_preserves_previous_calls_for_usage_accounting():
    from osworld_agent.knowledge_selection import KnowledgeSelectionError
    class FailingJudge(Judge):
        def decide(self,*args,**kw):
            result=super().decide(*args,**kw)
            return result if len(self.requests)==1 else {'status':'unknown','error_type':'TimeoutError'}
    judge=FailingJudge()
    with pytest.raises(KnowledgeSelectionError) as err:
        select_knowledge('校园',snapshot(*(entry(str(i)) for i in range(33))),method='jev',judge=judge)
    assert len(err.value.calls)==2
