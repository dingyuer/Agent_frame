"""Avoid extra model work without weakening file or whole-task verification."""
import copy
from types import SimpleNamespace

import pytest
from osworld_agent.actions import Action
from osworld_agent.download_review_scope import named_download_receipts
from osworld_agent.pipeline import AgentResult, Execution, Pipeline, StepState
from osworld_agent.memory import StructuredMemory
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model, observation, pipeline
from test_download_facts import receipt
from test_pipeline_recovery_wait import clock, EventEnv, metadata, make_pipeline, state_and_decision


def test_explicitly_disabled_transition_diagnostics_skips_advisory_judge(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier('test')
    pipe.decision_judge = SimpleNamespace(config=SimpleNamespace(transition_diagnostics=False), assess_transition=lambda *a, **k: pytest.fail('unnecessary model call'))
    state = StepState(task='test', obs=observation(), memory=StructuredMemory(), verification={'status':'passed'}, step_id=1)
    pipe._update_plan_progress(state, Execution(kind='browser_dom', obs=state.obs, info={'status':'executed'}),
                              action=Action('click', {'target':'button'}))
    assert state.step_review['status'] == 'passed' and state.step_review['method'] == 'code'


def test_download_events_wait_without_model_or_full_screenshots(clock):
    env = EventEnv([metadata(True), metadata(True), metadata(artifacts=[{'download_id':'d1','completed':True,'page_id':'task-page'}])])
    pipe = make_pipeline(env)
    pipe.decision_judge = SimpleNamespace(download_wait=lambda *a, **k: pytest.fail('events already own download state'))
    state, decision = state_and_decision()
    result = pipe._settle_execution(decision.action,
        Execution(kind='browser_dom', obs=state.obs, info={'status':'executed','dispatched':True}), state)
    assert result.info['framework_wait']['status'] == 'completed'
    assert 'download_jev_polls' not in result.info
    assert len(env.actions) == 0


def named_setup(tmp_path, monkeypatch, task='下载 report.pptx'):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    import zipfile
    path = tmp_path/'report.pptx'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('[Content_Types].xml','<Types/>')
        z.writestr('ppt/presentation.xml','<presentation/>')
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier(task)
    pipe.completion.observe({'info':{'artifacts':[receipt(path)]}})
    state = StepState(task=task, obs=observation(color='red'), memory=StructuredMemory())
    calls = []
    def review(state, step, execution, code, identity, before=None):
        calls.append(step['goal'])
        return {**identity,'method':'decision_model','status':'passed','reason':'身份及剩余要求已核实'}
    monkeypatch.setattr(pipe,'_review_visual',review)
    return pipe,state,path,calls


def test_named_download_rechecks_bytes_not_disappearing_notification(tmp_path, monkeypatch):
    pipe,state,path,calls = named_setup(tmp_path,monkeypatch)
    assert pipe._verify_finish(state,'done')[0]
    pipe.env.obs = observation(color='blue',value='notification closed')
    result=AgentResult(task=state.task,success=True)
    pipe._evaluate(result)
    assert len(calls)==1 and result.success
    assert result.reward_source=='independent_model_review' and not result.evaluation_available
    assert any(e['event_type']=='final_review_reused' for e in pipe.events)


@pytest.mark.parametrize('change',['delete','corrupt'])
def test_named_scope_never_hides_lost_or_changed_file(tmp_path,monkeypatch,change):
    pipe,state,path,calls = named_setup(tmp_path,monkeypatch)
    assert pipe._verify_finish(state,'done')[0]
    if change=='delete':path.unlink()
    else:path.write_bytes(b'wrong content')
    pipe.env.obs=observation(color='blue')
    result=AgentResult(task=state.task,success=True)
    pipe._evaluate(result)
    assert not result.success and len(calls)==1


def test_compound_task_keeps_fresh_whole_task_review(tmp_path,monkeypatch):
    pipe,state,path,calls = named_setup(tmp_path,monkeypatch,task='下载 report.pptx，然后填写表单')
    assert pipe._verify_finish(state,'done')[0]
    pipe.env.obs=observation(color='blue',value='form changed')
    pipe._evaluate(AgentResult(task=state.task,success=True))
    assert len(calls)==2


def test_receipt_scope_requires_exact_user_name_and_source():
    task='打开https://portal.test/home，去方案中心的方案库中下载 MBB Wireless Network swap SOP 3.0'
    fact={'artifact_id':'a','status':'passed','sha256':'x','path':r'C:\Downloads\MBB Wireless Network swap SOP 3.0.zip','source_urls':['https://portal.test/detail']}
    assert named_download_receipts(task,[fact])==['a']
    assert not named_download_receipts(task,[{**fact,'source_urls':['https://wrong.test/detail']}])
    assert not named_download_receipts(task,[{**fact,'path':'MBB Wireless Network swap SOP 3.0 old.zip'}])
    assert not named_download_receipts(task+'，并发送给同事',[fact])


def test_wall_clock_phases_cover_first_step_and_finalization(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status',None)
    pipe=pipeline(Model([{'type':'goto','url':'https://example.test/'}]),Env(),
        {'checks':[{'kind':'url_equals','url':'https://example.test/'}]})
    result=pipe.run('打开 https://example.test/')
    completed=[e for e in result.events if e['event_type']=='phase_timing' and e['status']=='completed']
    assert {'planning','observation','decision','dispatch','step_review','final_revalidation'} <= {e['phase'] for e in completed}
    assert all(e['elapsed_ms'] >= 0 for e in completed)
    assert result.run_manifest['phase_timings']
