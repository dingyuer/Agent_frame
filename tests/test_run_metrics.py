import json
from types import SimpleNamespace
import pytest

from osworld_agent.run_metrics import summarize_events


def call(ident, usage=None, role='decision', **extra):
    return {'event_type':'model_call','actor_role':role,
            'call':{'call_id':ident,'usage':usage,**extra}}


def test_deduplicate_canonical_calls_keep_missing_usage_and_zero_output():
    a=call('q',{'prompt_tokens':10,'completion_tokens':3})
    events=[{'event_type':'episode_start','timestamp':100},a,a,
            call('j',None,'verifier',response={'usage':{'input_tokens':20,'output_tokens':0}}),
            {'event_type':'jev_call','call':{'usage':{'input_tokens':20,'output_tokens':0}}},
            call('g',{'input_tokens':'invalid'},'grounder'),
            call('p',{'input_tokens':5,'output_tokens':2},'planner'),
            {'event_type':'episode_end','timestamp':130,'elapsed_s':29.5}]
    metrics=summarize_events(events)
    assert (metrics['input_tokens'],metrics['output_tokens'],metrics['calls'])==(35,5,4)
    assert metrics['elapsed_s']==29.5 and metrics['timing_source']=='monotonic'
    assert metrics['missing_usage_calls']==1 and not metrics['usage_complete']
    assert metrics['by_role']['verifier']['output_tokens']==0


def test_old_interrupted_and_exception_time_are_honest():
    start={'event_type':'episode_start','timestamp':100}
    assert summarize_events([])['elapsed_s'] is None
    partial=summarize_events([start,{'event_type':'execution','timestamp':120}])
    assert partial['elapsed_s']==20 and partial['timing_source']=='observed_span' and not partial['complete']
    failed=summarize_events([start,{'event_type':'episode_error','timestamp':125,'elapsed_s':24}])
    assert failed['elapsed_s']==24 and failed['complete']


def test_grounding_usage_records_coarse_fine_and_failed_request():
    from osworld_agent.model.grounding_usage import completion
    responses=iter([SimpleNamespace(id='a',usage={'prompt_tokens':30,'completion_tokens':2}),
                    SimpleNamespace(id='b',usage={'prompt_tokens':12,'completion_tokens':3}),RuntimeError('offline')])
    def create(**kw):
        response=next(responses)
        if isinstance(response,Exception):raise response
        return response
    owner=SimpleNamespace(model='grounder',last_calls=[],_get_client=lambda:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    completion(owner);completion(owner)
    with pytest.raises(RuntimeError):completion(owner)
    result=summarize_events([{'event_type':'model_call','actor_role':'grounder','call':c} for c in owner.last_calls])
    assert result['calls']==3 and result['input_tokens']==42 and result['output_tokens']==5
    assert result['missing_usage_calls']==1


def test_event_only_detail_and_list_share_statistics(monkeypatch,tmp_path):
    from osworld_agent.viz import server
    path=tmp_path/'events.jsonl'
    events=[{'event_type':'episode_start','timestamp':10,'manifest':{'task':'test'}},
            call('a',{'input_tokens':12,'output_tokens':1}),
            {'event_type':'episode_error','timestamp':15,'elapsed_s':5}]
    path.write_text('\n'.join(json.dumps(e) for e in events)+'\n{broken')
    monkeypatch.setattr(server,'_local_run_path',lambda run:(path,tmp_path,''))
    detail=server._read_local_traj('test')
    assert detail['ok'] and detail['tokens']['input_tokens']==12 and detail['tokens']['elapsed_s']==5
    (tmp_path/'run_timing.json').write_text(json.dumps({'elapsed_s':7.25}))
    updated=server._read_local_traj('test')['tokens']
    assert updated['elapsed_s']==7.25 and updated['pipeline_elapsed_s']==5
    assert updated['timing_source']=='process'


def test_process_reader_freezes_total_duration_and_persists(monkeypatch,tmp_path):
    from osworld_agent.viz import server
    proc=SimpleNamespace(stdout=[],wait=lambda:0)
    monkeypatch.setattr(server,'_agent_state',{'proc':proc,'status':'running','output':[]})
    monkeypatch.setattr(server.time,'perf_counter',lambda:25.0)
    server._reader_thread(proc,str(tmp_path/'run_timing.json'),10.0)
    assert server._agent_state['elapsed_final']==15
    assert json.loads((tmp_path/'run_timing.json').read_text())['elapsed_s']==15


def test_grounder_calls_enter_canonical_events_as_nontrainable():
    from test_pipeline_completion import Model,Env
    from osworld_agent.pipeline import Pipeline
    grounder=SimpleNamespace(last_calls=[])
    def locate(*args):
        grounder.last_calls=[{'usage':{'input_tokens':10,'output_tokens':1}},
                             {'usage':{'input_tokens':5,'output_tokens':2}}]
        return 'result'
    grounder.locate=locate
    pipe=Pipeline(model=Model([]),env=Env(),grounding=grounder,system_prompt='test')
    assert pipe._locate_recorded(None,'target')=='result'
    assert summarize_events(pipe.events)['input_tokens']==15
    assert all(e['actor_role']=='grounder' and e['call']['policy_trainable'] is False for e in pipe.events)


def test_failed_grounding_preparation_never_recounts_previous_usage():
    from test_pipeline_completion import Model,Env
    from osworld_agent.pipeline import Pipeline
    grounder=SimpleNamespace(last_calls=[{'usage':{'input_tokens':10,'output_tokens':1}}])
    def locate(*args):raise RuntimeError('image preparation failed before request')
    grounder.locate=locate
    pipe=Pipeline(model=Model([]),env=Env(),grounding=grounder,system_prompt='test')
    with pytest.raises(RuntimeError):pipe._locate_recorded(None,'target')
    assert not pipe.events
