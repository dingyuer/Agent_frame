import copy
import json
import pytest
from osworld_agent.replanning import recovery_brief,validate_repair
from osworld_agent.pipeline_types import StepState
from osworld_agent.memory import StructuredMemory
from osworld_agent.execution_queue import ExecutionQueue
from test_planning_service import Model,service,accepting


def current_state():
    steps=[{'goal':'进入门户','actions':['goto'],'expected_evidence':'门户显示'},
           {'goal':'点击附件入口','actions':['click'],'expected_evidence':'文件详情显示'}]
    state=StepState(task='下载附件',memory=StructuredMemory(plan={'steps':steps}))
    state.queue=ExecutionQueue(state.memory.plan)
    state.queue.accept_review({'plan_step_id':state.queue.current['id'],'status':'passed'})
    state.memory.add_failure('附件入口被遮挡，连续两次点击被拒绝')
    state.failure_class='blocked'
    return state


def test_completed_steps_are_preserved_and_failures_are_not_model_claims():
    state=current_state();brief=recovery_brief(state)
    assert brief['completed_steps']==['进入门户']
    assert brief['remaining_steps'][0]['goal']=='点击附件入口'
    assert '被遮挡' in brief['recent_failures'][0]


def test_rewording_only_evidence_does_not_bypass_repeat_guard():
    brief=recovery_brief(current_state());plan={'steps':copy.deepcopy(brief['remaining_steps'])}
    plan['steps'][0]['expected_evidence']='换一句验收话术'
    with pytest.raises(ValueError,match='重复'):
        validate_repair(plan,{'strategy_change':'我已经改变策略，现在继续同一个入口'},brief)
    with pytest.raises(ValueError,match='strategy_change'):
        validate_repair(plan,{},brief)


def test_identical_replan_is_rejected_then_obstruction_repair_accepted():
    state=current_state()
    repeated={'steps':copy.deepcopy(state.memory.plan['steps'][1:]),'strategy_change':'继续点击原入口以尝试打开文件'}
    changed={'steps':[{'goal':'关闭遮挡附件入口的弹窗','actions':['click'],'expected_evidence':'弹窗消失'},
                      *copy.deepcopy(state.memory.plan['steps'][1:])],
             'strategy_change':'两次点击被遮挡，先关闭弹窗，再进入附件详情核实文件'}
    model=Model([repeated,changed]);events=[];calls=[];accepted=[]
    planner,record=service(model,events)
    assert planner.plan(state.task,state,{'checks':[]},[],calls,{},accepting(events,record,accepted),repair=True)
    assert [c['valid_plan'] for c in calls]==[False,True]
    assert accepted[0]['strategy_change']==changed['strategy_change']
    assert '两次点击被拒绝' in model.calls[0][0][0]['content']
    assert state.memory.failure_memory  # surviving lessons are not cleared


def test_appending_a_new_tail_cannot_preserve_the_same_blocked_first_step():
    brief=recovery_brief(current_state())
    plan={'steps':brief['remaining_steps']+[{'goal':'检查结果','actions':['observe_browser']}]}
    with pytest.raises(ValueError,match='首个子任务'):
        validate_repair(plan,{'strategy_change':'先继续原来的点击然后检查新结果'},brief)


def test_different_observed_url_is_a_real_change_even_if_punctuation_differs():
    brief={'remaining_steps':[{'goal':'访问 https://campus-a.test','actions':['goto']}]}
    plan={'steps':[{'goal':'访问 https://campusa.test','actions':['goto']}]}
    validate_repair(plan,{'strategy_change':'之前入口失效，使用当前观察到的替代入口'},brief)


def test_knowledge_only_selected_entries_enter_planner_prompt():
    state=current_state();state.task_profile={}
    model=Model([{'steps':[{'goal':'查看官网通知','expected_evidence':'通知列表'}]}]);events=[];calls=[];accepted=[]
    planner,record=service(model,events)
    knowledge={'entries':[{'id':'school','title':'校园官网','url':'https://campus.test','content':'查看公告栏'}]}
    assert planner.plan('查看校园官网通知',state,{'checks':[]},[],calls,{},accepting(events,record,accepted),knowledge=knowledge)
    prompt=model.calls[0][0][0]['content']
    assert 'https://campus.test' in prompt and '不是完成证据' in prompt and '明确网址' in prompt
