"""Opt-in real Chromium frames + real remote Jev mode selection and dispatch."""
import json
import os

import pytest

from osworld_agent.actions import Action
from osworld_agent.browser_session import browser_thread
from osworld_agent.env import StepResult
from osworld_agent.model.judge_selection import RuntimeDecisionJudge
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.scene import browser_scene
from test_browser_live import live_browser, observe
from test_pipeline_completion import Model


def test_real_jev_selects_dom_then_gui_after_current_goal_changes(live_browser):
    endpoint=os.environ.get('AGENTFRAME_JEV_TEST_URL')
    if not endpoint:
        pytest.skip('Set AGENTFRAME_JEV_TEST_URL explicitly for real Jev calls')
    session,_=live_browser
    document='''<meta charset=utf-8><title>Mode selection fixture</title>
      <style>body{font:24px Arial;padding:35px}button{font-size:24px;padding:20px}
      #visual{position:absolute;left:80px;top:250px;width:400px;height:130px;background:#b8dcff;
       border:2px solid #2671c5;display:grid;place-items:center;cursor:pointer}</style>
      <h1>Choose a control</h1>
      <button id="dom" onclick="this.textContent='DOM action completed'">DOM Action</button>
      <div id="visual" onclick="this.textContent='Visual action completed'">Visual Card</div>'''
    browser_thread(lambda self:self._pick_page(self._connect()[1]).set_content(document))(session)

    class BrowserEnv:
        browser_only=True
        os_name='mac'
        def route_action(self,action,observation):
            # Deliberately try a semantic DOM match for unreferenced actions:
            # the pipeline must skip this path after Jev selected visual GUI.
            if not action.args.get('target_ref'):
                raise AssertionError('GUI target unexpectedly entered DOM router')
            return session.route(action,observation)
        def execute_routed(self,action,route):
            info=session.execute(action,route)
            return StepResult(observe(session),info=info)
        def step(self,action):
            info=session.execute_viewport_action(action,action.args.get('observation_metadata'))
            return StepResult(observe(session),info=info)

    pipe=Pipeline(model=Model([]),env=BrowserEnv(),grounding=None,system_prompt='test')
    pipe.decision_judge=RuntimeDecisionJudge({'url':endpoint,'timeout':5},selection_policy='maximum_probability')
    state=StepState(task='Click DOM Action, then click Visual Card.',step_id=1)
    from osworld_agent.execution_queue import ExecutionQueue
    state.queue=ExecutionQueue({'steps':[
        {'goal':'Click the DOM Action button.','expected_evidence':'Button text changes'},
        {'goal':'Click the Visual Card. Its div has no actionable AX ref.','expected_evidence':'Card text changes'}]})
    try:
        state.obs=observe(session)
        scene=browser_scene(state.obs.context[0],state.obs.screenshot).to_dict()
        state.obs.info['scene']=scene
        # Deliberately mark the statistical fallback GUI. Jev must still choose
        # the actual button represented in AX, without consulting this hint.
        state.obs.info['scene'].update(mode='browser_gui',reason='content_without_actionable_ax')
        pipe.build_input(state)
        first=state.prompt_metadata['prompt_profile']
        assert (first['mode'],first['source'])==('dom','jev'), first
        ref=next(n['ref'] for n in state.obs.context[0]['nodes'] if n.get('id')=='dom' and n.get('direct'))
        action=pipe.ground(Action('click',{'target':'DOM Action','target_ref':ref}),state)
        executed=pipe.execute(action,state)
        assert executed.info['dispatched'] and state.route['channel']=='browser_dom'
        current_id=state.queue.current['id']
        state.queue.accept_review({'plan_step_id':current_id,'status':'passed'})
        state.obs=executed.obs
        state.obs.info['scene']=browser_scene(state.obs.context[0],state.obs.screenshot).to_dict()
        state.obs.info['scene'].update(mode='browser_dom',reason='structured_page')
        state.step_id=2
        pipe.build_input(state)
        second=state.prompt_metadata['prompt_profile']
        assert (second['mode'],second['source'])==('gui','jev'), second
        # Grounding here is fixed to the real fixture bounds, isolating routing
        # from grounding model accuracy. The click itself is real CDP input.
        pipe._resolve=lambda action,obs:Action('click',{'x':280,'y':315})
        assert pipe._dom_reference_feedback(Action('click',{'target':'Visual Card'}),state) is None
        action=pipe.ground(Action('click',{'target':'Visual Card'}),state)
        assert state.route is None
        executed=pipe.execute(action,state)
        assert executed.info['dispatched']
        text=browser_thread(lambda self:self._pick_page(self._connect()[1]).locator('#visual').inner_text())(session)
        assert text=='Visual action completed'
        calls=[c for c in pipe.reviewer.calls if list(c.get('request',{}).get('questions',{}))==['next_mode']]
        assert len(calls)==2
        print(json.dumps({'real_jev_modes':[first,second],
                          'latency_ms':[c['latency_ms'] for c in calls]},ensure_ascii=False))
    finally:
        pipe.decision_judge.close()
