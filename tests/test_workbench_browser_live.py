import time
from pathlib import Path
import pytest
from test_browser_live import live_browser
from osworld_agent.browser_session import browser_thread
from osworld_agent.actions import Action
from osworld_agent.env import Observation


def test_preview_preserves_ax_refs_and_can_stop(live_browser):
    session,base=live_browser
    shot,snapshot,info=session.collect_observation()
    sid=snapshot['snapshot_id'];nodes=dict(session.nodes);frames=[]
    session.start_preview(frames.append)
    deadline=time.monotonic()+5
    while not frames and time.monotonic()<deadline:time.sleep(.05)
    session.stop_preview()
    assert frames and frames[0]['url'].startswith(base)
    assert frames[0]['screenshot'].startswith('data:image/jpeg;base64,')
    assert session.snapshot['snapshot_id']==sid and session.nodes==nodes
    count=len(frames);time.sleep(.2);assert len(frames)==count


def test_type_submit_handles_accessible_label_whitespace(live_browser):
    session,base=live_browser
    def content(owner):
        page=owner._pick_page(owner._connect()[1])
        page.set_content('<form action="'+base+'/child"><label>数据名称 \u00a0<input id="query" name="q"></label><button>搜索</button></form>')
    browser_thread(content)(session)
    shot,snapshot,info=session.collect_observation()
    node=next(n for n in snapshot['nodes'] if n.get('id')=='query')
    action=Action('type',{'target':'数据名称','target_ref':node['ref'],'text':'预算','overwrite':True,'submit_key':'Enter'})
    route=session.route(action,Observation(screenshot=shot,context=[snapshot],info=info))
    assert route['channel']=='browser_dom',route
    result=session.execute(action,route)
    assert result['status']=='executed',result
    _,snap,_=session.collect_observation()
    assert '/child?q=' in snap['url']


def test_workbench_navigation_streaming_replans_and_safe_output(request):
    if not request.config.getoption('--run-browser-integration'):pytest.skip('browser opt-in')
    from playwright.sync_api import sync_playwright,expect
    from osworld_agent.viz import server
    static=Path(server.__file__).parent/'static'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1400,'height':1000})
        page.set_content('<div id="view-live" class="active"><div id="timeline"></div></div><div id="view-local"><div id="local-body"></div></div>')
        page.add_style_tag(path=str(static/'workbench.css'));page.add_script_tag(path=str(static/'vendor/markdown-it.min.js'));page.add_script_tag(path=str(static/'workbench.js'))
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        def emit(event,**kw):page.evaluate('(p)=>AgentWorkbench.ingest(p)',{'event':event,'run_id':'test','ts':time.time(),**kw})
        emit('workbench_start',instruction='测试多步骤采集与总结')
        def record(kind,**kw):emit('workbench_record',record={'event_type':kind,'event_id':str(time.monotonic_ns()),**kw})
        record('plan_revision',revision=0,queue={'steps':[{'id':'p0_1','goal':'第一项','status':'doing'}]})
        record('action_context',step=1,plan_step_id='p0_1',plan_revision=0,plan_step={'goal':'第一项'},observation={'context':[{'kind':'browser_ax','nodes':[{'name':'真实 DOM 节点'}]}]},policy_input=[{'role':'system','content':'实际 system 模板'}])
        record('transition',transition={'step':1,'action':{'action':'click','args':{'target':'按钮'}},'policy_output':'{"next_action":{"type":"click"}}','routing':{'channel':'browser_dom'},'execution_info':{'status':'executed'}})
        record('action_context',step=2,plan_step_id='p0_1',plan_revision=0,plan_step={'goal':'第一项'})
        expect(page.locator('[data-action-select]')).to_have_value('2')
        page.locator('[data-wb="prev"]').click();expect(page.locator('[data-action-select]')).to_have_value('1')
        record('action_context',step=3,plan_step_id='p0_1',plan_revision=0)
        expect(page.locator('[data-action-select]')).to_have_value('1')
        page.locator('[data-wb="tab"][data-tab="input"]').click()
        expect(page.locator('.wb-evidence')).to_contain_text('实际 system 模板')
        page.locator('.wb-evidence summary').first.click();page.wait_for_timeout(230)
        assert page.locator('.wb-evidence details').first.get_attribute('open') is not None
        page.locator('.wb-evidence summary').first.click();page.wait_for_timeout(230)
        assert page.locator('.wb-evidence details').first.get_attribute('open') is None
        record('plan_revision',revision=1,queue={'steps':[{'id':'p0_1','goal':'第一项','status':'done'},{'id':'p1_1','goal':'改用另一入口','status':'doing'}]})
        expect(page.locator('[data-plan] option')).to_have_count(2)
        expect(page.locator('[data-action-select]')).to_have_value('1')
        page.locator('[data-wb="live"]').click();expect(page.locator('[data-action-select]')).to_have_value('3')
        emit('report_start',version=1,sources=[{'id':'S1','url':'https://example.org','title':'测试来源'}])
        emit('report_delta',version=1,sequence=1,delta='第一段',text='第一段')
        expect(page.locator('.wb-report-text')).to_contain_text('第一段')
        # Recover a missed chunk from cumulative text, then ignore duplicate.
        emit('report_delta',version=1,sequence=3,delta='第三段',text='第一段第二段第三段')
        emit('report_delta',version=1,sequence=3,delta='第三段',text='第一段第二段第三段')
        expect(page.locator('.wb-report-text')).to_have_text('第一段第二段第三段')
        emit('report_complete',version=1,status='complete',text='# 标题\n\n`/Users/test/名单.txt`\n\n---\n\n1. 团队甲\n2. 团队乙\n\n> 范围说明\n\n```txt\n代码正文\n```\n\n[危险](javascript:alert(1))\n\n<img src=x onerror="window.pwned=1">\n| 项目 | 值 |\n|---|---|\n| A | 40 [S1] |',sources=[{'id':'S1','url':'https://example.org','title':'来源'}])
        expect(page.locator('.wb-report-text table')).to_be_visible()
        expect(page.locator('.wb-report-text code').first).to_have_text('/Users/test/名单.txt')
        expect(page.locator('.wb-report-text hr')).to_have_count(1)
        expect(page.locator('.wb-report-text ol li')).to_have_count(2)
        expect(page.locator('.wb-report-text blockquote')).to_contain_text('范围说明')
        expect(page.locator('.wb-report-text pre code')).to_have_text('代码正文\n')
        assert page.locator('.wb-report-text a[href^="javascript:"]').count()==0
        assert page.evaluate('window.pwned') is None
        record('episode_end',outcome={'success':True,'termination_reason':'verified_by_model'},elapsed_s=9.5)
        assert page.locator('[data-wb="prev"]').is_enabled()
        assert not errors
        browser.close()


def test_workbench_actual_visual_mode_final_verdict_and_unexecuted_subtask(request):
    if not request.config.getoption('--run-browser-integration'):pytest.skip('browser opt-in')
    from playwright.sync_api import sync_playwright,expect
    from osworld_agent.viz import server
    static=Path(server.__file__).parent/'static'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1400,'height':900})
        page.set_content('<div id="view-live"></div><div id="view-local" class="active"><div id="local-body"></div></div>')
        page.add_style_tag(path=str(static/'workbench.css'));page.add_script_tag(path=str(static/'vendor/markdown-it.min.js'));page.add_script_tag(path=str(static/'workbench.js'))
        data={'instruction':'下载文件','metrics':{},'outcome':{'success':True},'report':{},'unassigned_calls':[],
            'revisions':[{'revision':0,'queue':{'steps':[{'id':'p0_1','goal':'点击下载','status':'done'},
                {'id':'p0_2','goal':'确认文件落盘','status':'pending','verification':[{'kind':'artifact_download'}]}]}}],
            'actions':[{'step':1,'plan_step_id':'p0_1','plan_revision':0,'action':{'action':'click','args':{'target':'下载图标'}},
                        'operation_mode':'gui','execution_info':{'channel':'browser_dom','resolved':'visual','status':'executed'},'step_reviews':[]},
                       {'step':2,'plan_step_id':'p0_1','plan_revision':0,'action':{'action':'request_finish','args':{}},'operation_mode':'framework',
                        'step_verification':{'status':'unknown','plan_step_id':'p0_1'},
                        'step_reviews':[{'status':'unknown','plan_step_id':'whole_task','method':'code'},
                                        {'status':'passed','plan_step_id':'whole_task','method':'decision_model'}]}]}
        page.evaluate('(d)=>AgentWorkbench.mountHistory("test",d)',data)
        expect(page.locator('.wb-stages')).to_contain_text('整项验收 · 通过')
        expect(page.locator('.wb-stages')).to_contain_text('提交验收')
        page.locator('[data-wb="prev"]').click()
        expect(page.locator('.wb-mode')).to_have_text('GUI')
        page.locator('[data-subtask="p0_2"]').click()
        expect(page.get_by_role('dialog',name='子任务详情')).to_contain_text('不能补造执行记录')
        page.get_by_role('button',name='关闭说明').click()
        expect(page.get_by_role('dialog',name='子任务详情')).to_have_count(0)
        for width in [1600,1280,850,390]:
            page.set_viewport_size({'width':width,'height':900})
            rects=page.locator('.wb-stages li').evaluate_all('(es)=>es.map(e=>e.getBoundingClientRect().y)')
            assert len(set(round(v) for v in rects))==1
        browser.close()


def test_replanned_actions_are_collapsed_history_not_pending_work(request):
    if not request.config.getoption('--run-browser-integration'):pytest.skip('browser opt-in')
    from playwright.sync_api import sync_playwright,expect
    from osworld_agent.viz import server
    static=Path(server.__file__).parent/'static'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True);page=browser.new_page()
        page.set_content('<div id="view-live"></div><div id="view-local" class="active"><div id="local-body"></div></div>')
        page.add_style_tag(path=str(static/'workbench.css'))
        page.add_script_tag(path=str(static/'vendor/markdown-it.min.js'))
        page.add_script_tag(path=str(static/'workbench.js'))
        data={'instruction':'查团队','metrics':{},'outcome':{'success':True},'unassigned_calls':[],
              'report':{'status':'complete','verification_status':'passed','text':'已收集名单'},
              'revisions':[{'revision':0,'queue':{'steps':[{'id':'old','goal':'旧入口','status':'doing'}]}},
                           {'revision':1,'queue':{'steps':[{'id':'new','goal':'首页读取','status':'done'},
                             {'id':'report','goal':'生成结果','phase':'report','status':'pending'}]}}],
              'actions':[{'step':1,'plan_revision':0,'plan_step_id':'old','action':{'action':'click','args':{'target':'旧入口'}},'execution_info':{'status':'executed','dispatched':True}},
                         {'step':2,'plan_revision':1,'plan_step_id':'new','action':{'action':'request_finish','args':{}}}]}
        page.evaluate('(d)=>AgentWorkbench.mountHistory("test",d)',data)
        history=page.locator('.wb-plan-history')
        expect(history.locator('summary')).to_contain_text('计划 1 的动作记录')
        assert not history.evaluate('(el)=>el.open')
        expect(page.locator('.wb-plan')).not_to_contain_text('其他版本 / 未记录关联')
        expect(page.locator('.wb-plan')).to_contain_text('框架交付 · 通过 · 查看结果')
        history.locator('summary').click()
        expect(history.locator('[data-step="1"]')).to_be_visible()
        history.locator('[data-step="1"]').click()
        expect(page.locator('[data-plan]')).to_have_value('0')
        expect(page.locator('[data-action-select]')).to_have_value('1')
        browser.close()
