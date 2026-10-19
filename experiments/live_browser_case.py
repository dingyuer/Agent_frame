"""Live Qwen + Jev + Chromium trial. No scripted policy or direct download API."""
import argparse, os, sys, json, time, socket, tempfile, shutil
from pathlib import Path
BOOT = time.perf_counter()
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--label',required=True);p.add_argument('--start-url',default='https://arxiv.org/abs/1706.03762');p.add_argument('--task');p.add_argument('--max-steps',type=int,default=12);p.add_argument('--fixture',action='store_true');p.add_argument('--record-history',action='store_true',help='Publish a meaningful multi-step scenario in Studio history; smoke checks stay separate');args=p.parse_args()
LIVE=Path(__file__).resolve().parents[1]
import importlib.util
spec=importlib.util.spec_from_file_location('osworld_agent',LIVE/'__init__.py',submodule_search_locations=[str(LIVE)])
package=importlib.util.module_from_spec(spec);sys.modules['osworld_agent']=package;spec.loader.exec_module(package)
# Credentials remain in the ignored local file/environment, never in output.
for line in ((LIVE/'.env').read_text().splitlines() if (LIVE/'.env').exists() else []):
 if line.startswith('DASHSCOPE_API_KEY='):os.environ.setdefault('DASHSCOPE_API_KEY',line.split('=',1)[1].strip())
from osworld_agent.config import ModelConfig, resolve_secret
from osworld_agent.model.decision_model import ChatModel
from osworld_agent.adapters.qwen_grounding import QwenPlusGrounding
from osworld_agent.browser_session import BrowserSession,BrowserOptions,browser_thread
from osworld_agent.browser_profile import prepare_pdf_download_profile
from osworld_agent.env import Observation,StepResult
from osworld_agent.scene import browser_scene
from osworld_agent.pipeline import Pipeline
from osworld_agent.prompts import build_system_prompt
from osworld_agent.trajectory import build_trajectory,dump_trajectory
from osworld_agent.viz.logging import LoggingChatModel,LoggingGrounding,log_status,flush_logs,set_results_dir,reset_collected
PROXY=os.getenv('https_proxy') or os.getenv('HTTPS_PROXY')
def client(self):
 if self._client is None:
  import httpx
  from openai import OpenAI
  # Public Qwen endpoint requires this host's existing HTTP proxy; Jev stays direct.
  self._client=OpenAI(base_url=self.cfg.url,api_key=resolve_secret(self.cfg.api_key),max_retries=0,
      http_client=httpx.Client(trust_env=False,proxy=PROXY,timeout=self.cfg.timeout))
 return self._client
ChatModel._get_client=client
class Env:
 browser_only=True;os_name='linux';evaluation_available=False;browser_start_url=args.start_url
 def __init__(self,session):self._browser_session=session;self.initial_state={};self._last=None;self.started=False
 def prepare_workspace(self,task_id):return self._browser_session.prepare_workspace(task_id)
 def reset(self,task):
  if not self.started:
   self.initial_state=self._browser_session.begin_episode(viewport={'width':1280,'height':800},start_url=self.browser_start_url)
   self.started=True
  return self.observe()
 def observe(self):
  shot,snapshot,info=self._browser_session.collect_observation()
  info['scene']=browser_scene(snapshot,shot,page_id=info.get('page_id',''),snapshot_id=info.get('snapshot_id','')).to_dict()
  info['active_surface']='browser';self._last=info
  return Observation(screenshot=shot,context=[snapshot] if snapshot else [],info=info)
 def route_action(self,a,o):return self._browser_session.route(a,o)
 def execute_routed(self,a,r):
  result=self._browser_session.execute(a,r)
  return StepResult(self.observe(),info=result)
 def prevent_duplicate_download(self,*a):return self._browser_session.prevent_duplicate_download(*a)
 def browser_event_metadata(self):return self._browser_session.event_metadata()
 def cancel_download(self,i):return self._browser_session.cancel_download(i)
 def step(self,a):
  session=self._browser_session;n=a.action
  if n=='goto':info=session.goto(a.args['url'])
  elif n=='observe_browser':info={'status':'executed','dispatched':False,'action':n}
  elif n in ('list_tabs','switch_tab','close_tab','handle_dialog'):info=getattr(session,n)(**a.args)
  elif n=='wait':time.sleep(min(2,float(a.args.get('seconds',1))));info={'status':'executed','action':n}
  elif n in ('click','double_click','right_click','type','scroll','press','hotkey','drag_and_drop'):
   info=session.execute_viewport_action(a,metadata=a.args.get('observation_metadata') or self._last)
  else:info={'status':'rejected','dispatched':False,'error_code':'headless_browser_only','reason':'This isolated test has no native desktop','action':n}
  return StepResult(self.observe(),info=info)
run_id='live_paper_'+time.strftime('%Y%m%d_%H%M%S')+'_'+args.label
output_root=LIVE/'local_runs' if args.record_history else LIVE.parent/'.runtime'/'smoke-tests'
out=output_root/run_id;out.mkdir(parents=True)
workspace=out/'workspace'
TASK=args.task or '打开 https://arxiv.org/abs/1706.03762 ，下载该页面的论文 Attention Is All You Need 的 PDF 到默认目录。'
model=LoggingChatModel(ChatModel(ModelConfig(name=os.getenv('WORKBENCH_DECISION_MODEL','qwen3.7-flash'),url=os.getenv('WORKBENCH_DECISION_URL','https://dashscope.aliyuncs.com/compatible-mode/v1'),api_key='env:DASHSCOPE_API_KEY',temperature=0,max_tokens=2048,timeout=60)))
ground=QwenPlusGrounding(url=model.inner.cfg.url,api_key='env:DASHSCOPE_API_KEY',model='qwen3.7-plus',width=1280,height=800)
def ground_client():
 if ground._client is None:
  from openai import OpenAI
  import httpx
  ground._client=OpenAI(base_url=ground.url,api_key=resolve_secret(ground.api_key),max_retries=0,
    http_client=httpx.Client(trust_env=False,proxy=PROXY,timeout=45))
 return ground._client
ground._get_client=ground_client
set_results_dir(str(out));reset_collected();log_status({'event':'task_start','mode':'local','instruction':TASK+' ['+args.label+']','max_steps':args.max_steps})
print(json.dumps({'event':'starting','label':args.label,'output':str(out),'real_models':True}),flush=True)
from playwright.sync_api import sync_playwright
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
profile=Path(tempfile.mkdtemp(prefix='paper-speed-chrome-'));prepare_pdf_download_profile(profile)
summary={'label':args.label,'run_id':run_id,'task':TASK,'policy':'real_qwen','jev':'real_d3_flash','source':args.source,'out_dir':str(out)}
try:
 with sync_playwright() as pw:
  launch=time.perf_counter()
  context=pw.chromium.launch_persistent_context(str(profile),headless=True,args=[f'--remote-debugging-port={port}'],
    viewport={'width':1280,'height':800},accept_downloads=True,ignore_https_errors=True,
    proxy={'server':PROXY,'bypass':'localhost,127.0.0.1'} if PROXY else None)
  summary['browser_launch_s']=round(time.perf_counter()-launch,3)
  session=BrowserSession(BrowserOptions(endpoint=f'http://127.0.0.1:{port}',require_focus=False,session_mode='persistent'),downloads_dir=str(workspace))
  pipe=Pipeline(model=model,env=Env(session),grounding=LoggingGrounding(ground),system_prompt=build_system_prompt(os_name='linux',browser_only=True),
    max_steps=args.max_steps,max_replans=1,enable_reflection=False,run_id=run_id,event_path=str(out/'events.jsonl'),
    wait_config={'operation_timeout_s':60,'task_timeout_s':300},
    decision_judge={'url':os.getenv('WORKBENCH_JEV_URL','http://114.212.10.51:8007/v1/systemone'),'model':'vllm-sr/d3-flash',
      'review_mode':'exclusive','selection_policy':'maximum_probability','timeout':15},
    run_manifest={'benchmark':{'label':args.label,'live_model':True,'headless_browser':True,'public_model_http_proxy':bool(PROXY)}})
  original_emit=pipe._emit
  def emit(kind,**data):
   result=original_emit(kind,**data)
   if kind in ('phase_timing','step_verification','planning_rejected','execution','episode_end'):
    if kind=='phase_timing':small={k:data.get(k) for k in ('phase','status','elapsed_ms')}
    elif kind=='step_verification':small={k:data.get('report',{}).get(k) for k in ('status','method','reason')}
    elif kind=='execution':small={'action':data.get('action'),'status':data.get('execution_info',{}).get('status')}
    else:small={'event_recorded':True}
    print(json.dumps({'event':kind,**small},ensure_ascii=False),flush=True)
   return result
  pipe._emit=emit
  started=time.perf_counter()
  try:
   result=pipe.run(TASK)
   summary.update(run_s=round(time.perf_counter()-started,3),success=result.success,steps=result.steps,termination=result.termination_reason,
     verification=result.verification,phase_timings=result.run_manifest.get('phase_timings',[]),
     planning_calls=[{k:c.get(k) for k in ('latency_ms','wall_time_ms','usage','valid_plan','error_type')} for c in result.planning_calls])
   if args.fixture:
    from urllib.parse import urlsplit
    assert urlsplit(args.start_url).hostname in ('127.0.0.1','localhost')
    summary['fixture_state']=browser_thread(lambda owner: owner._pick_page(owner._connect()[1]).evaluate("({url:location.href,value:document.querySelector('#query')?.value,results:document.querySelector('#results')?.textContent,expanded:document.querySelector('#query')?.getAttribute('aria-expanded')})"))(session)
    result.run_manifest['test_fixture']=True
   dump_trajectory(str(out/'trajectory.json'),build_trajectory(run_id,'browser',TASK,result))
   for n,shot in result.screenshots.items():
    if hasattr(shot,'save'):shot.save(out/f'step_{n}.png')
   log_status({'event':'verification_complete','success':result.success,'answer':result.final_answer})
  finally:
   cleanup=time.perf_counter();session.close();context.close();summary['cleanup_s']=round(time.perf_counter()-cleanup,3)
   if pipe.decision_judge:pipe.decision_judge.close()
except Exception as e:
 summary['error_type']=type(e).__name__
 print('ERROR',type(e).__name__,flush=True)
finally:
 shutil.rmtree(profile,ignore_errors=True)
 summary['process_wall_s']=round(time.perf_counter()-BOOT,3)
 (out/'benchmark.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 log_status({'event':'task_end','success':summary.get('success',False),'result':0,'termination_reason':summary.get('termination',summary.get('error_type'))})
 flush_logs()
 print(json.dumps({k:summary.get(k) for k in ('label','success','steps','run_s','process_wall_s','termination','error_type','out_dir')},ensure_ascii=False),flush=True)
