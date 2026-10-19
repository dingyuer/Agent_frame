"""FastAPI routes, background trials, durable reports and session settings."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from .config import ROOT, Settings
from .fixtures import ASSETS, BY_ID, CASES, MULTI_IDS, evidence, evidence_hash, public_case
from .protocol import ModelClients, PROTOCOL_VERSION

router=APIRouter(prefix='/jev/compare',tags=['Jev comparison'])
REPORTS=ROOT/'reports'

class Workbench:
    def __init__(self, reports=REPORTS):
        self.settings=Settings.load()
        self.models=ModelClients()
        self.reports=Path(reports); self.reports.mkdir(exist_ok=True,parents=True)
        self.trials={}; self.batches={}; self.tasks=set()
        for p in sorted(self.reports.glob('trial-*.json'))[-500:]:
            try:
                t=json.loads(p.read_text(encoding='utf-8'))
                if t.get('schema_version')==1 and t.get('case_id') in BY_ID: self.trials[t['id']]=t
            except (ValueError,KeyError,OSError): pass

    def spawn(self,coro):
        t=asyncio.create_task(coro); self.tasks.add(t); t.add_done_callback(self.tasks.discard)
        return t

    async def close(self):
        for t in list(self.tasks): t.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        await self.models.close()

    def new_trial(self,case_id,mode,settings,batch_id=None,repeat=1):
        c=BY_ID[case_id]; now=datetime.now(timezone.utc).isoformat()
        tid='trial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:10]
        t={'schema_version':1,'protocol_version':PROTOCOL_VERSION,'id':tid,'created_at':now,'case_id':case_id,'title':c['title'],'source':c['source'],
           'mode':mode,'status':'running','evidence_hash':evidence_hash(c),'expected':c['expected'],
           'questions':c['questions'],'state':c['state'],'assets':c['assets'],'settings':settings.public(),
           'prompt_language':c['prompt_language'],'question_ui':c['question_ui'],
           'batch_id':batch_id,'repeat':repeat,'results':{}}
        self.trials[tid]=t
        return t

    async def execute(self,t,settings):
        async def one(provider):
            t['results'][provider]=await self.models.run(BY_ID[t['case_id']],provider,settings,t['mode'])
        try:
            await asyncio.gather(one('jev'),one('vlm'))
            t['status']='complete'
        except asyncio.CancelledError:
            t['status']='interrupted'
            raise
        finally:
            t['finished_at']=datetime.now(timezone.utc).isoformat()
            self.save(t)
        return t

    def save(self,t):
        dest=self.reports/(t['id']+'.json'); temp=dest.with_suffix('.tmp')
        temp.write_text(json.dumps(t,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); temp.replace(dest)

    async def execute_batch(self,b,settings):
        try:
            for repeat in range(1,b['repeats']+1):
                for case_id in b['case_ids']:
                    if b.get('cancel_requested'):
                        b['status']='cancelled'; return
                    t=self.new_trial(case_id,b['mode'],settings,b['id'],repeat)
                    b['trial_ids'].append(t['id']); b['current_case']=case_id
                    await self.execute(t,settings)
                    b['completed']+=1
            b['status']='complete'
        except asyncio.CancelledError:
            b['status']='interrupted'; raise
        except Exception:
            b['status']='error'; b['error']='批次中断，已完成的结果仍可回放。'


workbench=Workbench()


def validate_run(body):
    if not isinstance(body,dict) or not isinstance(body.get('case_id'),str) or body.get('case_id') not in BY_ID: raise HTTPException(404,'案例不存在')
    mode=body.get('mode','json')
    if mode not in ('json','free'): raise HTTPException(422,'模式必须是 json 或 free')
    return body['case_id'],mode

@router.get('/cases')
def list_cases():
    return {'cases':[public_case(c) for c in CASES], 'recommended':[c['id'] for c in CASES if c['featured']], 'multi_ids':MULTI_IDS, 'protocol_version':PROTOCOL_VERSION, 'prompt_language':'zh-CN'}

@router.get('/assets/{name}.png')
def get_asset(name: str):
    if name not in {k for c in CASES for k in c['assets']}: raise HTTPException(404)
    return FileResponse(ASSETS/(name+'.png'),media_type='image/png',headers={'Cache-Control':'public, max-age=3600'})

@router.get('/settings')
def settings(): return workbench.settings.public()

@router.post('/settings')
async def update_settings(request: Request):
    if any(t['status']=='running' for t in workbench.trials.values()) or any(b['status']=='running' for b in workbench.batches.values()):
        raise HTTPException(409,'运行中暂不能更改连接设置，请等待当前调用结束。')
    try: workbench.settings=workbench.settings.updated(await request.json())
    except (ValueError,TypeError): raise HTTPException(422,'配置无效，请核对 URL、模型名和数值范围。')
    return workbench.settings.public()

@router.post('/trials')
async def start_trial(request: Request):
    if sum(t['status']=='running' for t in workbench.trials.values())>=3: raise HTTPException(429,'已有 3 组比较正在运行')
    case_id,mode=validate_run(await request.json())
    s=workbench.settings; t=workbench.new_trial(case_id,mode,s); workbench.spawn(workbench.execute(t,s))
    return t

@router.get('/trials')
def list_trials():
    return {'trials':sorted(workbench.trials.values(),key=lambda t:t['created_at'],reverse=True)[:300]}

@router.get('/trials/{tid}')
def get_trial(tid: str):
    if tid not in workbench.trials: raise HTTPException(404,'记录不存在')
    return workbench.trials[tid]

@router.get('/trials/{tid}/export')
def export_trial(tid: str):
    t=get_trial(tid)
    # Preserve the exact recorded evidence. Assets are bundled with the project.
    body={**t,'images':{k:'data:image/png;base64,'+__import__('base64').b64encode((ASSETS/(k+'.png')).read_bytes()).decode() for k in t['assets']}}
    return JSONResponse(body,headers={'Content-Disposition':f'attachment; filename="{tid}.json"'})

@router.post('/batches')
async def start_batch(request: Request):
    if any(b['status']=='running' for b in workbench.batches.values()): raise HTTPException(409,'已有批次正在运行')
    data=await request.json()
    if not isinstance(data,dict): raise HTTPException(422,'请求必须是 JSON 对象')
    ids=data.get('case_ids'); repeats=data.get('repeats',1); mode=data.get('mode','json')
    if not isinstance(ids,list) or not 1<=len(ids)<=len(CASES) or any(not isinstance(x,str) or x not in BY_ID for x in ids) or len(set(ids))!=len(ids) or type(repeats) is not int or not 1<=repeats<=5 or mode not in ('json','free'):
        raise HTTPException(422,'请选择有效案例、1–5 次重复以及有效模式')
    bid='batch-'+uuid.uuid4().hex[:12]
    b={'id':bid,'case_ids':ids,'repeats':repeats,'mode':mode,'total':len(ids)*repeats,'completed':0,'status':'running','trial_ids':[]}
    workbench.batches[bid]=b; workbench.spawn(workbench.execute_batch(b,workbench.settings)); return b

@router.get('/batches/{bid}')
def get_batch(bid: str):
    if bid not in workbench.batches: raise HTTPException(404)
    return workbench.batches[bid]

@router.post('/batches/{bid}/cancel')
def cancel_batch(bid: str):
    b=get_batch(bid); b['cancel_requested']=True
    return b

# Compatibility endpoints retained for existing integrations. A multi-step
# fixture is a sequence of independent observations, not a live agent rollout.
@router.post('/run')
async def run_legacy(request: Request):
    body=await request.json(); case_id,mode=validate_run(body); provider=body.get('model','jev')
    if provider not in ('jev','vlm'): raise HTTPException(422,'model must be jev or vlm')
    return {'model':provider,'result':await workbench.models.run(BY_ID[case_id],provider,workbench.settings,mode)}

@router.get('/multi_steps')
def multi_steps(): return {'steps':[public_case(BY_ID[x]) for x in MULTI_IDS],'source':'independent_synthetic_observations'}

@router.post('/run_multi')
async def run_multi(request: Request):
    data=await request.json(); provider=data.get('model','jev'); mode=data.get('mode','json')
    if provider not in ('jev','vlm') or mode not in ('json','free'): raise HTTPException(422,'Invalid provider or mode')
    results=[]
    s=workbench.settings
    for cid in MULTI_IDS:
        results.append({'id':cid,'title':BY_ID[cid]['title'],'result':await workbench.models.run(BY_ID[cid],provider,s,mode)})
    return {'steps':results,'count':len(results),'total_ms':round(sum(r['result']['latency_ms'] or 0 for r in results),2),'source':'independent_synthetic_observations'}
