"""Run reproducible batches against the local workbench and summarize all trials.

python -m comparison.benchmark --suite all --repeats 1
python -m comparison.benchmark --suite recommended --repeats 3
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import httpx
from .config import ROOT


def latency_stats(values):
    values=sorted(values)
    return {'n':len(values),'median_ms':round(statistics.median(values),2) if values else None,
            'min_ms':min(values) if values else None,'max_ms':max(values) if values else None,
            'p95_ms':values[math.ceil(.95*len(values))-1] if len(values)>=20 else None}


def summarize(trials):
    paired=[t for t in trials if all(t['results'].get(p,{}).get('status')=='ok' for p in ('jev','vlm'))]
    providers={}
    for p in ('jev','vlm'):
        rs=[t['results'][p] for t in trials if p in t['results']]
        scored=[r for r in rs if r.get('score') is not None]
        providers[p]={'requested_models':sorted({r['requested_model'] for r in rs}),
                      'reported_models':sorted({r['model'] for r in rs}),
                      'calls':len(rs),'valid_calls':sum(r['status']=='ok' for r in rs),
                      'errors':sum(r['status']=='error' for r in rs),
                      'correct_labels':sum(r['score']['correct'] for r in scored),
                      'total_labels':sum(r['score']['total'] for r in scored),
                      'accepted_labels':sum(r['score']['accepted'] for r in scored),
                      'accepted_correct':sum(r['score']['accepted_correct'] for r in scored),
                      'all_questions_correct':sum(r['status']=='ok' and r['score']['correct']==r['score']['total'] for r in scored),
                      'paired_latency':latency_stats([t['results'][p]['latency_ms'] for t in paired])}
    j=providers['jev']['paired_latency']['median_ms'];v=providers['vlm']['paired_latency']['median_ms']
    return {'trials':len(trials),'paired_valid_trials':len(paired),'providers':providers,
            'ratio_of_paired_medians_vlm_over_jev':round(v/j,3) if j and v else None,
            'jev_faster_pairs':sum(t['results']['jev']['latency_ms']<t['results']['vlm']['latency_ms'] for t in paired),
            'notes':['End-to-end client latency, different deployment and network paths.',
                     'Synthetic evidence, not complete agent-task success rate.',
                     'Errors remain in label denominators; latency excludes invalid pairs.',
                     'Highest-probability label accuracy and threshold acceptance are separate.',
                     'p95 uses nearest rank and is omitted for n < 20.']}


async def run(args):
    async with httpx.AsyncClient(base_url=args.server.rstrip('/')+'/jev/compare',trust_env=False,timeout=15) as c:
        response=await c.get('/cases');response.raise_for_status();catalog=response.json()
        ids=[x['id'] for x in catalog['cases']] if args.suite=='all' else catalog['recommended'] if args.suite=='recommended' else catalog['multi_ids']
        response=await c.post('/batches',json={'case_ids':ids,'repeats':args.repeats,'mode':args.mode});response.raise_for_status();batch=response.json()
        print('Batch:',batch['id'], '| comparisons:',batch['total'],flush=True)
        last=-1
        while batch['status']=='running':
            await asyncio.sleep(.8)
            response=await c.get('/batches/'+batch['id']);response.raise_for_status();batch=response.json()
            if batch['completed']!=last:
                print(f"  {batch['completed']}/{batch['total']} {batch['status']}",flush=True);last=batch['completed']
        trials=[]
        for tid in batch['trial_ids']:
            response=await c.get('/trials/'+tid);response.raise_for_status();trials.append(response.json())
    report={'created_at':datetime.now(timezone.utc).isoformat(),'batch':batch,'summary':summarize(trials),
            'case_summary':{cid:summarize([t for t in trials if t['case_id']==cid]) for cid in ids},'trial_ids':[t['id'] for t in trials]}
    path=ROOT/'reports'/('summary-'+batch['id']+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2),flush=True)
    print('Saved:',path,flush=True)
    if batch['status']!='complete':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',default='http://127.0.0.1:8095')
    p.add_argument('--suite',choices=['all','recommended','chain'],default='all')
    p.add_argument('--repeats',type=int,choices=range(1,6),default=1)
    p.add_argument('--mode',choices=['json','free'],default='json')
    asyncio.run(run(p.parse_args()))
