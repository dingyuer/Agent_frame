"""Real Qwen/Jev on controlled multipage websites; no scripted policy."""
import json, os, sys, subprocess, threading, hashlib, time
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'local_runs'/('workbench_validation_'+time.strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True,exist_ok=True)
visits=[]
CSV='department,budget\nLibrary,120000\nLab,85000\n'.encode()
STYLE='<meta charset="utf-8"><style>body{max-width:850px;margin:50px auto;font:20px/1.8 sans-serif;color:#263342}a{display:block;margin:15px 0}button,input{padding:12px;font-size:20px}h1{font-size:28px}</style>'
COURSES={'robot':('机器人实践','每周二 19:00–21:00','480 元','有 Python 基础','6 周'), 'web':('网页开发','每周六 09:00–12:00','360 元','零基础','4 周'), 'data':('数据分析','每周三 18:30–20:30','600 元','有 Python 基础','8 周')}
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  path=urlsplit(self.path).path;visits.append(path)
  if path=='/download/catalog.csv':
   self.send_response(200);self.send_header('Content-Type','text/csv');self.send_header('Content-Disposition','attachment; filename="catalog.csv"');self.send_header('Content-Length',str(len(CSV)));self.end_headers();self.wfile.write(CSV);return
  body='<h1>校园服务门户 · 集成测试站点</h1><a href="/learn">课程中心</a><a href="/open">公开数据</a>'
  if path=='/learn':body='<h1>课程中心</h1><a href="/catalog">查看本期课程目录</a>'
  elif path=='/catalog':body='<h1>2026 秋季课程</h1>'+''.join(f'<a href="/course/{k}">{v[0]}</a>' for k,v in COURSES.items())
  elif path.startswith('/course/'):
   key=path.rsplit('/',1)[-1];n,t,c,p,w=COURSES[key]
   body=f'<h1>{n}</h1><p>本期课程信息</p><a href="/schedule/{key}">课程安排与费用</a><a href="/catalog">返回课程目录</a>'
  elif path.startswith('/schedule/'):
   key=path.rsplit('/',1)[-1];n,t,c,p,w=COURSES[key]
   body=f'<h1>{n} · 课程安排</h1><p>时间：{t}</p><p>费用：{c}</p><button onclick="document.getElementById(\'more\').hidden=false;this.disabled=true">展开报名要求与周期</button><div id="more" hidden><p>报名要求：{p}</p><p>周期：{w}</p></div><a href="/catalog">返回课程目录</a>'
  elif path=='/open':body='<h1>公开数据</h1><a href="/search">搜索数据目录</a>'
  elif path=='/search':body='<h1>数据目录检索</h1><form action="/results"><label>数据名称 <input id="query" name="q"></label><button>搜索</button></form>'
  elif path=='/results':body='<h1>检索结果</h1><a href="/dataset">2026 部门预算清单</a>'
  elif path=='/dataset':body='<h1>2026 部门预算清单</h1><p>CSV 格式，包含 Library 与 Lab 两个部门。</p><button onclick="document.getElementById(\'gate\').hidden=false">查看下载说明</button><div id="gate" hidden><p>本清单为测试用数据。</p><a href="/download/catalog.csv" download>下载预算清单 CSV</a></div>'
  self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write((STYLE+body).encode())
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
cases=[('analysis_courses',f'进入 {base} 的课程中心，逐个查看机器人实践、网页开发、数据分析三门课的课程安排，并展开报名要求与周期。汇总比较三门课程的时间、费用、报名要求和周期，给零基础学习者推荐一门并说明依据，附上来源。',26),('download_catalog',f'进入 {base} 的公开数据栏目，搜索“2026 部门预算清单”，进入详情、查看下载说明，再点击下载预算清单 CSV 到默认目录。',12),('analysis_filtered',f'进入 {base} 的课程中心，查看机器人实践和数据分析的课程安排，分别展开报名要求与周期。比较这两门课每小时的费用，列出计算过程和来源，说明哪一门总费用更低。',16)]
results=[]
try:
 for label,task,steps in cases:
  start=len(visits)
  with (OUT/(label+'.log')).open('w') as log:
   r=subprocess.run([sys.executable,str(ROOT/'experiments/live_browser_case.py'),'--source',str(ROOT.parent),'--label','workbench_'+label,'--start-url',base,'--task',task,'--max-steps',str(steps),'--fixture','--record-history'],cwd=ROOT,stdout=log,stderr=log,timeout=420)
  summaries=[]
  for line in (OUT/(label+'.log')).read_text().splitlines():
   try:v=json.loads(line)
   except ValueError:continue
   if v.get('out_dir'):summaries.append(v)
  summary=summaries[-1] if summaries else {'error':'no_summary','returncode':r.returncode}
  summary['visited']=visits[start:];out=Path(summary.get('out_dir','/nonexistent'))
  if (out/'trajectory.json').exists():
   t=json.loads((out/'trajectory.json').read_text());report=t.get('run_manifest',{}).get('report_delivery',{});summary['report_status']=report.get('status');summary['report_text']=report.get('text');summary['report_source_count']=len(report.get('sources',[]));summary['report_delta_count']=sum(e.get('event_type')=='report_delta' for e in t['events']);summary['action_count']=len(t['steps']);summary['association_complete']=all(x.get('plan_step_id') for x in t['steps'])
   if label=='download_catalog':summary['file_oracle']=any(p.read_bytes()==CSV for p in (out/'workspace').rglob('*.csv'))
   else:summary['oracle_keywords']={k:k in (report.get('text') or '') for k in (['480','360','600','6','4','8','零基础'] if label=='analysis_courses' else ['480','600','40','37.5'])}
  results.append(summary);(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
finally:server.shutdown()
