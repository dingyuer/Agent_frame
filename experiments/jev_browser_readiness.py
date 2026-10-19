"""Paired Jev/VLM readiness experiment on a real, locally served browser page.

The page fixture controls its own true ready time; both verifiers see the same
browser UI and each runs outside the policy loop. This is a browser experiment,
not a claim that arxiv.org itself was reachable or that either model downloaded
a paper.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
from playwright.sync_api import sync_playwright

from ..config import resolve_secret
from ..model.decision_judge import DecisionJudge, DecisionJudgeConfig
from .jev_vs_vlm import JEV_URL, ask_vlm


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        delay = int(parse_qs(parsed.query).get('delay', ['1800'])[0])
        if parsed.path != '/readiness' or not 500 <= delay <= 5000:
            self.send_error(404)
            return
        html = '''<!doctype html><html><head><meta charset="utf-8"><title>Loading rehearsal</title>
<style>body{margin:0;font:22px Arial;background:#101827;color:#dbe7f3}
#loading{height:100vh;display:grid;place-content:center;text-align:center}
#ready{display:none;min-height:100vh;background:#f5f7f9;color:#24313a;padding:90px}
a{display:block;margin-top:40px;color:#125e92;font-size:28px}</style></head>
<body data-ready="0"><div id="loading"><div>Loading search results…</div><div>◌</div></div>
<main id="ready"><h1>Search results for Attention Is All You Need</h1>
<a href="https://export.arxiv.org/abs/1706.03762">Attention Is All You Need — Vaswani et al.</a></main>
<script>window.readyAt=null;setTimeout(()=>{
document.querySelector('#loading').style.display='none';
document.querySelector('#ready').style.display='block';
document.body.dataset.ready='1';window.readyAt=performance.now();
},''' + str(delay) + ''');</script></body></html>'''
        data = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        return


def _model_call(mode, image, judge, client, key):
    if mode == 'jev':
        response = judge.readiness(image)
        answer = response.get('answers', {}).get('readiness', {})
        return {'label': answer.get('label', 'unknown'), 'status': response['status'],
                'confidence': answer.get('confidence'), 'latency_ms': response['latency_ms']}
    return ask_vlm(client, image, 'loading', key)


def _trial(browser, base, delay_ms, mode, judge, client, key, poll_interval_ms):
    page = browser.new_page(viewport={'width': 1024, 'height': 640}, device_scale_factor=1)
    polls = []
    try:
        page.goto(f'{base}/readiness?delay={delay_ms}', wait_until='domcontentloaded')
        origin = time.monotonic()
        next_poll = origin
        deadline = origin + (delay_ms + 5000) / 1000
        while time.monotonic() < deadline:
            pause = next_poll - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            shot = page.screenshot(full_page=False)
            captured_ready = page.evaluate('document.body.dataset.ready === "1"')
            captured_perf = page.evaluate('performance.now()')
            response = _model_call(mode, shot, judge, client, key)
            response_perf = page.evaluate('performance.now()')
            ready_at = page.evaluate('window.readyAt')
            polls.append({'screenshot_ready': captured_ready, 'captured_perf_ms': round(captured_perf, 2),
                          'response_perf_ms': round(response_perf, 2), 'ready_at_ms': ready_at, **response})
            if response['status'] != 'answered' or response['label'] == 'ready':
                break
            next_poll += poll_interval_ms / 1000
            if next_poll < time.monotonic():
                next_poll = time.monotonic()
        last = polls[-1] if polls else {}
        false_ready = any(p['label'] == 'ready' and not p['screenshot_ready'] for p in polls)
        detected = last.get('label') == 'ready' and not false_ready
        lag = (last['response_perf_ms'] - last['ready_at_ms']) if detected and last.get('ready_at_ms') else None
        return {'delay_ms': delay_ms, 'mode': mode, 'polls': polls, 'calls': len(polls),
                'detected': detected, 'false_ready': false_ready,
                'detection_lag_ms': round(lag, 2) if lag is not None else None,
                'wall_ms': round((time.monotonic() - origin) * 1000, 2), 'policy_decisions': 0}
    finally:
        page.close()


def run(repetitions=2, poll_interval_ms=500):
    key = resolve_secret('env:DASHSCOPE_API_KEY')
    if not key:
        raise RuntimeError('DASHSCOPE_API_KEY is unavailable')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    judge = DecisionJudge(DecisionJudgeConfig(url=JEV_URL, model='vllm-sr/d3-flash',
                                               timeout=10, threshold=.8, margin=.15))
    trials = []
    try:
        with sync_playwright() as playwright, httpx.Client(trust_env=False, timeout=30) as client:
            chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            browser = playwright.chromium.launch(headless=True,
                **({'executable_path': str(chrome)} if chrome.is_file() else {}))
            try:
                for repeat in range(repetitions):
                    for delay in (1300, 2100, 2900):
                        modes = ('jev', 'vlm') if (repeat + delay) % 2 == 0 else ('vlm', 'jev')
                        for mode in modes:
                            trial = _trial(browser, base, delay, mode, judge, client, key, poll_interval_ms)
                            trial['repeat'] = repeat + 1
                            trials.append(trial)
                            print(f'{repeat+1}/{repetitions} {delay}ms {mode}: '
                                  f"{trial['calls']} calls, detected={trial['detected']}, "
                                  f"lag={trial['detection_lag_ms']}ms", flush=True)
            finally:
                browser.close()
    finally:
        judge.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    summary = {}
    for mode in ('jev', 'vlm'):
        group = [t for t in trials if t['mode'] == mode]
        latency = [p['latency_ms'] for t in group for p in t['polls'] if p['status'] == 'answered']
        lags = [t['detection_lag_ms'] for t in group if t['detection_lag_ms'] is not None]
        summary[mode] = {'trials': len(group), 'detected': sum(t['detected'] for t in group),
                         'false_ready': sum(t['false_ready'] for t in group),
                         'median_request_ms': round(statistics.median(latency), 2) if latency else None,
                         'median_detection_lag_ms': round(statistics.median(lags), 2) if lags else None,
                         'total_calls': sum(t['calls'] for t in group), 'policy_decisions': 0}
    return {'schema': 'jev_browser_readiness_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
            'fixture': 'locally served delayed browser page; real Chrome screenshots and remote model calls',
            'poll_interval_ms': poll_interval_ms, 'repetitions': repetitions, 'summary': summary,
            'trials': trials, 'note': 'Both verifiers run outside the policy loop and cost zero policy decisions. '
                'The page is controlled; results do not establish cross-site accuracy or live arXiv performance.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--poll-interval-ms', type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10 or not 100 <= args.poll_interval_ms <= 2000:
        parser.error('repetitions or poll interval out of range')
    result = run(args.repetitions, args.poll_interval_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
