"""Reproducible presentation pilot: identical visual fixtures, real model calls.

Run from the parent of osworld_agent:
    osworld_agent/.venv312/bin/python -m osworld_agent.experiments.jev_vs_vlm \
        --output /path/to/benchmark.json --repetitions 3

Fixtures are synthetic. Model latencies and labels are sampled live. No browser
actions or file downloads are implied by this benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

from ..config import resolve_secret
from ..model.decision_judge import DecisionJudge, DecisionJudgeConfig
from ..model.decision_model import encode_image

JEV_URL = "http://114.212.10.51:8007/v1/systemone"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
CRITERIA = {
    "loading": {
        "ready": "The article results page has visibly finished rendering and shows actionable results.",
        "waiting": "The page is blank, dark, spinning, or shows only an intermediate loading view.",
        "unknown": "The evidence does not establish whether results are ready.",
    },
    "identity": {
        "correct": "This is the original Attention Is All You Need paper by Vaswani and coauthors.",
        "wrong": "This is a different work that only mentions or paraphrases the target paper.",
        "unknown": "The visible title and authors do not establish identity.",
    },
    "overlay": {
        "clear": "The paper title and PDF action are visibly unobstructed and actionable.",
        "blocked": "A modal or overlay visibly covers the paper or PDF action.",
        "unknown": "The visible state does not establish whether the action is accessible.",
    },
}


def _font(size, bold=False):
    names = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame(kind):
    im = Image.new("RGB", (1024, 640), "#fafafa")
    d = ImageDraw.Draw(im)
    if kind in {"black", "spinner"}:
        d.rectangle((0, 0, 1024, 640), fill="#101827")
        if kind == "spinner":
            d.arc((476, 279, 548, 351), 30, 300, fill="#8ec9df", width=8)
            d.text((455, 375), "Loading...", font=_font(21), fill="#d9e5eb")
        return im
    d.rectangle((0, 0, 1024, 48), fill="#dee7e8")
    d.rounded_rectangle((108, 8, 916, 40), radius=8, fill="white")
    d.text((132, 15), "arxiv.org / research", font=_font(16), fill="#596b72")
    d.rectangle((0, 48, 1024, 105), fill="#232e34")
    d.text((34, 62), "arXiv", font=_font(27, True), fill="white")
    if kind == "results":
        d.text((80, 145), "Search results for: Attention Is All You Need", font=_font(28, True), fill="#28353e")
        d.rounded_rectangle((75, 220, 950, 435), radius=10, fill="white", outline="#d9e1e4", width=2)
        d.text((100, 250), "Attention Is All You Need", font=_font(27, True), fill="#24598a")
        d.text((100, 304), "Ashish Vaswani, Noam Shazeer, Niki Parmar et al.", font=_font(20), fill="#354750")
        d.text((100, 353), "arXiv:1706.03762  |  PDF", font=_font(20), fill="#517189")
    else:
        wrong = kind == "wrong_paper"
        title = "A Survey of Attention Mechanisms" if wrong else "Attention Is All You Need"
        d.text((68, 155), title, font=_font(31, True), fill="#243840")
        d.text((68, 211), "Jane Researcher and John Scholar" if wrong else
               "Ashish Vaswani, Noam Shazeer, Niki Parmar et al.", font=_font(20), fill="#3a525b")
        d.text((68, 265), "Abstract", font=_font(22, True), fill="#293c44")
        d.text((68, 310), "This survey discusses Attention Is All You Need." if wrong else
               "The Transformer is based solely on attention mechanisms.", font=_font(18), fill="#506570")
        d.rounded_rectangle((724, 158, 935, 214), radius=8, fill="#3c789e")
        d.text((775, 174), "View PDF", font=_font(20, True), fill="white")
        if kind == "overlay":
            d.rectangle((0, 105, 1024, 640), fill="#aab2b7")
            d.rounded_rectangle((245, 205, 785, 435), radius=16, fill="white", outline="#8999a0", width=2)
            d.text((295, 260), "Subscribe to continue", font=_font(29, True), fill="#304149")
            d.text((295, 325), "Close this dialog to access the paper.", font=_font(18), fill="#596d76")
    return im


CASES = [
    ("load_black", "loading", "black", "waiting"),
    ("load_spinner", "loading", "spinner", "waiting"),
    ("load_ready", "loading", "results", "ready"),
    ("wrong_paper", "identity", "wrong_paper", "wrong"),
    ("correct_paper", "identity", "paper", "correct"),
    ("overlay_blocked", "overlay", "overlay", "blocked"),
    ("overlay_clear", "overlay", "paper", "clear"),
]


def ask_jev(judge, image, criterion):
    result = judge.decide(
        "Classify the current browser screenshot using only the visible evidence.",
        {"outcome": {"type": "choice", "instructions": "Choose the state of the current screenshot.",
                     "criteria": CRITERIA[criterion]}}, [image],
    )
    answer = result.get("answers", {}).get("outcome", {})
    return {"label": answer.get("label", "unknown"), "top_label": answer.get("top_label"),
            "confidence": answer.get("confidence"), "latency_ms": result["latency_ms"],
            "status": result["status"], "error_type": result.get("error_type")}


def ask_vlm(client, image, criterion, key):
    question = "Classify this screenshot using only visible evidence. Criteria: " + json.dumps(CRITERIA[criterion])
    question += ". Respond with one JSON object {\"label\":\"one of the exact keys above\"}; use unknown if uncertain."
    started = time.perf_counter()
    try:
        response = client.post(QWEN_URL, headers={"Authorization": "Bearer " + key}, json={
            "model": "qwen3.7-flash", "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": encode_image(image)}}]}],
            "temperature": 0, "max_tokens": 80, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        })
        response.raise_for_status()
        body = response.json()
        raw = body["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        label = parsed.get("label", "unknown")
        if label not in CRITERIA[criterion]:
            label = "unknown"
        return {"label": label, "status": "answered", "latency_ms": round((time.perf_counter()-started)*1000, 2),
                "usage": body.get("usage", {})}
    except Exception as exc:
        return {"label": "unknown", "status": "error", "error_type": type(exc).__name__,
                "latency_ms": round((time.perf_counter()-started)*1000, 2)}


def run(repetitions):
    key = resolve_secret("env:DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is unavailable")
    judge = DecisionJudge(DecisionJudgeConfig(url=JEV_URL, model="vllm-sr/d3-flash",
                                               timeout=30, threshold=.8, margin=.15))
    records = []
    try:
        with httpx.Client(trust_env=False, timeout=60) as client:
            for repeat in range(repetitions):
                for case_id, criterion, image_kind, expected in CASES:
                    im = frame(image_kind)
                    jev = ask_jev(judge, im, criterion)
                    vlm = ask_vlm(client, im, criterion, key)
                    records.append({"repeat": repeat+1, "case": case_id, "criterion": criterion,
                                    "expected": expected, "jev": jev, "vlm": vlm})
                    print(f"{repeat+1}/{repetitions} {case_id}: Jev={jev['label']} {jev['latency_ms']}ms; "
                          f"VLM={vlm['label']} {vlm['latency_ms']}ms", flush=True)
    finally:
        judge.close()
    summary = {}
    for name in ("jev", "vlm"):
        valid = [r for r in records if r[name]["status"] == "answered"]
        correct = [r for r in valid if r[name]["label"] == r["expected"]]
        summary[name] = {"answered": len(valid), "correct": len(correct),
                         "total": len(records), "median_latency_ms": round(statistics.median([r[name]["latency_ms"] for r in valid]), 2) if valid else None}
    return {"schema": "jev_vs_vlm_pilot_v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "fixtures": "synthetic", "models": {"jev": "d3-flash via System One", "vlm": "qwen3.7-flash via DashScope"},
            "repetitions": repetitions, "summary": summary, "records": records,
            "method_note": "Both models receive the same fixture image and equivalent criteria. Latency includes network and server queue. "
                           "A VLM used only as verifier also costs zero policy decisions; extra decisions arise only when polling is routed through the policy loop. "
                           "Downloads should be judged by browser transfer events plus a durable file check, not by either visual model."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        parser.error("repetitions must be between 1 and 20")
    result = run(args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
