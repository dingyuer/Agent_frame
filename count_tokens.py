"""统计 events.jsonl 里每个模型调用已记录的输入/输出 token。

用法：
    python count_tokens.py <events.jsonl 路径>

按「角色 + 请求模型」分组累加 input/output tokens，兼容两种命名：
  - OpenAI 风格：prompt_tokens / completion_tokens
  - vLLM 风格：input_tokens / output_tokens
"""
import json
import sys
from collections import defaultdict


def usage_tokens(usage):
    """把任意 usage dict 归一化成 (input, output)。"""
    if not isinstance(usage, dict):
        return 0, 0
    inp = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    out = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return int(inp), int(out)


def iter_usage(event):
    """从一个事件里找出所有可能承载 usage 的位置，避免重复计数。"""
    # 决策/规划模型（ChatModel.chat）：call.usage
    call = event.get("call")
    if isinstance(call, dict):
        if isinstance(call.get("usage"), dict):
            yield call["usage"]
        # 验证/语义模型（DecisionJudge）：call.response.usage
        resp = call.get("response")
        if isinstance(resp, dict) and isinstance(resp.get("usage"), dict):
            yield resp["usage"]
    if isinstance(event.get("usage"), dict):
        yield event["usage"]


def main(path):
    per_role = defaultdict(lambda: [0, 0, 0])     # role -> [input, output, calls]
    per_model = defaultdict(lambda: [0, 0, 0])    # model -> [input, output, calls]
    total = [0, 0, 0]
    seen_calls = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 只统计 model_call：它是「一次模型调用」的规范记录；jev_call 是同一批
            # 验证调用在 framework 视角下的镜像事件，按 call_id 去重以免重复计数。
            if event.get("event_type") != "model_call":
                continue
            call = event.get("call") or {}
            call_id = call.get("call_id") if isinstance(call, dict) else None
            if call_id is not None:
                if call_id in seen_calls:
                    continue
                seen_calls.add(call_id)
            for usage in iter_usage(event):
                inp, out = usage_tokens(usage)
                if inp == 0 and out == 0:
                    continue
                role = str(event.get("actor_role") or "?")
                model = str(call.get("requested_model") or call.get("model") or "?")
                per_role[role][0] += inp
                per_role[role][1] += out
                per_role[role][2] += 1
                per_model[model][0] += inp
                per_model[model][1] += out
                per_model[model][2] += 1
                total[0] += inp
                total[1] += out
                total[2] += 1

    def dump(title, table):
        print(f"\n=== {title} ===")
        print(f"{'名称':<24}{'输入':>10}{'输出':>10}{'调用数':>8}")
        for name in sorted(table, key=lambda k: -(table[k][0] + table[k][1])):
            i, o, c = table[name]
            print(f"{name:<24}{i:>10}{o:>10}{c:>8}")

    dump("按角色", per_role)
    dump("按模型", per_model)
    print(f"\n合计：输入 {total[0]}，输出 {total[1]}，调用 {total[2]} 次")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "events.jsonl")
