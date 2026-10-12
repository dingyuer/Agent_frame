"""规划和兼容提示入口；每轮决策模板见 decision_context，独立核查见 step_verifier。"""
from __future__ import annotations

from typing import Dict, List, Optional

from .actions import ACTION_REGISTRY, ActionSpec


def build_system_prompt(action_registry=None, skill_registry=None, os_name="linux", browser_only=False) -> str:
    """Compatibility entry point; live decisions further scope this basic menu."""
    from .decision_context import scoped_system
    from .actions import BASIC_ACTIONS
    registry = ACTION_REGISTRY if action_registry is None else action_registry
    return scoped_system([n for n in registry if n in BASIC_ACTIONS], None, os_name)


def build_user_prompt(
    task_instruction: str,
    memory_text: str,
    skill_results: Optional[Dict[str, str]] = None,
    terminal_result: str = "",
    code_agent_result: str = "",
    page_source: str = "",
    a11y_tree: str = "",
    browser_result: str = "",
) -> str:
    parts = [f"## 任务\n{task_instruction}"]

    parts.append(f"## 结构化记忆\n{memory_text if memory_text else '（初始，尚无记忆）'}")

    # 页面信息：浏览器场景才有，非浏览器为空。引导由 _page_a11y 按类型动态生成
    # （有 selector → 引导 click_element；无 selector 的 AX-Tree → 引导用 GUI），这里只做中性展示。
    if a11y_tree:
        parts.append(f"## 当前页面信息\n{a11y_tree}")

    extras: List[str] = []
    if skill_results:
        for name, val in skill_results.items():
            extras.append(f"- 上次技能 {name} 结果 = {val}")
    if terminal_result:
        extras.append(f"- 上次终端命令输出 = {terminal_result}")
    if code_agent_result:
        extras.append(f"- 子Agent结果：\n{code_agent_result}")
    if browser_result:
        extras.append(f"- 上次浏览器操作结果 = {browser_result}")
    if page_source:
        extras.append(f"- 当前页面源码：\n{page_source}")
    if extras:
        parts.append("## 上一步结果\n" + "\n".join(extras))

    parts.append("请结合三张截图，先判断上一步动作是否成功，再输出一个 JSON（screen_analysis + observed_effect + action_status + state_update + next_action + expected_effect）。")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# 纠错反馈（动作解析/校验失败时，作为 assistant 历史追加，让模型重试）
# --------------------------------------------------------------------------- #
def build_format_feedback(error: str) -> str:
    return (
        f"你上一步的输出不符合要求，无法执行。错误原因：{error}\n"
        "请重新只输出一个合法的 JSON（screen_analysis + observed_effect + action_status + state_update + next_action + expected_effect），"
        "不要输出 Python 代码或多余文字。"
    )


# --------------------------------------------------------------------------- #
# 模型验收器（LLM-as-judge）：对「无法写成确定性代码验收」的要求做模型判断保底
# --------------------------------------------------------------------------- #
def build_verification_prompt(task: str, criteria: str) -> str:
    """验收用提示词：由模型根据当前截图/页面结构判断要求是否已满足。"""
    return (
        "你是任务完成度验收器。请结合当前浏览器截图和页面结构，客观判断下面的要求是否已经满足。\n"
        "\n"
        f"任务：{task}\n"
        f"待验收要求：\n{criteria}\n"
        "\n"
        "判断原则：只有截图或页面里能【明确看到】要求已满足时才判 true；"
        "拿不准、看不到证据、或只是模型自己声称完成了，都判 false。\n"
        "\n"
        "只输出一个 JSON 对象，不要输出任何其他文字：\n"
        '{"success": true, "reason": "简要说明你看到了什么证据"}\n'
        "\n"
        "若未完成，输出：\n"
        '{"success": false, "reason": "简要说明缺什么"}'
    )


# --------------------------------------------------------------------------- #
# 任务规划（planner）：把模糊任务拆成结构化 plan
# --------------------------------------------------------------------------- #
def build_planner_prompt(task: str, completion_spec=None, profile=None) -> str:
    """任务规划 prompt：把用户模糊任务拆解成 3~8 个可独立执行、按顺序完成的子步骤。"""
    import json
    contract = json.dumps(completion_spec or {"checks": []}, ensure_ascii=False)
    workflow = json.dumps(profile or {}, ensure_ascii=False)
    return f"""你是任务规划器。把任务拆解成 1~12 个具体、按顺序执行的子步骤。简单任务无需凑步骤，复杂任务必须覆盖全部要求。

拆解规则：
1. 第一步通常是「打开某个网站/应用」。搜索类任务要明确打开哪个搜索引擎（谷歌学术 scholar.google.com / 谷歌 / arXiv / Bing 等），不要只写"打开浏览器"；
2. 搜索类任务：必须自己提炼搜索关键词（论文名、主题词、作者等），写进 keywords 数组；
3. 每个子步骤要具体、可操作（含明确的操作对象和预期结果），不要写"完成任务"这种空话；
4. 最后给出「完成判据」final_check：一个可以客观核对的完成标志（如"downloads 目录下存在该论文的 PDF 文件"）。
5. 补全操作前提与基本步骤：先到正确网站，找到目标，再操作，最后核对结果。不能用模糊的“找到并完成”代替操作步骤。
6. 每步写 expected_evidence，说明执行后应观察到的具体变化。多对象任务逐个列出，不漏数量、筛选条件和保存要求。
7. 系统验收条件由调用方提供或由严格任务规则提取，你不能修改、删除或自行增加可信验收条件。check_ids 只引用下面已有 id；没有对应条件就填 []。
8. final_check 只是你的建议，不会作为任务完成的证据。任务结束由框架逐项核实。
9. 每个步骤增加 verification 数组：简单结果写明确谓词 url_equals(url)/url_host(host)/text_present(text)/app_open(app，真实进程名，不带路径)/element_state(name,role,states)；难以用这些条件完整核实的步骤用 {{"kind":"visual"}}。全部条件须同时满足，不能只检查最终目标的一小部分。例如“提交搜索且显示结果”不能仅验证在搜索网站域名。predicate 的期望值必须来自原始任务和计划目标，不允许使用“当前URL”等动态值。可选 actions 数组只列此步骤需要的基础交互动作。
10. 所有操作都通过 GUI/DOM：下载点链接或查看器工具栏；改名用另存为/文件管理器；打开应用用系统启动器或快捷键。禁止 download、rename_artifact、run_in_terminal、call_code_agent、UNO 写文件等代码捷径。
11. 若给出类型化任务流程，steps 必须覆盖 workflow 中每个 phase（用相同英文值），每步 expected_evidence 不能为空。论文搜索先核对身份，再下载；不要提前编造下载URL。

系统验收条件：{contract}
任务流程模板：{workflow}

任务：{task}

只输出一个 JSON 对象，不要多余文字：
{{"steps": [{{"idx": 1, "phase": "search", "goal": "...", "expected_evidence": "...", "check_ids": [], "verification": [{{"kind":"visual"}}], "actions": ["click", "type"]}}], "keywords": ["..."], "final_check": "..."}}"""
