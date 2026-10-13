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

    parts.append("请结合当前观察，只输出一个 JSON：next_action 为本次动作，summary 用一句话说明观察与行动计划。执行结果由框架核查。")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# 纠错反馈（动作解析/校验失败时，作为 assistant 历史追加，让模型重试）
# --------------------------------------------------------------------------- #
def build_format_feedback(error: str) -> str:
    return (
        f"本次候选动作未通过执行前校验，尚未执行：{error}\n"
        '请只修正本次输出，返回 {"next_action":{"type":"本轮动作名",...},"summary":"一句话观察与行动计划"}。'
        "保留原任务，不要把格式错误理解为上个已执行动作失败。"
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
    """Plan observable milestones; action-level decisions happen at execution time."""
    import json
    contract = json.dumps(completion_spec or {"checks": []}, ensure_ascii=False)
    workflow = json.dumps(profile or {}, ensure_ascii=False)
    from .actions import BASIC_ACTIONS
    menu = ", ".join(n for n in ACTION_REGISTRY if n in BASIC_ACTIONS)
    return f"""你是任务规划器。按任务的自然阶段形成最少但完整的、可独立核查的子目标，数量由任务复杂度决定，不按固定数量凑步骤。
一个子目标可以通过多个动作实现；不要把点击输入框、输入关键词、按回车拆成三个规划步骤，应合为“搜索目标并显示结果”。
保留身份核对、下载、改名等不同结果边界；简单任务可以只有一个子目标，复杂任务必须覆盖全部要求。
计划描述目标和必要前提，不预先编造尚未观察到的按钮、ref、搜索结果或论文身份；执行时依据真实反馈修订路径。

拆解规则：
1. 把打开正确网站/应用作为必要操作前提；若已经打开就直接利用当前状态。搜索子目标中明确首选入口（如 scholar.google.com、arxiv.org），无需把每个启动操作单列为子目标；
2. 搜索类任务：必须自己提炼搜索关键词（论文名、主题词、作者等），写进 keywords 数组；
3. 每个子目标要具体（含明确的操作对象和预期结果），不要写"完成任务"这种空话。不得把“搜索名为X的论文”改成“找一篇与X主题相关的论文”；短名先搜索核实身份，不凭常识猜成另一个对象；
4. 最后给出「完成判据」final_check：一个可以客观核对的完成标志（如"downloads 目录下存在该论文的 PDF 文件"）。
5. 补全操作前提与基本步骤：先到正确网站，找到目标，再操作，最后核对结果。不能用模糊的“找到并完成”代替操作步骤。
6. 每步写 expected_evidence，说明执行后应观察到的具体变化。goal 中写清本步所需的查询词、目标值、文件名等操作参数，不能只写“按要求填写”然后依赖 check_ids 代替指令；执行模型只看当前子目标。多对象任务逐个列出，不漏数量、筛选条件和保存要求。
7. 系统验收条件由调用方提供或由严格任务规则提取，你不能修改、删除或自行增加可信验收条件。check_ids 只引用下面已有 id；没有对应条件就填 []。
8. final_check 只是你的建议，不会作为任务完成的证据。任务结束由框架逐项核实。
9. 每个步骤增加 verification 数组：简单结果写明确谓词 url_equals(url)/url_host(host)/text_present(text)/app_open(app，真实进程名，不带路径)/element_state(name,role,states)；下载落盘使用 download_complete(check_id)，由框架核对对应产物；改名/最终验收另列步骤，不能用截图猜下载成功。难以用这些条件完整核实的步骤用 {{"kind":"visual"}}。全部条件须同时满足，不能只检查最终目标的一小部分。例如“提交搜索且显示结果”不能仅验证在搜索网站域名。predicate 的期望值必须来自原始任务和计划目标，不允许使用“当前URL”等动态值。可选 actions 数组只列此步骤需要的基础交互动作。
10. 所有操作都通过 GUI/DOM：下载点链接或查看器工具栏；改名用另存为/文件管理器；打开应用用系统启动器或快捷键。禁止 download、rename_artifact、run_in_terminal、call_code_agent、UNO 写文件等代码捷径。
11. 若给出类型化任务流程，steps 必须覆盖 workflow 中每个 phase（用相同英文值），每步 expected_evidence 不能为空。论文搜索先核对身份，再下载；不要提前编造下载URL。文件名要求放在 verify 阶段；框架会在 download 后自动插入独立的界面改名步骤，不必编造新的 phase。
12. 同时输出 requirements 数组，逐项保留原始任务要求。每项 kind 为 paper_download/rename/other，source_text 必须逐字引用任务中的连续原文；paper_download.value 为原文查询词，rename.value 为原文文件名，不能补造扩展名或论文ID。这些是候选解释，不是可信验收条件；框架会检查引用，最终仍核对完整原始任务。无法拆分的要求用 other 保留原文，不得省略否定、数量、目录或附加要求。
13. actions 使用以下精确动作名：{menu}。不要编造动作或调用后台下载/改名。类型提示与验收契约独立：即使系统 checks 为空，也必须保留用户目标与对应流程。

verification 的 JSON 写法示例（仅示范字段，选择与本步目标相符的条件，不要整组照抄）：
- 打开某域名：{{"kind":"url_host","host":"scholar.google.com"}}
- 打开准确网址：{{"kind":"url_equals","url":"https://arxiv.org/"}}
- 出现完整可见文字：{{"kind":"text_present","text":"示例完整标题"}}
- 打开应用：{{"kind":"app_open","app":"notepad"}}
- 激活应用：{{"kind":"app_foreground","app":"explorer"}}；打开后需继续键盘操作：{{"kind":"app_ready","app":"explorer"}}。app_open 仅证明窗口存在；app_ready 证明前台窗口有响应，不代表目录或业务内容已加载。Win+E 后由框架等待窗口就绪，超时先重新观察/选择窗口，不能盲发 Ctrl+L 或重复启动。
- 勾选指定控件：{{"kind":"element_state","name":"示例复选框","role":"checkbox","states":{{"checked":true}}}}
- 需要核对截图语义：{{"kind":"visual"}}
按示例写 host/url/text/app 等字段，不要统一写成 value；element_state 必须明确 name 与 states。
text_present 是完整可见文字匹配，不是关键词子串匹配；搜索短名不能证明论文身份，身份和全文结果需进一步核查。

系统验收条件：{contract}
任务流程模板：{workflow}

任务：{task}

只输出一个 JSON 对象，不要多余文字：
{{"steps": [{{"idx": 1, "phase": "search", "goal": "...", "expected_evidence": "...", "check_ids": [], "verification": [{{"kind":"visual"}}], "actions": ["click", "type"]}}], "keywords": ["..."], "final_check": "..."}}"""
