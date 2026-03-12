from __future__ import annotations

from pathlib import Path

from agents.business_consultant_worker import create_business_consultant_worker
from agents.data_analyst_worker import create_data_analyst_worker
from agents.report_writer_worker import create_report_writer_worker
from agents.result_utils import extract_agent_result_text
from agents.visualization_worker import create_visualization_worker


def _clean_text(value: str | None, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _selected_file_name(selected_file_path: str) -> str:
    normalized = str(selected_file_path or "").replace("\\", "/")
    return Path(normalized).name or normalized or "已选数据集"


def _is_meaningful_output(text: str, query: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if normalized.lower().startswith("error:"):
        return False

    cleaned_query = _clean_text(query)
    if cleaned_query and normalized == cleaned_query:
        return False
    if cleaned_query and len(normalized) < max(32, len(cleaned_query) + 12) and cleaned_query in normalized:
        return False
    return True


def _append_section(sections: list[str], title: str, content: str, query: str) -> bool:
    cleaned = _clean_text(content)
    if not _is_meaningful_output(cleaned, query):
        return False
    sections.append(f"## {title}\n{cleaned}")
    return True


def _build_dataset_profiler_task(query: str, selected_file_path: str) -> str:
    file_name = _selected_file_name(selected_file_path)
    user_request = _clean_text(query, "请分析这个训练数据集。")
    return f"""你现在是“训练数据质量分析师”，负责对大模型训练数据做结构化体检。

目标文件：{file_name}
用户原始需求：{user_request}

请直接读取这个文件，不要向用户追问文件路径，并且全程使用中文输出。

你的任务重点：
1. 判断数据集类型：是指令微调、对话、多轮聊天、偏好对、评测集、工具调用轨迹，还是混合格式。
2. 分析记录量、字段清单、schema 一致性，以及每个关键字段的含义。
3. 分析数据构成：角色分布、轮次分布、语言分布、evaluation 标签分布、来源类别、样本类型等。
4. 分析 token 相关字段，例如 input_tokens、output_tokens、total_tokens、长度分布、偏态和异常值。
5. 如果是 JSON/JSONL 对话数据，重点检查 messages：
   - role 分布是否合理
   - 平均轮次数
   - 是否存在空消息、重复 system prompt、异常消息结构
6. 结论必须尽量具体，给出数量、比例、分布、代表性样本模式，而不是泛泛而谈。

请输出一份扎实、详细、面向训练数据治理的数据画像报告，不要只做简短回应。必须使用中文回复"""


def _build_quality_audit_task(query: str, selected_file_path: str, dataset_profile: str) -> str:
    file_name = _selected_file_name(selected_file_path)
    user_request = _clean_text(query, "请分析这个训练数据集。")
    return f"""你现在是“训练数据质量分析员”，负责对大模型训练数据做严格的质量检查。

目标文件：{file_name}
用户原始需求：{user_request}

上一个阶段得到的数据画像如下：
{dataset_profile}

现在请重新读取并验证这个文件，不要盲信前一阶段的总结，并且全程使用中文输出。

重点审计以下问题：
1. 缺失、空值、空字符串、格式错误、schema 断裂。
2. 重复样本、近重复样本、重复提问、重复回答、模板化严重的样本。
3. messages 结构异常：role 错误、content 为空、轮次顺序混乱、list/object 结构不一致。
4. token 与长度问题：过短样本、过长样本、极端离群点、截断风险、长度分布严重失衡。
5. 标签与 evaluation 质量：类别不平衡、噪声标签、不一致的评价值、逻辑上不可能的组合。
6. 训练风险信号：乱码、占位符、HTML/Markdown 垃圾、元数据泄漏、明显的隐私或敏感内容线索。

输出要求：
- 按严重程度分为：严重 / 高 / 中 / 低
- 每一类问题尽量给出证据、数量、占比和典型模式
- 最后给出一句质量结论和一个按优先级排序的修复清单
- 避免空泛表达，要像真正的数据质检报告一样有判断力 必须使用中文回复"""


def _build_readiness_review_task(query: str, selected_file_path: str, dataset_profile: str, quality_audit: str) -> str:
    file_name = _selected_file_name(selected_file_path)
    user_request = _clean_text(query, "请分析这个训练数据集。")
    return f"""你现在是“训练就绪度评审专家”，负责判断这份数据是否真的适合用于大模型训练。

目标文件：{file_name}
用户原始需求：{user_request}

数据画像：
{dataset_profile}

质量审计结果：
{quality_audit}

请站在 SFT / 指令微调 / 对话训练数据治理的角度，全程使用中文输出，并给出有决策价值的评审意见。

你必须输出：
1. 0 到 100 的训练就绪度评分
2. 结论标签：可直接训练 / 有条件可训练 / 风险较高 / 不建议直接训练
3. 这份数据的主要优点
4. 会拖累训练效果的主要阻塞项
5. 一份按影响优先级排序的修复计划
6. 针对清洗、去重、均衡、长度控制、扩充策略的务实建议

不要机械复述前面的分析，请做真正的综合判断。"""


def _build_report_writer_task(query: str, selected_file_path: str) -> str:
    file_name = _selected_file_name(selected_file_path)
    user_request = _clean_text(query, "请分析这个训练数据集。")
    return f"""请把前面几个阶段的结果整合为一份专业的“训练数据审计报告”。

目标文件：{file_name}
用户原始需求：{user_request}

要求全程使用中文，报告风格要像一份严肃的训练数据质量评审，不要写成普通聊天回复。

请使用清晰的 Markdown 结构，并至少包含这些部分：
1. 执行摘要
2. 数据集构成画像
3. 数据质量审计结论
4. 训练就绪度评分卡
5. 关键风险与失败模式
6. 修复与治理建议
7. 最终结论

要求：
- 保留重要数字、比例、分布、异常模式和关键证据
- 如果前面生成了图表，可以自然引用
- 报告需要有技术感、判断力和可执行性
- 除了可选保存文件外，必须直接在当前回答中输出完整报告正文"""


def _build_visualization_request(query: str, selected_file_path: str) -> str:
    file_name = _selected_file_name(selected_file_path)
    user_request = _clean_text(query, "请分析这个训练数据集。")
    return (
        "请生成对训练数据审计真正有帮助的图表，并使用中文说明。\n"
        f"目标文件：{file_name}\n"
        f"用户需求：{user_request}\n\n"
        "优先考虑：token 或长度分布、evaluation 标签平衡、messages 轮次分布、role 构成、关键字段分布。\n"
        "只生成能被当前审计结果真实支撑的图表，避免空洞可视化。"
    )


async def run_training_data_audit(
    query: str,
    selected_file_path: str,
    *,
    include_visualization: bool = False,
) -> str:
    selected_file = _clean_text(selected_file_path)
    if not selected_file:
        return ""

    collected_sections: list[str] = []
    file_name = _selected_file_name(selected_file)

    profiler_response = await create_data_analyst_worker(
        task_description=_build_dataset_profiler_task(query, selected_file),
        file_paths=[selected_file],
        display_task_description=f"分析训练数据的结构、字段与样本构成：{file_name}",
        agent_name="DatasetProfiler",
    )
    profiler_output = extract_agent_result_text(profiler_response).strip()
    has_profiler_output = _append_section(collected_sections, "数据集画像", profiler_output, query)

    quality_output = ""
    if has_profiler_output:
        quality_response = await create_data_analyst_worker(
            task_description=_build_quality_audit_task(query, selected_file, profiler_output),
            file_paths=[selected_file],
            display_task_description=f"审计数据质量、messages 结构与训练风险：{file_name}",
            agent_name="QualityAuditor",
        )
        quality_output = extract_agent_result_text(quality_response).strip()
        _append_section(collected_sections, "质量审计", quality_output, query)

    readiness_output = ""
    readiness_source = "\n\n".join(section for section in [profiler_output, quality_output] if _clean_text(section))
    if _clean_text(readiness_source):
        readiness_response = await create_business_consultant_worker(
            task_description=_build_readiness_review_task(query, selected_file, profiler_output, quality_output),
            analysis_data=readiness_source,
            display_task_description=f"评估训练就绪度并给出优先级修复建议：{file_name}",
            agent_name="TrainingReadinessReviewer",
        )
        readiness_output = extract_agent_result_text(readiness_response).strip()
        _append_section(collected_sections, "训练就绪度评审", readiness_output, query)

    if include_visualization and _clean_text(readiness_source):
        visualization_response = await create_visualization_worker(
            analysis_data=readiness_source,
            custom_requirements=_build_visualization_request(query, selected_file),
        )
        visualization_output = extract_agent_result_text(visualization_response).strip()
        _append_section(collected_sections, "审计图表", visualization_output, query)

    report_input = "\n\n".join(collected_sections).strip()
    if report_input:
        report_response = await create_report_writer_worker(
            task_description=_build_report_writer_task(query, selected_file),
            all_results=report_input,
            display_task_description=f"生成最终训练数据审计报告：{file_name}",
            agent_name="AuditReportWriter",
        )
        report_output = extract_agent_result_text(report_response).strip()
        if _is_meaningful_output(report_output, query):
            return report_output

    return report_input
