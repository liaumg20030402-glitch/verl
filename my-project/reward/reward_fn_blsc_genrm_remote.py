"""病历审查（blsc）生成式奖励函数——**外部 GenRM server 版**。

actor 任务：根据【医患对话】生成【病历文书】（病历生成）。本函数用 prompts/blsc.md 让外部
GenRM 逐字段审核 actor 生成的病历文书（**无 ground_truth**，审核依据是对话本身），把审核
结果标签按三档映射成分数：
  - `审核结果` 含任一「严重错误」 → 0.0
  - 仅含「可接受错误」          → 0.5
  - 「合格」                    → 1.0
  - 解析失败 / 标签无法判定      → genrm_failed=1（软丢弃，优势置 0）

与 reward_fn_medexam_genrm_remote 的区别仅在：judge 模板（blsc.md）、字段抽取（医患对话取自
extra_info.dialogue、病历文书取自 actor 输出）、以及标签→分数映射。远程调用 / JSON 修复 /
落盘等基础设施直接复用 reward_fn_medexam_genrm_remote，避免重复实现、保证行为一致。

跑前提：同 med-exam，已用 start_genrm_server.sh 起好 GenRM 服务，
训练脚本 export GENRM_BASE_URL / GENRM_MODEL_NAME（或经 reward_kwargs.grm_base_url 下发）。
"""

import os
import sys
import time

# verl 通过文件路径动态加载本模块（非包导入），同目录模块需手动加 sys.path 再导入。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 复用 med-exam 远程版的通用基础设施（HTTP 重试+超时 / 地址解析 / 思考清洗 / JSON 修复 / 落盘）
from reward_fn_medexam_genrm_remote import (  # noqa: E402
    _chat_completions,
    _resolve_base_url,
    _clean_answer_text,
    _sanitize_judge_response,
    _loads_or_repair,
    _dump_judge_debug,
)


# blsc judge prompt 模板（病历审查），延迟加载只读一次
_BLSC_PROMPT_FILE = "/train21/medcog/permanent/jycai6/jmli27/reward/prompts/blsc.md"
_BLSC_PROMPT_TEMPLATE: str | None = None


def _load_blsc_template() -> str:
    global _BLSC_PROMPT_TEMPLATE
    if _BLSC_PROMPT_TEMPLATE is None:
        with open(_BLSC_PROMPT_FILE, "r", encoding="utf-8") as f:
            _BLSC_PROMPT_TEMPLATE = f.read().strip()
    return _BLSC_PROMPT_TEMPLATE


def _build_blsc_prompt(dialogue: str, medical_record: str) -> str:
    """填充 blsc.md 的 {aaa}=医患对话、{bbb}=病历文书。

    用 replace 而非 str.format：模板「输出格式」段含单花括号 JSON 示例，format 会冲突。
    """
    tpl = _load_blsc_template()
    return tpl.replace("{aaa}", dialogue).replace("{bbb}", medical_record)


def _parse_blsc_audit(text: str) -> tuple[list[str], str, bool, str]:
    """解析 GenRM 审核输出，返回 (labels, basis, parsed_ok, clean)。

    labels = `审核结果` 标签列表（统一成 list[str]）；basis = `错误情况依据`（落作 judge_reason）。
    parsed_ok=True 仅当能解析出 dict（含 json_repair 修复成功）。
    """
    sanitized = _sanitize_judge_response(text)
    obj, clean = _loads_or_repair(sanitized)
    if isinstance(obj, dict):
        raw = obj.get("审核结果", [])
        if isinstance(raw, str):
            raw = [raw]
        labels = [str(x).strip() for x in raw if str(x).strip()] if isinstance(raw, list) else []
        basis = str(obj.get("错误情况依据", "") or "")
        return labels, basis, True, clean
    return [], "", False, clean


def _labels_to_score(labels: list[str]) -> tuple[float | None, str]:
    """三档映射。返回 (score, tier)；无法判定 → (None, 'unknown')。

    优先级：严重错误 > 可接受错误 > 合格。只要出现任一严重错误即 0.0；否则仅可接受错误 0.5；
    否则为合格 1.0。靠子串匹配，对标签是否带「字段-」前缀、连字符是否规范都鲁棒。
    """
    if not labels:
        return None, "unknown"
    joined = "".join(labels)
    if "严重错误" in joined:
        return 0.0, "severe"
    if "可接受错误" in joined:
        return 0.5, "acceptable"
    if any("合格" in lab for lab in labels):   # 含 "合格"（"不合格" 不在 blsc 标签集内）
        return 1.0, "qualified"
    return None, "unknown"


async def compute_score_blsc_genrm_remote(
    data_source: str,
    solution_str: str,
    ground_truth: str,            # blsc 无 GT，占位忽略
    extra_info: dict,
    reward_router_address: str | None = None,
    **kwargs,
):
    """病历审查 GenRM 奖励函数（外部 server 版，async）。"""
    base_url = _resolve_base_url(reward_router_address, kwargs)

    info = extra_info or {}
    # 医患对话：convert_data_to_verl_rl.convert_blsc_row 把它存进 extra_info.dialogue
    dialogue = str(info.get("dialogue") or info.get("question") or "").strip()
    medical_record = _clean_answer_text(solution_str)   # actor 生成的病历文书（去掉思考段）

    model_name = (
        kwargs.get("grm_model_name")
        or os.environ.get("GENRM_MODEL_NAME")
        or os.environ.get("GRM_MODEL_NAME")
        or "genrm_remote"
    )

    prompt = _build_blsc_prompt(dialogue=dialogue, medical_record=medical_record)
    enable_thinking = kwargs.get("grm_enable_thinking", True)
    max_tokens = int(os.environ.get("GRM_MAX_TOKENS", kwargs.get("grm_max_tokens", 1024)))
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(os.environ.get("GRM_TEMPERATURE", kwargs.get("grm_temperature", 0.8))),
        "top_p": float(os.environ.get("GRM_TOP_P", kwargs.get("grm_top_p", 1.0))),
        "top_k": int(kwargs.get("grm_top_k", -1)),
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    # 防复读惩罚（仅非中性值才下发；与 med-exam 共用同一套 GRM_* 配置）。
    min_p = float(os.environ.get("GRM_MIN_P", kwargs.get("grm_min_p", 0.0)))
    repetition_penalty = float(os.environ.get("GRM_REPETITION_PENALTY", kwargs.get("grm_repetition_penalty", 1.0)))
    presence_penalty = float(os.environ.get("GRM_PRESENCE_PENALTY", kwargs.get("grm_presence_penalty", 0.0)))
    frequency_penalty = float(os.environ.get("GRM_FREQUENCY_PENALTY", kwargs.get("grm_frequency_penalty", 0.0)))
    if min_p > 0:
        payload["min_p"] = min_p
    if repetition_penalty != 1.0:
        payload["repetition_penalty"] = repetition_penalty
    if presence_penalty != 0.0:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty != 0.0:
        payload["frequency_penalty"] = frequency_penalty

    request_timeout = float(kwargs.get("grm_request_timeout", 300.0))
    max_retries = int(kwargs.get("grm_max_retries", 2))
    try:
        output = await _chat_completions(
            base_url, payload, max_retries=max_retries, request_timeout=request_timeout
        )
    except Exception as e:  # noqa: BLE001
        return {
            "score": 0.0,
            "pred": "",
            "judge_reason": f"genrm_request_failed: {e!r}",
            "genrm_failed": 1.0,
        }

    usage = output.get("usage", {}) if isinstance(output, dict) else {}
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    choices = output.get("choices", []) if isinstance(output, dict) else []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message", {}) or {}
        judge_resp = str(msg.get("content", "") or "")
        finish_reason = str(choices[0].get("finish_reason", "") or "")
    else:
        judge_resp = ""
        finish_reason = ""

    labels, basis, parsed_ok, clean = _parse_blsc_audit(judge_resp)
    score, tier = _labels_to_score(labels)

    if parsed_ok and score is not None:
        return {
            "score": float(score),
            "pred": ";".join(labels),                 # 命中的审核标签
            "judge_reason": f"{tier} | {basis}".strip(" |"),
            "genrm_failed": 0.0,
        }

    # 解析失败 或 标签无法判定 → 软丢弃 + 落盘排查
    dump_path = _dump_judge_debug(
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": data_source,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "max_tokens": max_tokens,
            "dialogue": dialogue,
            "model_answer": medical_record,    # actor 生成的病历文书
            "judge_resp_raw": judge_resp,
            "clean": clean,
            "parsed_labels": labels,
        },
        kwargs,
    )
    reason = "genrm_parse_failed" if not parsed_ok else f"genrm_unknown_labels:{labels}"
    return {
        "score": 0.0,
        "pred": ";".join(labels),
        "judge_reason": (
            f"{reason} | finish={finish_reason} "
            f"| gen_tokens={completion_tokens}/{max_tokens} | dump={dump_path or '-'}"
        ),
        "genrm_failed": 1.0,
    }
