import json
import re


# 允许的 JSON 键组合及对应的结论字段名。
# 键集合必须与模型输出完全匹配其中一种，才视为格式合法。
_KEY_MODES = (
    ({"质控结论", "质控结论的可解释性", "质控结论的推理过程"}, "质控结论"),
    ({"质控结论", "质控结论的推理过程"}, "质控结论"),
    ({"推理过程", "结论"}, "结论"),
    ({"质控结论", "质控结论的可解释性"}, "质控结论"),
    ({"质控结论", "质控结论的推理过程", "质控结论的可解释性", "质控原文溯源"}, "质控结论"),
)


def _normalize_target(text: str) -> str:
    """规范化标签：去除全角引号后返回"合格"或"不合格"，其余原样返回。"""
    s = str(text or "").strip().replace("\u201c", "").replace("\u201d", "").replace('"', "")
    if s in ("合格",):
        return "合格"
    if s in ("不合格",):
        return "不合格"
    return s


def _sanitize_answer(text: str) -> str:
    """清洗模型输出：移除 <think>...</think> 思考块及 Markdown 代码围栏。
    """
    s = str(text or "")
    # 移除思考块，只保留 </think> 之后的最终回答部分
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    # 若存在 markdown 代码块，优先提取其内部内容；否则仅去掉围栏标记
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence_match:
        s = fence_match.group(1)
    else:
        s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_object(text: str) -> str:
    """按花括号深度提取首个顶层 JSON 对象，忽略字符串内部花括号。"""
    s = str(text or "").strip()
    if not s:
        return ""
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return ""


def judge_blzk_answer(answer: str, target: str) -> tuple[float, dict]:
    """
    病历质控规则打分：
    1. 清洗模型输出并提取 JSON 对象
    2. 键集合必须与预定义 _KEY_MODES 之一完全一致
    3. 对应结论字段须为"合格"或"不合格"
    4. 预测结论与目标一致得 1.0，否则 0.0
    """
    clean_answer = _sanitize_answer(answer)
    target_norm = _normalize_target(target)
    if target_norm not in {"合格", "不合格"}:
        return 0.0, {"reason": "invalid_target"}

    json_text = _extract_first_json_object(clean_answer)
    if not json_text:
        return 0.0, {"reason": "json_not_found"}

    try:
        obj = json.loads(json_text)
    except Exception:
        return 0.0, {"reason": "json_parse_failed"}

    if not isinstance(obj, dict):
        return 0.0, {"reason": "json_not_object"}

    keys = set(obj.keys())
    for required_keys, conclusion_key in _KEY_MODES:
        if keys == required_keys:
            model_conclusion = _normalize_target(obj.get(conclusion_key, ""))
            if model_conclusion not in {"合格", "不合格"}:
                return 0.0, {
                    "reason": "invalid_model_conclusion",
                    "pred": model_conclusion
                }

            hit = model_conclusion == target_norm
            return (1.0 if hit else 0.0), {
                "reason": "ok" if hit else "mismatch",
                "pred": model_conclusion
            }

    return 0.0, {"reason": "key_mode_not_matched"}


def compute_score_blzk_rule(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):
    """
    verl 自定义奖励函数入口（同步版本）。
    函数签名遵循 dapo/naive reward manager 约定。
    """
    target = ground_truth
    if not target and isinstance(extra_info, dict):
        target = extra_info.get("target", "")

    score, detail = judge_blzk_answer(solution_str, target)
    return {
        "score": float(score),
        "pred": detail.get("pred", ""),
        "judge_reason": detail.get("reason", "unknown"),
    }
