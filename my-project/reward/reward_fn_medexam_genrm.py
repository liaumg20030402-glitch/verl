import json
import os
import re

import aiohttp
from transformers import PreTrainedTokenizer


_SCORE_RE = re.compile(r"\b(?:0(?:\.0+)?|0\.5|1(?:\.0+)?)\b")


def _clean_answer_text(text: str) -> str:
    s = str(text or "")
    # <unused7> 常作为“思考内容/最终答案”分隔符，通常其后才是最终回答。
    if "<unused7>" in s:
        s = s.split("<unused7>")[-1]
    s = s.replace("<end>", "")
    s = s.replace("<unused6>", "")
    s = s.replace("<ret>", "\n")
    s = s.replace("\\r", "")
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", " ")
    # 移除 Qwen3 风格思考块。
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    return s.strip()


def _normalize_option_answer(text: str) -> str:
    """将选项归一化为“去重 + 排序”的大写字符串。
    使用字母序排序，保证 'CBA' 与 'ABC' 归一化后一致。
    """
    s = str(text or "").upper()
    s = s.replace(" ", "").replace(",", "")
    s = re.sub(r"[^A-Z]", "", s)
    return "".join(sorted(set(s)))


def _preprocess_jsonish_text(text: str) -> str:
    """在抽取 JSON 前，先归一化空白并修正常见 JSON 标点错误。"""
    if not isinstance(text, str):
        return ""
    s = text.replace("\xa0", " ").replace("\u200b", "")
    s = s.replace("}，", "},").replace("]，", "],")
    s = s.replace('"：', '":').replace("”：", "”:")
    return s


def _balanced_json_object(text: str) -> str:
    """按花括号深度提取首个顶层 JSON 对象，并忽略字符串内部花括号。"""
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


def extract_json_from_text(text: str) -> str | None:
    """
    从模型输出中抽取 JSON 对象子串。
    兼容 markdown 代码块、前后自然语言及常见标点噪声。
    """
    s = _preprocess_jsonish_text(text)
    if not s:
        return None

    # 1) 处理 Markdown ```json ... ``` 或 ``` ... ``` 代码块
    fence = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    for m in fence.finditer(s):
        inner = m.group(1).strip()
        obj = _balanced_json_object(inner)
        if obj:
            return obj

    # 2) 从首个 '{' 到末尾 '}' 的旧式兜底（多对象场景可能不准确）
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        obj = _balanced_json_object(candidate)
        if obj:
            return obj

    # 3) 对整段文本做平衡扫描兜底
    obj = _balanced_json_object(s)
    return obj if obj else None


def _parse_json_dict(text: str) -> dict | None:
    """对抽取出的 JSON 进行 loads，成功且为 dict 则返回，否则返回 None。"""
    raw = extract_json_from_text(text)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_score(score_raw) -> float:
    try:
        s = float(score_raw)
    except Exception:
        return 0.0
    if s >= 0.75:
        return 1.0
    if s >= 0.25:
        return 0.5
    return 0.0


def _parse_judge_fields(text: str) -> tuple[str, str, float]:
    s = str(text or "").strip()
    extracted = ""
    reasoning = ""
    score = 0.0

    obj = _parse_json_dict(s)
    if obj is not None:
        extracted = str(obj.get("extracted_answer", "") or "")
        reasoning = str(obj.get("reasoning", "") or "")
        score = _normalize_score(obj.get("score", 0.0))
        return extracted, reasoning, score

    m = _SCORE_RE.findall(s)
    if m:
        score = _normalize_score(m[-1])
    return extracted, reasoning, score


async def _chat_completions(router_address: str, payload: dict) -> dict:
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return json.loads(await resp.text())


JUDGE_PROMPT_TEMPLATE = """
你是一位专业的医学考试评分专家，评估考生对医学考试题目的回答。

[题目]
{question}

[标准答案]
{ground_truth}

[模型答案]
{model_answer}

评分规则：
- 单选题：完全正确1.0，否则0.0
- 多选题：完全正确1.0，部分正确0.5，完全错误或多选少选超过一个0.0
输出格式：只输出以下json格式内容，不要输出其他，如果 reasoning 内部需要引号，必须使用单引号（' '）
```json
{
  "extracted_answer": "从考生回答中提取的选项，如'A'或'AB'，若无法提取则填'无法识别'",
  "reasoning": "对比标准答案的评分理由，简洁说明",
  "score": 分数
}
```
""".strip()


async def compute_score_medexam_genrm(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str | None = None,
    reward_model_tokenizer: PreTrainedTokenizer | None = None,
    **kwargs,
):
    """
    医考任务生成式奖励函数（GenRM）。
    返回值中的 score 已归一化到 [0, 1]，可直接用于 PPO/GRPO。
    """
    if not reward_router_address:
        raise ValueError(
            "GenRM requires reward.reward_model.enable=True so reward_router_address is available."
        )

    question = (extra_info or {}).get("question") or (extra_info or {}).get("query") or ""
    clean_solution = _clean_answer_text(solution_str)
    gt = _normalize_option_answer(ground_truth)

    model_name = (
        os.environ.get("GRM_MODEL_NAME")
        or os.environ.get("GRM_MODEL_PATH")
        or kwargs.get("grm_model_name")
    )
    if not model_name:
        model_name = reward_model_tokenizer.name_or_path if reward_model_tokenizer else None
    if not model_name:
        raise ValueError("Please set GRM_MODEL_NAME/GRM_MODEL_PATH or provide grm_model_name in reward_kwargs.")

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        ground_truth=gt,
        model_answer=clean_solution,
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(os.environ.get("GRM_TEMPERATURE", kwargs.get("grm_temperature", 0.0))),
        "top_p": float(os.environ.get("GRM_TOP_P", kwargs.get("grm_top_p", 1.0))),
        "max_tokens": int(os.environ.get("GRM_MAX_TOKENS", kwargs.get("grm_max_tokens", 128))),
    }
    output = await _chat_completions(reward_router_address, payload)
    # 容错处理：避免服务端异常返回导致 KeyError/IndexError 直接中断训练。
    choices = output.get("choices", []) if isinstance(output, dict) else []
    if choices and isinstance(choices[0], dict):
        judge_resp = str((choices[0].get("message", {}) or {}).get("content", "") or "")
    else:
        judge_resp = ""

    extracted_answer, reasoning, score = _parse_judge_fields(judge_resp)
    pred = _normalize_option_answer(extracted_answer)
    local_exact = float(pred == gt and pred != "")

    return {
        "score": score,
        "acc": bool(score > 0.5),
        "pred": pred,
        "target": gt,
        "reasoning": reasoning,
        "extracted_answer": extracted_answer,
        "local_exact": local_exact,
        "genrm_response": judge_resp,
        "genrm_score": score,
        "data_source": str(data_source),
    }
