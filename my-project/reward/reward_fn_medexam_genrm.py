import json
import os
import re

import aiohttp
from transformers import PreTrainedTokenizer


_SCORE_RE = re.compile(r"\b(?:0(?:\.0+)?|0\.5|1(?:\.0+)?)\b")

# prompt 模板文件路径（直接指定绝对路径）
_PROMPT_FILE = "/train21/medcog/permanent/jycai6/jmli27/verl/verl/my-project/reward/prompts/medexam_judge_prompt.md"

# 模块级缓存，避免每次调用都重复读取磁盘
_JUDGE_PROMPT_TEMPLATE: str | None = None


def _load_judge_prompt_template() -> str:
    """从 markdown 文件加载 GenRM 评分 prompt 模板（延迟加载，只读一次）。"""
    global _JUDGE_PROMPT_TEMPLATE
    if _JUDGE_PROMPT_TEMPLATE is None:
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            _JUDGE_PROMPT_TEMPLATE = f.read().strip()
    return _JUDGE_PROMPT_TEMPLATE


def _build_judge_prompt(question: str, ground_truth: str, model_answer: str) -> str:
    """用 replace 方式填充 prompt 模板，避免与模板中的花括号冲突。"""
    tpl = _load_judge_prompt_template()
    return (
        tpl
        .replace("{question}", question)
        .replace("{ground_truth}", ground_truth)
        .replace("{model_answer}", model_answer)
    )


def _clean_answer_text(text: str) -> str:
    """清洗模型输出：移除 <think>...</think> 思考块，保留最终答案文本。
    """
    s = str(text or "")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    return s.strip()


def _normalize_option_answer(text: str) -> str:
    """将选项归一化为"去重 + 字母排序"的大写字符串。
    确保 'CBA' 与 'ABC' 归一化后完全一致。
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
    s = s.replace('"：', '":').replace("\u201c：", "\u201c:")
    return s


def _balanced_json_object(text: str) -> str:
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


def extract_json_from_text(text: str) -> str | None:
    """从模型输出中抽取 JSON 对象子串。
    兼容 markdown 代码块、前后自然语言及常见标点噪声。
    """
    s = _preprocess_jsonish_text(text)
    if not s:
        return None

    # 优先处理 Markdown ```json ... ``` 或 ``` ... ``` 代码块
    fence = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    for m in fence.finditer(s):
        inner = m.group(1).strip()
        obj = _balanced_json_object(inner)
        if obj:
            return obj

    # 从首个 '{' 到末尾 '}' 兜底（多对象场景可能不准确）
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        obj = _balanced_json_object(candidate)
        if obj:
            return obj

    # 对整段文本做平衡扫描兜底
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


def _parse_judge_fields(text: str) -> tuple[str, str, float]:
    """从裁判模型回复中解析出 extracted_answer、reasoning 和 score。
    解析失败时 score 默认 0.0
    """
    s = str(text or "").strip()
    extracted = ""
    reasoning = ""
    score = 0.0

    obj = _parse_json_dict(s)
    if obj is not None:
        extracted = str(obj.get("extracted_answer", "") or "")
        reasoning = str(obj.get("reasoning", "") or "")
        try:
            score = float(obj.get("score", 0.0))
        except Exception:
            score = 0.0
        return extracted, reasoning, score

    # JSON 解析失败时，用正则从文本中直接抓取分数（_SCORE_RE 本身只匹配 0/0.5/1）
    m = _SCORE_RE.findall(s)
    if m:
        score = float(m[-1])
    return extracted, reasoning, score


async def _chat_completions(router_address: str, payload: dict) -> dict:
    """向 vLLM router 发送 chat completions 请求并返回响应 dict。"""
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return json.loads(await resp.text())


async def compute_score_medexam_genrm(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str | None = None,
    reward_model_tokenizer: PreTrainedTokenizer | None = None,
    **kwargs,
):
    """医考任务生成式奖励函数（GenRM）。

    通过调用 vLLM router 的 /v1/chat/completions 接口，
    让裁判模型对比标准答案给出评分。
    """
    if not reward_router_address:
        raise ValueError(
            "GenRM 需要启用 reward.reward_model.enable=True，以便获取 reward_router_address。"
        )

    question = (extra_info or {}).get("question") or (extra_info or {}).get("query") or ""
    clean_solution = _clean_answer_text(solution_str)
    gt = _normalize_option_answer(ground_truth)

    # 优先从环境变量获取裁判模型名，其次从 kwargs，最后从 tokenizer
    model_name = (
        os.environ.get("GRM_MODEL_NAME")
        or os.environ.get("GRM_MODEL_PATH")
        or kwargs.get("grm_model_name")
    )
    if not model_name:
        model_name = reward_model_tokenizer.name_or_path if reward_model_tokenizer else None
    if not model_name:
        raise ValueError(
            "请通过 GRM_MODEL_NAME/GRM_MODEL_PATH 环境变量，"
            "或 reward_kwargs.grm_model_name，或 reward_model_tokenizer 提供裁判模型名称。"
        )

    prompt = _build_judge_prompt(
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

    # 容错处理：避免服务端异常返回导致 KeyError/IndexError 中断训练
    choices = output.get("choices", []) if isinstance(output, dict) else []
    if choices and isinstance(choices[0], dict):
        judge_resp = str((choices[0].get("message", {}) or {}).get("content", "") or "")
    else:
        judge_resp = ""

    extracted_answer, reasoning, score = _parse_judge_fields(judge_resp)
    pred = _normalize_option_answer(extracted_answer)
    # 保证 score 是可比较的数值，避免异常值污染训练
    try:
        score = float(score)
    except Exception:
        score = 0.0

    return {
        "score": score,
        "pred": pred,
        "reasoning": reasoning,
    }
