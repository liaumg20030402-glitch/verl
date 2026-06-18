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
    """清洗 actor 模型输出：保留 </think> 之后的最终回答。

    覆盖三种情形：
      1) 标准 `<think>...</think>实际答案`
      2) Qwen3.5 / Qwen3 原生 thinking：chat template 在 prompt 末尾就追加了
         `<think>`，模型实际只输出 `thinking...</think>实际答案`，**没有 `<think>` 开始
         标签**。旧实现用 `<think>.*?</think>` 成对正则会漏掉这种情况。
      3) 没有 `</think>` 的非 thinking 输出，整段保留。
    """
    s = str(text or "")
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[-1]
    return s.strip()


def _sanitize_judge_response(text: str) -> str:
    """清洗 GenRM judge 模型的输出，便于后续直接 ``json.loads``。

    处理两件事：
      1. **thinking 块**：若文本中出现 ``</think>``，只保留最后一个 ``</think>`` 之后的内容。
         兼容带 thinking 的 GenRM（如 Qwen3-Instruct 系列）。
      2. **Markdown 代码围栏**：若有 ```json ... ```，提取围栏内内容；否则剥掉残缺围栏标记。
    """
    s = str(text or "")
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[-1]
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence_match:
        s = fence_match.group(1)
    else:
        s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_judge_fields(text: str) -> tuple[str, str, float]:
    """从裁判模型回复中解析出 extracted_answer、reasoning 和 score。

    简化逻辑：sanitize 后直接 ``json.loads``。
    失败时用正则从**原始文本**兜底抓 score，避免 reward 全 0 导致梯度消失。
    """
    extracted = ""
    reasoning = ""
    score = 0.0

    clean = _sanitize_judge_response(text)
    if clean:
        try:
            obj = json.loads(clean)
            if isinstance(obj, dict):
                extracted = str(obj.get("extracted_answer", "") or "")
                reasoning = str(obj.get("reasoning", "") or "")
                try:
                    score = float(obj.get("score", 0.0))
                except Exception:
                    score = 0.0
                return extracted, reasoning, score
        except Exception:
            pass

    # JSON 解析失败兜底：从原始文本里直接抓 0 / 0.5 / 1 这三种合法 score
    # （_SCORE_RE 在文件顶部定义，仅匹配这三个合法值，避免误抓）
    m = _SCORE_RE.findall(str(text or ""))
    if m:
        try:
            score = float(m[-1])
        except Exception:
            score = 0.0
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
    gt = str(ground_truth or "").strip()

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
    pred = str(extracted_answer or "").strip()
    # 保证 score 是可比较的数值，避免异常值污染训练
    try:
        score = float(score)
    except Exception:
        score = 0.0

    info = extra_info or {}
    return {
        "score": score,
        "pred": pred,
        "judge_reason": reasoning,
    }
