"""医考任务生成式奖励函数（GenRM）——**外部 server 版**。

与 `reward_fn_medexam_genrm.py`（verl 托管 standalone 版）的唯一区别：
  - 托管版：地址由 verl 注入 `reward_router_address`（需 reward.reward_model.enable=True）。
  - 外部版（本文件）：**地址来自环境变量 `GENRM_BASE_URL`**，指向你自己用
    `start_genrm_server.sh` 起的独立 vLLM/SGLang OpenAI 兼容服务。verl 不管 GenRM 的卡。

跑前提：
  1. 已用 start_genrm_server_sglang.sh 把 GenRM 起成服务，并 curl 验证过；
  2. 训练脚本里 export GENRM_BASE_URL / GENRM_MODEL_NAME 指向它。
"""

import asyncio
import json
import os
import re
import socket
import time

import aiohttp

# JSON 容错修复：裁判（尤其开思考的复杂任务）偶尔吐出非严格 JSON（字符串内用了双引号、
# 缺逗号、尾随逗号等）。json_repair 能把这些常见问题修好再解析。环境没装也不影响导入。
try:
    import json_repair
except ImportError:
    json_repair = None


# prompt 模板文件路径（直接指定绝对路径）
_PROMPT_FILE = "/train21/medcog/permanent/jycai6/jmli27/reward/prompts/medexam_judge_prompt.md"

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
    """清洗 actor 模型输出：保留 </think> 之后的最终回答。"""
    s = str(text or "")
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[-1]
    return s.strip()


def _sanitize_judge_response(text: str) -> str:
    """清洗 GenRM judge 模型的输出，便于后续直接 ``json.loads``。"""
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


def _loads_or_repair(clean: str):
    """先严格 json.loads；失败再用 json_repair 尝试修复后解析。都失败返回 None。"""
    if not clean:
        return None
    try:
        return json.loads(clean)
    except Exception:
        pass
    if json_repair is not None:
        try:
            return json_repair.loads(clean)
        except Exception:
            return None
    return None


def _parse_judge_fields(text: str) -> tuple[str, str, float, bool, str]:
    """从裁判模型回复中解析 extracted_answer / reasoning / score，并返回 parsed_ok 和 clean。

    parsed_ok=True 仅当 sanitize 后能解析出 dict（含 json_repair 修复成功）。
    解析失败（json_repair 也修不了）→ parsed_ok=False，上层据此标 genrm_failed=1 软丢弃。
    clean = sanitize 后（去 </think>/围栏、json_repair 修复前）的文本，失败时回传用于定位原因。
    """
    clean = _sanitize_judge_response(text)
    obj = _loads_or_repair(clean)
    if isinstance(obj, dict):
        extracted = str(obj.get("extracted_answer", "") or "")
        reasoning = str(obj.get("reasoning", "") or "")
        try:
            score = float(obj.get("score", 0.0))
        except Exception:
            score = 0.0
        return extracted, reasoning, score, True, clean
    return "", "", 0.0, False, clean


def _resolve_base_url(reward_router_address: str | None, kwargs: dict) -> str:
    """确定 GenRM 服务地址（单实例：start_genrm_server_sglang.sh 起的那个单 server）。

    优先级：reward_kwargs.grm_base_url（最稳：经 verl 配置下发，多机各 worker 都拿得到）
            > 环境变量 GENRM_BASE_URL > verl 注入的 reward_router_address（兼容托管模式）。
    返回 'http://host:port' 根地址（不含 /v1/...）。
    多机要点：环境变量未必能传到各节点的 Ray reward worker，所以**首选 grm_base_url 这条**。
    """
    base = kwargs.get("grm_base_url") or os.environ.get("GENRM_BASE_URL")
    if not base and reward_router_address:
        # 兼容：万一仍走 verl 托管路径，router 地址通常是 'host:port'（无 scheme）
        base = reward_router_address
        if not base.startswith("http"):
            base = f"http://{base}"
    if not base:
        raise ValueError(
            "外部 server 模式需要 GenRM 地址：优先用 reward_kwargs.grm_base_url（推荐，多机可达），"
            "或环境变量 GENRM_BASE_URL，例如 http://100.85.97.73:8000"
        )
    return base.rstrip("/")


async def _chat_completions(
    base_url: str,
    payload: dict,
    max_retries: int = 4,
    base_delay: float = 2.0,
    request_timeout: float = 300.0,
) -> dict:
    """向外部 vLLM/SGLang 的 /v1/chat/completions 发请求，带重试 + 总超时。

    外部 server 偶发连接重置 / 服务重启，重试能避免单点抖动直接打挂训练。
    ⚠️ 必须有总超时(total)：否则单条 judge 请求一旦卡住会无限等待，
       整个 reward batch 的 asyncio.gather 永不返回 → 训练 step 卡死 → 训练侧 GPU 空闲被集群杀。
    """
    url = f"{base_url}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=request_timeout, connect=30)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return json.loads(await resp.text())
        except Exception as e:  # noqa: BLE001 — 连接重置/超时/5xx 都重试
            last_exc = e
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
    # 重试用尽：抛出，让上层 reward manager 决定（也可改成 return 0 分容错）
    raise RuntimeError(f"GenRM 请求失败（已重试 {max_retries} 次）: {url} | {last_exc!r}")


def _dump_judge_debug(record: dict, kwargs: dict) -> str:
    """把裁判原始回复落盘到调试目录，便于排查超长/解析失败的样本。

    目录来自 reward_kwargs.grm_debug_dir 或环境变量 GRM_DEBUG_DIR；未设则不落盘、返回 ""。
    每个进程一个 jsonl 文件追加（ray reward worker 单进程单事件循环 → 同进程追加串行、安全；
    跨进程各写各的文件 → 不冲突）。调试完把 GRM_DEBUG_DIR 取消即可停。
    """
    debug_dir = kwargs.get("grm_debug_dir") or os.environ.get("GRM_DEBUG_DIR")
    if not debug_dir:
        return ""
    try:
        os.makedirs(debug_dir, exist_ok=True)
        fpath = os.path.join(debug_dir, f"genrm_debug_{socket.gethostname()}_{os.getpid()}.jsonl")
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return fpath
    except Exception:
        return ""


async def compute_score_medexam_genrm_remote(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str | None = None,  # 兼容托管模式；外部模式留空即可
    **kwargs,
):
    """医考 GenRM 奖励函数（外部 server 版）。

    通过 GENRM_BASE_URL 指向的独立 vLLM 服务的 /v1/chat/completions 打分。
    """
    base_url = _resolve_base_url(reward_router_address, kwargs)

    question = (extra_info or {}).get("question") or (extra_info or {}).get("query") or ""
    clean_solution = _clean_answer_text(solution_str)
    gt = str(ground_truth or "").strip()

    # 模型名 = 你给 vllm serve 的 --served-model-name（外部模式默认 'genrm'）
    model_name = (
        kwargs.get("grm_model_name")          # 首选：reward_kwargs 下发（多机可达）
        or os.environ.get("GENRM_MODEL_NAME")
        or os.environ.get("GRM_MODEL_NAME")
        or "genrm_remote"
    )

    prompt = _build_judge_prompt(
        question=question.strip(),
        ground_truth=gt,
        model_answer=clean_solution,
    )
    enable_thinking = kwargs.get("grm_enable_thinking", True)
    max_tokens = int(os.environ.get("GRM_MAX_TOKENS", kwargs.get("grm_max_tokens", 1024)))
    # ⚠️ thinking 模式必须用采样，禁止 temp=0（贪心+思考会无限复读崩溃）。
    # 复读主要受温度影响：实测 temp 0.6→0.7→0.8 复读递减，top_k(20/50/-1) 无明显差异。
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(os.environ.get("GRM_TEMPERATURE", kwargs.get("grm_temperature", 0.8))),
        "top_p": float(os.environ.get("GRM_TOP_P", kwargs.get("grm_top_p", 1.0))),
        "top_k": int(kwargs.get("grm_top_k", -1)),
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    request_timeout = float(kwargs.get("grm_request_timeout", 300.0))
    # 最坏空等 = request_timeout × max_retries，必须 < 集群 idle-kill 阈值(5min)。
    max_retries = int(kwargs.get("grm_max_retries", 2))
    try:
        output = await _chat_completions(
            base_url, payload, max_retries=max_retries, request_timeout=request_timeout
        )
    except Exception as e:  # noqa: BLE001
        # 失败兜底：返回 0（不抛异常/不卡死整个 step）。
        # 注意：只有把 verl 优势补丁同步到集群后，genrm_failed=1 才会让这些样本被
        # 排除组统计+优势置 0（软丢弃）；未打补丁时这条就是 0 分，会有少量污染——
        # 所以目标是让超时几乎不发生（靠 DP 调大），而不是依赖兜底。
        return {
            "score": 0.0,
            "pred": "",
            "judge_reason": f"genrm_request_failed: {e!r}",
            "genrm_failed": 1.0,   # 标记（给 verl 优势补丁用；未打补丁时仅作元数据）
        }

    usage = output.get("usage", {}) if isinstance(output, dict) else {}
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)   # 真实生成 token 数（同 max_tokens 量纲）

    choices = output.get("choices", []) if isinstance(output, dict) else []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message", {}) or {}
        judge_resp = str(msg.get("content", "") or "")
        # finish_reason="length" → 被 max_tokens 截断（思考太长没轮到输出 JSON 的铁证）。
        finish_reason = str(choices[0].get("finish_reason", "") or "")
    else:
        judge_resp = ""
        finish_reason = ""

    extracted_answer, reasoning, score, parsed_ok, clean = _parse_judge_fields(judge_resp)
    pred = str(extracted_answer or "").strip()

    if parsed_ok:
        # 结构化解析成功（含 json_repair 修复）：正常打分
        try:
            score = float(score)
        except Exception:
            score = 0.0
        return {"score": score, "pred": pred, "judge_reason": reasoning, "genrm_failed": 0.0}

    # json_repair 也修不了 → 软丢弃（genrm_failed=1 → 不进组统计、优势置 0），避免污染梯度。
    #   gen_tokens 逼近 max_tokens + finish=length → 被截断（多半思考太长）。
    #   完整 raw 回复 + actor 答案 + 标准答案落盘到 GRM_DEBUG_DIR，便于排查为什么这么长。
    dump_path = _dump_judge_debug(
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": data_source,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "max_tokens": max_tokens,
            "question": question,
            "ground_truth": gt,
            "model_answer": clean_solution,   # 喂给裁判的 actor 答案（看是不是太长导致裁判想很久）
            "judge_resp_raw": judge_resp,     # 裁判原始 content（完整，不截断）
            "clean": clean,                   # sanitize 后、json_repair 前
        },
        kwargs,
    )
    return {
        "score": 0.0,
        "pred": pred,
        "judge_reason": (
            f"genrm_parse_failed | finish={finish_reason} "
            f"| gen_tokens={completion_tokens}/{max_tokens} | dump={dump_path or '-'}"
        ),
        "genrm_failed": 1.0,
    }
