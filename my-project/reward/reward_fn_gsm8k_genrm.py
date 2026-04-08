# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GSM8K generative reward: call the colocated RM vLLM router at /v1/chat/completions."""

import json
import logging
import os
import re

import aiohttp
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

GRM_PROMPT_TEMPLATE = """
You are given a math problem and a proposed solution.

Problem:
{problem}

Proposed solution:
{solution}

Evaluate how well the solution solves the problem. Give a score from 1 to 10, where:
- 1 means completely wrong or irrelevant.
- 5 means partially correct or incomplete.
- 10 means fully correct and well reasoned.

Reply with only one integer from 1 to 10, no other text.
""".strip()

_SCORE_RE = re.compile(r"\b([1-9]|10)\b")


def _parse_score_1_to_10(text: str) -> float:
    if not text:
        return 1.0
    text = text.strip()
    try:
        return float(int(text.split()[-1]))
    except (ValueError, IndexError):
        pass
    matches = _SCORE_RE.findall(text)
    if matches:
        return float(int(matches[-1]))
    return 1.0


async def _chat_completions(router_address: str, payload: dict) -> dict:
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return json.loads(await resp.text())


async def compute_score_gsm8k_genrm(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str | None = None,
    reward_model_tokenizer: PreTrainedTokenizer | None = None,
    **kwargs,
):
    """Generative RM for GSM8K-style data. Expects `question` or `query` in extra_info."""
    if not reward_router_address:
        raise ValueError(
            "GenRM requires reward.reward_model.enable=True so reward_router_address is set. "
            "Check RM deployment and resource pool."
        )

    problem = (extra_info or {}).get("question") or (extra_info or {}).get("query") or ""
    if not problem:
        logger.warning("extra_info has no 'question' or 'query'; GenRM prompt may be empty.")

    grm_prompt = GRM_PROMPT_TEMPLATE.format(problem=problem.strip(), solution=solution_str)
    messages = [{"role": "user", "content": grm_prompt}]

    model_name = (
        os.environ.get("GRM_MODEL_NAME")
        or os.environ.get("GRM_MODEL_PATH")
        or kwargs.get("grm_model_name")
    )
    if not model_name:
        model_name = reward_model_tokenizer.name_or_path if reward_model_tokenizer else None
    if not model_name:
        raise ValueError(
            "Set GRM_MODEL_NAME in the environment, or pass grm_model_name via reward.custom_reward_function.reward_kwargs, "
            "or ensure reward_model_tokenizer is available."
        )

    temperature = float(os.environ.get("GRM_TEMPERATURE", kwargs.get("grm_temperature", 0.3)))
    max_tokens = int(os.environ.get("GRM_MAX_TOKENS", kwargs.get("grm_max_tokens", 256)))
    top_p = float(os.environ.get("GRM_TOP_P", kwargs.get("grm_top_p", 0.9)))

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    output = await _chat_completions(reward_router_address, payload)
    grm_response = output["choices"][0]["message"]["content"] or ""
    raw_score = _parse_score_1_to_10(grm_response)
    raw_score = max(1.0, min(10.0, raw_score))
    # Normalize to [0, 1] for PPO/GRPO reward scale
    score = raw_score / 10.0

    return {
        "score": score,
        "acc": raw_score >= 9.0,
        "genrm_response": grm_response,
        "genrm_score_1_10": raw_score,
    }
