import json
import re


# KIE（关键信息提取）规则奖励函数。
#
# 数据约定（见 convert_data_to_verl_rl.py 的 convert_kie_row）：
#   target / ground_truth 形如：
#       ```json
#       {"output": ["5"]}
#       ```
#   即一个 JSON 对象，"output" 是抽取结果列表（可为多值）。
#
# 评分逻辑：
#   1) 清洗模型输出（同 blzk：只去 </think> 思考块 + markdown 围栏）；
#   2) 模型输出和 target 都尝试 json.loads；
#      - 都解析成功 → compare_obj 递归比较，完全一致得 1.0，否则 0.0；
#      - 任一解析失败 → 退化为「清洗后字符串完全相等」比较（兜底原始行为）。


def _sanitize_answer(text: str) -> str:
    """清洗模型输出：移除思考块及 Markdown 代码围栏（逻辑与 reward_fn_blzk_rule 一致）。

    思考块处理：若文本中出现 ``</think>``，只保留最后一个 ``</think>`` 之后的内容。
    这能同时覆盖标准 `<think>...</think>答案`，以及 Qwen3.5 原生 thinking 模式
    （chat template 已在 prompt 末尾追加 `<think>`，模型实际只输出 `thinking...</think>答案`，
    没有 `<think>` 起始标签）。没有 `</think>` 的普通输出整段保留。
    """
    s = str(text or "")
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[-1]
    # 若存在 markdown 代码块，优先提取其内部内容；否则仅去掉围栏标记
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence_match:
        s = fence_match.group(1)
    else:
        s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def compare_obj(obj1, obj2) -> bool:
    """对象一致性对比（沿用旧框架逻辑）。

    - dict：键集合 + 长度一致，按 key 排序后逐值递归比较；
    - list：长度一致；元素为 dict 时，先把每个 dict 按 key 排序序列化成字符串、
            对字符串列表排序后再逐一递归比较；元素为标量时，直接 sorted 后逐一比较
            （顺序不敏感）；
    - 标量及其他类型：直接相等比较。
    todo: 嵌套 dict 的多 key-value(list) 组合排序。
    """
    try:
        if type(obj1) is not type(obj2):
            return False

        # dict compare
        if isinstance(obj1, dict):
            if len(obj1) != len(obj2):
                return False
            for key in obj1:
                # 对比 dict 的 key 是否一致
                if key not in obj2:
                    return False
                # 若 value 类型为 list，检查 list 长度是否一致
                if isinstance(obj1[key], list) and len(obj1[key]) != len(obj2[key]):
                    return False
            # dict 基于 key 重排序后对比
            obj1 = {key: obj1[key] for key in sorted(obj1)}
            obj2 = {key: obj2[key] for key in sorted(obj2)}
            return all(k in obj2 and compare_obj(v, obj2[k]) for k, v in obj1.items())

        # list compare
        elif isinstance(obj1, list):
            if not isinstance(obj2, list):
                return False
            if len(obj1) == len(obj2) == 0:
                return True
            if len(obj1) != len(obj2):
                return False
            # 对列表内 dict 排序后再比较
            if isinstance(obj1[0], dict) or isinstance(obj2[0], dict):
                obj1_ = [json.dumps({key: item[key] for key in sorted(item)}, ensure_ascii=False) for item in obj1]
                obj2_ = [json.dumps({key: item[key] for key in sorted(item)}, ensure_ascii=False) for item in obj2]
                return all(compare_obj(json.loads(i), json.loads(j)) for i, j in zip(sorted(obj1_), sorted(obj2_)))
            # 对列表内部排序后再比较
            return all(compare_obj(i, j) for i, j in zip(sorted(obj1), sorted(obj2)))

        # 字符串或其他非常见数据类型，直接对比
        else:
            return obj1 == obj2
    except Exception:
        return False


def judge_kie_answer(answer: str, target: str) -> tuple[float, dict]:
    clean_answer = _sanitize_answer(answer)
    clean_target = _sanitize_answer(target)

    if not clean_answer:
        return 0.0, {"reason": "json_not_found", "pred": ""}

    answer_obj = target_obj = None
    answer_ok = target_ok = False
    try:
        answer_obj = json.loads(clean_answer)
        answer_ok = True
    except Exception:
        answer_ok = False
    try:
        target_obj = json.loads(clean_target)
        target_ok = True
    except Exception:
        target_ok = False

    if answer_ok and target_ok:
        hit = compare_obj(answer_obj, target_obj)
        return (1.0 if hit else 0.0), {
            "reason": "ok" if hit else "mismatch",
            "pred": clean_answer,
        }

    # 兜底：任一侧 JSON 解析失败，退化为清洗后字符串完全相等比较
    hit = clean_answer == clean_target
    return (1.0 if hit else 0.0), {
        "reason": "str_match" if hit else "json_parse_failed",
        "pred": clean_answer,
    }


def compute_score_kie_rule(
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
    # ground_truth 和 extra_info["target"] 在 convert_data_to_verl_rl.py 里同源，
    # 直接用 ground_truth 即可。
    score, detail = judge_kie_answer(solution_str, ground_truth)
    return {
        "score": float(score),
        "pred": detail.get("pred", ""),
        "judge_reason": detail.get("reason", "unknown"),
    }
