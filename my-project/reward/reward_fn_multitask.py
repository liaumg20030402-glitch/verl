"""多任务联合训练分发奖励函数（med-exam / blzk / kie / zyzl-blzk）。

verl 的 custom_reward_function 对**每条样本**调用一次，并把该行 parquet 里的
`data_source` 作为第一个参数传进来。多任务混训时只需一个分发函数：按 data_source
路由到各任务自己的打分逻辑即可。

本项目路由：
  - med-exam   → reward_fn_medexam_genrm_remote.compute_score_medexam_genrm_remote（async，外部 GenRM）
  - blzk       → reward_fn_blzk_rule.compute_score_blzk_rule
  - kie-science        → reward_fn_kie_rule.compute_score_kie_rule
  - zyzl-blzk  → reward_fn_zyzl_blzk_rule.compute_score_zyzl_blzk_rule

⚠️ 两个关键点：
  1) 本函数是 async：med-exam 走异步 HTTP 打分需要 await；rule 任务是同步函数，
     在 async 里直接调用即可（不 await）。verl 会自动 await 协程型 reward 函数。
  2) **统一输出键 schema**：verl 的 dapo/naive reward manager 会把每条样本返回的
     额外字段汇总成 reward_extra_info（按键聚合成 list），要求一个 batch 内各样本
     的键集合一致。各任务原函数返回的键并不相同（med-exam 带 genrm_failed，zyzl
     没有 pred 等），所以这里统一归一成固定的四个键再返回。
"""

import os
import sys

# verl 通过文件路径动态加载本模块（非包导入），相对 import 不可用。
# 把本文件所在目录加入 sys.path，按模块名导入同目录下的各任务奖励函数。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reward_fn_medexam_genrm_remote import compute_score_medexam_genrm_remote  # noqa: E402
from reward_fn_blzk_rule import compute_score_blzk_rule  # noqa: E402
from reward_fn_kie_rule import compute_score_kie_rule  # noqa: E402
from reward_fn_zyzl_blzk_rule import compute_score_zyzl_blzk_rule  # noqa: E402


# 同步规则任务的 data_source → 打分函数
_RULE_DISPATCH = {
    "blzk": compute_score_blzk_rule,
    "kie-science": compute_score_kie_rule,
    "zyzl-blzk": compute_score_zyzl_blzk_rule,
}


def _normalize(result: dict, data_source: str) -> dict:
    """把各任务返回的 dict 归一成固定键，保证 batch 内 reward_extra_info 对齐。"""
    if not isinstance(result, dict):
        result = {"score": float(result or 0.0)}
    try:
        score = float(result.get("score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    # ⚠️ 不要用 "data_source" 作为返回键：它是数据集保留字段，会被写进 non_tensor_batch
    # 后与原字段冲突，验证阶段 DataProto.union 比对时报
    #   AssertionError: `data_source` ... are not the same object
    # 想保留"这条样本属于哪个任务"的调试信息，用一个不冲突的键名（reward_data_source）。
    return {
        "score": score,
        "pred": str(result.get("pred", "") or ""),
        "judge_reason": str(result.get("judge_reason", "") or ""),
        "genrm_failed": float(result.get("genrm_failed", 0.0) or 0.0),
        "reward_data_source": str(data_source or ""),
    }


async def compute_score_multitask(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):
    """多任务分发入口（verl custom_reward_function，async）。"""
    ds = str(data_source or "").strip()

    if ds == "med-exam":
        # 异步：外部 GenRM 打分。reward_kwargs（grm_base_url / grm_model_name / ...）
        # 经 **kwargs 透传；verl 注入的 reward_router_address 也在 kwargs 里。
        result = await compute_score_medexam_genrm_remote(
            data_source, solution_str, ground_truth, extra_info, **kwargs
        )
    elif ds in _RULE_DISPATCH:
        # 同步规则评分：直接调用，不 await。
        result = _RULE_DISPATCH[ds](
            data_source, solution_str, ground_truth, extra_info, **kwargs
        )
    else:
        result = {"score": 0.0, "judge_reason": f"unknown_data_source:{ds}"}

    return _normalize(result, ds)
