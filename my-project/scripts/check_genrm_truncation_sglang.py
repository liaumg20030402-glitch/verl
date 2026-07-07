#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenRM 截断率诊断（SGLang 离线推理版）。

目的：验证 med-exam 的 GenRM 裁判"超长被截断 / 复读"到底是不是**模型调用方式（HTTP server）**
的问题。做法：用和线上**完全一致**的 judge prompt 和采样参数，但走**离线 sgl.Engine**，
统计截断率 / JSON 解析失败率 / 生成 token 分布。若离线截断率也很高 → 不是 server 的锅，
是模型/采样/prompt 本身；若离线明显更低 → 问题出在 server 调用路径（如 chat_template_kwargs 没生效）。

输入：verl 落盘的 validation_data / rollout_data（.parquet 或 .jsonl），或任何含
  (question, ground_truth, model_answer, data_source) 的表。脚本会自动探测字段，探测不到
  可用 --*-field 覆盖。只取 data_source == --data-source 的行。

faithful 要点（与 reward_fn_medexam_genrm_remote 对齐）：
  - judge prompt 复用同一份 _build_judge_prompt（同一个 prompt 模板文件）；
  - model_answer 用 _clean_answer_text 去掉 actor 的 </think>（与 server 一致）；
  - 构造 messages=[{"role":"user", ...}] 再 apply_chat_template(enable_thinking=True)，镜像 server；
  - 解析复用 _parse_judge_fields（json + json_repair），口径与线上一致。

用法示例：
  python check_genrm_truncation_sglang.py \
      --model /train21/.../pretrain_models/Qwen3.5-27B \
      --input /train21/.../verl_grpo_qwen3_5_27b_multitask/.../validation_data/*.parquet \
      --data-source med-exam \
      --temperature 0.8 --top-p 1.0 --top-k -1 --max-new-tokens 8192 \
      --enable-thinking true --presence-penalty 1.5 --repetition-penalty 1.1 \
      --tp 1 --dp 8 \
      --out /train21/.../genrm_trunc_check
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 复用奖励函数里的 prompt 构造 / 清洗 / 解析（保证与线上一致）
# --------------------------------------------------------------------------- #
def load_reward_module(path: str):
    spec = importlib.util.spec_from_file_location("_genrm_reward_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_table(path: str) -> list[dict]:
    """读 parquet / jsonl，返回 records 列表。"""
    if path.endswith(".parquet"):
        return pd.read_parquet(path).to_dict("records")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# 只处理 verl validation_data / rollout_data 这一种 dump 格式（字段固定）：
#   input  : "system\n{系统提示}\nuser\n{题目}\nassistant\n<think>\n"
#   output : actor 回答（含 <think>...</think>）
#   gts    : 标准答案
#   reward_data_source / data_source : 任务标识
def extract_data_source(row: dict) -> str:
    v = row.get("reward_data_source")
    if v is None:
        v = row.get("data_source")
    return str(v or "")


def extract_ground_truth(row: dict) -> str:
    return str(row.get("gts", "") or "").strip()


def extract_model_answer(row: dict) -> str:
    return str(row.get("output", "") or "")


def extract_question(row: dict) -> str:
    """从 input 串里抠 user 轮题目：...\nuser\n{题目}\nassistant\n<think>\n。"""
    inp = str(row.get("input", "") or "")
    if not inp:
        return ""
    # 纯角色标签：取 user 行之后、assistant 行之前
    m = re.search(r"(?:^|\n)user\s*\n(.*?)\n\s*assistant\b", inp, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 只有 user 没 assistant：取 user 之后全部
    m = re.search(r"(?:^|\n)user\s*\n(.*)$", inp, re.DOTALL)
    if m:
        return m.group(1).strip()
    return inp.strip()


def _finish_type(meta: dict) -> str:
    """SGLang meta_info.finish_reason 可能是 dict({'type': 'length'/'stop'}) 或字符串。"""
    fr = meta.get("finish_reason") if isinstance(meta, dict) else None
    if isinstance(fr, dict):
        return str(fr.get("type", ""))
    return str(fr or "")


def _rep_ngram_rate(text: str, n: int = 10) -> float:
    """字符级 n-gram 复读率 = 1 - 去重n-gram数/总n-gram数。

    不依赖截断：值越高说明重复子串越多（复读越严重），中英混排都适用。
    """
    s = str(text or "")
    if len(s) < n + 1:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _gen_batch(engine, prompts: list[str], sp: dict, bs: int) -> list[dict]:
    """分批调用 engine.generate，返回 [{text, finish_reason, completion_tokens}, ...]。"""
    out = []
    for s in range(0, len(prompts), bs):
        outs = engine.generate(prompt=prompts[s:s + bs], sampling_params=sp)
        for o in outs:
            text = o.get("text", "") if isinstance(o, dict) else ""
            meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
            out.append({
                "text": text,
                "finish_reason": _finish_type(meta),
                "completion_tokens": int(meta.get("completion_tokens", 0) or 0),
            })
        print(f"[gen] {min(s + bs, len(prompts))}/{len(prompts)}", flush=True)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="GenRM 截断率诊断（SGLang 离线）")
    ap.add_argument("--model", required=True, help="GenRM 模型 HF 路径（和线上 server 同一个）")
    ap.add_argument("--input", nargs="+", required=True, help="validation/rollout dump（parquet/jsonl，可多个）")
    ap.add_argument("--data-source", default="med-exam", help="只取该 data_source 的行")
    ap.add_argument("--reward-fn-path",
                    default="/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_medexam_genrm_remote.py",
                    help="奖励函数文件，复用其 _build_judge_prompt / _clean_answer_text / _parse_judge_fields")
    # 采样：默认和训练脚本 multi 分支一致
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--enable-thinking", default="true", help="true/false：和线上 GRM_ENABLE_THINKING 一致")
    # ===== 防复读/防崩 可选手段（默认中性=不启用；逐个开做对比实验）=====
    ap.add_argument("--min-p", type=float, default=0.0, help="min-p 采样，>0 启用（如 0.05）")
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help="token 级，>1 生效（如 1.05/1.1）")
    ap.add_argument("--presence-penalty", type=float, default=0.0, help="OpenAI 风格，>0 生效（如 1.0/1.5）")
    ap.add_argument("--frequency-penalty", type=float, default=0.0, help="随出现次数累加，>0 生效（如 0.5/1.0）")
    ap.add_argument("--max-samples", type=int, default=-1, help="只测前 N 条，-1 用全部")
    ap.add_argument("--batch-size", type=int, default=256)
    # SGLang 引擎
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--mem-fraction", type=float, default=0.9)
    ap.add_argument("--mamba-backend", default="triton", help="Qwen3.5 GDN 必须 triton；留空则不传")
    ap.add_argument("--trust-remote-code", default="true")
    ap.add_argument("--out", required=True, help="输出目录")
    return ap.parse_args()


def main():
    args = parse_args()
    enable_thinking = args.enable_thinking.strip().lower() in ("1", "true", "yes", "y")
    trust_remote_code = args.trust_remote_code.strip().lower() in ("1", "true", "yes", "y")
    os.makedirs(args.out, exist_ok=True)

    rmod = load_reward_module(args.reward_fn_path)
    build_judge_prompt = rmod._build_judge_prompt
    clean_answer_text = rmod._clean_answer_text
    parse_judge_fields = rmod._parse_judge_fields

    # ---------- 读数据 + 过滤 data_source + 抽字段 ----------
    rows = []
    for p in args.input:
        rows.extend(read_table(p))
    print(f"[data] 读入 {len(rows)} 行（{len(args.input)} 个文件）")

    items = []          # {question, ground_truth, model_answer}
    empty_q = 0
    for row in rows:
        if extract_data_source(row) != args.data_source:
            continue
        q = extract_question(row)
        gt = extract_ground_truth(row)
        ma = clean_answer_text(extract_model_answer(row))
        if not q:
            empty_q += 1
        items.append({"question": q, "ground_truth": gt, "model_answer": ma})

    if args.max_samples > 0:
        items = items[: args.max_samples]
    if not items:
        raise SystemExit(f"没找到 data_source=={args.data_source} 的样本；检查 --data-source 和输入文件")
    print(f"[data] data_source={args.data_source} 命中 {len(items)} 条"
          + (f"；⚠️ 其中 {empty_q} 条没抽到 question（input 结构异常，judge prompt 会缺题干）"
             if empty_q else ""))

    # ---------- 构造 judge prompt（与 server 一致：user 单轮 + apply_chat_template enable_thinking） ----------
    judge_prompts = [
        build_judge_prompt(question=it["question"].strip(),
                           ground_truth=it["ground_truth"],
                           model_answer=it["model_answer"])
        for it in items
    ]

    # processor 渲染（和同事 infer_sglang / passk 一致；缺视觉文件则回退 AutoTokenizer）
    try:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(args.model, trust_remote_code=trust_remote_code)
    except Exception as e:
        print(f"[warn] AutoProcessor 失败（{e}），回退 AutoTokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code)

    def render(judge_prompt: str) -> str:
        msgs = [{"role": "user", "content": judge_prompt}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    rendered = [render(jp) for jp in judge_prompts]

    # ---------- 起 SGLang Engine，批量生成 ----------
    import sglang as sgl
    engine_kwargs = dict(model_path=args.model, tp_size=args.tp, dp_size=args.dp,
                         mem_fraction_static=args.mem_fraction, trust_remote_code=trust_remote_code)
    if args.mamba_backend:
        engine_kwargs["mamba_backend"] = args.mamba_backend
    engine = sgl.Engine(**engine_kwargs)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }
    # 可选防复读项：仅在非中性值时下发（老版本 SGLang 可能不认某些键，不下发就不会报错）
    if args.min_p > 0:
        sampling_params["min_p"] = args.min_p
    if args.repetition_penalty != 1.0:
        sampling_params["repetition_penalty"] = args.repetition_penalty
    if args.presence_penalty != 0.0:
        sampling_params["presence_penalty"] = args.presence_penalty
    if args.frequency_penalty != 0.0:
        sampling_params["frequency_penalty"] = args.frequency_penalty
    print(f"[sampling] {sampling_params}")

    # 生成（普通单段）
    gen = _gen_batch(engine, rendered, sampling_params, max(1, args.batch_size))
    engine.shutdown()

    raw_path = os.path.join(args.out, "judge_outputs.jsonl")
    n_trunc = 0          # finish_reason == length
    n_parse_fail = 0     # _parse_judge_fields parsed_ok=False
    n_no_think_close = 0 # 输出里没有 </think>（思考没闭合，多半截断）
    n_heavy_rep = 0      # 复读率 > 0.5 的样本（重度复读）
    comp_tokens = []
    rep_rates = []

    with open(raw_path, "w", encoding="utf-8") as fout:
        for idx, g in enumerate(gen):
            text = g["text"]
            ftype = g["finish_reason"]
            ctok = g["completion_tokens"]
            comp_tokens.append(ctok)

            is_trunc = (ftype == "length")
            no_close = ("</think>" not in text)
            rep10 = _rep_ngram_rate(text, n=10)   # 复读率（不依赖截断）
            rep_rates.append(rep10)
            # clean = json_repair 修复后的文本（与线上奖励函数同口径）
            _, _, _, parsed_ok, clean = parse_judge_fields(text)

            n_trunc += int(is_trunc)
            n_parse_fail += int(not parsed_ok)
            n_no_think_close += int(no_close)
            n_heavy_rep += int(rep10 > 0.5)

            fout.write(json.dumps({
                "finish_reason": ftype,
                "completion_tokens": ctok,
                "truncated": is_trunc,
                "no_think_close": no_close,
                "rep10_rate": round(rep10, 4),
                "parsed_ok": parsed_ok,
                "judge_prompt": judge_prompts[idx],   # 送给裁判打分的完整 prompt（_build_judge_prompt 拼好的）
                "judge_resp": text,
                "clean": clean,                       # json_repair 修复后
            }, ensure_ascii=False) + "\n")

    # ---------- 汇总 ----------
    n = len(rendered)
    arr = np.array(comp_tokens, dtype=np.float64) if comp_tokens else np.array([0.0])
    rep_arr = np.array(rep_rates, dtype=np.float64) if rep_rates else np.array([0.0])
    summary = {
        "model": args.model,
        "data_source": args.data_source,
        "n_samples": n,
        "enable_thinking": enable_thinking,
        # 本次试验用的全部采样/防复读配置（每个 OUT 自带，方便对比表）
        "sampling": {
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
        },
        "truncation_rate(finish=length)": round(n_trunc / n, 4),
        "no_think_close_rate": round(n_no_think_close / n, 4),
        "json_parse_fail_rate": round(n_parse_fail / n, 4),
        "heavy_repeat_rate(rep10>0.5)": round(n_heavy_rep / n, 4),
        "rep10_rate_mean": round(float(rep_arr.mean()), 4),
        "completion_tokens": {
            "mean": round(float(arr.mean()), 1),
            "median": round(float(np.median(arr)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "max": int(arr.max()),
            "pct_at_max_tokens": round(float((arr >= args.max_new_tokens - 1).mean()), 4),
        },
        "finish_reason_counts": dict(Counter(
            json.loads(l)["finish_reason"] for l in open(raw_path, encoding="utf-8"))),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==================== GenRM 截断率诊断 ====================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[save] 每条裁判输出 -> {raw_path}")
    print(f"[save] 汇总 -> {os.path.join(args.out, 'summary.json')}")
    print("==========================================================")


if __name__ == "__main__":
    main()
