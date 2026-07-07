#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GenRM **判分一致性** 诊断（SGLang 离线推理版，病历审查 blsc 任务）。

目的：衡量 GenRM 裁判在不同温度下的「判分稳定性」——同一条 (医患对话, 病历文书)，固定其余采样
参数、只改温度，**同参数跑 N 次**，看 N 次审核结论是否一致。med-exam 因为把 ground_truth 塞进了
judge prompt，裁判几乎是确定性比对、对温度不敏感、测不出区别；blsc 没有 GT、要真正逐字段推理审核，
判分一致率才会随温度变化，能真正衡量"高温降复读"与"判分稳定"之间的权衡。

数据：直接用 blsc 原始 messages 数据里的 (user=医患对话, assistant=参考病历) 作为待审样本——
  这样不需要先跑 actor rollout，现成的参考病历就能当"被审病历文书"喂给裁判。

faithful：judge prompt / 病历清洗 / 审核解析 / 标签→分数映射 全部复用 reward_fn_blsc_genrm_remote，
与线上 blsc 奖励函数完全一致。

一次加载引擎、内部循环全部温度，结果汇总进**单个 summary.json**（含 per_temperature 对比表）。

用法示例：
  python check_genrm_consistency_sglang.py \
      --model /train21/.../pretrain_models/Qwen3.5-27B \
      --input /train21/.../dataset/blsc/blsc_val.jsonl \
      --temperatures 1.0 0.9 0.8 0.7 0.6 --n-runs 5 \
      --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --repetition-penalty 1.0 \
      --max-new-tokens 8192 --enable-thinking true --tp 1 --dp 8 \
      --out /train21/.../genrm_consistency_check
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 自包含的通用工具（不依赖兄弟脚本，避免集群上版本不一致导致 ImportError）
# --------------------------------------------------------------------------- #
def load_reward_module(path: str):
    """按文件路径动态加载 reward 模块，复用其 prompt/解析/评分函数。"""
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


def _finish_type(meta: dict) -> str:
    """SGLang meta_info.finish_reason 可能是 dict({'type':...}) 或字符串。"""
    fr = meta.get("finish_reason") if isinstance(meta, dict) else None
    if isinstance(fr, dict):
        return str(fr.get("type", ""))
    return str(fr or "")


def _rep_ngram_rate(text: str, n: int = 10) -> float:
    """字符级 n-gram 复读率 = 1 - 去重n-gram数/总n-gram数（值越高复读越严重）。"""
    s = str(text or "")
    if len(s) < n + 1:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _gen_batch(engine, prompts: list[str], sp: dict, bs: int) -> list[dict]:
    """分批调用 engine.generate，返回 [{text, finish_reason, completion_tokens}, ...]（保序）。"""
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


def extract_blsc_dialogue_record(row: dict) -> tuple[str, str]:
    """从 blsc messages 行抽 (dialogue=user, record=assistant)。

    blsc 原始数据格式：{"messages": [{role:system,...},{role:user,...},{role:assistant,...}], "loss":...}
    """
    msgs = row.get("messages")
    if msgs is None:
        return "", ""
    dialogue = record = ""
    for m in list(msgs):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        content = str(m.get("content", "") or "")
        if role == "user" and not dialogue:
            dialogue = content
        elif role == "assistant" and not record:
            record = content
    return dialogue, record


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _mean_pairwise_jaccard(sets: list) -> float:
    n = len(sets)
    if n < 2:
        return 1.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += _jaccard(sets[i], sets[j])
            cnt += 1
    return tot / cnt if cnt else 1.0


def parse_args():
    ap = argparse.ArgumentParser(description="GenRM 判分一致性诊断（SGLang 离线，blsc）")
    ap.add_argument("--model", required=True, help="GenRM 模型 HF 路径（和线上 server 同一个）")
    ap.add_argument("--input", nargs="+", required=True, help="blsc 原始数据（messages 格式 jsonl/parquet，可多个）")
    ap.add_argument("--reward-fn-path",
                    default="/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_blsc_genrm_remote.py",
                    help="blsc 奖励函数，复用其 _build_blsc_prompt / _parse_blsc_audit / _labels_to_score / _clean_answer_text")
    # 一致性实验主参数
    ap.add_argument("--temperatures", nargs="+", type=float, default=[1.0, 0.9, 0.8, 0.7, 0.6],
                    help="温度档位列表，引擎只加载一次、内部依次扫描")
    ap.add_argument("--n-runs", type=int, default=5, help="同一条样本同参数重复推理次数 N")
    # 固定采样（默认 = 实验4 约定）
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--enable-thinking", default="true")
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
    build_blsc_prompt = rmod._build_blsc_prompt
    parse_blsc_audit = rmod._parse_blsc_audit
    labels_to_score = rmod._labels_to_score
    clean_answer_text = rmod._clean_answer_text

    # ---------- 读数据 + 抽 (对话, 病历) ----------
    rows = []
    for p in args.input:
        rows.extend(read_table(p))
    print(f"[data] 读入 {len(rows)} 行（{len(args.input)} 个文件）")

    items = []   # (dialogue, record)
    skipped = 0
    for row in rows:
        d, r = extract_blsc_dialogue_record(row)
        r = clean_answer_text(r)   # 参考病历去掉可能的思考段
        if not d or not r:
            skipped += 1
            continue
        items.append((d, r))
    if args.max_samples > 0:
        items = items[: args.max_samples]
    if not items:
        raise SystemExit("没抽到有效的 (user 医患对话, assistant 病历) 样本，检查输入是否为 blsc messages 格式")
    print(f"[data] 有效样本 {len(items)} 条" + (f"；跳过 {skipped} 条（缺 user/assistant）" if skipped else ""))

    judge_prompts = [build_blsc_prompt(dialogue=d, medical_record=r) for d, r in items]

    # processor 渲染（与 server / 截断脚本一致：user 单轮 + apply_chat_template enable_thinking）
    try:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(args.model, trust_remote_code=trust_remote_code)
    except Exception as e:
        print(f"[warn] AutoProcessor 失败（{e}），回退 AutoTokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code)

    def render(jp: str) -> str:
        msgs = [{"role": "user", "content": jp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    rendered = [render(jp) for jp in judge_prompts]
    k = len(rendered)
    N = max(1, args.n_runs)

    # ---------- 起引擎（只加载一次）----------
    import sglang as sgl
    engine_kwargs = dict(model_path=args.model, tp_size=args.tp, dp_size=args.dp,
                         mem_fraction_static=args.mem_fraction, trust_remote_code=trust_remote_code)
    if args.mamba_backend:
        engine_kwargs["mamba_backend"] = args.mamba_backend
    engine = sgl.Engine(**engine_kwargs)

    def base_sp(temp: float) -> dict:
        sp = {"temperature": temp, "top_p": args.top_p, "top_k": args.top_k,
              "max_new_tokens": args.max_new_tokens}
        if args.min_p > 0:
            sp["min_p"] = args.min_p
        if args.repetition_penalty != 1.0:
            sp["repetition_penalty"] = args.repetition_penalty
        if args.presence_penalty != 0.0:
            sp["presence_penalty"] = args.presence_penalty
        if args.frequency_penalty != 0.0:
            sp["frequency_penalty"] = args.frequency_penalty
        return sp

    fixed_sampling = {
        "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
        "presence_penalty": args.presence_penalty, "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": args.frequency_penalty, "max_new_tokens": args.max_new_tokens,
    }
    print(f"[setup] 样本={k}  N={N}  温度档={args.temperatures}  固定采样={fixed_sampling}")

    per_temp = []

    # ---------- 逐温度扫描 ----------
    for temp in args.temperatures:
        sp = base_sp(temp)
        # N 份拷贝拼成一个大 batch：位置 p -> 样本 p%k，第 p//k 次
        flat = rendered * N
        print(f"\n[temp={temp}] 生成 {len(flat)} 条（{k}×{N}）...")
        gen = _gen_batch(engine, flat, sp, max(1, args.batch_size))

        # 解析一次，按 [run][sample] 存好；同时把每个 (温度,第几次跑) 的明细写进各自的小文件夹
        # 目录名注明温度和第几次跑：<OUT>/T{temp}_run{r}/outputs.jsonl
        parsed = [[None] * k for _ in range(N)]
        for run in range(N):
            run_dir = os.path.join(args.out, f"T{temp}_run{run + 1}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
                for i in range(k):
                    g = gen[run * k + i]
                    text = g["text"]
                    labels, basis, ok, clean = parse_blsc_audit(text)
                    score, tier = labels_to_score(labels)
                    ok_final = bool(ok and score is not None)
                    rep10 = _rep_ngram_rate(text, n=10)
                    info = {
                        "finish_reason": g["finish_reason"],
                        "completion_tokens": g["completion_tokens"],
                        "rep10_rate": round(rep10, 4),
                        "parsed_ok": ok_final,
                        "labels": list(labels),
                        "score": (float(score) if ok_final else None),
                        "tier": tier,
                    }
                    parsed[run][i] = info
                    f.write(json.dumps(
                        {"sample_idx": i, **info, "judge_resp": text},
                        ensure_ascii=False) + "\n")

        # 按样本聚合 N 次结果（跨同一温度的 N 个 run）
        n_trunc = n_parse_fail = 0
        comp_tokens, rep_rates = [], []
        label_exact = score_exact = 0
        jac_list, score_std_list = [], []

        for i in range(k):
            label_sets, scores = [], []
            for run in range(N):
                info = parsed[run][i]
                comp_tokens.append(info["completion_tokens"])
                rep_rates.append(info["rep10_rate"])
                n_trunc += int(info["finish_reason"] == "length")
                n_parse_fail += int(not info["parsed_ok"])
                if info["parsed_ok"]:
                    label_sets.append(frozenset(info["labels"]))
                    scores.append(float(info["score"]))
                else:
                    label_sets.append(frozenset({"<FAIL>"}))   # 解析失败当成独立结论，如实计入不一致
                    scores.append(-1.0)

            label_exact += int(len(set(label_sets)) == 1)
            score_exact += int(len(set(scores)) == 1)
            jac_list.append(_mean_pairwise_jaccard(label_sets))
            score_std_list.append(float(np.std(scores)))

        tot_gen = k * N
        arr = np.array(comp_tokens, dtype=np.float64) if comp_tokens else np.array([0.0])
        rep_arr = np.array(rep_rates, dtype=np.float64) if rep_rates else np.array([0.0])
        rec = {
            "temperature": temp,
            "label_exact_match_rate": round(label_exact / k, 4),   # N 次审核标签集完全一致的样本占比
            "score_exact_match_rate": round(score_exact / k, 4),   # N 次三档分数完全一致的样本占比（最贴近 reward 稳定性）
            "mean_pairwise_jaccard": round(float(np.mean(jac_list)), 4),
            "score_std_mean": round(float(np.mean(score_std_list)), 4),
            "parse_fail_rate": round(n_parse_fail / tot_gen, 4),
            "truncation_rate(finish=length)": round(n_trunc / tot_gen, 4),
            "rep10_rate_mean": round(float(rep_arr.mean()), 4),
            "completion_tokens": {
                "mean": round(float(arr.mean()), 1),
                "p90": round(float(np.percentile(arr, 90)), 1),
                "max": int(arr.max()),
            },
        }
        per_temp.append(rec)
        print(f"[temp={temp}] label_exact={rec['label_exact_match_rate']} "
              f"score_exact={rec['score_exact_match_rate']} jaccard={rec['mean_pairwise_jaccard']} "
              f"rep10={rec['rep10_rate_mean']} trunc={rec['truncation_rate(finish=length)']}")

    engine.shutdown()

    # ---------- 汇总 ----------
    summary = {
        "model": args.model,
        "task": "blsc",
        "n_samples": k,
        "n_runs": N,
        "enable_thinking": enable_thinking,
        "temperatures": list(args.temperatures),
        "fixed_sampling": fixed_sampling,
        "per_temperature": per_temp,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 文本对比表（一温度一行）
    print("\n==================== GenRM 判分一致性（blsc，越高越稳）====================")
    hdr = f"{'temp':>5} | {'label一致':>8} | {'score一致':>8} | {'Jaccard':>7} | {'score_std':>9} | {'parse失败':>8} | {'超长':>6} | {'rep10':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in per_temp:
        print(f"{r['temperature']:>5} | {r['label_exact_match_rate']:>8} | {r['score_exact_match_rate']:>8} | "
              f"{r['mean_pairwise_jaccard']:>7} | {r['score_std_mean']:>9} | {r['parse_fail_rate']:>8} | "
              f"{r['truncation_rate(finish=length)']:>6} | {r['rep10_rate_mean']:>6}")
    print(f"\n[save] 每个(温度,第几次跑)的明细 -> {os.path.join(args.out, 'T<temp>_run<r>/outputs.jsonl')}")
    print(f"[save] 汇总 -> {os.path.join(args.out, 'summary.json')}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
