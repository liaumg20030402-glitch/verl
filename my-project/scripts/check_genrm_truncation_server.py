#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenRM 截断率诊断（**HTTP server 版**）——和实际 RL 训练完全同一条调用路径。

与 check_genrm_truncation_sglang.py（离线 Engine 版）的区别：
  - 离线版：本进程起 sgl.Engine 直接生成；
  - 本版：向你**已起好的 SGLang server** 的 /v1/chat/completions 发 HTTP 请求（同
    reward_fn_medexam_genrm_remote 的调用方式）。

用途：
  1) 验证 server 端是否支持各惩罚参数（直接看会不会 400）；
  2) 测这些参数在 server 路径下对截断率/复读率的实际效果（训练就是走 server）。

faithful：judge prompt / model_answer 清洗 / 解析全部复用 reward 模块，与线上一致。
payload 默认与 reward_fn 相同（temperature/top_p/max_tokens/chat_template_kwargs）；
额外的 top_k/min_p/penalties **仅在非中性值时**加进请求体——
这样默认 run == 训练请求；要测哪个就开哪个，server 不支持的键会以 HTTP 400 暴露出来。

用法示例：
  python check_genrm_truncation_server.py \
      --base-url http://100.85.97.73:8000 --model-name genrm_remote \
      --input /train21/.../validation_data/*.parquet --data-source med-exam \
      --temperature 0.8 --max-new-tokens 8192 --enable-thinking true \
      --presence-penalty 1.5 --repetition-penalty 1.1 \
      --out /train21/.../genrm_trunc_check_server/prespen1.5
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter

import aiohttp
import numpy as np

# 复用离线 check 脚本里的取数/解析工具，保证两版口径完全一致
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_genrm_truncation_sglang import (  # noqa: E402
    load_reward_module, read_table, extract_data_source, extract_ground_truth,
    extract_model_answer, extract_question, _rep_ngram_rate,
)


def _finish_type(choice: dict) -> str:
    """OpenAI 兼容响应里 choices[0].finish_reason 一般是字符串（'length'/'stop'）。"""
    fr = choice.get("finish_reason") if isinstance(choice, dict) else None
    if isinstance(fr, dict):
        return str(fr.get("type", ""))
    return str(fr or "")


async def _chat_one(sem, session, url, payload, timeout, retries):
    """发一条 chat 请求。返回 {"data": resp} 或 {"error": msg}。
    HTTP 4xx（如参数不支持）直接返回错误不重试；网络抖动才重试。
    """
    async with sem:
        last = None
        for attempt in range(max(1, retries)):
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        return {"error": f"HTTP {resp.status}: {body[:400]}"}
                    return {"data": json.loads(body)}
            except Exception as e:  # 连接重置/超时
                last = e
                if attempt < retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
        return {"error": f"request_failed: {last!r}"}


def parse_args():
    ap = argparse.ArgumentParser(description="GenRM 截断率诊断（HTTP server 版）")
    ap.add_argument("--base-url", required=True, help="如 http://100.85.97.73:8000（不含 /v1/...）")
    ap.add_argument("--model-name", default="genrm_remote", help="= server 的 --served-model-name")
    ap.add_argument("--input", nargs="+", required=True, help="validation/rollout dump（parquet/jsonl）")
    ap.add_argument("--data-source", default="med-exam")
    ap.add_argument("--reward-fn-path",
                    default="/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_medexam_genrm_remote.py")
    # 采样（默认对齐 reward_fn / multi 分支）
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--enable-thinking", default="true")
    # 防复读/防崩（仅非中性值才加进请求体 → 默认 run == 训练请求；server 不支持的键会 400 暴露）
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    # HTTP 并发/超时
    ap.add_argument("--concurrency", type=int, default=64, help="并发请求数")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


async def gather_all(args, payloads):
    """并发发全部请求，asyncio.gather **保序**返回（results[i] 对应 payloads[i]）。"""
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    total = len(payloads)
    done = 0

    async with aiohttp.ClientSession() as session:
        async def one(pl):
            nonlocal done
            r = await _chat_one(sem, session, url, pl, args.request_timeout, args.max_retries)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[req] {done}/{total}", flush=True)
            return r
        return await asyncio.gather(*[one(pl) for pl in payloads])


def main():
    args = parse_args()
    enable_thinking = args.enable_thinking.strip().lower() in ("1", "true", "yes", "y")
    os.makedirs(args.out, exist_ok=True)

    rmod = load_reward_module(args.reward_fn_path)
    build_judge_prompt = rmod._build_judge_prompt
    clean_answer_text = rmod._clean_answer_text
    parse_judge_fields = rmod._parse_judge_fields

    # ---------- 读数据 + 过滤 + 抽字段 ----------
    rows = []
    for p in args.input:
        rows.extend(read_table(p))
    items = []
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
        raise SystemExit(f"没找到 data_source=={args.data_source} 的样本")
    print(f"[data] data_source={args.data_source} 命中 {len(items)} 条"
          + (f"；⚠️ {empty_q} 条没抽到 question" if empty_q else ""))

    judge_prompts = [
        build_judge_prompt(question=it["question"].strip(),
                           ground_truth=it["ground_truth"], model_answer=it["model_answer"])
        for it in items
    ]

    # ---------- 组装 payload（默认=训练请求；额外键仅非中性值才加）----------
    def make_payload(jp: str) -> dict:
        pl = {
            "model": args.model_name,
            "messages": [{"role": "user", "content": jp}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if args.top_k != -1:
            pl["top_k"] = args.top_k
        if args.min_p > 0:
            pl["min_p"] = args.min_p
        if args.repetition_penalty != 1.0:
            pl["repetition_penalty"] = args.repetition_penalty
        if args.presence_penalty != 0.0:
            pl["presence_penalty"] = args.presence_penalty
        if args.frequency_penalty != 0.0:
            pl["frequency_penalty"] = args.frequency_penalty
        return pl

    payloads = [make_payload(jp) for jp in judge_prompts]
    # 打印一条样例 payload 的键，方便确认发了什么
    print(f"[payload keys] {sorted(payloads[0].keys())}")
    print(f"[url] {args.base_url.rstrip('/')}/v1/chat/completions  model={args.model_name}")

    # ---------- 并发请求（保序）----------
    results = asyncio.run(gather_all(args, payloads))

    # ---------- 解析 + 落盘 + 统计 ----------
    raw_path = os.path.join(args.out, "judge_outputs.jsonl")
    n = len(results)
    n_err = n_trunc = n_parse_fail = n_no_close = n_heavy = 0
    comp_tokens, rep_rates = [], []
    first_err = ""

    with open(raw_path, "w", encoding="utf-8") as fout:
        for idx, r in enumerate(results):
            rec = {"judge_prompt": judge_prompts[idx]}
            if r is None or "error" in r:
                n_err += 1
                err = (r or {}).get("error", "none")
                if not first_err:
                    first_err = err
                rec.update({"error": err})
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            out = r["data"]
            usage = out.get("usage", {}) if isinstance(out, dict) else {}
            ctok = int(usage.get("completion_tokens", 0) or 0)
            choices = out.get("choices", []) if isinstance(out, dict) else []
            ch0 = choices[0] if choices and isinstance(choices[0], dict) else {}
            text = str((ch0.get("message", {}) or {}).get("content", "") or "")
            ftype = _finish_type(ch0)

            comp_tokens.append(ctok)
            is_trunc = (ftype == "length")
            no_close = ("</think>" not in text)
            rep10 = _rep_ngram_rate(text, n=10)
            rep_rates.append(rep10)
            _, _, _, parsed_ok, clean = parse_judge_fields(text)

            n_trunc += int(is_trunc)
            n_parse_fail += int(not parsed_ok)
            n_no_close += int(no_close)
            n_heavy += int(rep10 > 0.5)

            rec.update({
                "finish_reason": ftype,
                "completion_tokens": ctok,
                "truncated": is_trunc,
                "no_think_close": no_close,
                "rep10_rate": round(rep10, 4),
                "parsed_ok": parsed_ok,
                "judge_resp": text,
                "clean": clean,
            })
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_ok = n - n_err
    arr = np.array(comp_tokens, dtype=np.float64) if comp_tokens else np.array([0.0])
    rep_arr = np.array(rep_rates, dtype=np.float64) if rep_rates else np.array([0.0])
    denom = max(1, n_ok)
    summary = {
        "base_url": args.base_url, "model_name": args.model_name,
        "data_source": args.data_source, "n_samples": n, "n_ok": n_ok, "n_error": n_err,
        "enable_thinking": enable_thinking,
        "sampling": {
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens, "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty, "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
        },
        "error_rate": round(n_err / n, 4),
        "first_error": first_err,   # 某惩罚键不支持时这里会显示 HTTP 400 报错原文
        "truncation_rate(finish=length)": round(n_trunc / denom, 4),
        "no_think_close_rate": round(n_no_close / denom, 4),
        "json_parse_fail_rate": round(n_parse_fail / denom, 4),
        "heavy_repeat_rate(rep10>0.5)": round(n_heavy / denom, 4),
        "rep10_rate_mean": round(float(rep_arr.mean()), 4),
        "completion_tokens": {
            "mean": round(float(arr.mean()), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "max": int(arr.max()),
            "pct_at_max_tokens": round(float((arr >= args.max_new_tokens - 1).mean()), 4),
        },
        "finish_reason_counts": dict(Counter(
            (json.loads(l).get("finish_reason", "ERROR")) for l in open(raw_path, encoding="utf-8"))),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==================== GenRM 截断率诊断（server）====================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if n_err:
        print(f"\n⚠️ 有 {n_err} 条请求失败。若 first_error 是 HTTP 400，"
              f"多半是某个惩罚键 server 不支持，看报错原文。")
    print(f"[save] {raw_path}\n[save] {os.path.join(args.out, 'summary.json')}")
    print("==================================================================")


if __name__ == "__main__":
    main()
