#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准（无偏）Pass@K 离线评测脚本 —— SGLang 推理版（blzk 规则任务）。

和 verl 自带的 eval_passk_blzk.sh 的区别：
  - verl 的 best@k 是「对 K 条做有放回 bootstrap 取 max」的**插值估计**，k 接近 n 时偏低；
  - 本脚本走 **HumanEval(Chen et al. 2021) 无偏估计**：每题采 n 条（n 可 > 你关心的 K），
    数对 c 条，pass@k = 1 - C(n-c, k) / C(n, k)，再对题平均。和论文口径一致。

整体流程：
  1) 读 verl 训练用的 *_verl.parquet（字段：prompt / reward_model.ground_truth / data_source / extra_info）；
  2) 用 tokenizer.apply_chat_template 还原成和训练**完全相同**的输入文本；
  3) 用 SGLang 离线引擎，每题采 n 条；
  4) 用项目自带的规则奖励函数 compute_score_blzk_rule 给每条打 0/1；
  5) 按 HumanEval 无偏公式算 pass@k（k 列表可配），按 data_source 汇总打印；
  6) 把每题 (n, c) 和原始生成落盘，之后改 k 不用重新推理。

⚠️ 模型路径必须是 **HF 格式**目录：
  - base 模型：直接给 HF 路径；
  - 训练 ckpt：优先看 global_step_N/actor/ 下有没有 verl 自动导出的 **huggingface/** 子目录
    （含 model.safetensors-*-of-* + config.json + tokenizer.*）。有就直接：
        --model /.../global_step_100/actor/huggingface
  - 只有 dist_ckpt/、没有 huggingface/ 时，SGLang 读不了 dist_checkpoint，需先 merge：
        python -m verl.model_merger merge \
            --backend megatron \
            --local_dir /.../global_step_100/actor \
            --target_dir /.../hf_export/maxrl_step100

⚠️ 公平对比：base / GRPO / MaxRL 三个模型务必用**同一** n / temperature / top_p / top_k /
    max_new_tokens / 同一验证集 / 同一 chat 模板（enable_thinking 一致）。

用法示例（27B 单卡放得下 → dp=8/tp=1 吞吐最高；放不下再用 tp）：
  python passk_sglang_blzk.py \
      --model /path/to/global_step_137/actor/huggingface --model_id maxrl_step137 \
      --val_path /train21/.../dataset/blzk/blzk_val_fast_verl.parquet \
      --n 64 --k_list 1,2,4,8,16,32,64 \
      --temperature 0.7 --top_p 1.0 \
      --max_new_tokens 16384 --tp 1 --dp 8 \
      --out_dir /train21/.../passk_unbiased/maxrl_step137

一般不用单独调它——用 run_passk_compare.sh 一键把 base/grpo/maxrl/dapo 顺序测完并画图。
"""

import argparse
import importlib.util
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def load_reward_fn(path: str, name: str):
    """从任意 .py 文件动态加载奖励函数。"""
    spec = importlib.util.spec_from_file_location("_blzk_reward_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def load_done_generations(path: str) -> list[dict]:
    """读已完成的 generations.jsonl，返回有效记录列表（用于断点续跑）。
    遇到末尾损坏行（上次崩溃写一半）即停止解析，其后内容丢弃、后续重新生成。
    """
    if not os.path.exists(path):
        return []
    good = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                good.append(json.loads(line))
            except Exception:
                break  # 末尾半行，停止
    return good


def pass_at_k(n: int, c: int, k: int) -> float:
    """HumanEval 无偏 Pass@K：1 - C(n-c, k)/C(n, k)。

    n: 该题总采样数；c: 答对数；k: 目标 k（要求 k <= n）。
    数值稳定写法（避免大组合数溢出）：1 - prod_{i=n-c+1..n} (1 - k/i)。
    """
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def majority_vote_correct(preds: list[str], scores: list[float]) -> float:
    """全样本多数投票准确率（maj@n，单点参考值）：
    对 pred 计票，取票最多的 pred，看它对应的样本是否判对（score==1）。
    """
    if not preds:
        return float("nan")
    tally = defaultdict(int)
    for p in preds:
        tally[p] += 1
    winner = max(tally.items(), key=lambda kv: kv[1])[0]
    # winner 这个 pred 是否正确：找一条 pred==winner 的样本看其 score
    for p, s in zip(preds, scores):
        if p == winner:
            return 1.0 if s >= 0.5 else 0.0
    return 0.0


def normalize_label(x) -> str:
    """规范化为 '合格'/'不合格'；其它（含解析失败的空串）归为 '其它'。"""
    s = str(x or "").strip().replace('"', "").replace("“", "").replace("”", "")
    if s == "合格":
        return "合格"
    if s == "不合格":
        return "不合格"
    return "其它"


def majority_label(preds) -> str:
    """N 条预测的多数投票决策（maj@N）；只在有效标签里投票，全无效则 '其它'。"""
    valid = [normalize_label(p) for p in preds]
    valid = [p for p in valid if p in ("合格", "不合格")]
    if not valid:
        return "其它"
    return Counter(valid).most_common(1)[0][0]


def binary_prf(items, qi2gt, positive: str = "不合格") -> dict:
    """每题用 maj@N 决策 vs 真标签，建混淆矩阵，算 positive 类的 precision/recall/F1 + 整体 accuracy。

    - 决策：N 条预测的多数投票（majority_label）。
    - 无效预测('其它')视为'非 positive'（= 没把它判成 positive，对召回是漏检 FN，不会虚增 FP）。
    - 真标签非 合格/不合格 的题跳过（invalid_gt）。
    """
    tp = fp = fn = tn = skipped = 0
    for it in items:
        true = normalize_label(qi2gt.get(it["qi"]))
        if true not in ("合格", "不合格"):
            skipped += 1
            continue
        pred_pos = (majority_label(it["preds"]) == positive)
        true_pos = (true == positive)
        if true_pos and pred_pos:
            tp += 1
        elif (not true_pos) and pred_pos:
            fp += 1
        elif true_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    n_used = tp + fp + fn + tn
    acc = (tp + tn) / n_used if n_used else 0.0
    return {
        "positive": positive,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "skipped(invalid_gt)": skipped,
        "accuracy(maj@n)": acc,
        f"precision({positive})": prec,
        f"recall({positive})": rec,
        f"f1({positive})": f1,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="无偏 Pass@K（SGLang 离线推理，blzk 规则任务）")
    ap.add_argument("--model", required=True, help="HF 格式模型目录（有 huggingface/ 直接用，否则先 merge）")
    ap.add_argument("--model_id", type=str, default="", help="模型标识，写进 summary 用于画图图例")
    ap.add_argument("--val_path", required=True, help="*_verl.parquet 验证集")
    ap.add_argument("--n", type=int, default=100, help="每题采样条数（建议 > 你关心的最大 k）")
    ap.add_argument("--k_list", type=str, default="1,2,4,8,16,32,64", help="逗号分隔的 k 列表")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=16384, help="对齐训练 max_response_length")
    ap.add_argument("--enable_thinking", type=str, default="true",
                    help="true/false：和训练 chat 模板保持一致（训练默认开 thinking）")
    ap.add_argument("--max_prompts", type=int, default=-1, help="只取前 N 题调试，-1 用全部")
    # 27B 单卡放得下 → tp=1/dp=8 吞吐最高；放不下再调 tp（如 tp=8/dp=1）
    ap.add_argument("--tp", type=int, default=1, help="SGLang tensor parallel size")
    ap.add_argument("--dp", type=int, default=8, help="SGLang data parallel size（多副本提吞吐）")
    ap.add_argument("--gen_batch_size", type=int, default=512, help="每批送进引擎的请求数（仅为进度/显存）")
    ap.add_argument("--mem_fraction_static", type=float, default=0.85)
    # GDN/SSM 内核后端：务必 triton（flashinfer 对 Qwen3.5 GDN 有 mbarrier 死锁，
    # 见 notes/vllm-gdn-hang-and-triton-backend.md）。等价于 server 端 --mamba-backend triton。
    # 留空("")则不传该 kwarg（某些 sglang 版本可能不认这个参数）。
    ap.add_argument("--mamba_backend", type=str, default="triton")
    # 自定义 tokenizer（Qwen3.5）必须 trust_remote_code，否则 SGLang 内部 detokenizer
    # 会报 "Tokenizer class ... does not exist / Failed to load the tokenizer"。
    ap.add_argument("--trust_remote_code", type=str, default="true", help="true/false")
    # 可选：单独指定 tokenizer 目录。verl 导出的 ckpt tokenizer 若加载失败，
    # 可指向 base 模型路径借用（RL 不改 tokenizer，等价且更稳）。
    ap.add_argument("--tokenizer_path", type=str, default="")
    # 奖励函数（默认与训练同一份）
    ap.add_argument("--reward_fn_path", type=str,
                    default="/train21/medcog/permanent/jycai6/jmli27/reward/reward_fn_blzk_rule.py")
    ap.add_argument("--reward_fn_name", type=str, default="compute_score_blzk_rule")
    ap.add_argument("--out_dir", required=True, help="输出目录（落盘生成结果 + 指标）")
    args = ap.parse_args()

    k_list = sorted({int(x) for x in args.k_list.split(",") if x.strip()})
    assert max(k_list) <= args.n, f"k_list 最大值 {max(k_list)} 不能超过 n={args.n}"
    enable_thinking = args.enable_thinking.strip().lower() in ("1", "true", "yes", "y")
    trust_remote_code = args.trust_remote_code.strip().lower() in ("1", "true", "yes", "y")
    tok_src = args.tokenizer_path or args.model  # tokenizer 来源（默认与模型同目录）
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1. 读数据 ----------
    df = pd.read_parquet(args.val_path)
    if args.max_prompts > 0:
        df = df.head(args.max_prompts)
    rows = df.to_dict("records")
    print(f"[data] {args.val_path} -> {len(rows)} 题；字段={list(df.columns)}")

    reward_fn = load_reward_fn(args.reward_fn_path, args.reward_fn_name)

    # ---------- 2. 还原 chat 模板（和训练一致）----------
    # 优先 AutoProcessor（Qwen3.5 多模态，和同事 infer_sglang.py 一致）；
    # verl 导出的 huggingface/ 可能缺视觉预处理文件 → 回退 AutoTokenizer（纯文本够用）。
    try:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(tok_src, trust_remote_code=trust_remote_code)
    except Exception as e:
        print(f"[warn] AutoProcessor 加载失败（{e}），回退 AutoTokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=trust_remote_code)

    def render(messages) -> str:
        msgs = [dict(m) for m in list(messages)]
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            # 老模板不认 enable_thinking 参数
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # 每题展开成 n 条（顺序：题0×n, 题1×n, ...）。meta 记录每条的题元信息；
    # prompt 文本每题只渲染一次（rendered 按 qi 索引），避免重复字符串占内存。
    meta = []  # 每条生成对应的题元信息
    for qi, row in enumerate(rows):
        gt = row["reward_model"]["ground_truth"]
        ds = row.get("data_source", "blzk")
        ei = row.get("extra_info", {})
        ei = dict(ei) if isinstance(ei, dict) else {}
        for _ in range(args.n):
            meta.append({"qi": qi, "data_source": ds, "ground_truth": gt, "extra_info": ei})
    total = len(meta)
    print(f"[gen] 总生成请求 = {len(rows)} 题 × n={args.n} = {total} 条")

    raw_path = os.path.join(args.out_dir, "generations.jsonl")

    # ---------- 3. 断点续：读已完成的 generations.jsonl，从断点继续 ----------
    done = load_done_generations(raw_path)
    start_index = len(done)
    if start_index > total:
        raise SystemExit(
            f"已有 {start_index} 条 > 应有 {total} 条：{args.out_dir} 里疑似旧配置(数据/n 不同)结果，请清空后重跑")
    if start_index > 0:
        # 用首尾两条的 qi 校验对齐，防止换了数据集/n 还接着旧文件续写
        if done[0].get("qi") != meta[0]["qi"] or done[start_index - 1].get("qi") != meta[start_index - 1]["qi"]:
            raise SystemExit(f"已有结果与当前数据/n 不一致（qi 对不上）：请清空 {args.out_dir} 后重跑")
        print(f"[resume] 已完成 {start_index}/{total} 条，从断点继续")
    # 重写一遍干净内容（丢掉可能存在的末尾半行），之后以 append 续写
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in done:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- 4. 生成→打分→增量落盘（每批 flush，可随时中断续跑）----------
    if start_index < total:
        rendered = [render(row["prompt"]) for row in rows]  # 每题 prompt 只渲染一次

        # 引擎参数对齐同事 infer_sglang.py 验证过的最小集，避免某些 sglang 版本不认的 kwarg。
        import sglang as sgl

        engine_kwargs = dict(
            model_path=args.model,
            tp_size=args.tp,
            dp_size=args.dp,
            mem_fraction_static=args.mem_fraction_static,
            trust_remote_code=trust_remote_code,  # 自定义 tokenizer 必须，否则 detokenizer 加载失败
        )
        # GDN→triton，规避 flashinfer mbarrier 死锁（CLI --mamba-backend 的 Engine 等价 kwarg）
        if args.mamba_backend:
            engine_kwargs["mamba_backend"] = args.mamba_backend
        # 可选：单独的 tokenizer 目录（ckpt tokenizer 坏时借 base 的）
        if args.tokenizer_path:
            engine_kwargs["tokenizer_path"] = args.tokenizer_path
        engine = sgl.Engine(**engine_kwargs)
        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        }

        bs = max(1, args.gen_batch_size)
        fout = open(raw_path, "a", encoding="utf-8")
        try:
            for start in range(start_index, total, bs):
                batch_meta = meta[start:start + bs]
                batch_prompts = [rendered[m["qi"]] for m in batch_meta]
                # 关键字 prompt= / sampling_params=，与同事脚本一致
                outputs = engine.generate(prompt=batch_prompts, sampling_params=sampling_params)
                for m, o in zip(batch_meta, outputs):
                    gen = o["text"]
                    res = reward_fn(
                        data_source=m["data_source"],
                        solution_str=gen,
                        ground_truth=m["ground_truth"],
                        extra_info=m["extra_info"],
                    )
                    if isinstance(res, dict):
                        score = float(res.get("score", 0.0))
                        pred = str(res.get("pred", ""))
                    else:
                        score = float(res)
                        pred = ""
                    fout.write(json.dumps(
                        {"qi": m["qi"], "data_source": m["data_source"], "score": score,
                         "pred": pred, "gen": gen}, ensure_ascii=False) + "\n")
                fout.flush()  # 每批落盘 → 有中间产出、崩了能续
                print(f"[gen] {min(start + bs, total)}/{total} -> {raw_path}", flush=True)
        finally:
            fout.close()
            engine.shutdown()
    else:
        print(f"[resume] generations.jsonl 已完整（{total} 条），跳过推理，直接汇总")
    print(f"[save] 原始生成 -> {raw_path}")

    # ---------- 5. 读回全部生成，按题聚合 ----------
    per_q = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qi = r["qi"]
            d = per_q.setdefault(qi, {"ds": r.get("data_source", "blzk"), "scores": [], "preds": []})
            d["scores"].append(float(r.get("score", 0.0)))
            d["preds"].append(str(r.get("pred", "")))

    # 真标签映射：qi -> ground_truth（用于 precision/recall 的混淆矩阵）
    qi2gt = {i: rows[i]["reward_model"]["ground_truth"] for i in range(len(rows))}

    # ---------- 6. 算无偏 pass@k，按 data_source 汇总 ----------
    # ds -> list of per-prompt (qi, n, c, scores, preds)
    by_ds = defaultdict(list)
    for qi, d in per_q.items():
        scores = d["scores"]
        n_q = len(scores)
        c_q = int(sum(1 for s in scores if s >= 0.5))
        by_ds[d["ds"]].append({"qi": qi, "n": n_q, "c": c_q, "scores": scores, "preds": d["preds"]})

    per_prompt_path = os.path.join(args.out_dir, "per_prompt.jsonl")
    with open(per_prompt_path, "w", encoding="utf-8") as fpp:
        for ds, items in by_ds.items():
            for qi_local, it in enumerate(items):
                fpp.write(json.dumps(
                    {"data_source": ds, "n": it["n"], "c": it["c"]}, ensure_ascii=False) + "\n")

    summary = {}
    print("\n==================== 无偏 Pass@K 结果 ====================")
    for ds, items in by_ds.items():
        n_prompts = len(items)
        metrics = {"n_prompts": n_prompts, "samples_per_prompt": args.n}
        # pass@1 = 平均通过率（= mean c/n）
        metrics["pass@1(mean_acc)"] = float(np.mean([it["c"] / it["n"] for it in items]))
        # 各 k 的无偏 pass@k
        for k in k_list:
            vals = [pass_at_k(it["n"], it["c"], k) for it in items]
            metrics[f"pass@{k}"] = float(np.nanmean(vals))
        # solve@n：至少对一次的题占比（= 经验 pass@n 上限）
        metrics["solve_rate(>=1 correct)"] = float(np.mean([1.0 if it["c"] > 0 else 0.0 for it in items]))
        # maj@n：全样本多数投票（单点参考）
        metrics["maj@n"] = float(np.nanmean(
            [majority_vote_correct(it["preds"], it["scores"]) for it in items]))

        # format_valid_rate：每题 N 条里能解析出 合格/不合格 的比例，再对题平均。
        # 它解释了下面两个 maj 口径的差距：base 格式合规差 → 含无效投票的 maj@n 被拖低。
        metrics["format_valid_rate"] = float(np.mean([
            sum(1 for p in it["preds"] if normalize_label(p) in ("合格", "不合格")) / it["n"]
            for it in items]))

        # 分类指标：以 maj@N 决策建混淆矩阵，算「不合格」的 precision/recall/F1（+整体 acc）
        prf = binary_prf(items, qi2gt, positive="不合格")
        metrics["classification"] = prf

        summary[ds] = metrics
        print(f"\n[data_source = {ds}]  题数={n_prompts}  每题采样={args.n}")
        print(f"  pass@1 (平均准确率)            = {metrics['pass@1(mean_acc)']:.4f}")
        for k in k_list:
            print(f"  pass@{k:<3d} (无偏)                = {metrics[f'pass@{k}']:.4f}")
        print(f"  solve_rate (>=1 correct)       = {metrics['solve_rate(>=1 correct)']:.4f}")
        print(f"  format_valid_rate              = {metrics['format_valid_rate']:.4f}")
        print(f"  maj@n (含无效输出投票)         = {metrics['maj@n']:.4f}")
        print(f"  --- 分类（maj@N 决策, 仅有效标签投票, 正类=不合格, 跳过无效真标签 {prf['skipped(invalid_gt)']} 题）---")
        print(f"  accuracy(maj@n, 仅有效投票)    = {prf['accuracy(maj@n)']:.4f}")
        print(f"  precision(不合格)              = {prf['precision(不合格)']:.4f}")
        print(f"  recall(不合格)                 = {prf['recall(不合格)']:.4f}")
        print(f"  f1(不合格)                     = {prf['f1(不合格)']:.4f}")
        print(f"  混淆: TP={prf['tp']} FP={prf['fp']} FN={prf['fn']} TN={prf['tn']}")

    summary_path = os.path.join(args.out_dir, "passk_summary.json")
    model_id = args.model_id or os.path.basename(os.path.normpath(args.out_dir))
    with open(summary_path, "w", encoding="utf-8") as fs:
        json.dump({"model_id": model_id, "args": vars(args), "summary": summary},
                  fs, ensure_ascii=False, indent=2)
    print(f"\n[save] 指标汇总 -> {summary_path}")
    print("=========================================================")


if __name__ == "__main__":
    main()
