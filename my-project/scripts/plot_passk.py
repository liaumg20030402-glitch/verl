#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取多个 passk_summary.json（每个模型一份），画无偏 Pass@K 对比曲线 + 打印对比表。

x 轴 = k（log2 刻度），y 轴 = pass@k；每个模型一条线，每个 data_source 一张图。
matplotlib 缺失时自动降级：只打印表格 + 写 CSV，不画图。

用法：
  python plot_passk.py --summaries /out_root/*/passk_summary.json --out /out_root/passk_compare.png
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


def _k_of(metric_key: str):
    """从 'pass@8' 取出 8；非 pass@<int> 返回 None。"""
    m = re.fullmatch(r"pass@(\d+)", metric_key)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="passk_summary.json 路径（支持 shell 通配展开，或自己传通配串）")
    ap.add_argument("--out", required=True, help="输出图片路径（多 data_source 会加后缀）")
    args = ap.parse_args()

    # 允许传未展开的通配串
    paths = []
    for p in args.summaries:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit(f"没找到任何 summary：{args.summaries}")

    # runs: model_id -> {data_source -> {k -> pass@k, "pass@1": ...}}
    runs = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        mid = obj.get("model_id") or os.path.basename(os.path.dirname(p))
        runs[mid] = obj["summary"]

    # base 排最前，其余按名字
    def order_key(mid):
        return (0 if mid.lower().startswith("base") else 1, mid)
    model_ids = sorted(runs.keys(), key=order_key)

    # 收集所有 data_source
    data_sources = sorted({ds for s in runs.values() for ds in s.keys()})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception as e:
        print(f"[warn] matplotlib 不可用（{e}），只输出表格/CSV")
        have_mpl = False

    out_base, out_ext = os.path.splitext(args.out)
    out_ext = out_ext or ".png"

    for ds in data_sources:
        # 统一 k 列表（取所有模型该 ds 下出现过的 k 的并集）
        kset = set()
        for mid in model_ids:
            metrics = runs[mid].get(ds, {})
            for key in metrics:
                kk = _k_of(key)
                if kk is not None:
                    kset.add(kk)
        ks = sorted(kset)
        if not ks:
            continue

        # -------- 打印对比表 --------
        print(f"\n==================== data_source = {ds} ====================")
        header = "model".ljust(22) + "".join([f"p@{k}".rjust(9) for k in ks])
        print(header)
        print("-" * len(header))
        csv_rows = []
        for mid in model_ids:
            metrics = runs[mid].get(ds, {})
            vals = [metrics.get(f"pass@{k}", float("nan")) for k in ks]
            line = mid.ljust(22) + "".join([f"{v:9.4f}" for v in vals])
            print(line)
            csv_rows.append([mid] + vals)

        # 写 CSV
        csv_path = f"{out_base}_{ds}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fc:
            w = csv.writer(fc)
            w.writerow(["model"] + [f"pass@{k}" for k in ks])
            w.writerows(csv_rows)
        print(f"[save] 表格 -> {csv_path}")

        # -------- 画图 --------
        if have_mpl:
            plt.figure(figsize=(7, 5))
            for mid in model_ids:
                metrics = runs[mid].get(ds, {})
                ys = [metrics.get(f"pass@{k}", float("nan")) for k in ks]
                plt.plot(ks, ys, marker="o", label=mid)
            plt.xscale("log", base=2)
            plt.xticks(ks, [str(k) for k in ks])
            plt.xlabel("k (samples)")
            plt.ylabel("Pass@k (unbiased)")
            plt.title(f"Unbiased Pass@k — {ds}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            fig_path = f"{out_base}_{ds}{out_ext}" if len(data_sources) > 1 else args.out
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[save] 图 -> {fig_path}")


if __name__ == "__main__":
    main()
