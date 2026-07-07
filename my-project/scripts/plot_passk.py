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


def _isnan(v) -> bool:
    return isinstance(v, float) and v != v


def _disp(s: str) -> str:
    """把 model_id 转成纯 ASCII 显示（系统无 CJK 字体时避免方块）。卡→gpu，其余非 ASCII 丢弃。"""
    s = str(s).replace("卡", "gpu")
    out = s.encode("ascii", "ignore").decode()
    return out or s


def _get_cls(metrics: dict):
    """从单模型某 data_source 的 metrics 里取分类指标：(acc, precision, recall, f1, maj@n, fmt_valid)。"""
    cls = metrics.get("classification", {}) or {}
    prec = next((v for k, v in cls.items() if k.startswith("precision")), float("nan"))
    rec = next((v for k, v in cls.items() if k.startswith("recall")), float("nan"))
    f1 = next((v for k, v in cls.items() if k.startswith("f1")), float("nan"))
    acc = cls.get("accuracy(maj@n)", float("nan"))
    majn = metrics.get("maj@n", float("nan"))
    fmt = metrics.get("format_valid_rate", float("nan"))
    return acc, prec, rec, f1, majn, fmt


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

    # base 排最前；其余按「自然排序」：方法名分组 + step 按数字升序（step50→step100→step137），
    # 而不是字符串序（否则 step100 会排在 step50 前面）。
    def _natkey(s: str):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s.lower())]

    def order_key(mid):
        return (0 if mid.lower().startswith("base") else 1, _natkey(mid))
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
                plt.plot(ks, ys, marker="o", label=_disp(mid))
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

            # 放大版：去掉 base（它把 y 轴拉太宽），y 轴自动贴合 RL 模型，差别才看得清
            rl_ids = [mid for mid in model_ids if not mid.lower().startswith("base")]
            zoom_vals = [runs[mid].get(ds, {}).get(f"pass@{k}", float("nan"))
                         for mid in rl_ids for k in ks]
            zoom_vals = [v for v in zoom_vals if not _isnan(v)]
            if len(rl_ids) >= 2 and zoom_vals:
                plt.figure(figsize=(8, 5))
                for mid in rl_ids:
                    metrics = runs[mid].get(ds, {})
                    ys = [metrics.get(f"pass@{k}", float("nan")) for k in ks]
                    plt.plot(ks, ys, marker="o", label=_disp(mid))
                lo, hi = min(zoom_vals), max(zoom_vals)
                pad = max((hi - lo) * 0.08, 0.003)
                plt.ylim(lo - pad, hi + pad)
                plt.xscale("log", base=2)
                plt.xticks(ks, [str(k) for k in ks])
                plt.xlabel("k (samples)")
                plt.ylabel("Pass@k (unbiased)")
                plt.title(f"Unbiased Pass@k (RL only, zoomed) — {ds}")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8)
                plt.tight_layout()
                zoom_fig = f"{out_base}_{ds}_zoom{out_ext}"
                plt.savefig(zoom_fig, dpi=150)
                plt.close()
                print(f"[save] 放大图 -> {zoom_fig}")

        # -------- 分类指标（不合格 precision/recall/F1）表 + 柱状图 --------
        cls_data = [(mid, *_get_cls(runs[mid].get(ds, {}))) for mid in model_ids]
        if any(not _isnan(r[2]) for r in cls_data):  # 至少一个模型有 precision 才输出
            print(f"\n--------- 分类指标（maj@N 决策, 正类=不合格） data_source = {ds} ---------")
            cols = ["acc", "precision", "recall", "f1", "maj@n", "fmt_valid"]
            hdr = "model".ljust(22) + "".join(c.rjust(11) for c in cols)
            print(hdr)
            print("-" * len(hdr))
            cls_csv = []
            for mid, acc, prec, rec, f1, majn, fmt in cls_data:
                vals = [acc, prec, rec, f1, majn, fmt]
                print(mid.ljust(22) + "".join(f"{v:11.4f}" for v in vals))
                cls_csv.append([mid] + vals)

            cls_csv_path = f"{out_base}_{ds}_clf.csv"
            with open(cls_csv_path, "w", newline="", encoding="utf-8") as fc:
                w = csv.writer(fc)
                w.writerow(["model", "accuracy", "precision(不合格)", "recall(不合格)",
                            "f1(不合格)", "maj@n", "format_valid_rate"])
                w.writerows(cls_csv)
            print(f"[save] 分类表 -> {cls_csv_path}")

            if have_mpl:
                # precision / recall / f1 各画一张折线图（y 轴按数据范围放大，差别才看得清）。
                # 图内全用 ASCII，避免无 CJK 字体时显示成方块；NG = 不合格(buhege) 正类。
                labels = [_disp(r[0]) for r in cls_data]
                xs = list(range(len(labels)))
                for col_idx, name in ((2, "precision"), (3, "recall"), (4, "f1")):
                    ys = [r[col_idx] for r in cls_data]
                    valid = [v for v in ys if not _isnan(v)]
                    if not valid:
                        continue
                    plt.figure(figsize=(max(7, 1.4 * len(labels)), 5))
                    plt.plot(xs, ys, marker="o")
                    lo, hi = min(valid), max(valid)
                    pad = max((hi - lo) * 0.15, 0.01)  # 留边距，把差别放大
                    plt.ylim(max(0.0, lo - pad), min(1.0, hi + pad))
                    plt.xticks(xs, labels, rotation=30, ha="right")
                    plt.ylabel(name)
                    plt.title(f"{name} of NG (buhege) class, maj@N — {ds}")
                    plt.grid(True, alpha=0.3)
                    for x, v in zip(xs, ys):  # 点上标数值，近距离也能读数
                        if not _isnan(v):
                            plt.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                                         xytext=(0, 5), ha="center", fontsize=8)
                    plt.tight_layout()
                    fig_p = f"{out_base}_{ds}_{name}{out_ext}"
                    plt.savefig(fig_p, dpi=150)
                    plt.close()
                    print(f"[save] {name} 图 -> {fig_p}")


if __name__ == "__main__":
    main()
