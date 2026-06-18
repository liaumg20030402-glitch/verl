#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 id 去重（重复 id 只保留第一条）。支持 .parquet / .json / .jsonl 读写。

id 位置自动识别（--id_field auto，默认）：
  - 转换后的 verl 格式：id 在每行 extra_info["id"] 里；
  - 原始数据（convert 之前）：id 在顶层 "id" 列。
也可用 --id_field <列名> 强制指定顶层列，或 --id_field extra_info 强制取 extra_info.id。

输出格式默认跟随输入后缀（parquet→parquet, json→json, jsonl→jsonl），可用 --output 指定。

空 id 处理（--empty_policy）：
  keep(默认)  空 id 各行视为唯一、全保留（不会误并）
  by_prompt   空 id 行按整行内容哈希去重
  drop        直接丢弃空 id 行

用法：
  python dedup_parquet_by_id.py --input /.../hard_val_verl.parquet
  python dedup_parquet_by_id.py --input /.../raw_hard.jsonl              # 原始 jsonl，按顶层 id
  python dedup_parquet_by_id.py --input in.json --output out.json
  python dedup_parquet_by_id.py --input in.parquet --empty_policy by_prompt
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


# --------------------------- 读写（多格式） --------------------------- #
def read_any(path: str) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    if p.endswith(".json"):
        # 探测首个非空白字符：'[' 标准数组，'{' 当 JSONL
        with open(p, "r", encoding="utf-8") as f:
            head = ""
            while True:
                ch = f.read(1)
                if ch == "" or not ch.isspace():
                    head = ch
                    break
        return pd.read_json(p) if head == "[" else pd.read_json(p, lines=True)
    raise SystemExit(f"不支持的输入格式（需 .parquet/.json/.jsonl）：{p}")


def write_any(df: pd.DataFrame, path: str) -> None:
    p = str(path)
    if p.endswith(".parquet"):
        df.to_parquet(p, index=False)
    elif p.endswith(".jsonl"):
        df.to_json(p, orient="records", lines=True, force_ascii=False)
    elif p.endswith(".json"):
        df.to_json(p, orient="records", force_ascii=False, indent=2)
    else:
        raise SystemExit(f"不支持的输出格式（需 .parquet/.json/.jsonl）：{p}")


# --------------------------- id / 内容 提取 --------------------------- #
def id_from_extra(extra_info) -> str:
    if isinstance(extra_info, dict):
        return str(extra_info.get("id", "") or "").strip()
    return ""


def make_id_series(df: pd.DataFrame, id_field: str) -> pd.Series:
    if id_field == "auto":
        if "extra_info" in df.columns:
            return df["extra_info"].apply(id_from_extra)
        if "id" in df.columns:
            return df["id"].astype(str).str.strip()
        raise SystemExit(f"自动识别失败：既无 extra_info 也无 id 列；字段={list(df.columns)}")
    if id_field == "extra_info":
        return df["extra_info"].apply(id_from_extra)
    # 否则当成顶层列名
    if id_field not in df.columns:
        raise SystemExit(f"指定的 id 列 '{id_field}' 不存在；字段={list(df.columns)}")
    return df[id_field].astype(str).str.strip()


def row_hash(row: dict) -> str:
    try:
        s = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        s = str(row)
    return "h:" + hashlib.md5(s.encode("utf-8")).hexdigest()


# --------------------------------- 主 --------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="按 id 去重（parquet/json/jsonl）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="", help="默认输入同目录加 _dedup，后缀同输入")
    ap.add_argument("--id_field", default="auto", help="auto / extra_info / <顶层列名>")
    ap.add_argument("--empty_policy", choices=["keep", "by_prompt", "drop"], default="keep")
    args = ap.parse_args()

    in_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_name(f"{in_path.stem}_dedup{in_path.suffix}")

    df = read_any(in_path)
    n0 = len(df)
    ids = make_id_series(df, args.id_field)
    empty_mask = ids.eq("")
    n_empty = int(empty_mask.sum())

    keys = ids.copy()
    if args.empty_policy == "keep":
        keys = keys.mask(empty_mask, pd.Series([f"__uniq__{i}" for i in range(len(df))], index=df.index))
    elif args.empty_policy == "by_prompt":
        rh = df.apply(lambda r: row_hash(r.to_dict()), axis=1)
        keys = keys.mask(empty_mask, rh)
    elif args.empty_policy == "drop":
        df = df[~empty_mask]
        keys = ids[~empty_mask]

    keep_mask = ~keys.duplicated(keep="first")
    df_out = df[keep_mask].reset_index(drop=True)
    write_any(df_out, out_path)

    n1 = len(df_out)
    n_unique_id = int(ids[~empty_mask].nunique())
    print(f"[dedup] 输入: {in_path}  共 {n0} 行（空 id {n_empty} 行）")
    print(f"[dedup] 非空 id 唯一数 = {n_unique_id}")
    print(f"[dedup] empty_policy={args.empty_policy} → 输出 {n1} 行（删除 {n0 - n1} 行）")
    print(f"[dedup] 输出: {out_path}")


if __name__ == "__main__":
    main()
