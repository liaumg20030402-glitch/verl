#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 check_genrm_truncation_sglang.py 产出的 judge_outputs.jsonl 转成 infer_sglang.py 能吃的格式。

目的：用 infer_sglang.py（同事那套离线推理）在**完全相同的 judge prompt** 上再跑一遍，
做严格 apples-to-apples 对照——确认"判分复读/截断"是任务本身导致，还是 infer_sglang 的
调用路径（engine kwargs / 渲染）与 check 有差异。

转换逻辑：
  judge_outputs.jsonl 每行的 `judge_prompt`（_build_judge_prompt 拼好的完整裁判 query）
  → infer_sglang 行：{"id": <序号>, "messages": [{"role": "user", "content": <judge_prompt>}]}

  id 用行号，方便之后把 infer_sglang 的 predictions.jsonl（sample_id）和这里的 judge_outputs
  按序号对回去比对。

用法：
  python convert_judge_outputs_to_infer_sglang.py \
      --input /train21/.../genrm_trunc_check/.../judge_outputs.jsonl \
      --output /train21/.../genrm_infer_check/genrm_judge_prompts.jsonl

之后用 infer_sglang.sh / infer_sglang.py 跑这个输出文件即可（注意采样参数和 enable_thinking
要与 check 对齐：--temperature 0.8 --top-p 1.0 --top-k -1 --max-new-tokens 8192 --thinking-mode slow）。
"""

import argparse
import json
import os


def read_jsonl(path: str):
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


def main():
    ap = argparse.ArgumentParser(description="judge_outputs.jsonl → infer_sglang 输入格式")
    ap.add_argument("--input", nargs="+", required=True,
                    help="check 脚本产出的 judge_outputs.jsonl（可多个，按给定顺序拼接）")
    ap.add_argument("--output", required=True, help="输出 jsonl（喂给 infer_sglang.py）")
    ap.add_argument("--prompt-field", default="judge_prompt",
                    help="取哪个字段作为裁判 query（默认 judge_prompt）")
    args = ap.parse_args()

    rows = []
    for p in args.input:
        rows.extend(read_jsonl(p))
    print(f"[in] 读入 {len(rows)} 行（{len(args.input)} 个文件）")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    kept = skipped = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            prompt = str(r.get(args.prompt_field, "") or "")
            if not prompt:
                skipped += 1
                continue
            rec = {
                "id": i,                                   # 行号，便于和 judge_outputs 对回去
                "messages": [{"role": "user", "content": prompt}],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[out] 写出 {kept} 行 -> {args.output}" + (f"（跳过 {skipped} 条空 prompt）" if skipped else ""))


if __name__ == "__main__":
    main()
