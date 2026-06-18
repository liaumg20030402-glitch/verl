"""
合并并随机打乱多个 JSONL 文件。

适用场景：文件后缀可能是 .json 或 .jsonl，但内容都是「每行一个 JSON 对象」
（即 JSONL 格式）。脚本逐行读取、合并、随机打乱后写出。

用法：
    # 直接改下面的 INPUT_PATHS / OUTPUT_PATH 后运行
    python merge_jsonl.py

    # 或用命令行覆盖（最后一个为输出，前面的都是输入；至少 2 个输入）
    python merge_jsonl.py a.json b.json merged.jsonl --seed 42
"""

import argparse
import json
import random


# -------------------- 固定配置（按需改这里，或用命令行覆盖） --------------------
INPUT_PATHS = [
    "/train21/medcog/permanent/jycai6/jmli27/dataset/a.json",
    "/train21/medcog/permanent/jycai6/jmli27/dataset/b.json",
]
OUTPUT_PATH = "/train21/medcog/permanent/jycai6/jmli27/dataset/merged.jsonl"
SEED = 42


def read_jsonl(path: str) -> list[dict]:
    """逐行读取 JSONL（忽略空行）。后缀无所谓，只按行解析 JSON 对象。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON：{e}") from e
    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    """写出 JSONL（每行一个对象，UTF-8，不转义中文）。"""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def merge_and_shuffle(input_paths: list[str], output_path: str, seed: int) -> None:
    """读取全部输入文件，合并、随机打乱后写出。"""
    merged: list[dict] = []
    per_file_counts = []
    for path in input_paths:
        records = read_jsonl(path)
        per_file_counts.append((path, len(records)))
        merged.extend(records)

    random.seed(seed)
    random.shuffle(merged)

    write_jsonl(output_path, merged)

    print("[完成] 合并 + 随机打乱")
    for path, n in per_file_counts:
        print(f"  输入: {path}  ({n} 条)")
    print(f"  输出: {output_path}  (共 {len(merged)} 条, seed={seed})")


def main() -> None:
    parser = argparse.ArgumentParser(description="合并并随机打乱多个 JSONL 文件")
    parser.add_argument(
        "paths",
        nargs="*",
        help="至少 2 个输入文件 + 1 个输出文件（最后一个为输出）；省略则用脚本顶部配置。",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"随机种子（默认 {SEED}）")
    args = parser.parse_args()

    if args.paths:
        if len(args.paths) < 3:
            parser.error("命令行模式需要至少 2 个输入 + 1 个输出，共 ≥3 个路径。")
        input_paths = args.paths[:-1]
        output_path = args.paths[-1]
    else:
        input_paths = INPUT_PATHS
        output_path = OUTPUT_PATH

    merge_and_shuffle(input_paths, output_path, args.seed)


if __name__ == "__main__":
    main()
