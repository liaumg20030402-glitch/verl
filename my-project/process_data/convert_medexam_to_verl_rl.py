"""
医考题目（medexam）数据转换脚本
将讯飞星火格式的原始数据转换为 verl 训练所需的 parquet 格式。

用法：
    python convert_medexam_to_verl_rl.py [--train 输入] [--val 输入] \
        [--train-out 输出] [--val-out 输出] [--workers N] [--verify]
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ──────────────────────────── 解析 ────────────────────────────

def _clean_spark_text(text: str) -> str:
    """清洗原始 Spark 格式转义字符，为后续正则切分做准备。"""
    s = str(text or "")
    s = s.replace("<ret>", "\n")
    s = s.replace("\\r", "")
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", " ")
    return s.strip()


def parse_spark_input(raw_input: str) -> tuple[str, str]:
    """将讯飞星火格式的 input 字符串解析为 (system_content, user_content)。

    原始数据格式固定为：<System>内容<end><User>内容<end><Bot>
    直接用 <end> 作为段落边界提取，无需多重回退逻辑。

    【关于 <unused6>/<unused7> → <think></think> 的替换】
    系统提示中含有讯飞格式的思考模式指令，例如：
      "请以 <unused6> 开头，在结尾处以 <unused7> 标注结束"
    此处将其替换为标准思考标签，使训练数据与 Qwen3 等模型格式对齐。
    """
    s = _clean_spark_text(raw_input)

    # 提取 System 内容：<System>...<end>
    system_m = re.search(r"<System>(.*?)<end>", s, re.DOTALL | re.IGNORECASE)
    system_content = system_m.group(1).strip() if system_m else ""
    # 将星火思考 token 替换为标准思考标签
    system_content = system_content.replace("<unused6>", "<think>").replace("<unused7>", "</think>")

    # 提取 User 内容：<User>...<end>
    user_m = re.search(r"<User>(.*?)<end>", s, re.DOTALL | re.IGNORECASE)
    user_content = user_m.group(1).strip() if user_m else ""

    return system_content, user_content


# ──────────────────────────── 规范化 ────────────────────────────

def _normalize_category(category: str) -> str:
    """将原始 category 字段规范化为统一的内部标识。"""
    c = str(category or "").strip().lower()
    if c in {"med-exam-multi", "med_exam_multi"}:
        return "med_exam_multi"
    if c in {"med-exam", "med_exam"}:
        return "med_exam_single"
    return "med_exam_unknown"


def _normalize_target(target: str) -> str:
    """将答案选项规范化为"去重 + 字母排序"的大写字符串。
    确保 'CBA' 与 'ABC' 归一化后完全一致，支持超出 E 的长选项题。
    """
    t = str(target or "").strip().upper().replace(" ", "").replace(",", "")
    t = re.sub(r"[^A-Z]", "", t)
    return "".join(sorted(set(t)))


# ──────────────────────────── 转换 ────────────────────────────

def convert_to_verl_row(row: dict) -> dict | None:
    """单条数据转换为 verl 标准格式。
    target 为空时返回 None，由调用方统计跳过数量。
    """
    target = _normalize_target(row.get("target", ""))
    if not target:
        return None

    raw_input = row.get("input", "")
    system_content, user_content = parse_spark_input(raw_input)

    raw_category = str(row.get("category", "")).strip()
    norm_category = _normalize_category(raw_category)

    prompt = []
    if system_content:
        prompt.append({"role": "system", "content": system_content})
    prompt.append({"role": "user", "content": user_content})

    return {
        "data_source": norm_category,
        "prompt": prompt,
        "ability": "med_exam",
        "reward_model": {
            "style": "model",
            "ground_truth": target,
        },
        "extra_info": {
            "id":            str(row.get("id", "")),
            "category":      raw_category,
            "category_norm": norm_category,
            "hardness":      str(row.get("hardness", "")),
            "question":      user_content,
            "target":        target,
            "raw_target":    str(row.get("target", "")),
        },
    }


# ──────────────────────────── I/O ────────────────────────────

def _read_df(input_path: str) -> pd.DataFrame:
    """根据文件扩展名自动选择读取方式，支持 parquet/json/jsonl/csv。"""
    if input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    if input_path.endswith(".json"):
        return pd.read_json(input_path)
    if input_path.endswith(".jsonl"):
        return pd.read_json(input_path, lines=True)
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    raise ValueError(f"不支持的文件格式: {input_path}")


def process_dataset(input_path: str, output_path: str, num_workers: int = 4) -> None:
    """读取原始数据、并行转换、保存为 verl 所需 parquet 文件。

    参数：
        input_path  : 输入文件路径（parquet/json/jsonl/csv）
        output_path : 输出 parquet 文件路径
        num_workers : 并行进程数；设为 1 时退化为单进程（便于调试）
    """
    df = _read_df(input_path)
    rows = df.to_dict("records")
    total = len(rows)
    filename = Path(input_path).name

    records: list[dict] = []
    skipped = 0

    if num_workers > 1:
        # 多进程并行转换，适合数据集较大的场景
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(convert_to_verl_row, rows, chunksize=200),
                    total=total,
                    desc=f"转换 {filename}",
                    unit="条",
                )
            )
    else:
        # 单进程串行，方便调试或小数据集
        results = [
            convert_to_verl_row(row)
            for row in tqdm(rows, desc=f"转换 {filename}", unit="条")
        ]

    for item in results:
        if item is None:
            skipped += 1
        else:
            records.append(item)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path, index=False)

    # 统计各 data_source 分布
    by_source: dict[str, int] = {}
    for r in records:
        key = r["data_source"]
        by_source[key] = by_source.get(key, 0) + 1

    print(f"\n[完成] {filename} -> {Path(output_path).name}")
    print(f"  保留: {len(records)} 条  |  跳过: {skipped} 条")
    print(f"  类别分布: {by_source}")


def verify_output(parquet_path: str, n: int = 3) -> None:
    """验证转换后的 parquet 文件内容与格式是否正确。"""
    df = pd.read_parquet(parquet_path)
    print(f"\n[验证] {parquet_path}")
    print(f"  总条数: {len(df)}")
    print(f"  字段列表: {list(df.columns)}")
    for i, row in df.head(n).iterrows():
        print(f"\n  --- 第 {i + 1} 条 ---")
        print(f"  data_source : {row['data_source']}")
        print(f"  ability     : {row['ability']}")
        print(f"  ground_truth: {row['reward_model']['ground_truth']}")
        print(f"  prompt 段数 : {len(row['prompt'])}")
        print(f"  user 内容头 : {str(row['prompt'][-1]['content'])[:120]}...")


# ──────────────────────────── 入口 ────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="医考数据转换：Spark 格式 → verl parquet")
    parser.add_argument("--train",     default="/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train.parquet")
    parser.add_argument("--val",       default="/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val.parquet")
    parser.add_argument("--train-out", default="/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train_verl.parquet")
    parser.add_argument("--val-out",   default="/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val_verl.parquet")
    parser.add_argument("--workers",   type=int, default=4, help="并行进程数，1 表示单进程（便于调试）")
    parser.add_argument("--verify",    action="store_true", help="转换完成后验证输出文件")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    process_dataset(args.train,     args.train_out, num_workers=args.workers)
    process_dataset(args.val,       args.val_out,   num_workers=args.workers)

    if args.verify:
        verify_output(args.train_out)
        verify_output(args.val_out)
