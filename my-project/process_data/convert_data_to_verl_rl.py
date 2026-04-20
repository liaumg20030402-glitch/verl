"""
数据转换脚本（medexam + blzk）
将原始数据转换为 verl 强化学习训练所需的 parquet 格式。

"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm


# -------------------- 固定路径配置（按需改这里） --------------------
WORKERS = 4
VERIFY_AFTER_CONVERT = True

TASK_CONFIGS = [
    {
        "task_name": "medexam_train",
        "task_type": "med-exam",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train.parquet",
        "output_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train_verl.parquet",
    },
    {
        "task_name": "medexam_val",
        "task_type": "med-exam",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val.parquet",
        "output_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val_verl.parquet",
    },
    {
        "task_name": "blzk_train",
        "task_type": "blzk",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train.parquet",
        "output_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train_verl.parquet",
    },
    {
        "task_name": "blzk_val",
        "task_type": "blzk",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val.parquet",
        "output_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val_verl.parquet",
    },
]


# -------------------- 通用解析 --------------------
def _clean_spark_text(text: str) -> str:
    """清洗 Spark 导出的转义字符，便于后续切分。"""
    s = str(text or "")
    s = s.replace("<ret>", "\n")
    s = s.replace("\\r", "")
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", " ")
    return s.strip()


def parse_spark_input(raw_input: str) -> tuple[str, str]:
    """解析 <System>...<end><User>...<end><Bot> 格式输入。"""
    s = _clean_spark_text(raw_input)

    system_m = re.search(r"<System>(.*?)<end>", s, re.DOTALL | re.IGNORECASE)
    system_content = system_m.group(1).strip() if system_m else ""
    # 将历史思考标记替换为统一标签，保持和现有训练格式一致
    system_content = system_content.replace("<unused6>", "<think>").replace("<unused7>", "</think>")

    user_m = re.search(r"<User>(.*?)<end>", s, re.DOTALL | re.IGNORECASE)
    user_content = user_m.group(1).strip() if user_m else ""
    return system_content, user_content


def _read_df(input_path: str) -> pd.DataFrame:
    """根据后缀读取 parquet/json/jsonl/csv。"""
    if input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    if input_path.endswith(".json"):
        return pd.read_json(input_path)
    if input_path.endswith(".jsonl"):
        return pd.read_json(input_path, lines=True)
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    raise ValueError(f"不支持的文件格式: {input_path}")


def _build_prompt(raw_input: str) -> list[dict]:
    """构建 verl 所需的对话 prompt。"""
    system_content, user_content = parse_spark_input(raw_input)
    prompt = []
    if system_content:
        prompt.append({"role": "system", "content": system_content})
    prompt.append({"role": "user", "content": user_content})
    return prompt


# -------------------- 两类任务转换逻辑 --------------------
def convert_medexam_row(row: dict) -> dict | None:
    """medexam 单条转换：直接使用原始 category/target。"""
    target = str(row.get("target", "") or "").strip()
    if not target:
        return None

    raw_category = str(row.get("category", "") or "").strip()
    user_content = parse_spark_input(row.get("input", ""))[1]

    return {
        "data_source": raw_category or "med-exam",
        "prompt": _build_prompt(row.get("input", "")),
        "ability": "med_exam",
        "reward_model": {
            "style": "model",
            "ground_truth": target,  # 不再做字母排序/去重，直接保留原始答案
        },
        "extra_info": {
            "id": str(row.get("id", "")),
            "category": raw_category,
            "hardness": str(row.get("hardness", "")),
            "question": user_content,
            "target": target,
        },
    }


def convert_blzk_row(row: dict) -> dict | None:
    """blzk 单条转换：保留原始 target。"""
    target = str(row.get("target", "") or "").strip()
    if not target:
        return None

    return {
        "data_source": str(row.get("category", "med-blzk")),
        "prompt": _build_prompt(row.get("input", "")),
        "ability": "med-blzk",
        "reward_model": {
            "style": "rule",
            "ground_truth": target,
        },
        "extra_info": {
            "id": str(row.get("id", "")),
            "category": str(row.get("category", "")),
            "hardness": str(row.get("hardness", "")),
            "target": target,
            "type": str(row.get("type", "")),
            "质控项类型"：str(row.get("质控项类型", "")),
        },
    }


def convert_row(row: dict, task_type: str) -> dict | None:
    """根据任务类型路由到对应转换函数。"""
    if task_type == "medexam":
        return convert_medexam_row(row)
    if task_type == "blzk":
        return convert_blzk_row(row)
    raise ValueError(f"未知 task_type: {task_type}")


# -------------------- 批处理流程 --------------------
def process_one_dataset(task_name: str, task_type: str, input_path: str, output_path: str, workers: int) -> None:
    """处理单个数据集并写出 parquet。"""
    df = _read_df(input_path)
    rows = df.to_dict("records")
    total = len(rows)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                tqdm(
                    executor.map(convert_row, rows, [task_type] * total, chunksize=200),
                    total=total,
                    desc=f"转换 {task_name}",
                    unit="条",
                )
            )
    else:
        results = [
            convert_row(row, task_type)
            for row in tqdm(rows, desc=f"转换 {task_name}", unit="条")
        ]

    records = [x for x in results if x is not None]
    skipped = total - len(records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)

    print(f"\n[完成] {task_name}")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  保留: {len(records)} 条 | 跳过: {skipped} 条")


def verify_output(parquet_path: str, n: int = 2) -> None:
    """简单抽样验证输出结构。"""
    df = pd.read_parquet(parquet_path)
    print(f"\n[验证] {parquet_path}")
    print(f"  总条数: {len(df)}")
    print(f"  字段列表: {list(df.columns)}")
    for i, row in df.head(n).iterrows():
        print(f"  样本{i + 1}: data_source={row['data_source']} ground_truth={row['reward_model']['ground_truth']}")


def main() -> None:
    """按固定配置批量执行全部任务。"""
    for cfg in TASK_CONFIGS:
        process_one_dataset(
            task_name=cfg["task_name"],
            task_type=cfg["task_type"],
            input_path=cfg["input_path"],
            output_path=cfg["output_path"],
            workers=WORKERS,
        )
        if VERIFY_AFTER_CONVERT:
            verify_output(cfg["output_path"])


if __name__ == "__main__":
    main()
