"""
数据转换脚本（medexam / blzk / kie / zyzl-blzk）
将原始数据转换为 verl 强化学习训练所需的 parquet 格式。

任务分两类：
  - medexam：style=model，额外带 question 字段（convert_medexam_row）；
  - 规则评分类（blzk / kie / zyzl-blzk）：结构一致、仅 data_source 不同，
    走通用 convert_rule_row，新增同类任务只需在 RULE_TASK_DATA_SOURCE 登记。
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
import re

import pandas as pd
from tqdm import tqdm


# -------------------- 固定路径配置（按需改这里） --------------------
WORKERS = 4
VERIFY_AFTER_CONVERT = True

# output_path 自动由 input_path 派生（同目录、文件名加 _verl.parquet 后缀），
# 因此这里只需维护 task_type 和 input_path。
TASK_CONFIGS = [
    {
        "task_type": "med-exam",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train.parquet",
    },
    {
        "task_type": "med-exam",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val.parquet",
    },
    {
        "task_type": "blzk",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train.parquet",
    },
    {
        "task_type": "blzk",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val.parquet",
    },
    {
        "task_type": "kie",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/kie/kie_train.parquet",
    },
    {
        "task_type": "kie",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/kie/kie_val.parquet",
    },
    {
        "task_type": "zyzl-blzk",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/zyzl_blzk/zyzl_blzk_train.parquet",
    },
    {
        "task_type": "zyzl-blzk",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/zyzl_blzk/zyzl_blzk_val.parquet",
    },
]


def _derive_output_path(input_path: str) -> str:
    """由 input_path 派生 output_path：同目录，文件名去掉原后缀后加 _verl.parquet。

    例：.../medexam_train.parquet -> .../medexam_train_verl.parquet
    """
    p = PurePosixPath(input_path)
    return str(p.with_name(f"{p.stem}_verl.parquet"))



UNIFIED_SYSTEM_PROMPT = "你能够回答用户的各种问题，回答问题能够多角度全面、表述专业、重点突出。"


def _clean_spark_text(text: str) -> str:
    """清洗 Spark 导出的转义字符，便于后续切分。"""
    s = str(text or "")
    s = s.replace("<ret>", "\n")
    s = s.replace("\\r", "")
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", " ")
    return s.strip()


def parse_spark_input(raw_input: str) -> tuple[str, str]:
    """解析 <System>...<end><User>...<end><Bot> 格式输入。

    返回 (system_content, user_content)。其中 system_content 固定使用
    UNIFIED_SYSTEM_PROMPT，丢弃原始 <System> 段（含 <unused6>/<unused7>
    思考标记），原因：Qwen3.5-MoE 自带 thinking，不需要外部 system 提示
    再去声明 <think></think> 这种格式。
    """
    s = _clean_spark_text(raw_input)

    user_m = re.search(r"<User>(.*?)<end>", s, re.DOTALL | re.IGNORECASE)
    user_content = user_m.group(1).strip() if user_m else ""
    return UNIFIED_SYSTEM_PROMPT, user_content


def _read_df(input_path: str) -> pd.DataFrame:
    """根据后缀读取 parquet/json/jsonl/csv。"""
    if input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    if input_path.endswith(".json"):
        # 读取首个非空白字符：
        # - '[' 开头：标准 JSON（通常是数组）
        # - '{' 开头：按 JSONL（每行一个 JSON 对象）读取
        with open(input_path, "r", encoding="utf-8") as f:
            first_non_ws = ""
            while True:
                ch = f.read(1)
                if ch == "":
                    break
                if not ch.isspace():
                    first_non_ws = ch
                    break

        if first_non_ws == "[":
            return pd.read_json(input_path)
        if first_non_ws == "{":
            return pd.read_json(input_path, lines=True)
        raise ValueError(f"无法识别 JSON 文件结构: {input_path}")
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


# -------------------- 任务转换逻辑 --------------------
# task_type 直接作为 data_source 使用。规则评分类任务（blzk / kie / zyzl-blzk）结构
# 完全一致，target 一律原样保留（清洗/格式识别交给各自的奖励函数）；新增同类任务
# 只需把 task_type 登记到这里，并在 TASK_CONFIGS 增加配置。
RULE_TASKS = {"blzk", "kie", "zyzl-blzk"}


def convert_medexam_row(row: dict) -> dict | None:
    """medexam 单条转换：style=model，额外带 question 字段。"""
    target = str(row.get("target", "") or "").strip()
    if not target:
        return None

    raw_category = str(row.get("category", "") or "").strip()
    user_content = parse_spark_input(row.get("input", ""))[1]

    return {
        "data_source": "med-exam",
        "prompt": _build_prompt(row.get("input", "")),
        "ability": "med-exam",
        "reward_model": {
            "style": "model",
            "ground_truth": target,  # 不再做字母排序/去重，直接保留原始答案
        },
        "extra_info": {
            "id": str(row.get("id", "")),
            "category": raw_category,
            "question": user_content,
            "target": target,
        },
    }


def convert_rule_row(row: dict, data_source: str) -> dict | None:
    """规则评分类任务（blzk / kie / zyzl-blzk）通用转换：原样保留 target。"""
    target = str(row.get("target", "") or "").strip()
    if not target:
        return None

    return {
        "data_source": data_source,
        "prompt": _build_prompt(row.get("input", "")),
        "ability": data_source,
        "reward_model": {
            "style": "rule",
            "ground_truth": target,
        },
        "extra_info": {
            "id": str(row.get("id", "")),
            "category": str(row.get("category", "")),
            "target": target,
            "type": str(row.get("type", "")),
        },
    }


def convert_row(row: dict, task_type: str) -> dict | None:
    """根据任务类型路由到对应转换函数（task_type 即 data_source）。"""
    if task_type == "med-exam":
        return convert_medexam_row(row)
    if task_type in RULE_TASKS:
        return convert_rule_row(row, task_type)
    raise ValueError(f"未知 task_type: {task_type}")


# -------------------- 批处理流程 --------------------
def process_one_dataset(task_type: str, input_path: str, output_path: str, workers: int) -> None:
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
                    desc=f"转换 {task_type}",
                    unit="条",
                )
            )
    else:
        results = [
            convert_row(row, task_type)
            for row in tqdm(rows, desc=f"转换 {task_type}", unit="条")
        ]

    records = [x for x in results if x is not None]
    skipped = total - len(records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)

    print(f"\n[完成] {task_type}")
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
        output_path = _derive_output_path(cfg["input_path"])
        process_one_dataset(
            task_type=cfg["task_type"],
            input_path=cfg["input_path"],
            output_path=output_path,
            workers=WORKERS,
        )
        if VERIFY_AFTER_CONVERT:
            verify_output(output_path)


if __name__ == "__main__":
    main()
