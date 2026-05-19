"""
训练集 / 验证集划分脚本

输入：convert_data_to_verl_rl.py 处理过的 parquet 文件（带 prompt / reward_model / extra_info 字段）
输出：按指定比例切分出 train / val 两个 parquet

设计要点：
1. **比例可配置**：默认 48:1（val ~2%），可改成 9:1 (10%) / 19:1 (5%) 等常见比例
2. **可分层抽样（stratified）**：按 hardness / category 等字段分层切，保证 val 集分布和 train 一致
3. **固定随机种子**：保证可复现
4. **保留所有原字段**：split 不改数据,只洗牌+切分
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================================
# 配置区（按需改这里）
# ============================================================================

# 比例参考（train : val）：
#   48 : 1  → val ~2%   
DEFAULT_TRAIN_VAL_RATIO = 48
DEFAULT_SEED = 42

# 是否分层抽样（按某个 categorical 字段分桶后各桶按比例切）
# - None: 整体随机切（默认,直接按比例切）
# - "hardness" / "category" / "data_source": 按对应字段分层
# 分层字段必须在 row 顶层或 extra_info 里有非空值,否则会被归到 "UNKNOWN" 单组
DEFAULT_STRATIFY_BY: str | None = None

SPLIT_CONFIGS = [
    {
        "name": "blzk",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train_verl.parquet",
        "output_train_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train_fast_verl.parquet",
        "output_val_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val_fast_verl.parquet",
        "train_val_ratio": DEFAULT_TRAIN_VAL_RATIO,
        "stratify_by": DEFAULT_STRATIFY_BY,
        "seed": DEFAULT_SEED,
    },
    {
        "name": "medexam",
        "input_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train_verl.parquet",
        "output_train_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train_fast_verl.parquet",
        "output_val_path": "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val_fast_verl.parquet",
        "train_val_ratio": DEFAULT_TRAIN_VAL_RATIO,
        "stratify_by": DEFAULT_STRATIFY_BY,
        "seed": DEFAULT_SEED,
    },
]

VERIFY_AFTER_SPLIT = True


# ============================================================================
# 切分核心逻辑
# ============================================================================

def _extract_stratify_key(row: dict, stratify_by: str) -> str:
    """从一条记录里抽取分层 key。支持 extra_info 嵌套字段。

    优先级：top-level 字段 > extra_info 里的字段 > 字符串 'UNKNOWN'。
    """
    if stratify_by in row and row[stratify_by] is not None:
        return str(row[stratify_by])
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict) and stratify_by in extra:
        return str(extra[stratify_by] or "UNKNOWN")
    return "UNKNOWN"


def _split_one_group(df: pd.DataFrame, ratio: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对单个 group(或整体)做 ratio:1 切分。

    数学：ratio:1 即 val 占 1/(ratio+1)。
    最少保留 1 条 val(否则 val 集可能为空)。
    """
    df_shuf = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_total = len(df_shuf)
    n_val = max(1, n_total // (ratio + 1))   # ratio+1 因为 ratio:1 总份数是 ratio+1
    val_df = df_shuf.head(n_val)
    train_df = df_shuf.tail(n_total - n_val)
    return train_df, val_df


def split_dataset(
    input_path: str,
    output_train_path: str,
    output_val_path: str,
    train_val_ratio: int,
    stratify_by: str | None,
    seed: int,
    name: str = "",
) -> None:
    """切分一个 parquet 数据集到 train / val。

    Args:
        input_path: 输入 parquet 路径(必须是 convert_data_to_verl_rl.py 的输出格式)
        output_train_path: 训练集输出路径
        output_val_path: 验证集输出路径
        train_val_ratio: train:val 比例(int),如 48 表示 48:1
        stratify_by: 分层字段名(None = 不分层),从 row 顶层或 extra_info 里找
        seed: 随机种子
        name: 数据集名(只用于日志)
    """
    print(f"\n=========== 切分 [{name}] ===========")
    print(f"  输入: {input_path}")

    df = pd.read_parquet(input_path)
    total = len(df)
    print(f"  总条数: {total}")
    print(f"  字段列表: {list(df.columns)}")
    print(f"  比例: {train_val_ratio} : 1  (val 占 {100.0 / (train_val_ratio + 1):.2f}%)")
    print(f"  分层字段: {stratify_by or '(不分层,整体随机切)'}")
    print(f"  随机种子: {seed}")

    if stratify_by is None:
        # 整体切
        train_df, val_df = _split_one_group(df, train_val_ratio, seed)
    else:
        # 按 stratify_by 分组,每组各自按比例切,最后拼接
        keys = df.apply(lambda row: _extract_stratify_key(row, stratify_by), axis=1)
        df_with_key = df.assign(_strat_key=keys)

        train_parts, val_parts = [], []
        group_stats = []

        for key, group in tqdm(df_with_key.groupby("_strat_key"), desc=f"分层切分 {name}", unit="组"):
            group = group.drop(columns=["_strat_key"])
            t_df, v_df = _split_one_group(group, train_val_ratio, seed)
            train_parts.append(t_df)
            val_parts.append(v_df)
            group_stats.append((key, len(group), len(t_df), len(v_df)))

        train_df = pd.concat(train_parts, ignore_index=True).sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)
        val_df = pd.concat(val_parts, ignore_index=True).sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)

        # 打印分层后的分布
        print(f"  分层切分明细（按 {stratify_by} 分组）:")
        print(f"    {'key':<25}  {'total':>8}  {'train':>8}  {'val':>6}")
        for key, total_n, train_n, val_n in sorted(group_stats):
            print(f"    {str(key)[:25]:<25}  {total_n:>8}  {train_n:>8}  {val_n:>6}")

    # 落盘
    Path(output_train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_val_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_train_path, index=False)
    val_df.to_parquet(output_val_path, index=False)

    print(f"\n  [完成]")
    print(f"  Train: {len(train_df)} 条 → {output_train_path}")
    print(f"  Val:   {len(val_df)} 条 → {output_val_path}")
    print(f"  实际比例: {len(train_df)} : {len(val_df)} ≈ {len(train_df) / max(1, len(val_df)):.2f} : 1")


def verify_split(train_path: str, val_path: str, n_preview: int = 1) -> None:
    """简单验证 train / val 输出结构。"""
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    print(f"\n  [验证]")
    print(f"  Train 字段: {list(train_df.columns)} | 条数: {len(train_df)}")
    print(f"  Val   字段: {list(val_df.columns)} | 条数: {len(val_df)}")
    print(f"  Train 与 Val 字段一致: {list(train_df.columns) == list(val_df.columns)}")

    # 抽查 id 不重叠（如果 extra_info 里有 id 字段）
    if "extra_info" in train_df.columns:
        train_ids = set(
            row.get("id", "") if isinstance(row, dict) else ""
            for row in train_df["extra_info"]
        )
        val_ids = set(
            row.get("id", "") if isinstance(row, dict) else ""
            for row in val_df["extra_info"]
        )
        overlap = train_ids & val_ids - {""}
        print(f"  Train / Val id 重叠数: {len(overlap)}（应为 0）")
        if overlap:
            print(f"    ⚠️ 发现重叠 id,前 5 个: {list(overlap)[:5]}")

    if n_preview > 0 and len(val_df) > 0:
        sample = val_df.iloc[0]
        print(f"  Val 样本预览: data_source={sample.get('data_source', '?')}, "
              f"ground_truth={sample.get('reward_model', {}).get('ground_truth', '?')[:30]}...")


def main() -> None:
    """按 SPLIT_CONFIGS 批量切分。"""
    for cfg in SPLIT_CONFIGS:
        input_path = cfg["input_path"]
        if not Path(input_path).exists():
            print(f"\n[跳过 {cfg['name']}] 输入文件不存在: {input_path}")
            continue

        split_dataset(
            input_path=input_path,
            output_train_path=cfg["output_train_path"],
            output_val_path=cfg["output_val_path"],
            train_val_ratio=cfg["train_val_ratio"],
            stratify_by=cfg.get("stratify_by"),
            seed=cfg.get("seed", DEFAULT_SEED),
            name=cfg["name"],
        )

        if VERIFY_AFTER_SPLIT:
            verify_split(cfg["output_train_path"], cfg["output_val_path"])


if __name__ == "__main__":
    main()
