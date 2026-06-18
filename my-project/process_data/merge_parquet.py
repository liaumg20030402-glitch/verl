"""
随机合并多个 verl parquet 文件（用于多任务联合训练）。

把各任务转换好的 *_verl.parquet（带 data_source / prompt / reward_model / extra_info
列）拼成一个大文件，并随机打乱顺序。每行自带 data_source，训练时由分发奖励函数
reward_fn_multitask.py 按 data_source 路由打分。

注意：
  - 各输入 parquet 的列结构需一致（本项目 convert_data_to_verl_rl.py 产出的都一致）；
  - train 和 val 分别合并（下面配置了两组），不要把 val 混进 train。

用法：
    # 改下面的 TRAIN_INPUTS / VAL_INPUTS / 输出路径后直接运行
    python merge_parquet.py

    # 或命令行（最后一个为输出，前面都是输入；输入 ≥1 个）
    python merge_parquet.py a.parquet b.parquet merged.parquet --seed 42
"""

import argparse

import pandas as pd


# -------------------- 固定配置（按需改这里，或用命令行覆盖） --------------------
SEED = 42
_DATASET_ROOT = "/train21/medcog/permanent/jycai6/jmli27/dataset"

TRAIN_INPUTS = [
    f"{_DATASET_ROOT}/medexam/medexam_train_verl.parquet",
    f"{_DATASET_ROOT}/blzk/blzk_train_verl.parquet",
    f"{_DATASET_ROOT}/kie/kie_train_verl.parquet",
    f"{_DATASET_ROOT}/zyzl_blzk/zyzl_blzk_train_verl.parquet",
]
TRAIN_OUTPUT = f"{_DATASET_ROOT}/multitask/multitask_train_verl.parquet"

VAL_INPUTS = [
    f"{_DATASET_ROOT}/medexam/medexam_val_verl.parquet",
    f"{_DATASET_ROOT}/blzk/blzk_val_verl.parquet",
    f"{_DATASET_ROOT}/kie/kie_val_verl.parquet",
    f"{_DATASET_ROOT}/zyzl_blzk/zyzl_blzk_val_verl.parquet",
]
VAL_OUTPUT = f"{_DATASET_ROOT}/multitask/multitask_val_verl.parquet"


def merge_and_shuffle(input_paths: list[str], output_path: str, seed: int) -> None:
    """读取全部输入 parquet，按行拼接、随机打乱后写出一个 parquet。"""
    frames = []
    per_file = []
    for path in input_paths:
        df = pd.read_parquet(path)
        per_file.append((path, len(df), df["data_source"].iloc[0] if "data_source" in df.columns and len(df) else "?"))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    # frac=1 全量重排；reset_index 丢掉旧索引。固定 seed 可复现。
    merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print("[完成] 合并 + 随机打乱")
    for path, n, ds in per_file:
        print(f"  输入: {path}  ({n} 条, data_source={ds})")
    print(f"  输出: {output_path}  (共 {len(merged)} 条, seed={seed})")
    if "data_source" in merged.columns:
        print("  各 data_source 占比:")
        for ds, cnt in merged["data_source"].value_counts().items():
            print(f"    {ds}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="随机合并多个 verl parquet 文件")
    parser.add_argument(
        "paths",
        nargs="*",
        help="若干输入 parquet + 1 个输出 parquet（最后一个为输出）；省略则用脚本顶部 TRAIN/VAL 配置。",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"随机种子（默认 {SEED}）")
    args = parser.parse_args()

    if args.paths:
        if len(args.paths) < 2:
            parser.error("命令行模式需要至少 1 个输入 + 1 个输出，共 ≥2 个路径。")
        merge_and_shuffle(args.paths[:-1], args.paths[-1], args.seed)
    else:
        merge_and_shuffle(TRAIN_INPUTS, TRAIN_OUTPUT, args.seed)
        merge_and_shuffle(VAL_INPUTS, VAL_OUTPUT, args.seed)


if __name__ == "__main__":
    main()
