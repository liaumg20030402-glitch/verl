#!/usr/bin/env python3
"""
脚本功能:
1. 将 checkpoint 目录复制到同级目录下的新目录（名称为 `[原目录名]-resave`）
   - 排除 iter_ 开头的子目录（optimizer 等参数状态）
2. 将预训练模型目录中的 `.jinja` 和 `.json` 文件复制到新目录（排除 `model.safetensors.index.json`）
"""

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def copy_file(args):
    """复制单个文件的辅助函数，用于多进程调用"""
    src, dst = args
    shutil.copy2(src, dst)
    return src


def main():
    # ============ 配置变量 ============
    # checkpoint 目录路径 (修改此变量为目标 checkpoint 路径)
    input_path = "/train21/medcog/permanent/jycai6/med_sft_train_swift/exps/qwen27b_sft_jkys_20260424/model_output/qwen27b_sft_jkys_20260522/v1-20260522-152105/checkpoint-232"

    # 预训练模型目录路径
    pretrained_path = "/train21/medcog/permanent/jycai6/med_sft_train_swift/pretrained_ckpts/Qwen3.5-27B"
    # ==================================

    # 转换为 Path 对象
    input_path = Path(input_path)
    pretrained_path = Path(pretrained_path)

    # 验证输入路径是否存在
    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint 目录不存在: {input_path}")
    if not pretrained_path.exists():
        raise FileNotFoundError(f"预训练模型目录不存在: {pretrained_path}")

    # 构建输出路径: 父目录/[原目录名]-resave
    output_path = input_path.parent / f"{input_path.name}-resave"

    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"预训练模型路径: {pretrained_path}")
    print("-" * 50)

    # 步骤 3: 多进程复制 checkpoint 目录
    print("正在复制 checkpoint 目录...")
    if output_path.exists():
        print(f"输出目录已存在, 正在删除: {output_path}")
        shutil.rmtree(output_path)

    # 收集需要复制的文件
    copy_tasks = []
    for src_path in input_path.rglob('*'):
        # 检查是否在排除目录中
        rel_path = src_path.relative_to(input_path)
        if any(part.startswith('iter_') for part in rel_path.parts):
            continue
        
        if src_path.is_file():
            dst_path = output_path / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            copy_tasks.append((str(src_path), str(dst_path)))

    # 多进程复制
    num_workers = min(8, os.cpu_count() or 1)
    print(f"使用 {num_workers} 个进程复制 {len(copy_tasks)} 个文件...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(copy_file, task): task for task in copy_tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  已复制 {completed}/{len(copy_tasks)} 个文件...")

    print(f"✓ 已复制 checkpoint 目录到: {output_path}")

    # 步骤 4: 复制预训练模型的配置文件
    print("\n正在复制预训练模型配置文件...")
    copied_files = []
    skipped_files = []

    for file_path in pretrained_path.iterdir():
        if file_path.is_file():
            # 检查是否为 .jinja 或 .json 文件
            if file_path.suffix in ['.jinja', '.json']:
                # 排除 model.safetensors.index.json
                if file_path.name == 'model.safetensors.index.json':
                    skipped_files.append(file_path.name)
                    continue

                # 复制文件（覆盖同名文件）
                dest_path = output_path / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_files.append(file_path.name)

    print(f"✓ 已复制 {len(copied_files)} 个配置文件:")
    for f in sorted(copied_files):
        print(f"  - {f}")

    if skipped_files:
        print(f"✓ 已跳过 {len(skipped_files)} 个文件:")
        for f in sorted(skipped_files):
            print(f"  - {f}")

    # 步骤 5: 输出结果摘要
    print("\n" + "=" * 50)
    print("操作完成!")
    print(f"输出目录: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()