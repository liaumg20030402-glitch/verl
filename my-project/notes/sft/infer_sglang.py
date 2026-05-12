"""
使用 SGLang Engine 对 Qwen3.5-27B 模型进行批量推理
输入：JSONL 文件，每行包含 messages 和可选的 images 字段
输出：JSONL 文件，每行追加 response 字段
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import sglang as sgl
from transformers import AutoProcessor


# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------

def read_image_base64(image_path: str) -> str:
    """读取本地图片，返回纯 base64 字符串（不含 data-URI 前缀）。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_processor_messages(raw_messages: list, images: list[str]) -> tuple[list[dict], list[str]]:
    """
    将原始 messages + images 转换为：
      - processor_messages: 供 processor.apply_chat_template 使用的多模态消息列表
      - image_b64_list: 与消息中图片顺序一致的 base64 字符串列表

    <image> 占位符替换为 {"type": "image"} block（transformers 标准格式）。
    """
    image_iter = iter(images)
    processor_messages = []
    image_b64_list = []

    for msg in raw_messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str) and "<image>" in content:
            parts = re.split(r"(<image>)", content)
            new_content = []
            for part in parts:
                if part == "<image>":
                    img_path = next(image_iter, None)
                    if img_path is None:
                        raise ValueError("images 数量少于 <image> 占位符数量")
                    image_b64_list.append(read_image_base64(img_path))
                    # transformers processor 识别的图片 block 格式
                    new_content.append({"type": "image"})
                elif part:
                    new_content.append({"type": "text", "text": part})
            processor_messages.append({"role": role, "content": new_content})

        elif isinstance(content, list):
            # 已经是 content list 格式，收集其中的图片路径
            new_content = []
            for block in content:
                if block.get("type") == "image":
                    img_path = next(image_iter, None)
                    if img_path is None:
                        raise ValueError("images 数量不足")
                    image_b64_list.append(read_image_base64(img_path))
                    new_content.append({"type": "image"})
                else:
                    new_content.append(block)
            processor_messages.append({"role": role, "content": new_content})

        else:
            processor_messages.append({"role": role, "content": content})

    return processor_messages, image_b64_list


def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {lineno} 行 JSON 解析失败，已跳过: {e}", file=sys.stderr)
    return samples


def save_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_thinking_and_prediction(response: str) -> tuple[str | None, str]:
    """从 response 中提取 thinking 和 prediction

    模型输出格式: 思考内容</think>答案内容
    （<think> 在推理时已通过 prompt 拼接，模型只输出 </think> 结束标记）
    """
    thinking = None
    prediction = response

    # 匹配 ...</think> 格式（模型只输出结束标签）
    pattern = r"(.*?)</think>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        # prediction 为 </think> 标签后的内容
        prediction = response[match.end():].strip()

    return thinking, prediction


def get_last_user_message(messages: list) -> str:
    """提取 messages 中最后一条用户消息"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # 如果 content 是列表，提取文本部分
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return " ".join(text_parts)
            return content
    return ""


def build_output_path(output_dir: str, model_id: str, input_path: str, mode: str) -> Path:
    """构建输出路径: {output_dir}/{model_id}/{task}/{mode}/{subtask}/predictions.jsonl"""
    input_path = Path(input_path)
    task = input_path.parent.name      # 父目录名
    subtask = input_path.stem          # 文件名（不含扩展名）

    output_path = Path(output_dir) / model_id / task / mode / subtask / "predictions.jsonl"
    return output_path


# ---------------------------------------------------------
# 推理核心
# ---------------------------------------------------------

def run_inference(
    engine: sgl.Engine,
    processor: AutoProcessor,
    samples: list[dict],
    sampling_params,
    batch_size: int = 8,
    enable_thinking=False,
) -> list[dict]:
    results = []
    total = len(samples)

    for start in range(0, total, batch_size):
        batch = samples[start: start + batch_size]

        # 每个元素是 (prompt_str, image_b64_list) 或 None（表示构建失败）
        prepared = []
        for item in batch:
            try:
                proc_msgs, image_b64_list = build_processor_messages(
                    item["messages"], item.get("images", [])
                )
                # apply_chat_template 生成带特殊 token 的 prompt 字符串
                # add_generation_prompt=True 会在末尾加上 <|im_start|>assistant\n
                # print(f"enable_thinking: {enable_thinking}")
                prompt_str = processor.apply_chat_template(
                    proc_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
                prepared.append((prompt_str, image_b64_list))
            except (FileNotFoundError, ValueError, StopIteration) as e:
                print(f"[错误] 构建样本失败: {e}", file=sys.stderr)
                prepared.append(None)

        valid_indices = [i for i, p in enumerate(prepared) if p is not None]

        if valid_indices:
            prompts     = [prepared[i][0] for i in valid_indices]
            image_datas = [prepared[i][1] for i in valid_indices]

            # ✅ 正确的 Engine.generate 调用方式：
            #   - prompts    : 文本字符串列表（已含图片占位 token）
            #   - image_data : 每条样本对应的 base64 列表，外层列表与 prompts 等长
            outputs = engine.generate(
                prompt=prompts,
                sampling_params=sampling_params,
                image_data=image_datas,
            )
            out_iter = iter(outputs)
        else:
            out_iter = iter([])

        for local_idx, item in enumerate(batch):
            result = dict(item)
            if local_idx in valid_indices:
                out = next(out_iter)
                result["response"] = out["text"]
            else:
                result["response"] = None
            results.append(result)

        done = min(start + batch_size, total)
        print(f"进度: {done}/{total}", end="\r", flush=True)

    print()
    return results


# ---------------------------------------------------------
# 入口
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SGLang + Qwen2.5-VL 批量推理脚本")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 HuggingFace Hub ID")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="输入 JSONL 文件路径（支持多个）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出根目录")
    parser.add_argument("--model-id", type=str, required=True,
                        help="模型标识符，用于构建输出路径")
    parser.add_argument("--thinking-mode", type=str, choices=["fast", "slow", "all"], default="all",
                        help="思考模式: fast (无思考), slow (带思考), 或 all (两种都运行)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor Parallel 大小（默认 1）")
    parser.add_argument("--dp", type=int, default=1,
                        help="Data Parallel 大小（默认 1）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="每批处理样本数（默认 8）")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="最大生成 token 数（默认 1024）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--mem-fraction", type=float, default=0.9,
                        help="GPU 显存占用比例（默认 0.9）")

    return parser.parse_args()


def main():
    args = parse_args()

    # 确定需要运行的思考模式
    if args.thinking_mode == "all":
        modes_to_run = ["fast", "slow"]
    else:
        modes_to_run = [args.thinking_mode]

    # 2. 加载 processor (用于 apply_chat_template，不做 tokenize)
    print(f"[1/4] 加载 processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # 3. 启动 SGLang Engine
    print(f"[2/4] 启动 SGLang Engine (tp={args.tp}, dp={args.dp})")
    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp,
        dp_size=args.dp,
        mem_fraction_static=args.mem_fraction,
    )

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    # 4. 遍历所有输入文件进行推理
    print(f"[3/4] 开始推理 (batch_size={args.batch_size}, thinking_mode={args.thinking_mode}) ...")

    for mode in modes_to_run:
        enable_thinking = (mode == "slow")
        print(f"\n{'='*50}")
        print(f"运行思考模式: {mode}")
        print(f"{'='*50}")

        for input_path in args.input:
            print(f"\n处理文件: {input_path}")

            # 构建输出路径
            output_path = build_output_path(args.output_dir, args.model_id, input_path, mode)

            # 检查是否已完成
            if output_path.exists():
                print(f"  ⏭  输出文件已存在，跳过: {output_path}")
                continue

            # 加载数据
            samples = load_jsonl(input_path)
            print(f"  共 {len(samples)} 条样本")

            # 推理
            results = run_inference(engine, processor, samples, sampling_params,
                                    batch_size=args.batch_size, enable_thinking=enable_thinking)

            # 转换输出格式
            output_records = []
            for item in results:
                sample_id = item.get("id", "")
                query = get_last_user_message(item.get("messages", []))
                response = item.get("response", "")
                thinking, prediction = extract_thinking_and_prediction(response)

                output_records.append({
                    "sample_id": sample_id,
                    "query": query,
                    "thinking": thinking,
                    "prediction": prediction
                })

            # 创建输出目录并保存结果
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(output_records, str(output_path))
            print(f"  ✅ 完成，结果已保存至: {output_path}")

    print(f"\n[4/4] 所有文件处理完成")
    engine.shutdown()


if __name__ == "__main__":
    main()