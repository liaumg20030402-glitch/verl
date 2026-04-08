import re
import json
import pandas as pd


def parse_spark_input(raw_input: str):
    """
    把讯飞星火格式的 input 字符串解析成 system content 和 user content
    """
    # 仅清洗转义字符；<end>/<bot> 在切分后再移除，避免影响边界判断。
    raw_input = raw_input.replace("<ret>", "\n")
    raw_input = raw_input.replace("\\r", "")
    raw_input = raw_input.replace("\\n", "\n")
    raw_input = raw_input.replace("\\t", " ")

    # 提取 System 内容：优先 <System>...<User>，缺失时回退到 <System>...<end>
    system_match = re.search(r"<System>(.*?)<User>", raw_input, re.DOTALL | re.IGNORECASE)
    if system_match:
        system_content = system_match.group(1).strip()
    else:
        system_end_match = re.search(r"<System>(.*?)<end>", raw_input, re.DOTALL | re.IGNORECASE)
        system_content = system_end_match.group(1).strip() if system_end_match else ""
    system_content = system_content.replace("<unused6>", "<think>").replace("<unused7>", "</think>")
    system_content = re.sub(r"<end>|<bot>", "", system_content, flags=re.IGNORECASE).strip()

    # 提取 User 内容：优先 <User>...<Bot>，缺失时取 <User> 后全部内容
    user_match = re.search(r"<User>(.*?)<Bot>", raw_input, re.DOTALL | re.IGNORECASE)
    if user_match:
        user_content = user_match.group(1).strip()
    else:
        user_fallback = re.search(r"<User>(.*)$", raw_input, re.DOTALL | re.IGNORECASE)
        user_content = user_fallback.group(1).strip() if user_fallback else ""
    user_content = re.sub(r"<end>|<bot>", "", user_content, flags=re.IGNORECASE).strip()

    return system_content, user_content


def convert_to_verl_format(row: dict) -> dict:
    """
    单条数据转换为 verl 标准格式
    """
    raw_input = row.get("input", "")
    system_content, user_content = parse_spark_input(raw_input)

    # 构建 prompt messages
    prompt = []
    if system_content:
        prompt.append({"role": "system", "content": system_content})
    prompt.append({"role": "user", "content": user_content})

    return {
        "data_source": row.get("category", "med-blzk"),      
        "prompt": prompt,
        "ability": "med-blzk",                            # 任务类型标识
        "reward_model": {
            "style": "rule",                              # 使用规则奖励函数
            "ground_truth": row.get("target", ""),        # "合格" 或 "不合格"
        },
        # 以下为可选的额外信息，会通过 extra_info 传给 reward function
        "extra_info": {
            "id":       row.get("id", ""),
            "category": row.get("category", ""),
            "hardness": row.get("hardness", ""),
            "target": row.get("target", ""),
            "type": row.get("type",""),
        }
    }


def process_dataset(input_path: str, output_path: str):
    """
    读取原始数据，转换格式，保存为 verl 所需的 parquet 文件
    支持输入格式：parquet / json / jsonl / csv
    """
    # 根据文件格式读取
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path)
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"不支持的文件格式: {input_path}")

    records = []
    skipped = 0

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # 跳过 target 为空的数据
        if not row_dict.get("target", "").strip():
            skipped += 1
            continue

        # 跳过 target 不合法的数据
        if row_dict.get("target", "") not in ["合格", "不合格"]:
            skipped += 1
            continue

        records.append(convert_to_verl_format(row_dict))

    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path, index=False)

    print(f"转换完成：共 {len(records)} 条，跳过 {skipped} 条")
    print(f"合格数量：{sum(1 for r in records if r['reward_model']['ground_truth'] == '合格')}")
    print(f"不合格数量：{sum(1 for r in records if r['reward_model']['ground_truth'] == '不合格')}")
    print(f"已保存至：{output_path}")


def verify_output(parquet_path: str, n: int = 2):
    """
    验证转换后的 parquet 文件格式是否正确
    """
    df = pd.read_parquet(parquet_path)
    print(f"\n总条数: {len(df)}")
    print(f"字段列表: {list(df.columns)}")
    print(f"\n前 {n} 条示例：")
    for i, row in df.head(n).iterrows():
        print(f"\n--- 第 {i+1} 条 ---")
        print(f"data_source : {row['data_source']}")
        print(f"ability     : {row['ability']}")
        print(f"ground_truth: {row['reward_model']['ground_truth']}")
        print(f"prompt[0]   : {row['prompt'][0]}")   # system
        print(f"prompt[1]   : {str(row['prompt'][1])[:100]}...")  # user 截断显示


if __name__ == "__main__":
    train_input = "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train.parquet"
    val_input = "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val.parquet"
    train_out = "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_train_verl.parquet"
    val_out = "/train21/medcog/permanent/jycai6/jmli27/dataset/blzk/blzk_val_verl.parquet"

    process_dataset(train_input, train_out)
    process_dataset(val_input, val_out)

    verify_output(train_out)
    verify_output(val_out)