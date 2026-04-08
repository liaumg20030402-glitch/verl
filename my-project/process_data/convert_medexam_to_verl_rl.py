import re
from pathlib import Path

import pandas as pd


def _clean_spark_text(text: str) -> str:
    s = str(text or "")
    # 仅清洗转义字符；<end>/<bot> 在切分后再移除，避免影响边界判断。
    s = s.replace("<ret>", "\n")
    s = s.replace("\\r", "")
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", " ")
    return s.strip()


def parse_spark_input(raw_input: str) -> tuple[str, str]:
    """
    Parse Spark-formatted `input` into system and user contents.
    Supports <System> ... <User> ... <Bot> style.
    """
    s = _clean_spark_text(raw_input)
    system_match = re.search(r"<System>(.*?)<User>", s, re.DOTALL | re.IGNORECASE)
    if system_match:
        system_content = system_match.group(1).strip()
    else:
        system_end_match = re.search(r"<System>(.*?)<end>", s, re.DOTALL | re.IGNORECASE)
        system_content = system_end_match.group(1).strip() if system_end_match else ""
    system_content = system_content.replace("<unused6>", "<think>").replace("<unused7>", "</think>")
    system_content = re.sub(r"<end>|<bot>", "", system_content, flags=re.IGNORECASE).strip()
    user_match = re.search(r"<User>(.*?)<Bot>", s, re.DOTALL | re.IGNORECASE)
    if user_match:
        user_content = user_match.group(1).strip()
    else:
        user_fallback = re.search(r"<User>(.*)$", s, re.DOTALL | re.IGNORECASE)
        user_content = user_fallback.group(1).strip() if user_fallback else s
    user_content = re.sub(r"<end>|<bot>", "", user_content, flags=re.IGNORECASE).strip()
    return system_content, user_content


def _normalize_category(category: str) -> str:
    c = str(category or "").strip().lower()
    if c in {"med-exam-multi", "med_exam_multi"}:
        return "med_exam_multi"
    if c in {"med-exam", "med_exam"}:
        return "med_exam_single"
    return "med_exam_unknown"


def _normalize_target(target: str) -> str:
    """Normalize answer letters to sorted, deduplicated uppercase string.

    Supports options beyond E for longer multi-choice questions.
    Alphabetical sort ensures 'CBA' and 'ABC' both normalise to 'ABC'.
    """
    t = str(target or "").strip().upper().replace(" ", "").replace(",", "")
    t = re.sub(r"[^A-Z]", "", t)
    return "".join(sorted(set(t)))


def convert_to_verl_row(row: dict) -> dict | None:
    raw_input = row.get("input", "")
    system_content, user_content = parse_spark_input(raw_input)

    target = _normalize_target(row.get("target", ""))
    if not target:
        return None

    raw_category = str(row.get("category", "")).strip()
    norm_category = _normalize_category(raw_category)
    data_source = norm_category

    prompt = []
    if system_content:
        prompt.append({"role": "system", "content": system_content})
    prompt.append({"role": "user", "content": user_content})

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "med_exam",
        "reward_model": {
            "style": "custom",
            "ground_truth": target,
        },
        "extra_info": {
            "id": str(row.get("id", "")),
            "category": raw_category,
            "category_norm": norm_category,
            "hardness": str(row.get("hardness", "")),
            "question": user_content,
            "target": target,
            "raw_target": str(row.get("target", "")),
        },
    }


def _read_df(input_path: str) -> pd.DataFrame:
    if input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    if input_path.endswith(".json"):
        return pd.read_json(input_path)
    if input_path.endswith(".jsonl"):
        return pd.read_json(input_path, lines=True)
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    raise ValueError(f"Unsupported file type: {input_path}")


def process_dataset(input_path: str, output_path: str) -> None:
    df = _read_df(input_path)
    records = []
    skipped = 0
    for _, row in df.iterrows():
        item = convert_to_verl_row(row.to_dict())
        if item is None:
            skipped += 1
            continue
        records.append(item)

    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path, index=False)

    print(f"[done] {input_path} -> {output_path}")
    print(f"kept={len(records)}, skipped={skipped}")
    if records:
        by_source = {}
        for r in records:
            by_source[r["data_source"]] = by_source.get(r["data_source"], 0) + 1
        print(f"data_source distribution: {by_source}")


def verify_output(parquet_path: str, n: int = 3) -> None:
    df = pd.read_parquet(parquet_path)
    print(f"\n[verify] {parquet_path}")
    print(f"rows={len(df)}")
    print(f"columns={list(df.columns)}")
    for i, row in df.head(n).iterrows():
        print(f"\n--- sample {i} ---")
        print(f"data_source: {row['data_source']}")
        print(f"ability: {row['ability']}")
        print(f"ground_truth: {row['reward_model']['ground_truth']}")
        print(f"prompt_size: {len(row['prompt'])}")
        print(f"user_head: {str(row['prompt'][-1]['content'])[:120]}...")


def main():

    train_input = "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train.parquet"
    val_input = "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val.parquet"
    train_out = "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_train_verl.parquet"
    val_out = "/train21/medcog/permanent/jycai6/jmli27/dataset/medexam/medexam_val_verl.parquet"

    process_dataset(train_input, train_out)
    process_dataset(val_input, val_out)

    verify_output(train_out)
    verify_output(val_out)


if __name__ == "__main__":
    main()
