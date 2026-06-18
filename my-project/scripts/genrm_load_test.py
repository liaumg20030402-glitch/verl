#!/usr/bin/env python3
"""GenRM 负载压测——模拟训练 rollout 的判分请求，持续高并发打 GenRM 服务。

用途：和 start_genrm_server_sglang.sh 同时跑，观察
  1) GenRM 在【真实负载】下 GPU 利用率能不能打高；
  2) 吞吐 / 延迟分位 / 失败(超时)情况，用来对比不同配置。

⭐ 关键设计：每条请求的 prompt 内容都**随机生成、互不相同**（模板里的占位符每次填不同的
  随机文本）。原因：vLLM/SGLang 都默认开 prefix caching；只有占位符内容每条都不同（像真实
  rollout 数据），prefill 才不会命中缓存、才会跑满、压出真实负载。模板那一小段固定前缀会被
  缓存——这和真实训练一致（真实场景也是模板固定、内容变），无需特殊处理。

调用方式与奖励函数一致：POST {base_url}/v1/chat/completions，messages=[{role:user, content:prompt}]，
  temperature/top_p/max_tokens/超时/重试 对齐训练侧。**单端点**（SGLang DP 路由在 server 内部，
  不需要客户端做多地址轮询）。

⚠️ 关于尺寸单位：--fill-words 是**“随机词块”个数，不是模型 token 数**。一个词块是 1~4 个汉字 /
  2~7 个字母 / 一个数字，实测 **1 词块 ≈ 2~3 个 token**。所以一条 prompt 的大致 token 量 ≈
  占位符个数 × fill_words × ~2.6 + 模板固定部分。别把 fill-words 当 token 数。

用法示例（病历审核任务 blsc，思考开、max_tokens 给足）：
  export GENRM_BASE_URL="http://<ip>:8000"          # SGLang 单地址
  export GENRM_MODEL_NAME="genrm_remote"
  python3 genrm_load_test.py \
      --template /train21/medcog/permanent/jycai6/jmli27/reward/prompts/blsc.md \
      --concurrency 128 --duration 1800 --fill-words 1500 --max-tokens 3072
"""

import argparse
import asyncio
import json
import os
import random
import re
import time

import aiohttp

# 造随机文本用的“词表”：混中文片段 + 数字 + 字母，保证每条 prompt 都不同、不命中前缀缓存。
_CN = list("患者诊断治疗药物剂量症状体征检查化验影像手术麻醉感染发热咳嗽血压心率肝肾功能"
           "病史既往家族过敏用药方案预后随访指南循证一线二线禁忌适应不良反应机制代谢排泄"
           "主诉现病史体格检查诊疗意见对话医生建议时间咳痰胸闷腹痛恶心呕吐头晕乏力")
_EN = list("abcdefghijklmnopqrstuvwxyz")

# 匹配模板里的 {aaa} {bbb} {question} 这类占位符；**不**匹配 {{ }}（双花括号，JSON 示例用）。
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# 没给 --template 时的内联默认模板（仿 blsc：医患对话 + 病历文书 → 带思考的 JSON 审核）。
# 真实测试请用 --template 指向你的 blsc.md，这里只是兜底。
_DEFAULT_TEMPLATE = (
    "# 任务\n你是门诊病历审核专家，依据【医患对话】逐字段审核【病历文书】，给出分析过程与审核结论。\n\n"
    "# 医患对话\n{aaa}\n\n# 病历文书\n{bbb}\n\n"
    "# 输出要求\n以 JSON 输出，除 JSON 外不要输出其他内容：\n"
    "{{\"病历文书审核分析过程\": \"...\", \"审核结果\": [\"...\"], \"错误情况依据\": \"...\"}}"
)


def _rand_words(n_words: int, seed: str) -> str:
    """生成约 n_words 个“词块”的随机文本（按 seed 确定，每条请求 seed 不同 → 内容唯一）。

    n_words 是词块数，不是 token 数；实际 token ≈ n_words × 2~3（中文按字、字母按 BPE）。
    """
    rnd = random.Random(seed)
    parts = []
    for _ in range(max(1, n_words)):
        r = rnd.random()
        if r < 0.6:
            parts.append("".join(rnd.choices(_CN, k=rnd.randint(1, 4))))
        elif r < 0.85:
            parts.append("".join(rnd.choices(_EN, k=rnd.randint(2, 7))))
        else:
            parts.append(str(rnd.randint(0, 99999)))
    return "".join(parts)


def _build_prompt(template_text: str, fill_words: int, req_id: int) -> str:
    """把模板里每个 {占位符} 填成**本条请求独有**的随机文本（每个占位符各 fill_words 个词块）。

    模板固定部分（占位符以外的文字）保持不变——这和真实判分一致（模板固定、内容变），
    那一小段固定前缀被 prefix cache 命中是正常的、可忽略；大头随机内容不命中、跑满 prefill。
    """
    uniq = f"{req_id}-{random.randint(0, 1 << 30)}"

    def _fill(m: "re.Match") -> str:
        name = m.group(1)
        return _rand_words(fill_words, f"{uniq}-{name}")

    return _PLACEHOLDER_RE.sub(_fill, template_text)


def _pct(sorted_vals, q):
    """从已排序列表取 q 分位（0~1）。"""
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


class Stats:
    def __init__(self):
        self.ok = 0
        self.fail = 0
        self.inflight = 0
        self.lat = []            # 最近窗口成功请求的延迟（秒），打印后清空
        self.all_lat = []        # 全程所有成功请求的延迟（秒），最后算总分位
        self.out_tokens = 0      # 累计输出 token（来自 usage.completion_tokens）
        self.errors = {}         # 失败原因 -> 次数（暴露真实错误，别再吞异常）

    def snapshot_and_reset_window(self):
        lat = self.lat
        self.lat = []
        return lat

    def top_error(self):
        if not self.errors:
            return ""
        k = max(self.errors, key=self.errors.get)
        return f"{k} (×{self.errors[k]})"


async def _one_request(session, url, payload, args, stats: Stats):
    full = f"{url}/v1/chat/completions"
    for attempt in range(args.retries):
        try:
            stats.inflight += 1
            t0 = time.monotonic()
            timeout = aiohttp.ClientTimeout(total=args.timeout, connect=30)
            async with session.post(full, json=payload, timeout=timeout) as resp:
                body = await resp.text()
                if resp.status != 200:
                    # 把服务端返回的报错正文带出来（4xx 常见：prompt 超长、模型名不对）
                    raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                data = json.loads(body)
            dt = time.monotonic() - t0
            stats.ok += 1
            stats.lat.append(dt)
            stats.all_lat.append(dt)
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            stats.out_tokens += int(usage.get("completion_tokens", 0) or 0)
            return
        except Exception as e:
            if attempt >= args.retries - 1:
                stats.fail += 1
                key = f"{type(e).__name__}: {e}"
                stats.errors[key] = stats.errors.get(key, 0) + 1
                if stats.fail <= 3:   # 头几条失败立刻打印，便于秒级定位
                    print(f"  [fail] {url} -> {key}", flush=True)
            else:
                await asyncio.sleep(1.0)
        finally:
            stats.inflight -= 1


async def _worker(session, base_url, args, stats: Stats, deadline, counter):
    while time.monotonic() < deadline:
        req_id = counter[0]
        counter[0] += 1
        prompt = _build_prompt(args.template_text, args.fill_words, req_id)
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        if args.no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        await _one_request(session, base_url, payload, args, stats)


async def _reporter(stats: Stats, deadline, interval, start):
    while time.monotonic() < deadline:
        await asyncio.sleep(interval)
        window = stats.snapshot_and_reset_window()
        elapsed = time.monotonic() - start
        if window:
            window.sort()
            p50 = window[len(window) // 2]
            p95 = window[min(len(window) - 1, int(len(window) * 0.95))]
            rps = len(window) / interval
        else:
            p50 = p95 = rps = 0.0
        tok_s = stats.out_tokens / elapsed if elapsed > 0 else 0.0
        err = stats.top_error()
        print(f"[{elapsed:6.0f}s] ok={stats.ok} fail={stats.fail} inflight={stats.inflight} "
              f"rps={rps:5.1f} p50={p50:6.1f}s p95={p95:6.1f}s out_tok/s={tok_s:7.0f}"
              + (f"  | top_err: {err}" if err else ""),
              flush=True)


async def main_async(args):
    base = args.base_url or os.environ.get("GENRM_BASE_URL")
    if not base:
        raise SystemExit("缺少地址：--base-url 或环境变量 GENRM_BASE_URL")
    # 单端点：SGLang 的 DP 路由在 server 内部，客户端不需要多地址轮询。若误填逗号列表，取第一个。
    base_url = base.split(",")[0].strip().rstrip("/")
    if "," in base:
        print(f"[load] ⚠️ 检测到逗号列表，只用第一个地址：{base_url}（SGLang 单端点即可）", flush=True)

    # 预加载模板（只读一次）
    if args.template:
        with open(args.template, "r", encoding="utf-8") as f:
            args.template_text = f.read().strip()
        tpl_src = args.template
    else:
        args.template_text = _DEFAULT_TEMPLATE
        tpl_src = "<内联默认模板>"
    placeholders = sorted(set(_PLACEHOLDER_RE.findall(args.template_text)))
    est_tok = len(placeholders) * args.fill_words * 2.6   # 粗估输入 token

    print(f"[load] 目标地址：{base_url}")
    print(f"[load] 模板：{tpl_src}  占位符={placeholders}")
    print(f"[load] 并发={args.concurrency} 时长={args.duration}s 模型={args.model} "
          f"max_tokens={args.max_tokens} fill_words={args.fill_words}"
          f"(每占位符)≈输入{est_tok:.0f}token no_thinking={args.no_thinking}", flush=True)

    # ===== 预检：开压前先确认连得通、模型名对不对（秒级定位“全失败”）=====
    print("[load] 预检 /v1/models ...", flush=True)
    async with aiohttp.ClientSession() as s:
        try:
            async with s.get(f"{base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as r:
                print(f"  {base_url}  HTTP {r.status}  {(await r.text())[:160]}", flush=True)
                if r.status != 200:
                    raise SystemExit(f"[load] ❌ /v1/models 返回 {r.status}，server 没就绪。")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"[load] ❌ 连不通 {base_url}：{type(e).__name__}: {e}。"
                             f"（server 没起/被杀，或本节点到该 IP 不通）")

    stats = Stats()
    start = time.monotonic()
    deadline = start + args.duration
    counter = [0]
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, limit_per_host=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(_worker(session, base_url, args, stats, deadline, counter))
                 for _ in range(args.concurrency)]
        tasks.append(asyncio.create_task(_reporter(stats, deadline, args.report_interval, start)))
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - start
    total = stats.ok + stats.fail
    fail_rate = (stats.fail / total * 100) if total else 0.0
    print("\n" + "=" * 70)
    print(f"[汇总] 时长={elapsed:.0f}s  并发={args.concurrency}  目标={base_url}  "
          f"no_thinking={args.no_thinking}")
    print(f"[汇总] ok={stats.ok}  fail={stats.fail}  失败率={fail_rate:.2f}%  "
          f"平均 rps={stats.ok / elapsed:.2f}  平均 out_tok/s={stats.out_tokens / elapsed:.0f}")
    sl = sorted(stats.all_lat)
    if sl:
        avg_out = stats.out_tokens / stats.ok if stats.ok else 0
        print(f"[汇总] 成功延迟(秒)  p50={_pct(sl, 0.50):.1f}  p90={_pct(sl, 0.90):.1f}  "
              f"p95={_pct(sl, 0.95):.1f}  p99={_pct(sl, 0.99):.1f}  max={sl[-1]:.1f}")
        print(f"[汇总] 平均每条输出 {avg_out:.0f} token（贴近 max_tokens 说明被思考/分析填满）")
    # 失败按类型细分：超时(TimeoutError) / 连接断开(ServerDisconnected) / 4xx 等 → 即“超时样本数”
    if stats.errors:
        print("[汇总] 失败按类型：")
        for k, v in sorted(stats.errors.items(), key=lambda kv: -kv[1]):
            print(f"         {v:6d}  {k[:140]}")
    else:
        print("[汇总] 无失败。")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="GenRM 真实负载压测（模板驱动、随机内容、单端点）")
    p.add_argument("--base-url", default=None, help="GenRM 地址（单个）；默认读环境变量 GENRM_BASE_URL")
    p.add_argument("--model", default=os.environ.get("GENRM_MODEL_NAME", "genrm_remote"))
    p.add_argument("--concurrency", type=int, default=128, help="持续在飞的并发请求数")
    p.add_argument("--duration", type=int, default=1800, help="压测时长(秒)")
    p.add_argument("--max-tokens", type=int, default=3072,
                   help="每条生成 token 上限。复杂任务(病历审核含思考+长分析)给足；不够再加大(4096/8192)")
    p.add_argument("--template", default=None,
                   help="judge prompt 模板文件（如 blsc.md）。会把里面每个 {占位符} 填成随机内容；不给则用内联默认模板")
    p.add_argument("--fill-words", type=int, default=1500,
                   help="每个占位符填多少【词块】(非token! 1词块≈2~3token)。输入token≈占位符数×fill_words×2.6")
    p.add_argument("--temperature", type=float, default=float(os.environ.get("GRM_TEMPERATURE", 0.0)))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("GRM_TOP_P", 1.0)))
    p.add_argument("--timeout", type=float, default=240.0, help="单条总超时(秒)，对齐训练 grm_request_timeout")
    p.add_argument("--retries", type=int, default=1, help="重试次数，对齐训练 grm_max_retries")
    p.add_argument("--no-thinking", action="store_true",
                   help="加 enable_thinking=False。默认不加（保持思考开，符合复杂判分任务）")
    p.add_argument("--report-interval", type=float, default=5.0, help="统计打印间隔(秒)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
