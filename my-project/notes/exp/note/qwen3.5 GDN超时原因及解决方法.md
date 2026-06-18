# Qwen3.5 多机训练 hang 根因总结：FlashInfer GDN kernel 

> **问题**：Qwen3.5 系列在 vLLM TP=8、多机 RL 训练（GRPO 等）下随机 step 挂死
> ——根因在 FlashInfer 的 `chunk_gated_delta_rule` kernel，CUDA Graph replay 下
> 触发 **warp-specialized mbarrier 死锁**，导致 TP rank 间 desync → NCCL allgather watchdog 触发挂死。
> **解决方法**：vLLM 配置 `--gdn-prefill-backend triton`，把 GDN 前向从 FlashInfer 换成 Triton。


---
三件相互印证的 issue：
* flashinfer#3329： 	根因：cuda-gdb 抓到 mbarrier 死锁 + 给出 triton workaround
* ms-swift#8506:	症状：rank 1 stuck in GDN，rank 0+2-7 stuck in all_reduce
* vllm#41862:	旁证：Qwen3.5 EP=8 也挂在 GDN，TP 走 triton backend 100% 工作
* 我的报错:	匹配：blzk 长序列、大 batch、Qwen3.5 → 必崩

## 一、症状总结

### 现象
- Qwen3.5（27B Dense / 35B-A3B / Next 等带 GDN 的版本）
- 多机训练（一般 2 节点 16 卡起）
- vLLM rollout，TP=8，async 模式
- 训练跑 **几步到几十步随机挂死**
- GPU 利用率掉到 0%，无报错，**10 分钟后 NCCL watchdog 触发 timeout**
- 报错样子：
  ```
  WorkNCCL(SeqNum=N, OpType=_ALLGATHER_BASE,
           NumelIn=M, NumelOut=8M, Timeout(ms)=600000)
  ran for 600007 milliseconds before timing out.
  ```
   
```
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m (EngineCore_DP0 pid=4761) ERROR 05-22 14:41:15 [core.py:1102]   File "/home3/medcog/jycai6/miniforge3/envs/verl_rl/lib/python3.12/contextlib.py", line 137, in __enter__
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m (EngineCore_DP0 pid=4761) ERROR 05-22 14:41:15 [core.py:1102]     return next(self.gen)
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m (EngineCore_DP0 pid=4761) ERROR 05-22 14:41:15 [core.py:1102]            ^^^^^^^^^^^^^^
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m (EngineCore_DP0 pid=4761) ERROR 05-22 14:41:15 [core.py:1102]   File "/home3/medcog/jycai6/miniforge3/envs/verl_rl/lib/python3.12/site-packages/vllm/distributed/device_communicators/shm_broadcast.py", line 537, in acquire_read
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m (EngineCore_DP0 pid=4761) ERROR 05-22 14:41:15 [core.py:1102]     raise RuntimeError("cancelled")
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m (EngineCore_DP0 pid=4761) ERROR 05-22 14:41:15 [core.py:1102] RuntimeError: cancelled
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708] AsyncLLM output_handler failed.
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708] Traceback (most recent call last):
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708]   File "/home3/medcog/jycai6/miniforge3/envs/verl_rl/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 664, in output_handler
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708]     outputs = await engine_core.get_output_async()
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708]   File "/home3/medcog/jycai6/miniforge3/envs/verl_rl/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 1009, in get_output_async
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708]     raise self._format_exception(outputs) from None
[36m(vLLMHttpServer pid=4064, ip=172.19.60.164)[0m ERROR 05-22 14:41:15 [async_llm.py:708] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```


### py-spy 抓到的关键证据（来自 [ms-swift#8506](https://github.com/modelscope/ms-swift/issues/8506)）

```text
Rank 1（卡死的元凶）:
    gdn_prefill (flashinfer/gdn_prefill.py:52)
    chunk_gated_delta_rule (flashinfer/gdn_prefill.py:196)
    fi_chunk_gated_delta_rule (vllm/.../qwen3_next.py:138)
    gdn_attention_core (vllm/.../qwen3_next.py:1451)

Rank 0, 2-7（等 Rank 1 的 7 个）:
    all_reduce (vllm/.../symm_mem.py:148)
    all_reduce (vllm/.../cuda_communicator.py:201)
    _all_reduce_out_place (vllm/.../parallel_state.py:514)
```

**Rank 1 还在算 GDN attention kernel；其他 7 个已经过了 GDN，在后面那个 all_reduce 上等死**。
TP 内 rank 之间本应 lockstep —— 但 GDN kernel **打破了这个假设**。

---

## 二、根因（来自 [flashinfer#3329](https://github.com/flashinfer-ai/flashinfer/issues/3329)）

cuda-gdb 调试发现：

> "Kernel deadlocks on a **single SM block**; other blocks of the same launch complete normally.
> cuda-gdb shows the stuck warp parked on an **mbarrier wait** —
> looks like a warp-specialized **producer/consumer mbarrier mis-arrival**."

翻译：
- GDN kernel 用了 warp-specialized 编程模式（warp 分工：一组 producer 取数据，一组 consumer 算）
- 它们用 CUDA `mbarrier`（memory barrier）同步
- **某个 SM 的某个 warp 永远等不到对面的 mbarrier 信号** → 永久挂死
- **只在 CUDA Graph capture + replay 路径下触发，直接调 kernel 不挂**
- hang 概率跟 batch 形状强相关：特定 N 值下 60-100% 触发

### 跟我的数据相关吗？是

flashinfer#3329 给的触发 shape：`cu_seqlens = [prefill_T] + [6]*(N-1)`
- 一个 prefill 序列 + N-1 个 decode 步
- N=73, 81, 97 时 hang 概率 60-100%

vLLM continuous batching 模式下，**每一步都是 1 prefill + 几百 decode**。
长 response（max=16384）= active request 数 N 经常落到 80-100+ 的高 hang 概率区间。
**长尾数据 = 高 hang 概率**。

---

## 三、解药：切到 Triton 后端

flashinfer#3329 原文：

> "Switching the prefill backend to **Triton/FLA** (vLLM's `--gdn-prefill-backend triton`)
> at the exact same shapes: **0/35 hangs**.
> Costs ~1.7-1.9× per-call kernel time vs flashinfer."

**35 次重试 0 次挂，1.7-1.9× kernel 慢（整体 rollout 慢 5-10%）**。

### 在 verl 里怎么开（vLLM 后端）

`ROLLOUT` 数组里加一行：

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
```

vLLM 0.18.0 源码确认（[engine/arg_utils.py:618](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/engine/arg_utils.py#L618)）：

```python
gdn_prefill_backend: Literal["flashinfer", "triton"] | None = None
```

### 在 verl 里怎么开（SGLang 后端）

**SGLang 也有同样的 bug**——默认在 H100/H200 上用 FlashInfer 跑 GDN，
所以 SGLang 用户同样需要切到 triton。

SGLang `server_args.py:275`：
```python
LINEAR_ATTN_KERNEL_BACKEND_CHOICES = ["triton", "cutedsl", "flashinfer"]
```

SGLang 默认行为（`server_args.py:3073`）："defaulting --linear-attn-decode-backend to flashinfer"
——SM90 (H100/H200) 上**默认走 FlashInfer**，跟 vLLM 一模一样。

CLI 三个相关 flag：
- `--linear-attn-backend` — 同时设 prefill + decode
- `--linear-attn-prefill-backend` — 单独覆盖 prefill（**bug 在这条路径**）
- `--linear-attn-decode-backend` — 单独覆盖 decode

verl 配置（`rollout.name=sglang` 时）：
```bash
# 只切 prefill（最小变量）
+actor_rollout_ref.rollout.engine_kwargs.sglang.linear_attn_prefill_backend=triton

# 或者 prefill + decode 都切（更激进的兜底）
+actor_rollout_ref.rollout.engine_kwargs.sglang.linear_attn_backend=triton
```

### vLLM vs SGLang 对照表（统一参考）

| | vLLM | SGLang |
|---|---|---|
| 默认 GDN 后端（H200）| FlashInfer | FlashInfer |
| 触发 mbarrier bug | ✅ | ✅ |
| 切 triton 的 CLI flag | `--gdn-prefill-backend triton` | `--linear-attn-prefill-backend triton` |
| verl 透传的 hydra key | `engine_kwargs.vllm.gdn_prefill_backend=triton` | `engine_kwargs.sglang.linear_attn_prefill_backend=triton` |
| 性能代价 | ~5-10% rollout 慢 | ~5-10% rollout 慢 |



