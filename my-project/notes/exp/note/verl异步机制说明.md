# verl 的「异步」到底指什么——三个轴 & colocate vs standalone

> 背景：很多人把 verl 里的"异步"当成一个开关，其实它有**三个互相独立的轴**。
> 只有「GenRM 和 rollout 并行重叠」这一种是和 standalone（独立资源池）绑定的，其余两种跟资源池无关。

---

## TL;DR

| 异步类型 | 是什么 | colocate | standalone |
|---|---|---|---|
| **① rollout 异步**（`rollout.mode=async`）| 策略生成走 AgentLoop + vLLM server、continuous batching | ✅ | ✅ |
| **② GenRM 与 rollout 异步重叠**（streaming reward loop）| reward 打分和策略生成**并行重叠** | ❌ | ✅ |
| **③ 全异步 policy**（`experimental/fully_async_policy`）| 训练与 rollout 整体解耦、各占独立卡 | ❌（本质要独立分卡）| ✅ |

**结论**：
- "异步"不能笼统说"只有 standalone 才有"——①（rollout 异步）colocate 也一直在用。
- 真正绑定 standalone 的是 **②（GenRM 异步重叠）**。
- 想让 **GenRM 和 rollout 真并行 → 必须 standalone**（`reward.reward_model.enable_resource_pool=True`）。

---

## ① rollout 异步：`rollout.mode=async`

- 含义：**单步生成内部**的调度——策略 rollout 通过 AgentLoop + vLLM server（continuous batching）服务请求。
- 跟资源池**无关**，colocate / standalone 都支持。
- 注意它**不改变外层训练循环的同步性**：整体仍是「生成全部 rollout → 训练 → 同步权重 → 再生成」串行。训练时生成卡着、生成时训练卡着，GPU 有"气泡"。真正消除这个串行是 ③。

## ② GenRM 与 rollout 异步重叠：streaming reward loop（绑定 standalone）

同步后的 verl 把新的 reward loop（`verl/experimental/reward_loop/`）接进了标准 `RayPPOTrainer`。是否开启"reward 打分与 rollout 重叠"由这行决定：

```python
# verl/trainer/ppo/ray_trainer.py
enable_agent_reward_loop = (not self.use_rm) or self.config.reward.reward_model.enable_resource_pool
```

解读——只有以下两种情况才会流式重叠：
- **没有 reward model**（rule / 自定义函数奖励）→ 本来就 True；
- **reward model 用独立资源池（standalone）** → True。

**唯一不开的情况：有 reward model + colocate。**

### 为什么 colocate 的 GenRM 拿不到真异步（物理决定，非框架偷懒）

- **colocate**：RM 和 policy **共用同一批卡**，同一时刻一张卡只能干一件事——RM 打分时 policy 得让出显存/算力（sleep/wake 切换），两者**无法同时跑**。所以 reward 只能"rollout 跑完 → 再阻塞式算"。
- **standalone**：RM 有**独立的卡**，可以在 policy 还在 rollout 的同时并行打分 → 这才是真正的异步重叠。

### standalone 模式 verl 内部会自动做的事
走 `enable_resource_pool=True`（`RewardModelManager`）时，**你什么都不用手动起**：
- `_initialize_llm_servers()` → 对每个副本 `init_standalone()` 起 vLLM server；
- `_initialize_router()` → 起 router；
- reward worker 通过 router 地址异步 post 请求。

> 对比：只有 `recipe/genrm_remote`（`RemoteRewardManager`）那条路才要你自己 `vllm serve` 起**外部**服务、填 `BASE_URL`。standalone(resource_pool) 是 verl 在同一个 Ray job 里自动分卡 + 起服务。

## ③ 全异步 policy：`experimental/fully_async_policy`

- 把"生成样本"和"更新参数"拆成**两个永不停歇、并行运行的服务**（Rollouter / Trainer），中间用 MessageQueue 连接，参数用 NCCL 定期同步。
- 核心是**资源隔离**（Rollouter 和 Trainer 各占独立 GPU）+ **可控 staleness**（`async_training.staleness_threshold`，允许用略旧参数生成的样本）+ PartialRollout。
- 代价：**off-policy**，需要重要性采样校正（TIS/MIS，见 `rollout_correction` 配置 / `*_mis.sh`），调参更复杂；experimental，仅支持 megatron/fsdp + vllm server 模式。
- 报告收益：Qwen2.5-7B / 128 卡 2.35~2.67× 加速。
- 它解决的是 **policy 侧 GPU 因长尾 rollout 闲置**，和 ② 的 GenRM 异步是**两个不同的轴**。

---

## 决策建议（本项目：policy 27B + GenRM 27B，≤48×H200）

1. **要 GenRM 真异步（和 rollout 重叠）→ 用 standalone**（`RM_NNODES>0`，即 `enable_resource_pool=True`）。
   - 顺带解决双 27B colocate 的显存 OOM（两个模型不再挤同一张卡）。
   - GenRM 独享卡，`GRM_GPU_MEM` 可拉到 0.85~0.9，`GEN_RM_TP≥4`。
   - 别忘了 GenRM 也是 GDN，rollout 要 `+reward.reward_model.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton`。
2. **用规则奖励（rule）** 时本来就异步重叠（无 reward model），不需要资源池，`RM_NNODES=0`。
3. **fully_async_policy** 先别急着上——确认 rollout 是瓶颈、GPU 利用率低之后再考虑，且要配重要性采样校正、从小 staleness 起步。

---

## 一句话总结

> 普通 rollout 异步（`mode=async`）colocate 也有；但**「GenRM 与 rollout 并行重叠」这种异步只有 standalone 才有**——因为共卡的两个模型物理上没法同时跑。判定逻辑就一行：`enable_agent_reward_loop = (not use_rm) or enable_resource_pool`。

---

## ④ 换个视角：异步的本质是三个角色的解耦——rollout / train / reward

前面三个轴，其实对应**三个可以各自独立、并行的角色**：

| 角色 | 干什么 | 同步框架(RayPPOTrainer) | 全异步(fully_async_policy) |
|---|---|---|---|
| **Rollouter** | 生成样本 | 单步内 async（`mode=async`），但外层和 train 串行 | 独立 GPU 池，持续生成 |
| **Trainer** | 更新参数 | 串行：等齐一个 batch 才训 | 独立 GPU 池，持续训练 |
| **RewardModel** | 打分 | 规则/standalone-RM 可与 rollout 重叠；colocate-RM 阻塞 | 独立 GPU 池，HTTP 打分 |

### 「rollout/train/reward 谁能异步」对照

| 你的设置 | rollout 异步 | reward 与 rollout 重叠 | **train 异步** |
|---|---|---|---|
| 标准 trainer + 规则奖励 | ✅ | ✅（无 RM，`not use_rm`=True） | ❌ 仍串行 |
| 标准 trainer + GenRM colocate | ✅ | ❌（共卡阻塞） | ❌ |
| 标准 trainer + GenRM standalone | ✅ | ✅ | ❌ **train 仍同步** |
| **fully_async_policy** | ✅ | ✅（PR #6044 后支持 GenRM） | ✅ |

**关键认知**：
- 「standalone（`reward_model.enable_resource_pool`）」**只让"奖励模型"单独分卡**——它解决的是 ②（reward 与 rollout 重叠），**train 依然同步**。
- **要让 train 也异步**（和 rollout 解耦、各跑各的），必须上 **fully_async_policy**：`hybrid_engine=False` + rollout / train **各占独立 GPU 池**。这是比"reward standalone"更彻底的隔离。
- 规则奖励里"rollout 和 reward 重叠"和 colocate/standalone **无关**——因为规则奖励压根没有 reward model 要分卡，`not use_rm=True` 直接让 reward loop 重叠。但这仍在同步 trainer 内，**train 没异步**。

### issue #5949 / PR #6044：给 fully async 补上 GenRM

- **#5949（需求）**：fully_async_policy 之前写死 `use_rm=False`，**只支持规则奖励**，没法在全异步管线里用 GenRM（用户想自己托管裁判、不依赖外部 API）。
- **#6044（实现）**：`fully_async_rollouter` 里创建 `RewardLoopManager` + standalone GenRM——**GenRM 作为独立 vLLM server 占自己的 GPU，rollout 期间走 HTTP 打分**。于是 fully async 支持 **三方资源隔离：rollout / train / reward 各占独立卡**。（附带：加了校验，并禁止 `use_trainer_do_validate` 与 GenRM 同时用。）

### fully_async + GenRM 的三方隔离写法（`run_fully_async_policy_genrm.sh`）

入口是 `fully_async_main`，三组卡分别配置：
```bash
python3 -m verl.experimental.fully_async_policy.fully_async_main \
  actor_rollout_ref.hybrid_engine=False \              # ← rollout 与 train 分离（fully async 前提）
  rollout.nnodes=1  rollout.n_gpus_per_node=${n_gpus_rollout} \      # rollout 独立池
  trainer.nnodes=1  trainer.n_gpus_per_node=${n_gpus_training} \     # train 独立池
  reward.reward_model.enable=True \
  reward.reward_model.enable_resource_pool=True \
  reward.reward_model.nnodes=1 \
  reward.reward_model.n_gpus_per_node=${n_gpus_genrm} \              # genrm 独立池
  reward.custom_reward_function.path=... \                          # genrm 仍需 async 奖励函数
  async_training.staleness_threshold=0.5 \      # 允许多旧的样本（off-policy 程度）
  async_training.trigger_parameter_sync_step=4 \# 每训几轮同步一次权重给 rollouter
  async_training.partial_rollout=True           # 同步时 sleep/resume 半成品，省长尾等待
```
测试用例：3×H100 = 1 rollout + 1 train + 1 genrm，各占一张卡。注意这里多了 `rollout.nnodes / rollout.n_gpus_per_node` 这组**rollout 专属资源**配置（同步框架里没有，因为同步框架 rollout 和 train 共用 hybrid engine）。

> 代价照旧：fully async 是 **off-policy**，要靠 `staleness_threshold` 控新鲜度 + 重要性采样校正（TIS/MIS）。先确认 rollout 是瓶颈再上。
