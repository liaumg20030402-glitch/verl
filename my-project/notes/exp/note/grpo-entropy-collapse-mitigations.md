# GRPO 熵坍塌的非 KL / Entropy Loss 处理办法

这篇只讨论你已经知道的 `kl_loss` 和 `entropy_loss` 之外的办法。核心思路是：熵坍塌通常不是单个 loss 系数能完全解决的问题，而是 rollout 多样性、group reward 方差、reward 噪声、PPO 更新幅度和数据难度共同造成的。

## 0. 先判断是不是真的 entropy collapse

不要只看一条 `actor/entropy` 曲线。建议同时看：

| 指标 | 异常信号 | 含义 |
|---|---|---|
| `actor/entropy` | 快速降到很低，并持续下降 | policy 变得过早确定 |
| 同 prompt 的 `rollout.n` 条回答 | 内容高度相同 | 组内探索消失 |
| `critic/advantages/std` 或 group reward std | 大量为 0 | GRPO 没有有效相对优势 |
| `response_length/clip_ratio` | 很高 | 大量样本被截断，reward 可能有噪声 |
| `actor/pg_clipfrac` / ratio 相关指标 | 长期很高 | policy update 过猛，PPO clip 在硬拦 |
| train reward vs val accuracy | train reward 升、val 不升或下降 | reward overfit / reward hacking |

如果只是 entropy 变低，但验证集稳定上升、rollout 仍有组内差异，不一定要强行拉高 entropy。真正危险的是“熵低 + 组内回答趋同 + advantage 信号消失”。

## 1. 优先级最高的办法

### 1.1 Clip-Higher：放宽正向探索 token 的 PPO 上界

DAPO 论文指出，普通 GRPO/PPO 的对称 clip 会限制低概率 token 被正向强化。对于本来概率很低、但拿到正 advantage 的探索 token，`1 + clip_ratio` 的上界太紧，会让它们很难被抬起来，采样空间越来越窄。

做法是把 PPO clip 上下界解耦：

```bash
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
```

直觉：

- `clip_ratio_low` 控制“抑制坏 token”的幅度，保持保守。
- `clip_ratio_high` 控制“鼓励好 token”的幅度，略微放开。
- 这不是 entropy bonus，但会让探索 token 更容易从低概率区被拉出来。

风险：

- `clip_ratio_high` 太大时，policy drift 会变快，KL / ratio / clipfrac 可能爆。
- 如果 reward 很脏，放大正向更新会更快学坏。

建议起点：`0.2 / 0.28`。如果熵仍快速塌，先别盲目加到很大，优先检查 reward 噪声和 group 方差。

### 1.2 Dynamic Sampling：过滤 reward 全相同的 group

GRPO 的 advantage 来自同一 prompt 下多条回答的 reward 均值和方差：

```text
A_i = (r_i - mean(r_1...r_G)) / std(r_1...r_G)
```

如果一个 prompt 的 G 条回答全对或全错，`std=0`，这个 group 没有有效学习信号。继续把这种 group 塞进 batch，会降低有效 batch size，让训练越来越依赖少量有方差样本。

做法：

- 同 prompt 采样 G 条。
- 如果 reward 全 0 或全 1，丢掉这个 group。
- 继续补采样，直到 batch 里有足够多“组内有差异”的 prompt。

这本质上是把训练集中“当前模型已经太简单 / 当前模型完全不会”的样本动态跳过。它比训练前固定难度过滤更适合 RL，因为模型能力一直在变。

注意：

- 这是算法侧的采样策略，不是单纯 reward manager 能完整解决的事。
- 你这个仓库里能看到 `FilterGroupsConfig` 定义，字段包括 `enable / metric / max_num_gen_batches`，但具体是否接入要看当前使用的 trainer 分支。
- 如果当前脚本没有真正启用动态过滤，至少要离线统计每个 step 的 group reward std 占比，避免大量无效 group。

### 1.3 增大 group size，而不是只加正则

`rollout.n` 太小会导致两个问题：

- 很多 prompt 刚好全 0 或全 1，advantage 直接没信号。
- 即使有正负样本，std 估计也很噪。

建议：

```bash
actor_rollout_ref.rollout.n=8   # 最低可用
actor_rollout_ref.rollout.n=16  # 更稳，成本更高
```

判断依据：

- 如果 `n=8` 时 group std 为 0 的比例很高，优先试 `n=16`。
- 如果显存/吞吐不允许，配合 Dynamic Sampling，比单纯加大 batch 更有效。

### 1.4 保持 rollout 采样足够开放

训练 rollout 不要用 greedy 或低温：

```bash
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1
actor_rollout_ref.rollout.do_sample=True
```

说明：

- `temperature < 1` 会让分布更尖，组内回答更像，容易让 GRPO 没有方差。
- `top_p < 1` / `top_k` 会截断尾部 token，降低探索。
- 验证集可以 deterministic，但训练 rollout 应优先保证组内多样性。

如果熵已经塌了，单纯把 temperature 调高通常救不回来，因为模型 logits 已经很尖。它更适合作为预防手段。

### 1.5 用 token-level loss，避免长回答的梯度被稀释

原始 GRPO 如果按 sequence 先平均 token loss，再平均 sample，会让长 response 的每个 token 权重变低。长 CoT 场景下，这会影响对推理模式、重复模式、坏模式的学习和抑制。

verl 默认配置里已经有：

```bash
actor_rollout_ref.actor.loss_agg_mode=token-mean
```

建议保持 `token-mean`，不要随便改成 `seq-mean-token-mean`。如果你怀疑长度相关问题，可以对比：

- `response_length/mean`
- `response_length/clip_ratio`
- `actor/entropy`
- val accuracy

健康状态不是越长越好，也不是 entropy 越高越好，而是长度、entropy、reward、val accuracy 同步改善。

### 1.6 Overlong Reward Shaping：不要让截断样本制造噪声

长 CoT 训练里，response 被 `max_response_length` 截断时，常见问题是：

- 本来推理方向对，但因为没来得及输出答案，被记成 0。
- 模型学到“少写”而不是“写对”。
- reward 在长度边界处硬跳变，增大方差。

DAPO 的做法是对接近最大长度的样本做软惩罚，或者过滤截断样本的 loss。

verl 里可以用 `dapo` reward manager 的 overlong buffer：

```bash
reward.reward_manager.name=dapo
+reward.reward_kwargs.overlong_buffer_cfg.enable=True
+reward.reward_kwargs.overlong_buffer_cfg.len=4096
+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
+reward.reward_kwargs.overlong_buffer_cfg.log=False
+reward.reward_kwargs.max_resp_len=${max_response_length}
```

经验：

- `overlong_buffer.len` 通常设成 `max_response_length` 的 1/4 左右。
- 如果 `response_length/clip_ratio` 很高，先处理 overlong，再谈 entropy。
- 如果模型被压得太短，降低 `penalty_factor` 或加大 `max_response_length`。

## 2. 数据和 reward 侧办法

### 2.1 做动态难度课程，而不是固定全量训练

熵坍塌经常出现在这两类数据上：

- 太简单：模型很快全答对，group 内无差异。
- 太难：模型长期全答错，只能从噪声里学。

处理方式：

- 用当前 policy 定期 rollout，估每个 prompt 的 pass rate。
- 优先训练 pass rate 在中间区间的 prompt，例如 `0.2 ~ 0.8`。
- 对 pass rate 长期为 0 的 prompt，先放到后期或加 SFT / hint / curriculum。
- 对 pass rate 长期为 1 的 prompt，降低采样权重。

这和 Dynamic Sampling 不冲突：前者是数据池层面的 curriculum，后者是 batch 构造层面的有效样本过滤。

### 2.2 降低 reward 噪声

脏 reward 会把 policy 推向少数偶然高分模式，熵很快变低。重点排查：

- 答案解析是否稳定，尤其是单位、格式、同义表达、JSON/正则边界。
- `EOS` / chat template token 是否污染 reward parser。
- 格式 reward 是否压过 correctness reward。
- 长答案是否更容易被 parser 判错。
- 是否存在某个固定模板能骗过 reward。

建议把 reward 拆成多个可观测字段：

```text
acc / format_score / final_answer_found / overlong / parse_error / repetition
```

不要只记录一个总分。熵坍塌时，通常能在这些字段里看到模型钻了哪个空子。

### 2.3 加 partial credit 或过程信号

纯 0/1 outcome reward 很容易导致：

- 大量 group std 为 0。
- 偶然正确样本 advantage 极大。
- 模型只强化最后答案附近的表面模式。

可选方案：

- 对格式、答案抽取、关键步骤分别给分，但控制权重，不能让 format 压过 correctness。
- 对数学/代码任务加单元测试通过率、子问题得分。
- 对医学/问答类任务，把“命中关键点数量”作为辅助 reward。
- 对明显重复、乱码、空泛模板加惩罚。

注意：partial credit 的目标是减少 reward 方差和噪声，不是把 reward 写得越来越复杂。每加一个 reward 分量，都要单独画分布。

### 2.4 保留少量 SFT / expert rehearsal

这不是 KL loss。做法是在 RL update 中混入少量高质量 SFT token loss，或周期性用高质量轨迹做 replay，防止模型语言分布和推理格式退化。

适用场景：

- RL 后模型开始固定模板化输出。
- answer parser 被固定话术骗过。
- 多样性下降，同时 val accuracy 不涨。

风险：

- SFT 比例太高会压住 RL 的探索收益。
- SFT 数据如果太短，会抑制长 CoT。

建议从很小比例开始，例如每几个 RL step 插一个小 SFT batch，或总 loss 里只给很小的 SFT 权重。

## 3. 优化器和更新节奏

### 3.1 降低每轮 policy update 强度

熵快速塌缩常见于 update 太猛：

- learning rate 过大。
- 每批 rollout 被重复训练太多 epoch。
- mini-batch 太小，梯度噪声大。
- `clip_ratio_high` 放太大。

优先尝试：

- 降 actor learning rate。
- 减少每个 rollout batch 的 PPO epoch / update 次数。
- 增大有效 train batch。
- 开启或收紧 grad norm clipping。
- 监控 `pg_clipfrac`，如果长期很高，说明 PPO clip 已经在大量截断更新。

### 3.2 使用 checkpoint 回滚和早停阈值

不要等完全 collapse 后再救。建议设硬阈值：

```text
如果 actor/entropy 连续 N 个 step 低于阈值
或同 prompt rollout 重复率高于阈值
或 val accuracy 连续下降
则回滚到上一个 checkpoint，降低 LR / 增大 n / 打开动态过滤 / 调整 reward。
```

熵坍塌后继续训练，通常只会把坏模式压得更实。

## 4. 可以进一步研究的算法替代

### 4.1 DAPO

DAPO 可以理解为 GRPO 的工程增强版，四个重点是：

1. Clip-Higher：提升探索 token 的正向更新空间。
2. Dynamic Sampling：过滤无 group 方差样本。
3. Token-Level Policy Gradient Loss：更适合长 CoT。
4. Overlong Reward Shaping：降低截断 reward 噪声。

如果当前问题是 GRPO 熵坍塌，DAPO 是最直接的第一候选。

### 4.2 GSPO

GSPO 把 importance ratio、clip 和优化粒度改到 sequence level，目标是提升大模型 RL 训练稳定性，尤其是 MoE 场景。它不只是调参，而是换了 policy optimization 形式。

适合考虑的情况：

- token-level ratio 指标波动很大。
- MoE 路由或 token 级更新不稳定。
- GRPO/DAPO 已经调过，仍然不稳。

### 4.3 OPEFO / entropy-flow 类方法

OPEFO 从 token-level entropy flow 角度解释 collapse：训练中 entropy-decreasing update 系统性压过 entropy-increasing update。它不是简单加 entropy bonus，而是按 token 对 entropy 变化的贡献自适应重缩放更新。

这个方向更偏研究实现，但思想有用：不要只看 batch 平均 entropy，要看哪些 token/update 在持续压低 entropy。

### 4.4 EP-GRPO / entropy-progress 类方法

EP-GRPO 关注 GRPO 的 token-level credit assignment 问题：不同 token 的信息价值不同，单纯 outcome reward 会把同一个 advantage 粗糙地摊到所有 token 上。它用 entropy-gated modulation、policy divergence 过程信号和 cumulative entropy mapping 来补 token-level 信号。

适合启发：

- 高 entropy 的关键决策 token 应该比模板 token 更值得学习。
- 只有最终 0/1 reward 时，可以从 policy 自身变化中挖过程信号。

## 5. 我建议的排查顺序

如果你现在已经看到熵坍塌，按这个顺序改：

1. 先确认训练 rollout 是 `temperature=1.0, top_p=1.0, top_k=-1, do_sample=True`，不要低温。
2. 统计 group reward std 为 0 的比例；高的话加 `rollout.n` 或做 Dynamic Sampling。
3. 打开 DAPO 的 overlong shaping，处理 `response_length/clip_ratio` 高的问题。
4. 用 Clip-Higher：`clip_ratio_low=0.2, clip_ratio_high=0.28`。
5. 保持 `loss_agg_mode=token-mean`。
6. 如果 `pg_clipfrac` 高或 KL 飙，降 LR / 减 update epoch / 增 batch，而不是继续加探索。
7. 拆 reward 日志，确认不是 parser、格式分、截断或重复模板在骗分。
8. 仍不稳时，再考虑 SFT rehearsal、GSPO 或 entropy-flow 类算法。

## 6. 参考资料

- DeepSeekMath 提出 GRPO，并说明它是 PPO 的 critic-free 变体：[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1 展示纯 RL 激发 reasoning 能力的路线：[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- DAPO 论文，明确讨论 naive GRPO 的 entropy collapse、reward noise、training instability，并提出四个工程技巧：[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- GSPO 用 sequence-level ratio / clipping / optimization 改善稳定性：[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)
- OPEFO 从 token-level entropy flow 解释和处理 entropy collapse：[arXiv:2605.11491](https://arxiv.org/abs/2605.11491)
- EP-GRPO 用 entropy-progress 和隐式过程信号改善 GRPO credit assignment：[arXiv:2605.04960](https://arxiv.org/abs/2605.04960)
