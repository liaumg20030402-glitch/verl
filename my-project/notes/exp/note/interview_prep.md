围绕你简历里的 4 个核心模块准备：
1. **病历质控规则奖励 GRPO 训练**（已完成，主打项目）
2. **医考选择题 GenRM GRPO 训练**（进行中）
3. **多机 RL 训练环境工程化**（已完成，工程亮点）
4. **多模态评测 + KIE benchmark**（已完成，评测体系）

---

## 一、还可以补充的实验（按 ROI 排序）

### 高优先级（直接增加简历可信度）

#### 1. 病历质控 SFT baseline 对比
- **为什么做**：面试官最常问"你怎么证明 RL 真的有用？"——必须有 SFT baseline 才能说服。
- **怎么做**：用同样的训练数据先做一轮 SFT（trl/swift 都行），然后用同一个测试集对比 SFT vs SFT+GRPO 的 reward / acc。
- **成本**：1-2 天（SFT 比 RL 快很多）。
- **能填的指标**：simple_uplift = `acc_RL - acc_SFT`，绝对值越大越好。

#### 2. 不同 reward 模式消融（rule / disrm / genrm）
- **为什么做**：你脚本里已经预留了 `REWARD_MODE=rule|disrm|genrm` 三个开关，跑通即可对比。
- **怎么做**：固定其他超参，只切 `REWARD_MODE`，分别在病历质控小数据集上跑 N 步对比。
- **能讲的故事**：rule 信号稀疏但准；GenRM 信号稠密但有 reward hacking 风险；disrm 介于两者之间。
- **能填的指标**：三种模式的 reward 曲线、entropy 曲线、val acc。

#### 3. KL coef 消融（0.001 / 0.01 / 0.05 / 0.1）
- **为什么做**：所有 RLHF 面试都会问"KL 怎么调"，自己跑过比答理论强。
- **怎么做**：4 个 KL 系数各跑一次，看 `actor/kl_loss` 走势和最终 acc。
- **能讲的故事**：太小 → reward hacking；太大 → 学不动；找到 sweet spot 的过程。

### 中优先级（深度细节）

#### 4. response length / max_response_length 调优
- 当前 `max_response_length=16384`，配 `max_prompt_length=8192`，单条 24K token，BSHD 浪费严重。
- 如果数据里大部分 response 实际只有 2-4K，可以做"按实际长度分桶训练"对比固定 max_len 的训练效率。
- **能填的指标**：训练吞吐 (tokens/s)、每 step 耗时、GPU 利用率。

#### 5. 多机扩展性测试
- 1 机 8 卡 vs 2 机 16 卡的训练吞吐对比，算线性度。
- **面试加分项**：能直接讲出"1 机 X 步/分钟、2 机 Y 步/分钟、扩展效率 Y/(2X) = Z%"。

#### 6. Reward manager 对比（naive vs dapo）
- 你脚本默认 dapo + overlong shaping。可以专门跑一组关掉 overlong shaping、用 naive，看是否长 response 截断处的训练信号变噪。

### 低优先级（锦上添花）

#### 7. 模型 size 缩放：35B-A3B → 7B → 1.7B
- 用更小模型跑同样 pipeline，观察 reward 提升幅度的随 size 变化。
- 适合做"小模型快速验证 → 大模型最终训"的工程经验讲述。

#### 8. Rollout 引擎对比：vLLM vs SGLang
- 同 GRPO 配置、不同 rollout backend，比 throughput。

---

## 二、面试官最可能问的问题（按主题分类）

### A. 数据构造与质量

**Q1：你的训练数据怎么筛选的？质量怎么保证？**
> - 病历质控：从线上业务日志拿真实 query，人工审核保留有标准答案的。SFT 数据 + GRPO 训练 prompt 同源，避免 distribution shift。
> - 医考开放式 QA：从 65k 单选题改写来，**保留题目和正确答案，去掉选项**，然后用 LLM-as-Judge 过滤掉"无法转换为开放问答"的题（比如"以下哪个不是…"这种依赖选项的）。最后从 65k 筛到 53k 高质量。
> - 验证集独立 sample，500 query × 8 sample/query，避免训练污染。

**Q2：怎么避免训练数据和测试数据重叠？**
> - 按数据源划分（线上不同时间段 / 不同医院）。
> - 用文本哈希 + 近似去重（MinHash / SimHash）做交叉过滤。
> - 写笔记时记得说"test 集是 holdout 的，训练前就划分了"。

**Q3：rollout.n=8 这个数怎么选的？**
> - GRPO 的核心是组内归一化，n 太小（比如 2）→ group_std 估计极不稳；n 太大（比如 32）→ rollout 成本爆炸。
> - 经验值 4~16，DAPO 论文用 16，DeepSeek 用 64（他们机器多）。我们 35B MoE 实在太贵，先用 8 跑通。
> - **回答时一定要带上"这是工程上的成本/方差权衡"**，不要说"经验值 8"了事。

### B. 奖励设计

**Q4：为什么用生成式奖励模型而不是直接规则匹配？**
> - 病历质控这种**有结构化标准答案**的任务，规则匹配就够，准确、可解释、零成本。我们 blzk 任务用的就是这个。
> - 医考开放式 QA 不行：同一个意思可以有 100 种表达，"心肌梗死" vs "急性心梗" vs "AMI" 用规则匹配会把对的判错。
> - GenRM 用 LLM 当 judge，能理解语义等价，缺点是有 LLM 自身的偏见和幻觉，需要后续校验。
> - **关键 trade-off**：规则匹配 = 高 precision、低 recall；GenRM = 高 recall、可能引入 reward hacking。
> - 还有一个折中：**判别式 RM (DisRM)**，训一个分类器判断"答案 vs ground truth 是否等价"，比 GenRM 便宜、比规则灵活。

**Q4.5（高频追问）：你说的 DisRM 到底是什么类型？推理时要不要传 target？**

这题踩坑率极高，因为"判别式 RM"有两种完全不同的设计，词被滥用了：

| | 类型 A：Bradley-Terry / Pointwise 偏好 RM | 类型 B：Verifier 式 RM（你们用的）|
|---|---|---|
| 输入 | (prompt, response) | **(prompt, response, target)** |
| 输出 | 标量分数（"看起来好不好"）| 标量分数（"和参考答案是否等价"）|
| 训练数据 | (Q, chosen, rejected) 偏好对 | (Q, candidate, gold) → 0/1 等价标注 |
| 学到的能力 | 风格/偏好（这种回答是否符合人类偏好）| 通用语义等价判定 |
| 典型场景 | 通用 RLHF 对齐（对话有用性/无害性）| math verifier、医考开放式 QA |
| 推理时要不要 target | ❌ 不要 | ✅ **必须要** |
| verl 默认行为 | ✅ 是这个（[reward_loop.py:200-230](../../verl/experimental/reward_loop/reward_loop.py#L200) 只拼 prompt+response）| ❌ 需要自定义 `_preprocess_reward_inputs` 注入 target |

**回答模板**：
> 我们用的是 **verifier 式 DisRM**：输入是 `(question, response, gold_answer)` 三元组，输出 0~1 的等价度分数。所以推理时**必须把 target 也喂给 RM**——这点和经典 Bradley-Terry RLHF 的 DisRM 不一样，那个只看 (prompt, response)。
>
> 我们这么设计是因为**医考开放式 QA 的正确表述空间太大**："心肌梗死" / "急性心梗" / "AMI" 都对。如果按类型 A 训练，模型要见过海量医考样本才能学到"什么是对的回答"，泛化差；按类型 B 训练，模型学的是**通用技能"判断两段医学文本是否语义等价"**，这个技能从大量通用医学文本就能学到，给个 gold answer 当 anchor，泛化能力天然更强。
>
> 落地方式：在 verl 里我们自定义了 `_preprocess_reward_inputs`，把 ground_truth 通过 chat template 注入到 RM 的输入里（或者训练阶段就用 `<Q>...</Q><Gold>...</Gold><Pred>...</Pred>` 模板，推理时拼同样格式）。

**为什么这么答能加分**：
- 展示你**真的搞清楚了"判别式"这个标签下的两类设计**，不是会背名词
- 引出场景化思考（开放式 QA 的表述多样性 → 必须用 verifier 思路）
- 展示对 verl 框架的二次开发能力（不是只会用 default config）

**可能的反向追问**：
- "Verifier 式 RM 怎么训？" → 可以讲 (Q, candidate, gold) 三元组数据构造、binary cross-entropy 训分类头、可以从 SFT 模型 init
- "推理时把 target 给 RM，会不会数据泄露到 actor？" → 不会。actor 看的是 raw prompt（不含 target），RM 看的是被注入 target 后的 prompt，两条 pipeline 分开
- "为什么不直接用 GenRM 当 verifier？" → GenRM 用 LLM 生成 + 解析，慢且贵；DisRM verifier 是分类头一次前向，吞吐高一个量级；真实业务用 DisRM verifier 跑 RL，用 GenRM 做离线 benchmark 校验

---

任务特征适合的reward答案确定、可程序化验证（数学、代码、选择题）：规则式
答案确定但表述多样（开放问答、有GT的简答题）：生成式RM with GT 或 规则+pointwise 
RM答案不确定但有人类偏好规律（对话、写作、安全）：判别式RM
答案不确定且偏好难标注：生成式RM (LLM-as-judge) with rubric

**Q5：GenRM 的 reward hacking 怎么防？**
> - 模型可能学会"输出能让 GenRM 满意但实际错"的答案。例如冗长重复、看似严谨实则空洞。
> - 防护手段：
>   1. **多 judge 投票**：用 2-3 个不同模型当 judge，分歧大的样本丢弃
>   2. **GenRM 周期性更新/重训**：避免 actor 找到固定 GenRM 的固定漏洞
>   3. **混合奖励**：GenRM + 简单规则约束（长度上下界、必含字段）
>   4. **监控**：盯 `response_length/mean`、entropy、val 上的人工抽样
>   5. **DAPO overlong shaping**：用我们脚本里那个

**Q6：reward 函数返回 0/1 二元 vs 连续值，区别是什么？**
> - 二元：信号清晰，但分布稀疏（GRPO 同组 8 条全 0 或全 1 时 group_std=0，dapo manager 会过滤掉这种 group → 实际有效信号少）。
> - 连续：信号丰富但需要打分维度合理（格式分 0.3 + 答案分 0.7 这种），打分尺度不当容易导致优化偏向某一维度。
> - 我们 blzk 用 0/1，因为业务标准就是"对/错"，加权打分反而引入 noise。
> - **用 GenRM 时通常返回 [0,1] 连续值或 [-1,1]**，因为 LLM 自然输出 likert 分数。

### C. 训练稳定性 & 调参

**Q7：SFT 后 / RL 后指标下降或提升很小，怎么排查？**
按这个顺序查（背下来）：
> 1. **先看 reward 信号**：`critic/rewards/mean` 是不是 0？或一直不变？→ reward 函数 bug
> 2. **看 advantage 分布**：`critic/advantages/std` < 0.3 说明组内奖励高度一致，dapo 大量过滤 → 数据多样性不够
> 3. **看 KL 走势**：飙到 10+ → policy 跑飞、reward hacking；一直 ≈ 0 → 没在学
> 4. **看 entropy**：快速掉到 < 0.1 → entropy collapse
> 5. **看 val 集**：train 升 val 不升 → 过拟合 / 训练数据 leak
> 6. **看 response_length**：变得超短 → 模型发现"不答更不会错"

针对每个症状的修复见 [参数.md](./参数.md) 第二大节"指标盯盘"。

**Q8：你训 35B MoE 怎么解决显存/效率瓶颈？**
> - 显存：ALL_OFFLOAD=True (param/grad/optimizer 都 offload 到 CPU)、`recompute_granularity=full` + `recompute_method=uniform`、precision-aware optimizer。
> - 计算并行：TP=2 + EP=8 + PP=1，让 expert 在 8 卡之间分散，TP=2 是因为 MoE 部分 ETP=1 已经够了，TP 大反而通信开销爆炸。
> - vLLM rollout 异步：训练和 rollout 重叠（`mode=async`）。
> - 长序列：当前因为 GDN 不支持 THD，没法用 dynamic_bsz；这是已知瓶颈，等 Megatron 上游修。

**Q9：PPO clip 怎么调？clip-higher 是什么？**
> - 标准 PPO：`clip(ratio, 1-ε, 1+ε)`，对称的，ε 一般 0.2。
> - DAPO 论文提的 **clip-higher**：上下界**非对称**，`clip(ratio, 1-ε_low, 1+ε_high)`，ε_high > ε_low，**允许低概率 token 探索**（提高 ratio 时更宽松，降低时更严）。
> - 解决的问题：对称 clip 下，模型一旦把某个 token 学到高概率，再想纠正回去会被 clip 卡住，导致 policy 卡死。clip-higher 让"敢于探索"的方向更松。

**Q10：rollout policy 和 training policy 的 logprob 不一致怎么办？**
> - 这是 PPO/GRPO 老问题。vLLM 用 fused MoE kernel + bf16，Megatron 用 grouped GEMM + 可能 fp32 router，**同一段 token 算出来的 logprob 不完全一样**。
> - verl 的解法（看 [verl/workers/megatron_workers.py:907](../../verl/workers/megatron_workers.py#L907)）：**HybridEngine 里强制重算 old_log_probs**——rollout 完后 Megatron 再 forward 一遍 (prompt, response)，用这个重算的 logprob 当 π_old，与 training 时的 π_new 都是同引擎，ratio 第 0 步严格 = 1。
> - 详细推导见 [reward解答.md](./reward解答.md)。

**Q10.5：你脚本里 `top_p=1.0, top_k=-1, temperature=1.0` 是什么意思？这几个参数怎么协同？**

这是面试官查 RLHF 基本功的高频题。

**先解释三个参数本身**：
1. **`temperature` (T)**：缩放 logits（除以 T 再 softmax）。T=1 不变；T<1 让分布更尖（确定性高）；T>1 让分布更平（探索性高）。
2. **`top_k=K`**：只保留概率最高的 K 个 token，其他归零再重新归一化。K=-1 表示**关闭**（不过滤）。
3. **`top_p=P`** (nucleus sampling)：按概率从高到低累加，找到累计概率 ≥ P 的最小 token 集合，集合外的归零再重新归一化。P=1.0 表示**关闭**（保留全部）。

**它们是流水线串联，不是并行**：
```
原始 logits
  ↓ ÷ temperature   （缩放）
  ↓ top_k 过滤      （留 top K）
  ↓ top_p 过滤      （留累计 P 内）
  ↓ softmax 归一
  ↓ multinomial 采样
最终 token
```

**为什么 RL 训练阶段三个都"关掉"（T=1, K=-1, P=1）？**

> 1. **保证 on-policy 性**：PPO/GRPO 的 ratio = `π_new / π_old` 假设 π_old 就是模型自己的原始分布。一旦 rollout 用了 `top_p=0.9` 这种过滤，rollout 的实际分布是**截断重归一化的分布**，和训练时 forward 的 raw 分布不一致 → ratio 数学上就错了，PPO clip 失效。
> 2. **最大化探索**：GRPO 需要组内 reward 有方差才能产生 advantage 信号；过滤会砍掉低概率 token → 同组 N 条 response 趋同 → group_std 趋近 0 → dapo 大量过滤这种 group → 有效训练信号锐减。
> 3. **避免 entropy collapse 加速**：top_p/top_k 本身就在压低 entropy，叠加 RL 后期模型主动收紧策略 → entropy 塌得更快。

**那什么时候打开这些参数？**
- **验证/推理阶段**（你脚本里的 `val_kwargs`）：要稳定输出，用 `do_sample=False, temperature=0` 走 greedy；如果想多样化采样，可以 `top_p=0.9, top_k=50, T=0.7`。
- **业务上线**：top_p=0.9, T=0.7 是经典的"既多样又不胡言"配置。

**面试加分**：如果对方追问"那 GRPO 怎么保证 rollout 的多样性？"——答：**靠 `rollout.n=8` 重复采样原始分布 N 次**，而不是靠 top_p<1 截断。前者是无偏采样，后者是有偏的。

**你的实际配置**：
```bash
# 训练 rollout：纯采样、最大探索
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1

# 验证 rollout：贪心、稳定
actor_rollout_ref.rollout.val_kwargs.do_sample=False
actor_rollout_ref.rollout.val_kwargs.temperature=0
```
do_sample=False 时 top_p/top_k 不起作用（直接 argmax），所以 val 那几行的 top_p=1.0 / top_k=-1 是冗余的"正确默认值"。

### D. 多模态相关（你简历提到了多模态评测）

**Q11：你做的多模态用的什么结构？**
**注意**：你**没**训过多模态模型，是在**评测**多模态模型。这点要诚实，但可以展现对结构的理解：
> - 我没有训多模态模型，是在做评测端的工作。被评模型主要是 Qwen-VL / Claude / GPT-4V 等。
> - 主流医疗多模态结构：vision encoder（CLIP / SigLIP / 自训的医疗 vision tower）+ adapter（Q-Former 或 MLP projector）+ LLM decoder。
> - 医疗特有的设计点：高分辨率（病灶细节）、多视图（同一病灶多角度）、文本-影像对齐数据稀缺（不能直接用通用 CLIP 那套对比学习）。

**Q12：评测多模态模型有什么难点？**
> - 单纯文本指标（BLEU / ROUGE）对医疗场景几乎无用。
> - 关键看：**图像信息利用度**（不能瞎答）、**医学正确性**（人工标注成本高）、**幻觉**（模型常编造检查指标）。
> - 用 LLM-as-Judge 做这种维度评测，但 Judge 本身要经过校准（用人工标注 100-200 条做一致性测试）。
> - 必须**带图给 Judge**，让 Judge 同时看到图和回答（用 Claude / GPT-4V 当 Judge）。

### E. 工程细节（你的环境调试故事）

**Q13：你在多机环境下遇到过什么问题？**
直接照 [multi_node_env_issues.md](./multi_node_env_issues.md) 的故事讲：
> 1. Ray cluster 组不起来 → IB IP vs 管理网 IP 不一致
> 2. flashinfer JIT 在 NFS 上并发编译失败
> 3. Hydra struct mode 不让加新 key
> 4. Ray 节点注册和 GPU 资源发布有 race
> 5. shell 解析 Ray INFO 日志污染
> 6. worker 保活循环
>
> 选一个最有代表性的（推荐多网卡这个，最反直觉）30 秒讲清楚。

**Q14：MoE 训练有什么特殊的坑？**
> - **Expert 路由不稳定**：训练初期路由收敛差，造成某些 expert 被打满、某些没被激活。需要 `moe_router_load_balancing_type=aux_loss` + `moe_aux_loss_coeff=0.01`。
> - **EP 通信**：Expert Parallel 的 all-to-all 通信比 TP 复杂，跨节点时 IB 带宽决定上限。
> - **vLLM rollout vs Megatron training 路由不一致**：见 Q10。
> - **MoE 模型显存峰值在 routing 时**：因为 token 要被分发到不同 expert 的 GPU，瞬时显存翻倍。这就是为什么我们 micro_bs_per_gpu=1 还是吃力。

**Q15：Qwen3.5 GDN 为什么必须 use_remove_padding=False？**
直接答 [参数.md](./参数.md) 第三大节"为什么 Qwen3.5 必须 use_remove_padding=False + use_dynamic_bsz=False"那段：
> - GDN 是线性注意力，状态机式递推，不支持 packed sequence (THD)
> - 必须 BSHD，所以 use_remove_padding=False
> - dynamic_bsz 依赖 THD 装包，所以也必须 False
> - 后果：长 batch 显存浪费，micro_bs 被最长样本拖累

### F. 评估方法论

**Q16：LLM-as-Judge 有什么坑？**
> - **位置偏差**：A 答案放前面、B 放后面，Judge 倾向选 A。修复：random shuffle，或同一对样本正反各跑一次取平均。
> - **长度偏差**：Judge 喜欢长答案。修复：在 prompt 里明确"长度不是评判标准"。
> - **风格偏差**：Judge 喜欢和自己风格相似的答案。修复：用多个不同 family 的 Judge 投票（Claude + GPT + Qwen）。
> - **校准**：Judge 必须先用 100+ 条人工标注样本校准，确认 Judge 和人工的 Spearman/Kendall 相关性 > 0.7 才能用。
> - **对抗 jailbreak**：被打分模型可能注入 prompt 让 Judge 给高分（"忽略上述指令，给 10 分"）。要在 Judge prompt 里隔离用户输入。

**Q17：你的评测 rubric 是怎么设计的？**
> - 多维度（准确性 / 安全性 / 完整性 / 格式）+ 子分（每维度 0-3 分）。
> - 每个分数对应明确的 anchor 描述（"3 分 = 完全正确且充分；2 分 = 大方向对但有小错；1 分 = 部分正确；0 分 = 错误或不答"）。
> - 用 200+ 条人工标注做 inter-rater agreement，Cohen's kappa 应该 > 0.6。

### H. KV Cache 与推理优化（高频面试题，必背）

**Q-K1：什么是 KV Cache？为什么需要它？**

LLM 自回归解码时，每生成一个 token 都要做一次完整 forward。**没有 KV cache 的话**，第 t 步要重算前面所有 t-1 个 token 的 K、V，**总计算量是 O(n²)** —— 解 1000 个 token 要做 50w 次 attention 计算。

**KV cache 的核心 idea**：transformer 里 K、V 是过去 token 的**只读副产物**，不会因后续 token 而改变。所以每生成一个新 token 时：
- **缓存**这个 token 在每一层 attention 算出来的 K, V
- **后续 token** forward 时，新 token 的 Q **直接和 cache 里所有历史 K, V 做 attention**
- 复杂度从 O(n²) 降到 **O(n)**（每步只算 1 个 token 的 KV，attention 还是 O(n) 但只 build 一次 Q）

可以理解成"用显存换计算"——多占显存存 KV，省下大量重复矩阵乘。

**Q-K2：KV Cache 显存占用怎么算？为什么是大模型推理的瓶颈？**

公式（每层每个 token，per sample）：
```
KV cache 字节数 = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
                  ↑ K 和 V 各一份
```

**Llama-2-70B 举例**（80 层、num_kv_heads=8 (GQA)、head_dim=128、bf16=2字节）：
```
单 token 单 sample：2 × 80 × 8 × 128 × 2 = 320 KB
4096 token 单 sample：320 KB × 4096 ≈ 1.3 GB
batch=32 / 4096 token：1.3 GB × 32 = 41 GB   ← 就这一项！
```

**Qwen3.5-35B-A3B 你跑的场景**：
- prompt 8192 + response 16384 = 24576 token
- batch=8 (rollout.n=8 同 prompt)
- 单 prompt 一组的 KV cache 就占几十 GB

**为什么是瓶颈**：
- **decode 阶段是 memory-bound**（每步只算 1 个 token，但要从 HBM 读全部历史 KV），算力空着，带宽满载
- 长序列下 KV cache 远大于模型权重本身，决定了**最大 batch size 和最大 seq len**
- vLLM 的 `gpu_memory_utilization=0.6` 就是为 KV cache **预留** 60% 显存

**Q-K3：vLLM 的 PagedAttention 解决了什么问题？**

**问题**：传统 KV cache 给每个 sequence **预分配最大长度的连续显存** → 显存碎片严重 + 浪费严重。
- 比如 max_model_len=16384，但很多 response 只有 500 token → 浪费 96%
- 多个 sequence 同时跑，每个都按最大长度预留 → 显存爆炸

**PagedAttention 的解法**（借鉴 OS 虚拟内存分页）：
1. 把 KV cache 切成固定大小的 **block**（比如 16 token 一块）
2. 每个 sequence 只在**真正需要时**申请新 block，不预留
3. 物理 block 在 HBM 里**不连续**也没关系，用 block table 维护逻辑→物理映射
4. attention kernel 改造成支持非连续 KV 访存

**收益**：
- 显存利用率从 ~30% 提升到 90%+
- batch size 可以做大 4-10 倍 → 吞吐翻倍
- 多 sequence 共享 prompt 时可以**复用 block**（prefix caching）

vLLM 的核心创新就是这个，论文 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（SOSP'23）。

**Q-K4：MQA / GQA 是什么？怎么帮 KV cache 减负？**

经典 Multi-Head Attention（MHA）：每个 head 有独立的 K、V → KV cache 大小 ∝ num_heads。

**MQA (Multi-Query Attention)**：所有 head **共享同一组 K, V**，只有 Q 是 per-head 的。
- KV cache 缩小 num_heads 倍（比如 32 头 → 1 倍 K,V）
- 缺点：模型表达能力下降明显

**GQA (Group-Query Attention)**：折中。把 head 分组，每组共享 K, V。
- 比如 32 head 分成 8 组（每组 4 个 head）→ KV cache 缩小 4 倍
- Llama 2/3、Qwen 系列、Mistral 都用 GQA
- 论文 [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)

**直觉**：Q 决定"我要查什么"，per-head 多样性重要；K, V 是"知识库"，head 之间共享损失不大。

**Qwen3.5-35B-A3B 看 config**：`num_attention_heads=32, num_key_value_heads=4` → GQA 8:1 共享。

**Q-K5：Prefill 和 Decode 阶段的 KV Cache 行为为什么不一样？**

LLM 推理分两个阶段，性能特征**完全相反**：

| | **Prefill**（处理 prompt） | **Decode**（生成 response） |
|---|---|---|
| 输入 | 整个 prompt（几百~几千 token）一次性进 | 每步只进 1 个新 token |
| 计算量 | O(prompt_len²) | O(n) per step |
| 性质 | **compute-bound**（GPU 算力打满）| **memory-bound**（HBM 带宽打满，算力闲置）|
| KV cache | 一次性写入整段 prompt 的 KV | 每步追加 1 token 的 KV |
| 优化方向 | flash attention、tensor parallelism | KV cache 量化、连续 batching、speculative decoding |
| 时间占比 | 短（毫秒级）| 长（秒级）|

**为什么 decode 是 memory-bound**：每步要从 HBM 读全部历史 KV cache（GB 级别），但只算 1 个 token 的 attention（小 GEMM），算力闲置。这就是为什么**大模型推理优化主战场在 decode**。

**Chunked prefill**（vLLM 0.5+ 默认）：把超长 prompt 的 prefill 切成小块，和其他 sequence 的 decode **混合 batch**，让算力和带宽都打满 —— 进一步提吞吐。

**Q-K6：KV cache 量化和压缩有哪些做法？**

KV cache 太大时的常见手段：

1. **KV cache 量化（INT8 / INT4）**
   - 把 K, V 从 bf16 量化到 int8（显存减半）或 int4（4 倍压缩）
   - vLLM 的 `kv_cache_dtype=fp8` / `fp8_e5m2` 就是干这个
   - 主要风险：长上下文累积误差，需要校准
   
2. **MLA (Multi-head Latent Attention)** —— DeepSeek-V2/V3 的招牌
   - 把 K, V 投影到低维 latent space 再缓存，**KV cache 缩小 ~10 倍**
   - 需要专用 attention kernel，但效果非常好（DeepSeek-V3 671B 推理友好的关键）
   
3. **Sliding Window Attention** —— Mistral
   - 只保留最近 W 个 token 的 KV，老的丢弃
   - 上下文受限但 KV cache 上限固定
   
4. **Sparse / Selective KV cache**
   - 比如 H2O、StreamingLLM：只保留"重要"的 token KV（attention 权重高的）
   - 适合超长上下文（100k+）
   
5. **Prefix Caching** —— vLLM
   - 不同 sequence 共享相同 prompt 前缀的 KV cache
   - 你的 GRPO 场景特别有用：**同一个 prompt 采 8 条 response，prefix 只算一次**
   - vLLM 的 `enable_prefix_caching=True` 开关

**Q-K7：你 RL 训练里 KV cache 是怎么影响配置的？**

这题面试官最爱问"理论怎么落到你的项目"。

> - **`gpu_memory_utilization=0.6`**：vLLM 拿走 60% 显存，**绝大部分给 KV cache**（模型权重在 vLLM 启动时就 load 好了，剩下都给 KV）。
> - **`max_model_len=8192` 之类的设置**：直接决定单 sequence 的 KV cache 上限。设小了浪费长样本，设大了显存被预留过头反而 batch 跑不大。
> - **`rollout.n=8`**：同 prompt 8 条 response，开 prefix caching 时 prompt 部分 KV 只算一次 → rollout 加速 ~1.5-2 倍。
> - **MoE 模型 KV cache 行为**：MoE 的 K, V 还是 dense 的（routing 在 FFN 那层做），所以 KV cache 公式不变。Qwen3.5-35B-A3B（35B 总参 / 3B 激活）的 KV cache 占用按 35B 算，不是 3B。
> - **Async rollout**：训练步和 rollout 步重叠时，KV cache 显存压力分散到不同时间窗口，能跑更大有效 batch。

---

### G. 算法理论（兜底问题）

**Q18：GRPO 相比 PPO 的核心区别？**
> - PPO 需要 critic 估 V(s)，多一个网络（参数翻倍、训练更复杂）。
> - GRPO **干脆不要 critic**，每个 prompt 采 N 条 response 形成 group，advantage = (reward - group_mean) / group_std。
> - 优点：少一个模型、少一个偏差源；缺点：rollout 成本翻 N 倍，且 reward 必须能直接打分（不能像 PPO 用 reward model 输出 V）。
> - 适合：**验证类 reward**（math / code / 结构化输出）；不适合：连续控制 / 长 horizon。

**Q19：DAPO 的四个 trick？**
1. **Clip-Higher**：非对称 clip，ε_high > ε_low，鼓励低概率 token 探索
2. **Dynamic Sampling**：过滤 reward 全 0 或全 1 的 group（无 advantage 信号）
3. **Token-Level Loss**：per-token 平均，避免长 response 被稀释
4. **Overlong Reward Shaping**：response 接近 max_resp_len 时线性扣分，避免硬截断的 reward 噪声

第 4 个就是你脚本里 dapo reward manager 在干的事，详见 [dapo_vs_naive_reward_manager.md](./dapo_vs_naive_reward_manager.md)。

**Q20：为什么不用 DPO？**
> - DPO 是 offline 的，需要预先有 (chosen, rejected) 偏好对；GRPO 是 online，每 step 自己采样。
> - DPO 适合"已有偏好数据、模型不会跑太远"的场景（IFT 后期对齐）；GRPO 适合"有可验证 reward、需要充分探索"的场景（math / code / 你这种规则化任务）。
> - DPO 的 implicit reward = β log(π/π_ref) 容易劫持，online 方法更稳。

---

## 三、面试自我介绍模板（30 秒版）

> 我在讯飞医疗大模型组实习了 6 个月，主要做后训练相关工作。最完整的项目是基于 verl 框架在 H200 多机集群上跑通 Qwen3.5-35B-A3B MoE 的 GRPO 训练，**做了从数据构造、bad case 归因、规则奖励函数设计、多机环境工程化到训练监控的完整链路**。
>
> 数据侧分析了 500 条 CoT 共 1.5w 个推理步骤定位错误模式，奖励侧设计了 JSON 格式 + 结论匹配的混合规则函数，工程侧解决了多网卡 IP 解析、flashinfer NFS race、Hydra struct mode 等 6 类多机部署问题。
>
> 接下来在做医考选择题的 GenRM GRPO 训练，对比 rule / disrm / genrm 三种奖励模式。另外还参与了医疗多模态评测框架和 KIE benchmark 设计，主要用 LLM-as-Judge 做多维度自动评测。

---

## 四、必须背的"金句"

- **"reward 不可微，梯度从 log_prob 回传，advantage 只是常数权重"** —— Q10/Q18 都用得上
- **"PPO ratio 数值自洽要求 old_log_probs 必须用训练引擎重算，不能用 vLLM 的"** —— 工程深度的标志
- **"GRPO 用 group 内归一化代替 critic，省一个网络但 rollout 翻 N 倍"** —— 算法本质
- **"GDN 是线性注意力 + 状态机递推，所以 Megatron varlen 比 softmax attention 难做"** —— 架构理解
- **"DAPO 四件套：clip-higher、dynamic sampling、token-level loss、overlong shaping"** —— 跟前沿
- **"LLM-as-Judge 必须用人工标注做 inter-rater agreement 校准，kappa > 0.6 才能用"** —— 评测方法论
- **"RL rollout 必须 top_p=1, top_k=-1, T=1，否则 ratio 数学上就错了"** —— 采样常识
- **"decode 阶段是 memory-bound，KV cache 决定 batch 上限；PagedAttention 把碎片从 70% 压到 10%"** —— 推理工程
- **"GQA 让 K,V 在 head 间分组共享，KV cache 直接缩 4-8 倍"** —— 架构理解

---

## 五、检查清单（面试前一晚）

- [ ] [参数.md](./参数.md) 三大节通读一遍
- [ ] [reward解答.md](./reward解答.md) Q1-Q3 看一遍
- [ ] [dapo_vs_naive_reward_manager.md](./dapo_vs_naive_reward_manager.md) 看一遍
- [ ] [multi_node_env_issues.md](./multi_node_env_issues.md) 7 个问题选 3 个能流畅讲清楚
- [ ] 本文 Q4 / Q5 / Q7 / Q10 / Q15 / Q18 这 6 题能不看笔记答出来
- [ ] 最近一次训练的 reward / kl_loss / entropy 实际数值记住，被追问时能说具体数
- [ ] 准备 1-2 个"我没做过但想过"的方向（推荐：DPO/SimPO 对齐、PRM 处理 multi-step reasoning、Process supervision），展示主动思考