# reward 相关问答整理（verl / PPO / RewardManager）

这份笔记整理了我们刚才的高频问题与回答，目标是让你后续能快速回忆：
- `reward_fn` 在 `verl` 里如何接入
- `naive` / `dapo` / `critic` / `non_tensor_batch` 的核心含义
- 为什么会出现全 0 / 全 1 奖励，以及怎么修

---

## 1. 对话问答整理（你问 + 我答）

### Q1：`reward_fn_blzk_rule.py` 有问题吗？返回字典里也有 `data_source`，会和原始数据冲突吗？

**A：**
- 一般不会造成实际问题，但有“同名覆盖”机制。
- `RewardManager` 会把你 `compute_score` 返回 dict 的字段收集到 `reward_extra_info`，后续 trainer 会 `update` 到 `batch.non_tensor_batch`。
- 如果 key 同名（如 `data_source`），会覆盖原字段；但你返回的是同一个值，通常等价。
- 更稳妥做法：不返回 `data_source`，或改名为 `reward_data_source`。

---

### Q2：仓库里有没有自定义规则奖励函数示例？RewardManager 怎么工作？

**A：有，且资料较全。**

**文档/示例入口：**
- `docs/preparation/reward_function.rst`（自定义 reward 函数签名与配置）
- `docs/advance/reward_loop.rst`（reward loop 架构）
- `tests/special_e2e/ppo_trainer/run_function_reward.sh`（可运行 e2e）

**主流程（简化）：**
1. rollout 生成 response tokens  
2. RewardManager decode response，取 `data_source/ground_truth/extra_info`  
3. 调你自定义 `compute_score(...)`  
4. 得到 `score`（float 或 dict），写入 `reward_tensor`（通常写在最后有效 token）  
5. 额外字段放 `reward_extra_info`，后续写回 `non_tensor_batch`  
6. trainer 用 reward 算 advantage，更新 actor/critic  

---

### Q3：按当前 `reward_fn_blzk_rule.py`，从 rollout 到 reward 写回的时序是怎样？`dapo` 起什么作用？

**A：**
- 你的 `reward_fn_blzk_rule.py` 负责“业务打分规则”（JSON 提取、key 校验、结论匹配）。
- `dapo` 负责“管道与后处理”：
  - 读取样本并调用你的 `compute_score`
  - 聚合 `reward_extra_info`
  - 写 `reward_tensor`
  - 可选叠加 overlong penalty（超长惩罚）

一句话：**业务评分在 `reward_fn`，调度与 reward shaping 在 `dapo`。**

---

### Q4：只有 GRPO 需要 rollout 吗？PPO 需要吗？DPO 呢？

**A：**
- GRPO：需要 rollout（on-policy RL）。
- PPO：也需要 rollout（on-policy RL）。
- 标准 DPO：通常不需要在线 rollout（离线偏好学习）。

---

### Q5：PPO 的 critic model 起什么作用？

**A：**
- critic 预测每个时刻的状态价值 `V(s_t)`，用于算 advantage（如 GAE）。
- 作用是降低策略梯度方差、稳定训练、提高样本效率。
- 不是替代业务 reward，而是作为“基线估计器”。

---

### Q6：`dapo` 和 `naive` 的区别？

**A：**
- `naive`：直接用你的原始分数。
- `dapo`：在原始分数基础上可加 overlong penalty。
- 若你正在验证 reward 函数正确性，通常先用 `naive` 更容易排错；稳定后再考虑 `dapo` 做长度约束。

---

### Q7：BLZK 任务如果用 0/1 奖励，出现全 0 / 全 1 怎么办？

**A：这是稀疏奖励常见问题。**

- 全 0：几乎无学习信号，训练易卡住。
- 全 1：区分度不足，更新变慢。

**建议：**
1. 用分级奖励替代硬 0/1（按 `judge_reason` 给 0.0/0.2/0.6/1.0 等）
2. 拆成“格式分 + 结论分”
3. 监控分布：`mean/std/p(score==0)/p(score==1)`
4. curriculum（先易后难）
5. 训练用软分，评估仍保留硬 `acc`

---

### Q8：critic 对每个 token 位置给 value 预测是什么意思？

**A：**
- 对 response 的每个前缀状态 `s_t=[prompt, y_1...y_t]`，critic 预测 `V(s_t)`。
- 这样才能构造 token-level advantage，与 token-level PPO 更新对齐。
- 直观上，critic 在回答“从当前前缀继续写，预计还能拿多少最终回报”。

---

### Q9：`non_tensor_batch` 是什么？为什么要它？`token_level_scores / token_level_rewards` 是什么？`reward_tensor[i, last]` 是什么？

**A：**
- `batch`：放 tensor（模型训练主路径数据，如 `prompts/responses/attention_mask`）。
- `non_tensor_batch`：放字符串/字典/对象（如 `data_source/ground_truth/extra_info/judge_reason`）。
- 为什么要 `non_tensor_batch`：reward 函数需要这些业务上下文；并且回传诊断字段也要保留。
- `token_level_scores`：reward manager 原始输出。
- `token_level_rewards`：训练实际用的奖励（可能叠加 KL 修正）。
- `reward_tensor[i, valid_response_last_token]`：把终局 reward 写在该样本最后有效 response token 位置（典型 terminal reward 设计）。

---

### Q10：`batch` 放 tensor 怎么理解？tensor 不就是数组吗？

**A：**
- 可以把 tensor 看成“用于深度学习计算的数组”。
- 与普通数组相比，tensor 支持：
  - GPU 运算
  - 自动求导
  - 高效并行算子
- 所以训练主路径数据放 `batch`；元数据放 `non_tensor_batch`。

---

## 2. 一页速记（考试前 30 秒版本）

- `reward_fn` 决定业务分，`RewardManager` 决定如何接入训练管道。  
- `naive` = 原始分；`dapo` = 原始分 + 可选超长惩罚。  
- PPO/GRPO 都需要 rollout；标准 DPO 通常不用。  
- critic 估计 `V(s_t)`，用于优势函数，主要作用是降方差稳训练。  
- `batch` 是 tensor 训练数据，`non_tensor_batch` 是业务元信息。  
- 终局奖励常写在最后有效 token：`reward_tensor[i, last_valid_token]`。  
- 0/1 稀疏奖励容易全 0/全 1，建议分级奖励与分布监控。  

---

## 3. 大厂面试官风格“拷打题” + 参考答案

> 建议你先自己答，再看参考答案。

### 问题 1：为什么 `compute_score` 返回 dict，而不是只返回 float？

**参考答案：**
- float 仅用于训练 reward；dict 可携带可观测性信息（`pred/target/reason/key_mode`）。
- 这些字段可以进入 `reward_extra_info`，用于验证分析、错误定位、分组统计。
- 工程上这让“训练目标”和“诊断信息”解耦。

---

### 问题 2：`dapo` 的 overlong penalty 为什么放在 manager 而不是放在 reward_fn 里？

**参考答案：**
- 长度约束更像通用策略约束，不是任务语义本身。
- 放 manager 可复用且统一配置，不污染业务评分逻辑。
- 便于 A/B：同一个 `reward_fn` 可在 `naive` 与 `dapo` 间切换。

---

### 问题 3：如果 reward 函数严格 key 全匹配，线上会有什么副作用？

**参考答案：**
- 模型稍微多输出字段就会判错，导致“本质正确但得 0 分”。
- 训练早期会放大奖励稀疏性，导致全 0 风险更高。
- 可改为“必须字段是子集”并保留格式惩罚，兼顾鲁棒与约束。

---

### 问题 4：为什么 terminal reward 只打在最后 token 还能训练前面 token？

**参考答案：**
- PPO 用 return/GAE 会把终局信号回传到前序时间步。
- critic 提供每步 `V(s_t)` 基线，计算每步 advantage。
- 所以前面 token 仍有梯度信号，不是“只有最后 token 学习”。

---

### 问题 5：全 0/全 1 奖励你如何在线检测并自动告警？

**参考答案：**
- 统计滑窗指标：`mean/std/p0/p1`。
- 规则示例：连续 K 次 `p0>0.95` 或 `p1>0.95` 且 `std<阈值` 触发告警。
- 告警后可自动切换采样难度或启用软分策略。

---

### 问题 6：`non_tensor_batch` 的数据为什么后续还会被转 numpy？

**参考答案：**
- 便于批处理拼接、日志存储、跨模块统一接口。
- 这些字段仍是“非模型主干训练张量”，转 numpy 主要是工程传递便利。

---

### 问题 7：你会如何设计 BLZK 的奖励分层以避免 reward hacking？

**参考答案：**
- 分离“格式合规”和“语义正确”两条线，避免模型只学格式。
- 对格式给上限（如 0.4），语义正确给主要权重（如 0.6）。
- 加入反作弊规则（空推理、模板复述、无效 JSON 重复）扣分。

---

### 问题 8：当 `reward_fn` 与 `data_source` 逻辑耦合时，如何防止跨任务污染？

**参考答案：**
- 通过 `data_source` 路由到不同子打分器；
- 每个子打分器独立测试集与指标；
- 在 `reward_extra_info` 记录路由信息，便于审计。

---

## 4. 你的自测清单（建议打勾）

- [ ] 我能口述从 rollout 到 reward 写回的 6 个步骤。  
- [ ] 我能解释 `naive` 和 `dapo` 的边界责任。  
- [ ] 我能说明为什么 `critic` 需要 token-level value。  
- [ ] 我能解释 `batch` 与 `non_tensor_batch` 的数据分层。  
- [ ] 我能提出至少 2 种避免全 0/全 1 的具体方案。  
- [ ] 我能说清 `token_level_scores` 与 `token_level_rewards` 的差别。  

---

如果你愿意，我下一版可以再补一节：  
**“你当前 `reward_fn_blzk_rule.py` 的改进优先级（P0/P1/P2）”**，直接按可落地改动清单给你。  