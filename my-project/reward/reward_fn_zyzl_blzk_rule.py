"""
zyzl-blzk（智能助理-病历质控）规则奖励函数 —— 适配 verl + Qwen3.5。

与 blzk 不同：模型输出格式非常多样（表格 / JSON / 箭头 / Markdown / 列表 等约 28 种），
因此沿用旧框架的「智能识别 + 多格式解析」逻辑：parse_answer 自动判别格式 → 解析成
统一的 [{质检项名称, 质检结论, 推理过程}, ...] 结构 → 按质检项名称对齐后比对结论。

适配说明（相对旧框架）：
  - 去掉 sentencepiece tokenizer：verl 传进来的 solution_str / ground_truth 已是文本；
  - thinking 处理对齐 Qwen3.5 原生 thinking：只看 `</think>`，不再处理 `<unused6>/<unused7>`；
  - 入口函数改为 verl dapo/naive reward manager 约定的签名。

核心解析 / 评分逻辑（parse_answer / parse_format_* / zyzl_blzk_content_reward_item）
完全沿用旧框架代码。
"""

import json
import re

# 旧框架部分 parse_format_* 用到 json_repair；环境没装也不影响导入。
try:
    import json_repair
except ImportError:
    json_repair = None


# ==================== 格式0: 列表格式（带推理过程和结论）====================
def parse_format_0(text):
    """
    格式: <质检项名称>: 【推理过程】推理过程【质检结论】合格/不合格
    """
    result = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.replace(':', '：').strip()
        if not line:
            continue
        name_end = line.find('：')
        if name_end == -1:
            continue
        # 保留尖括号
        name = line[:name_end].strip()

        if '【推理过程】' in line and '【质检结论】' in line:
            reasoning_start = line.find('【推理过程】') + len('【推理过程】')
            reasoning_end = line.find('【质检结论】')
            reasoning = line[reasoning_start:reasoning_end].strip()

            conclusion_start = line.find('【质检结论】') + len('【质检结论】')
            conclusion = line[conclusion_start:].strip()

            if conclusion in ['合格', '不合格']:
                result.append({
                    '质检项名称': name,
                    '推理过程': reasoning,
                    '质检结论': conclusion
                })
    return result


# ==================== 格式1: Markdown表格（3列）====================
def parse_format_1(text):
    """
    格式: | 质检项 | 质检结论 | 质检依据 | 或 | 质检项 | 质检结论 | 质检解释说明 |
    """
    result = []
    lines = text.strip().split('\n')

    in_table = False
    for line in lines:
        line = line.strip()

        # 跳过空行
        if not line:
            continue

        # 支持多种表头交接
        if (line.replace(' ', '').startswith('|质检项|质检结论|质检依据|') or
                line.replace(' ', '').startswith('|质检项|质检结论|质检解释说明|')):
            in_table = True
            continue

        # 跳过分隔行
        if in_table and '---' in line:
            continue

        if in_table and line.startswith('|'):
            parts = line.split('|')
            # 清理单元格并过滤空值
            cells = [part.strip() for part in parts if part.strip()]

            # 确保有足够的列数（至少3列）
            if len(cells) >= 3:
                # 检查结论是否有效
                conclusion = cells[1]
                if conclusion in ['合格', '不合格']:
                    result.append({
                        '质检项名称': cells[0],
                        '质检结论': conclusion,
                        '推理过程': cells[2]
                    })
            else:
                # 如果列数不够，可能表格结束
                in_table = False

    return result


# ==================== 格式2: 箭头分隔格式 ====================
def parse_format_2(text):
    """
    格式: 质检项名称 -> 合格/不合格 -> 解释说明
    """
    result = []
    lines = text.strip().split('\n')
    for line in lines:
        parts = [part.strip() for part in line.split('->')]
        if len(parts) < 3:
            continue
        result.append({
            '质检项名称': parts[0],
            '质检结论': parts[1],
            '推理过程': parts[2]
        })
    return result


# ==================== 格式3: JSON格式 ====================
def parse_format_3(text):
    """
    格式: {"质检项名称": {"质控结论":"合格/不合格", "质控依据":""}}
    """
    try:
        res_tem = json.loads(text)

        # 如果是列表格式
        if isinstance(res_tem, list):
            return res_tem

        # 如果是字典格式
        res = []
        for k in res_tem:
            item_data = res_tem[k]
            if isinstance(item_data, dict):
                res.append({
                    '质检项名称': k,
                    '推理过程': item_data.get('质控依据', item_data.get('质检依据', '')),
                    '质检结论': item_data.get('质控结论', item_data.get('质检结论', ''))
                })
        return res
    except Exception as e:
        return []


# ============================================================================
# ↓↓↓ 以下为旧框架中被省略的 parse_format_* / 辅助解析函数占位 ↓↓↓
#     直接把你旧框架对应实现粘贴覆盖即可（函数名 / 签名保持不变）。
#     默认返回 []，未粘贴前 parse_answer 命中这些分支会判 0 分，不影响导入与运行。
# ============================================================================

def parse_answer_markdown_single(s):
    # TODO: 粘贴旧框架实现（### 推理过程 ... 单条 Markdown）
    return []


def parse_format_4(text):
    # TODO: 粘贴旧框架实现（### Markdown，带 **质检结论**/**质控结论**）
    return []


def parse_format_5(text):
    # TODO: 粘贴旧框架实现（简单列表，仅结论；带【】）
    return []


def parse_format_6(text):
    # TODO: 粘贴旧框架实现（<尖括号> 名称 + 结论）
    return []


def parse_format_7(text):
    # TODO: 粘贴旧框架实现（键值对：质检项名称: ...）
    return []


def parse_format_8(text):
    # TODO: 粘贴旧框架实现（### Markdown，无结论标记）
    return []


def parse_format_9(text):
    # TODO: 粘贴旧框架实现（2列表格：| 质检项 | 质检依据 |）
    return []


def parse_format_18(text):
    # TODO: 粘贴旧框架实现（### Markdown，带 **质检项编码**）
    return []


def parse_format_19(text):
    # TODO: 粘贴旧框架实现（4列表格，带编码：| 质检项编码 | ... |）
    return []


def parse_format_20(text):
    # TODO: 粘贴旧框架实现（键值对，含质检项编码）
    return []


def parse_format_21(text):
    # TODO: 粘贴旧框架实现（质检项编码: ... / 短字符串编码列表）
    return []


def parse_format_23(text):
    # TODO: 粘贴旧框架实现（质检项编码: ... 带解释说明）
    return []


def parse_format_24(text):
    # TODO: 粘贴旧框架实现（表格：| 质检项编码 | 质检依据 |）
    return []


def parse_format_27(text):
    # TODO: 粘贴旧框架实现（## 合格 / 不合格 分组格式）
    return []

# ============================================================================
# ↑↑↑ 占位结束 ↑↑↑
# ============================================================================


def parse_answer(s):
    """
    智能识别并解析各种格式的质检结果
    支持28种输出格式
    使用startswith开头判断确保不会被误判
    只提取结果，不自动补充质检结论
    """

    s = s.strip().strip("```markdown").strip("```json").strip("```").strip()
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    if not lines:
        return []
    if lines[0].startswith('### 推理过程'):
        return parse_answer_markdown_single(s)
    # 格式27: 分组格式（合格/不合格分开）- 检查开头
    if lines[0].startswith('## 合格'):
        return parse_format_27(s)

    # 格式3: JSON格式 - 检查开头
    first_line = lines[0].strip()
    if first_line.startswith('{'):
        if s.count('{') == 1:
            return json.loads(s)
        else:
            return parse_format_3(s)

    # 格式1, 9, 13, 17, 19, 24, 26: 表格格式 - 检查表格格式
    if first_line.startswith('|') or any(line.startswith('|') for line in lines[:3]):
        # 检查表头确定具体表格格式
        for line in lines[:5]:
            if line.startswith('|'):
                # 【修复点】调整顺序：将具体的模式放在前面
                if '| 质检项编码 | 质检依据 |' in line:
                    return parse_format_24(s)
                elif '| 质检项编码 |' in line:
                    return parse_format_19(s)  # 4列表格（带编码）
                elif ('| 质检项 | 质检结论 | 质检依据 |' in line or
                      '| 质检项 | 质检结论 | 质检解释说明 |' in line):
                    return parse_format_1(s)  # 3列表格
                elif '| 质检项 | 质检依据 |' in line:
                    return parse_format_9(s)

        # 默认根据列数判断
        if first_line.startswith('|'):
            cols = len([c for c in first_line.split('|')[1:-1] if c.strip()])
            if cols >= 4:
                return parse_format_19(s)
            elif cols == 3:
                return parse_format_1(s)
            else:
                return parse_format_9(s)

    # 格式2: 箭头分隔格式 - 检查开头
    if first_line.find('->') != -1 and len(first_line.split('->')) >= 3:
        return parse_format_2(s)

    # 格式4, 8, 12, 16, 18: Markdown格式 - 检查开头
    if first_line.startswith('###'):
        # 检查是否包含特定的标记
        if any('**质检项编码**' in line for line in lines[:10]):
            return parse_format_18(s)
        elif any('**质检结论**' in line or '**质控结论**' in line for line in lines[:10]):
            return parse_format_4(s)
        else:
            return parse_format_8(s)

    # 格式7, 11, 15, 20: 键值对格式 - 检查开头
    if first_line.startswith('质检项名称: '):
        if any('质检项编码' in line for line in lines[:5]):
            return parse_format_20(s)
        else:
            return parse_format_7(s)

    # 格式0: 带推理过程和结论的列表格式 - 检查内容特征
    if any('【推理过程】' in line and '【质检结论】' in line for line in lines[:5]):
        return parse_format_0(s)

    # 格式5: 简单列表（仅结论）- 检查开头和内容
    # 支持带尖括号和不带尖括号的格式
    if '】' in first_line and ('合格' in first_line or '不合格' in first_line):
        # 检查是否为带尖括号的"名称: 结论"格式
        if not first_line.startswith('|') and not first_line.startswith('#') and '->' not in first_line:
            return parse_format_5(s)
    elif first_line.startswith('<') and '>' in first_line and '】' in first_line:
        # 带尖括号且有结论的格式
        return parse_format_6(s)

    # 格式21-26: 编码相关格式 - 检查开头
    if first_line.startswith('质检项编码: '):
        if any('解释说明' in line for line in lines[:5]):
            return parse_format_23(s)
        else:
            return parse_format_21(s)

    if len(first_line) < 20 and not any(keyword in first_line for keyword in
                                        ['】', ':', '->', '|', '###', '##', '{', '[', '【', '】']):
        # 检查前几行是否都是短字符串（可能是编码）
        code_like_count = sum(1 for line in lines[:min(5, len(lines))]
                              if len(line) < 20 and not any(sep in line for sep in ['】', ':', '->', '|']))
        if code_like_count >= min(3, len(lines[:5])):
            return parse_format_21(s)

    return []


def zyzl_blzk_content_reward_item(answer, target):
    try:
        answer_list = parse_answer(answer)
        target_list = parse_answer(target)
    except Exception:
        return 0.0

    if isinstance(answer_list, dict) and isinstance(target_list, dict):
        answer_conclusion = answer_list.get('质检结论', '')
        target_conclusion = target_list.get('质检结论', '')
        return 1.0 if answer_conclusion == target_conclusion else 0.0
    else:
        answer_names = set(
            item.get('质检项名称', '')
            for item in answer_list
            if isinstance(item, dict) and item.get('质检项名称', '').strip()
        )
        target_names = set(
            item.get('质检项名称', '')
            for item in target_list
            if isinstance(item, dict) and item.get('质检项名称', '').strip()
        )

        all_names = answer_names | target_names
        denominator = len(all_names)

        if denominator == 0:
            return 0.0

        has_answer_conclusion = any(
            isinstance(item, dict) and '质检结论' in item
            for item in answer_list
        )
        has_target_conclusion = any(
            isinstance(item, dict) and '质检结论' in item
            for item in target_list
        )
        no_conclusion_field = not has_answer_conclusion and not has_target_conclusion

        if no_conclusion_field:
            numerator = len(answer_names & target_names)
        else:
            answer_dict = {
                item.get('质检项名称', ''): item.get('质检结论', '')
                for item in answer_list
                if isinstance(item, dict) and item.get('质检项名称', '').strip()
            }
            target_dict = {
                item.get('质检项名称', ''): item.get('质检结论', '')
                for item in target_list
                if isinstance(item, dict) and item.get('质检项名称', '').strip()
            }

            numerator = 0
            for name in all_names:
                if name in answer_dict and name in target_dict:
                    if answer_dict[name] == target_dict[name]:
                        numerator += 1

        # return numerator / denominator
        if numerator == denominator:
            return 1.0
        else:
            return 0.0


def _sanitize_answer(text: str) -> str:
    """Qwen3.5 输出清洗：只处理 `</think>`（对齐 reward_fn_blzk_rule）。

    Qwen3.5 原生 thinking 模式下 chat template 已在 prompt 末尾追加 `<think>`，
    模型实际只输出 `thinking...</think>答案`，没有起始 `<think>`，因此取最后一个
    `</think>` 之后的内容即为正式答案。markdown 围栏交给 parse_answer 内部处理。
    """
    s = str(text or "")
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[-1]
    return s.strip()


def compute_score_zyzl_blzk_rule(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):
    """
    verl 自定义奖励函数入口（同步版本）。
    函数签名遵循 dapo/naive reward manager 约定。

    注：zyzl 输出格式多达 28 种、且本文件部分 parse_format_* 仍是占位实现，
    解析结果不足以稳定还原一个可读的 pred，故不返回 pred（dispatcher 会补默认空串）。
    """
    answer = _sanitize_answer(solution_str)
    # ground_truth 与 extra_info["target"] 在 convert_data_to_verl_rl.py 里同源，
    # target 不含 thinking，直接交给 parse_answer（其内部会清洗围栏）。
    score = zyzl_blzk_content_reward_item(answer, str(ground_truth or ""))
    return {
        "score": float(score),
        "judge_reason": "ok" if score >= 1.0 else "mismatch",
    }
