#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
替换训练方法为委托调用
"""

import re

# 方法列表及其大致行号（用于定位）
METHODS_INFO = [
    ('_train_model', 1840),
    ('_training_finished', 2381),
    ('_set_random_seeds', 2387),
    ('_initialize_cuda_safely', 2447),
    ('start_ae_training', 3992),
    ('stop_ae_training', 4068),
    ('_run_three_stage_training', 4458),
    ('_run_end_to_end_training', 4487),
    ('_train_autoencoder_stage1', 4504),
    ('_train_parameter_mapping_stage2', 4625),
    ('_train_end_to_end_stage3', 4762),
    ('_train_full_end_to_end', 4890),
    ('_run_three_stage_training_v2', 5022),
    ('_run_end_to_end_training_v2', 5080),
    ('_train_autoencoder_stage1_v2', 5098),
    ('_train_parameter_mapping_stage2_v2', 5278),
    ('_train_end_to_end_stage3_v2', 5462),
    ('_train_full_end_to_end_v2', 5645),
    ('_open_loss_config_for_ae', 5782),
    ('_create_ae_training_config', 5788),
    ('_create_ae_optimizer_and_scheduler', 5833),
    ('_create_ae_loss_function', 5875),
    ('_create_end_to_end_loss_function', 5888),
    ('_ae_step_scheduler', 5909),
    ('_ae_log_training_progress', 5919),
]

def find_method_boundaries(lines, method_name, start_line_hint):
    """找到方法的开始和结束位置"""
    # 转换为0索引
    start_search = max(0, start_line_hint - 10)

    # 查找方法定义
    method_start = None
    for i in range(start_search, len(lines)):
        if f'    def {method_name}(' in lines[i]:
            method_start = i
            break

    if method_start is None:
        return None, None

    # 查找方法结束（下一个同级def或文件末尾）
    method_end = None
    for i in range(method_start + 1, len(lines)):
        line = lines[i]
        # 检查是否是新的同级方法（4空格缩进的def）
        if line.startswith('    def ') or line.startswith('def main('):
            method_end = i
            break

    if method_end is None:
        method_end = len(lines)

    return method_start, method_end

def extract_method_signature_and_params(line):
    """从方法定义行提取签名和参数"""
    # 提取参数（不包括self）
    match = re.search(r'\(self(?:,\s*([^)]*))?\):', line)
    if match and match.group(1):
        params = match.group(1).strip()
        return params
    return ''

def create_delegation(method_name, params):
    """创建委托调用代码"""
    if params:
        return f"        return self.training_manager.{method_name}({params})\n"
    else:
        return f"        return self.training_manager.{method_name}()\n"

def replace_method(lines, method_name, start_line_hint):
    """替换单个方法"""
    start, end = find_method_boundaries(lines, method_name, start_line_hint)

    if start is None:
        print(f"  [SKIP] {method_name} - not found")
        return lines, 0

    # 提取方法签名行
    signature_line = lines[start]

    # 提取文档字符串（如果有）
    docstring_lines = []
    i = start + 1
    if i < end and '"""' in lines[i]:
        # 找到文档字符串的结束
        docstring_lines.append(lines[i])
        i += 1
        while i < end and '"""' not in lines[i-1] or lines[i-1].count('"""') == 1:
            docstring_lines.append(lines[i])
            i += 1
            if '"""' in lines[i-1] and lines[i-1].count('"""') == 2:
                break

    # 提取参数
    params = extract_method_signature_and_params(signature_line)

    # 创建新方法体
    delegation = create_delegation(method_name, params)

    # 构建新方法
    new_method = [signature_line] + docstring_lines + [delegation, '\n']

    # 替换
    removed_lines = end - start
    new_lines = lines[:start] + new_method + lines[end:]

    print(f"  [OK] {method_name} ({removed_lines} -> {len(new_method)} lines)")

    return new_lines, removed_lines - len(new_method)

def main():
    print("Replacing training methods with delegations...")

    # 读取文件
    with open('gui.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    original_count = len(lines)
    total_removed = 0

    # 按逆序替换（从后往前，避免行号变化问题）
    for method_name, line_hint in reversed(METHODS_INFO):
        lines, removed = replace_method(lines, method_name, line_hint)
        total_removed += removed

    # 保存
    with open('gui.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    new_count = len(lines)

    print(f"\nDone!")
    print(f"  Original lines: {original_count}")
    print(f"  New lines: {new_count}")
    print(f"  Removed: {total_removed} ({100*total_removed/original_count:.1f}%)")

if __name__ == '__main__':
    main()
