#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将TrainingManager集成到gui.py
"""

import re

# 要替换的方法列表
METHODS = [
    '_train_model',
    '_training_finished',
    '_set_random_seeds',
    '_initialize_cuda_safely',
    'start_ae_training',
    'stop_ae_training',
    '_run_three_stage_training',
    '_run_end_to_end_training',
    '_train_autoencoder_stage1',
    '_train_parameter_mapping_stage2',
    '_train_end_to_end_stage3',
    '_train_full_end_to_end',
    '_run_three_stage_training_v2',
    '_run_end_to_end_training_v2',
    '_train_autoencoder_stage1_v2',
    '_train_parameter_mapping_stage2_v2',
    '_train_end_to_end_stage3_v2',
    '_train_full_end_to_end_v2',
    '_open_loss_config_for_ae',
    '_create_ae_training_config',
    '_create_ae_optimizer_and_scheduler',
    '_create_ae_loss_function',
    '_create_end_to_end_loss_function',
    '_ae_step_scheduler',
    '_ae_log_training_progress',
]

def read_gui_file():
    """读取gui.py"""
    with open('gui.py', 'r', encoding='utf-8') as f:
        return f.read()

def update_imports(content):
    """更新导入语句"""
    # 查找现有的manager导入语句
    import_pattern = r'from gui_managers\.managers import StatisticsManager, VisualizationManager'
    replacement = 'from gui_managers.managers import StatisticsManager, VisualizationManager, TrainingManager'

    if import_pattern in content:
        content = content.replace(import_pattern, replacement)
        print("✓ 更新导入语句")
    else:
        print("⚠ 警告: 未找到预期的导入语句")

    return content

def add_manager_initialization(content):
    """添加TrainingManager初始化"""
    # 查找visualization_manager初始化的位置
    pattern = r'(self\.visualization_manager = VisualizationManager\(self\))'
    replacement = r'\1\n        self.training_manager = TrainingManager(self)'

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✓ 添加TrainingManager初始化")
    else:
        print("⚠ 警告: 未找到visualization_manager初始化位置")

    return content

def replace_method_with_delegation(content, method_name):
    """将方法替换为委托调用"""
    # 构建方法的正则表达式模式
    # 需要匹配从def开始到下一个def或文件末尾的所有内容

    # 方法签名模式
    signature_pattern = rf'    def {re.escape(method_name)}\([^)]*\):'

    # 查找方法开始位置
    match = re.search(signature_pattern, content)
    if not match:
        print(f"  ⚠ 未找到方法: {method_name}")
        return content

    method_start = match.start()

    # 找到方法的文档字符串（如果有）
    docstring_match = re.search(r'        """[^"]*"""', content[match.end():])

    # 查找方法结束位置（下一个同级别的def或类定义，或文件末尾）
    # 从方法开始后搜索
    rest_content = content[match.end():]

    # 查找下一个同级别（4空格缩进）的def
    next_method_match = re.search(r'\n    def ', rest_content)

    if next_method_match:
        method_end = match.end() + next_method_match.start()
    else:
        # 如果没有下一个方法，查找文件末尾或主函数
        main_match = re.search(r'\ndef main\(\):', rest_content)
        if main_match:
            method_end = match.end() + main_match.start()
        else:
            method_end = len(content)

    # 提取方法签名
    original_signature = content[method_start:match.end()]

    # 提取方法的文档字符串
    docstring = ''
    if docstring_match:
        docstring = '\n' + content[match.end():match.end() + docstring_match.end()]

    # 提取参数列表（不包括self）
    params_match = re.search(r'\(self(?:,\s*([^)]*))?\)', original_signature)
    if params_match and params_match.group(1):
        params = params_match.group(1).strip()
        delegation_call = f"return self.training_manager.{method_name}({params})"
    else:
        delegation_call = f"return self.training_manager.{method_name}()"

    # 构建新的方法体
    new_method = f"{original_signature}{docstring}\n        {delegation_call}\n"

    # 替换内容
    content = content[:method_start] + new_method + content[method_end:]

    print(f"  ✓ 替换方法: {method_name}")
    return content

def main():
    print("开始集成TrainingManager到gui.py...")

    # 读取文件
    content = read_gui_file()
    original_lines = len(content.splitlines())

    # 1. 更新导入
    content = update_imports(content)

    # 2. 添加初始化
    content = add_manager_initialization(content)

    # 3. 替换所有方法
    print("\n替换方法为委托调用:")
    for method in METHODS:
        content = replace_method_with_delegation(content, method)

    # 保存文件
    with open('gui.py', 'w', encoding='utf-8') as f:
        f.write(content)

    new_lines = len(content.splitlines())

    print(f"\n完成！")
    print(f"  原始行数: {original_lines}")
    print(f"  新行数: {new_lines}")
    print(f"  减少行数: {original_lines - new_lines} (-{100*(original_lines-new_lines)/original_lines:.1f}%)")

if __name__ == '__main__':
    main()
