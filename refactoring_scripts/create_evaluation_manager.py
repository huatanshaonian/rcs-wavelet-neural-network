#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建EvaluationManager - 提取5个评估方法
"""

# 评估方法列表及行号
METHODS = [
    ('_evaluate_traditional_model', 2131, 2191),
    # _reconstruct_rcs (2192-2411) 属于步骤5，跳过
    ('_evaluate_autoencoder_model', 2413, 2533),
    ('_update_evaluation_display', 2535, 2547),
    ('_display_autoencoder_results', 2549, 2575),
    ('_display_traditional_results', 2577, 2600),
]

def read_gui():
    with open('gui.py', 'r', encoding='utf-8') as f:
        return f.readlines()

def extract_method(lines, start, end):
    """提取方法并替换self为self.gui"""
    method_lines = lines[start-1:end]

    processed = []
    for line in method_lines:
        # 替换self.为self.gui.，但保留self.gui本身
        if 'self.gui.' not in line and 'self.gui' not in line:
            line = line.replace('self.', 'self.gui.')
        # 修正过度替换
        line = line.replace('self.gui.gui.', 'self.gui.')
        processed.append(line)

    return ''.join(processed)

def main():
    lines = read_gui()

    header = '''"""
评估管理器
处理所有模型评估相关功能
"""

import numpy as np
import torch
from tkinter import messagebox, filedialog


class EvaluationManager:
    """评估管理器 - 负责所有模型评估功能"""

    def __init__(self, parent_gui):
        """
        初始化评估管理器

        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui

'''

    methods_code = []
    for name, start, end in METHODS:
        print(f"提取方法: {name} (行 {start}-{end})")
        code = extract_method(lines, start, end)
        methods_code.append(code)

    full_content = header + ''.join(methods_code)

    with open('gui_managers/managers/evaluation_manager.py', 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"\nEvaluationManager创建成功!")
    print(f"  文件: gui_managers/managers/evaluation_manager.py")
    print(f"  方法数: {len(METHODS)}")
    print(f"  总行数: {len(full_content.splitlines())}")

if __name__ == '__main__':
    main()
