#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建ReconstructionManager - 提取RCS重建方法
"""

# 重建方法：只有1个核心方法
METHODS = [
    ('_reconstruct_rcs', 2192, 2411),
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
重建管理器
处理RCS重建相关功能
"""

import torch
import numpy as np


class ReconstructionManager:
    """重建管理器 - 负责RCS重建功能"""

    def __init__(self, parent_gui):
        """
        初始化重建管理器

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

    with open('gui_managers/managers/reconstruction_manager.py', 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"\nReconstructionManager创建成功!")
    print(f"  文件: gui_managers/managers/reconstruction_manager.py")
    print(f"  方法数: {len(METHODS)}")
    print(f"  总行数: {len(full_content.splitlines())}")

if __name__ == '__main__':
    main()
