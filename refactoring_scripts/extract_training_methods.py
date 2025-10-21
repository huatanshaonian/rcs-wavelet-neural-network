#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动提取训练方法到TrainingManager
"""

import re

# 要提取的方法及其大致行号范围
METHODS_TO_EXTRACT = [
    # 传统模型训练 (4个)
    ('_train_model', 1840, 2380),
    ('_training_finished', 2381, 2386),
    ('_set_random_seeds', 2387, 2446),
    ('_initialize_cuda_safely', 2447, 2520),

    # AutoEncoder训练控制 (2个)
    ('start_ae_training', 3992, 4067),
    ('stop_ae_training', 4068, 4072),

    # AutoEncoder旧版训练 (6个)
    ('_run_three_stage_training', 4458, 4486),
    ('_run_end_to_end_training', 4487, 4503),
    ('_train_autoencoder_stage1', 4504, 4624),
    ('_train_parameter_mapping_stage2', 4625, 4761),
    ('_train_end_to_end_stage3', 4762, 4889),
    ('_train_full_end_to_end', 4890, 5021),

    # AutoEncoder v2版训练 (6个)
    ('_run_three_stage_training_v2', 5022, 5079),
    ('_run_end_to_end_training_v2', 5080, 5097),
    ('_train_autoencoder_stage1_v2', 5098, 5277),
    ('_train_parameter_mapping_stage2_v2', 5278, 5461),
    ('_train_end_to_end_stage3_v2', 5462, 5644),
    ('_train_full_end_to_end_v2', 5645, 5781),

    # AutoEncoder辅助方法 (7个)
    ('_open_loss_config_for_ae', 5782, 5787),
    ('_create_ae_training_config', 5788, 5832),
    ('_create_ae_optimizer_and_scheduler', 5833, 5874),
    ('_create_ae_loss_function', 5875, 5887),
    ('_create_end_to_end_loss_function', 5888, 5908),
    ('_ae_step_scheduler', 5909, 5918),
    ('_ae_log_training_progress', 5919, 5924),
]

def read_gui_file():
    """读取gui.py文件"""
    with open('gui.py', 'r', encoding='utf-8') as f:
        return f.readlines()

def extract_method(lines, method_name, start_line, end_line):
    """提取指定范围的方法"""
    # 转换为0索引
    start_idx = start_line - 1
    end_idx = end_line

    method_lines = lines[start_idx:end_idx]

    # 将 self. 替换为 self.gui.
    processed_lines = []
    for line in method_lines:
        # 替换self.xxx为self.gui.xxx，但保留self.gui本身
        # 需要排除已经是self.gui的情况
        if 'self.gui.' not in line and 'self.gui' not in line:
            line = line.replace('self.', 'self.gui.')
        # 修正过度替换：self.gui.gui -> self.gui
        line = line.replace('self.gui.gui.', 'self.gui.')

        processed_lines.append(line)

    return ''.join(processed_lines)

def create_training_manager():
    """创建TrainingManager文件"""
    gui_lines = read_gui_file()

    # 文件头部
    header = '''"""
训练管理器
处理所有训练相关功能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import threading
from datetime import datetime
from tkinter import messagebox
import tkinter as tk


class TrainingManager:
    """训练管理器 - 负责所有训练相关功能"""

    def __init__(self, parent_gui):
        """
        初始化训练管理器

        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui

'''

    # 提取所有方法
    methods_code = []
    for method_name, start, end in METHODS_TO_EXTRACT:
        print(f"提取方法: {method_name} (行 {start}-{end})")
        method_code = extract_method(gui_lines, method_name, start, end)
        methods_code.append(method_code)

    # 组装完整文件
    full_content = header + ''.join(methods_code)

    # 写入文件
    with open('gui_managers/managers/training_manager.py', 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"\n✅ TrainingManager创建成功！")
    print(f"   文件: gui_managers/managers/training_manager.py")
    print(f"   方法数: {len(METHODS_TO_EXTRACT)}")
    print(f"   总行数: {len(full_content.splitlines())}")

if __name__ == '__main__':
    create_training_manager()
