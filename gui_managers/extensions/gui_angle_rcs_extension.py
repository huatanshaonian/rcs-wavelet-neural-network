#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GUI Angle-based RCS扩展模块

为现有GUI添加基于角度编码的RCS预测功能：
1. 单点RCS预测（类似NeRF架构）
2. 角度编码 + 参数调制
3. 端到端训练
4. 任意角度插值

集成到现有的RCSWaveletGUI中
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import torch
import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class AngleRCSExtension:
    """Angle-based RCS功能扩展类"""

    def __init__(self, main_gui):
        """
        初始化扩展模块

        Args:
            main_gui: 主GUI实例
        """
        self.main_gui = main_gui
        self.angle_rcs_system = None  # 存储训练好的系统
        self.training_history = None
        self.training_thread = None
        self.is_training = False
        self.trainer = None  # 训练器引用（用于停止训练）

        # 数据加载相关
        self.train_loader = None
        self.val_loader = None
        self.sampler = None
        self.rcs_data = None  # 原始RCS数据
        self.param_data = None  # 原始参数数据

        # 扩展变量
        self._init_extension_vars()

    def _init_extension_vars(self):
        """初始化扩展变量"""
        # 模型配置
        self.angle_rcs_L = tk.IntVar(value=16)  # 傅里叶频率数量
        self.angle_rcs_param_embed_dim = tk.IntVar(value=128)  # 参数嵌入维度
        self.angle_rcs_activation = tk.StringVar(value="sin")  # 激活函数
        self.angle_rcs_dropout = tk.DoubleVar(value=0.1)  # Dropout率

        # 训练配置
        self.angle_rcs_epochs = tk.IntVar(value=200)  # 训练轮数
        self.angle_rcs_batch_size = tk.IntVar(value=256)  # 批次大小
        self.angle_rcs_lr = tk.DoubleVar(value=1e-4)  # 学习率
        self.angle_rcs_optimizer = tk.StringVar(value="adam")  # 优化器
        self.angle_rcs_scheduler = tk.StringVar(value="cosine")  # 学习率调度器
        self.angle_rcs_patience = tk.IntVar(value=50)  # Early stopping patience
        self.angle_rcs_weight_decay = tk.DoubleVar(value=1e-5)  # 权重衰减

        # 学习率调度器详细配置（从AE复用）
        self.angle_rcs_min_lr = tk.DoubleVar(value=1e-6)  # 最小学习率
        self.angle_rcs_restart_period = tk.IntVar(value=50)  # CosineRestart重启周期
        self.angle_rcs_num_lr_stages = tk.IntVar(value=3)  # multi_stage阶段数
        self.angle_rcs_lr_decay_factor = tk.DoubleVar(value=0.1)  # multi_stage LR衰减因子

        # 数据配置
        self.angle_rcs_train_split = tk.DoubleVar(value=0.8)  # 训练集比例
        self.angle_rcs_use_subset = tk.BooleanVar(value=False)  # 是否使用子集
        self.angle_rcs_subset_size = tk.IntVar(value=300000)  # 子集大小
        self.angle_rcs_normalize_params = tk.BooleanVar(value=True)  # 是否标准化参数
        self.angle_rcs_preload_gpu = tk.BooleanVar(value=True)  # 是否预加载数据到GPU（默认开启）

    def extend_angle_rcs_tab(self):
        """扩展Angle-based RCS标签页"""
        # 检查是否已有angle_rcs_frame，如果没有则创建新标签页
        if not hasattr(self.main_gui, 'angle_rcs_frame'):
            self.main_gui.angle_rcs_frame = ttk.Frame(self.main_gui.notebook)
            self.main_gui.notebook.add(self.main_gui.angle_rcs_frame, text="Angle-based RCS")

        # 清除现有内容
        for widget in self.main_gui.angle_rcs_frame.winfo_children():
            widget.destroy()

        # 创建新的布局
        self._create_angle_rcs_layout(self.main_gui.angle_rcs_frame)

    def _create_angle_rcs_layout(self, parent):
        """创建Angle-based RCS布局（参照autoencoder结构）"""
        # 主容器：左右分栏
        main_container = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧面板：配置和控制（增加宽度以避免遮挡）
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)

        # 右侧面板：状态和结果
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)

        # 构建左侧面板
        self._create_left_panel(left_panel)

        # 构建右侧面板
        self._create_right_panel(right_panel)

    def _create_left_panel(self, parent):
        """创建左侧配置面板（两列布局）"""
        # 创建可滚动框架
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 创建两列容器
        columns_frame = ttk.Frame(scrollable_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左列
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 右列
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # === 左列配置组 ===

        # 1. 模型配置组
        model_group = ttk.LabelFrame(left_column, text="🏗️ 模型配置")
        model_group.pack(fill=tk.X, pady=(0, 10))

        model_frame = ttk.Frame(model_group)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        # 傅里叶频率数量
        ttk.Label(model_frame, text="傅里叶频率L:").grid(row=0, column=0, sticky="w")
        ttk.Entry(model_frame, textvariable=self.angle_rcs_L, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(model_frame, text="(8/16/32)").grid(row=0, column=2, sticky="w", padx=(5, 0))

        # 参数嵌入维度
        ttk.Label(model_frame, text="参数嵌入维度:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(model_frame, textvariable=self.angle_rcs_param_embed_dim, width=8).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(model_frame, text="(64/128/256)").grid(row=1, column=2, sticky="w", padx=(5, 0), pady=(5, 0))

        # 激活函数
        ttk.Label(model_frame, text="激活函数:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        activation_combo = ttk.Combobox(model_frame, textvariable=self.angle_rcs_activation,
                                       values=["sin", "relu", "gelu", "swish", "tanh", "mish"],
                                       state="readonly", width=12)
        activation_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(5, 0))

        # Dropout率
        ttk.Label(model_frame, text="Dropout率:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(model_frame, textvariable=self.angle_rcs_dropout, width=8).grid(row=3, column=1, sticky="w", pady=(5, 0))
        ttk.Label(model_frame, text="(0.1-0.2)").grid(row=3, column=2, sticky="w", padx=(5, 0), pady=(5, 0))

        # 2. 训练配置组
        training_group = ttk.LabelFrame(left_column, text="🎯 训练配置")
        training_group.pack(fill=tk.X, pady=(0, 10))

        training_frame = ttk.Frame(training_group)
        training_frame.pack(fill=tk.X, padx=5, pady=5)

        # Epochs
        ttk.Label(training_frame, text="Epochs:").grid(row=0, column=0, sticky="w")
        ttk.Entry(training_frame, textvariable=self.angle_rcs_epochs, width=8).grid(row=0, column=1, sticky="w")

        # 批次大小
        ttk.Label(training_frame, text="批次大小:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Entry(training_frame, textvariable=self.angle_rcs_batch_size, width=8).grid(row=0, column=3, sticky="w")

        # 优化器
        ttk.Label(training_frame, text="优化器:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        optimizer_combo = ttk.Combobox(training_frame, textvariable=self.angle_rcs_optimizer,
                                      values=["adam", "adamw", "sgd", "lbfgs"],
                                      state="readonly", width=12)
        optimizer_combo.grid(row=1, column=1, sticky="w", pady=(5, 0))

        # 权重衰减
        ttk.Label(training_frame, text="权重衰减:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.angle_rcs_weight_decay, width=8).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # Patience
        ttk.Label(training_frame, text="Patience:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.angle_rcs_patience, width=8).grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Label(training_frame, text="(Early Stopping)").grid(row=2, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=(5, 0))

        # === 学习率调度配置组（从AE复用）===
        lr_group = ttk.LabelFrame(left_column, text="📈 学习率调度配置")
        lr_group.pack(fill=tk.X, pady=(0, 10))

        lr_frame = ttk.Frame(lr_group)
        lr_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：调度策略
        ttk.Label(lr_frame, text="调度策略:").grid(row=0, column=0, sticky="w")
        lr_scheduler_combo = ttk.Combobox(lr_frame, textvariable=self.angle_rcs_scheduler,
                                        values=['constant', 'cosine', 'cosine_restart', 'adaptive',
                                                'multi_stage', 'adaptive_multi_stage'],
                                        state="readonly", width=18)
        lr_scheduler_combo.grid(row=0, column=1, columnspan=3, sticky="ew")

        # 第二行：初始学习率和最小学习率
        ttk.Label(lr_frame, text="初始学习率:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.angle_rcs_lr, width=8).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(lr_frame, text="最小学习率:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.angle_rcs_min_lr, width=8).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # 第三行：重启周期和阶段数
        ttk.Label(lr_frame, text="重启周期:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.angle_rcs_restart_period, width=8).grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Label(lr_frame, text="阶段数:").grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.angle_rcs_num_lr_stages, width=8).grid(row=2, column=3, sticky="w", pady=(5, 0))

        # 第四行：学习率衰减因子（用于multi_stage策略）
        ttk.Label(lr_frame, text="LR衰减因子:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.angle_rcs_lr_decay_factor, width=8).grid(row=3, column=1, sticky="w", pady=(5, 0))
        ttk.Label(lr_frame, text="(multi_stage专用)", font=("", 8)).grid(row=3, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=(5, 0))

        # 第五行：multi_stage说明
        info_label = ttk.Label(lr_frame, text="💡 multi_stage: patience耗尽时降低LR，patience线性增长(30→60→90)",
                               font=("", 8), foreground="gray")
        info_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(5, 0))

        # === 右列配置组 ===

        # 3. 数据配置组
        data_group = ttk.LabelFrame(right_column, text="📊 数据配置")
        data_group.pack(fill=tk.X, pady=(0, 10))

        data_frame = ttk.Frame(data_group)
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        # 训练集比例
        ttk.Label(data_frame, text="训练集比例:").grid(row=0, column=0, sticky="w")
        ttk.Entry(data_frame, textvariable=self.angle_rcs_train_split, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(data_frame, text="(0.0-1.0)").grid(row=0, column=2, sticky="w", padx=(5, 0))

        # 参数标准化
        ttk.Checkbutton(data_frame, text="参数标准化",
                       variable=self.angle_rcs_normalize_params).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # GPU预加载
        ttk.Checkbutton(data_frame, text="预加载到GPU (推荐16G显存)",
                       variable=self.angle_rcs_preload_gpu).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 使用子集
        ttk.Checkbutton(data_frame, text="使用训练子集",
                       variable=self.angle_rcs_use_subset,
                       command=self._on_subset_toggle).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 子集大小
        ttk.Label(data_frame, text="子集大小:").grid(row=4, column=0, sticky="w", pady=(5, 0))
        self.subset_entry = ttk.Entry(data_frame, textvariable=self.angle_rcs_subset_size, width=12)
        self.subset_entry.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(5, 0))
        self.subset_entry.config(state='disabled')  # 初始禁用

        # 数据说明
        info_text = """
数据点数量：
• 200样本 × 91θ × 91φ × 3频率
• 总计: 4,968,600个数据点

采样策略：
• 全局混合采样（80-20划分）
• 支持子集训练（快速验证）
• GPU预加载: ~300MB显存，极速训练
"""
        ttk.Label(data_frame, text=info_text, justify=tk.LEFT,
                 font=('Courier', 8), foreground="gray").grid(row=5, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 4. 控制按钮组
        control_group = ttk.LabelFrame(right_column, text="🎮 控制")
        control_group.pack(fill=tk.X, pady=(0, 10))

        button_frame = ttk.Frame(control_group)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行按钮
        ttk.Button(button_frame, text="创建模型", command=self._create_model).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="加载数据", command=self._load_data).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        # 第二行按钮
        ttk.Button(button_frame, text="开始训练", command=self._start_training).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="停止训练", command=self._stop_training).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        # 第三行按钮
        ttk.Button(button_frame, text="急停", command=self._immediate_stop_training, style="Danger.TButton").grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="保存模型", command=self._save_model).grid(row=2, column=1, sticky="ew", padx=2, pady=2)

        # 第四行按钮
        ttk.Button(button_frame, text="加载模型", command=self._load_model).grid(row=3, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="评估模型", command=self._evaluate_model).grid(row=3, column=1, sticky="ew", padx=2, pady=2)

        # 第五行按钮
        ttk.Button(button_frame, text="可视化测试", command=self._visualize_test_data).grid(row=4, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        # 第六行按钮
        ttk.Button(button_frame, text="初始化Loss归一化", command=self._initialize_loss_normalization).grid(row=5, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        # 配置列权重
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

    def _create_right_panel(self, parent):
        """创建右侧状态和结果面板"""
        # 创建标签页管理器
        self.result_notebook = ttk.Notebook(parent)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)

        # 1. 训练日志标签页
        log_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(log_frame, text="训练日志")

        # 日志显示区域
        import tkinter.scrolledtext as scrolledtext
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 2. 训练曲线标签页
        curve_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(curve_frame, text="训练曲线")

        # 训练曲线显示区域
        self.curve_canvas_frame = ttk.Frame(curve_frame)
        self.curve_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 3. 可视化测试标签页
        viz_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(viz_frame, text="可视化测试")

        # 可视化显示区域
        self.viz_canvas_frame = ttk.Frame(viz_frame)
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _on_subset_toggle(self):
        """切换子集选项时的回调"""
        if self.angle_rcs_use_subset.get():
            self.subset_entry.config(state='normal')
        else:
            self.subset_entry.config(state='disabled')

    def _log(self, message):
        """输出日志到日志框和控制台"""
        # 输出到GUI日志框
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update()

        # 同时输出到控制台（方便排查问题）
        print(f"[Angle-RCS] {message}")

    def _create_model(self):
        """创建Angle-based RCS模型"""
        try:
            from angle_based_rcs.models.angle_rcs_network import AngleRCSNetwork

            # 获取频率数量
            if not hasattr(self.main_gui, 'rcs_data') or self.main_gui.rcs_data is None:
                messagebox.showwarning("警告", "请先加载数据！")
                return

            num_frequencies = self.main_gui.rcs_data.shape[-1]  # 从RCS数据形状获取频率数量

            # 创建模型
            model = AngleRCSNetwork(
                num_frequencies=num_frequencies,
                angle_L=self.angle_rcs_L.get(),
                param_dim=9,  # 固定9个设计参数
                param_embed_dim=self.angle_rcs_param_embed_dim.get(),
                activation=self.angle_rcs_activation.get(),
                dropout_rate=self.angle_rcs_dropout.get()
            )

            # 保存系统
            self.angle_rcs_system = {
                'model': model,
                'num_frequencies': num_frequencies
            }

            # 统计参数量
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self._log(f"✅ 模型创建成功！")
            self._log(f"  • 频率数量: {num_frequencies}")
            self._log(f"  • 傅里叶频率L: {self.angle_rcs_L.get()}")
            self._log(f"  • 参数嵌入维度: {self.angle_rcs_param_embed_dim.get()}")
            self._log(f"  • 激活函数: {self.angle_rcs_activation.get()}")
            self._log(f"  • Dropout率: {self.angle_rcs_dropout.get()}")
            self._log(f"  • 总参数量: {param_count:,}")
            self._log("")

        except Exception as e:
            error_msg = f"模型创建失败: {str(e)}"
            self._log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _load_data(self):
        """加载数据"""
        if self.angle_rcs_system is None:
            messagebox.showwarning("警告", "请先创建模型！")
            return

        try:
            from angle_based_rcs.data import create_dataloaders

            self._log("=" * 60)
            self._log("开始加载数据...")
            self._log("=" * 60)

            # 检查主GUI是否有数据
            if not hasattr(self.main_gui, 'rcs_data') or self.main_gui.rcs_data is None:
                messagebox.showwarning("警告", "请先在主界面加载RCS数据！")
                return

            if not hasattr(self.main_gui, 'param_data') or self.main_gui.param_data is None:
                messagebox.showwarning("警告", "请先在主界面加载设计参数！")
                return

            # 保存原始数据引用（用于可视化测试）
            self.rcs_data = self.main_gui.rcs_data
            self.param_data = self.main_gui.param_data

            # 获取配置
            num_frequencies = self.angle_rcs_system['num_frequencies']
            train_subset_size = self.angle_rcs_subset_size.get() if self.angle_rcs_use_subset.get() else None

            self._log(f"数据配置:")
            self._log(f"  • RCS数据形状: {self.rcs_data.shape}")
            self._log(f"  • 参数数据形状: {self.param_data.shape}")
            self._log(f"  • 频率数量: {num_frequencies}")
            self._log(f"  • 训练集比例: {self.angle_rcs_train_split.get()}")
            self._log(f"  • 标准化参数: {self.angle_rcs_normalize_params.get()}")
            self._log(f"  • GPU预加载: {self.angle_rcs_preload_gpu.get()}")
            if train_subset_size:
                self._log(f"  • 训练子集大小: {train_subset_size:,}")
            self._log("")

            # 创建DataLoader
            self.train_loader, self.val_loader, self.sampler = create_dataloaders(
                rcs_data=self.rcs_data,
                param_data=self.param_data,
                batch_size=self.angle_rcs_batch_size.get(),
                num_frequencies=num_frequencies,
                train_split=self.angle_rcs_train_split.get(),
                random_seed=42,
                train_subset_size=train_subset_size,
                normalize_params=self.angle_rcs_normalize_params.get(),
                num_workers=0,  # 固定为0（单进程）
                preload_to_gpu=self.angle_rcs_preload_gpu.get()
            )

            # 统计信息
            total_train = len(self.train_loader.dataset)
            total_val = len(self.val_loader.dataset)
            train_batches = len(self.train_loader)
            val_batches = len(self.val_loader)

            self._log(f"✅ 数据加载成功！")
            self._log(f"  • 训练样本数: {total_train:,}")
            self._log(f"  • 验证样本数: {total_val:,}")
            self._log(f"  • 训练批次数: {train_batches:,}")
            self._log(f"  • 验证批次数: {val_batches:,}")
            self._log("")

        except Exception as e:
            import traceback
            error_msg = f"数据加载失败: {str(e)}\n{traceback.format_exc()}"
            self._log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _start_training(self):
        """开始训练"""
        if self.angle_rcs_system is None:
            messagebox.showwarning("警告", "请先创建模型！")
            return

        if self.train_loader is None or self.val_loader is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        if self.is_training:
            messagebox.showwarning("警告", "训练正在进行中！")
            return

        # 在新线程中启动训练
        self.is_training = True
        self.training_thread = threading.Thread(target=self._train_model_thread, daemon=True)
        self.training_thread.start()

        self._log("🚀 训练已启动（后台线程）")

    def _train_model_thread(self):
        """训练线程（后台执行）"""
        try:
            from angle_based_rcs.training.angle_trainer import AngleRCSTrainer

            self._log("=" * 60)
            self._log("开始训练Angle-based RCS模型")
            self._log("=" * 60)

            # 使用已加载的DataLoader
            self._log(f"训练集: {len(self.train_loader)} batches, {len(self.train_loader.dataset):,} samples")
            self._log(f"验证集: {len(self.val_loader)} batches, {len(self.val_loader.dataset):,} samples")
            self._log("")

            # 创建训练器
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._log(f"使用设备: {device}")

            # 获取loss归一化系数（如果存在）
            loss_normalization_factor = self.angle_rcs_system.get('loss_normalization_factor', 1.0)

            if loss_normalization_factor != 1.0:
                self._log(f"🔧 使用Loss归一化系数: {loss_normalization_factor:.6f}")
            else:
                self._log("⚠️ 未初始化Loss归一化（使用默认系数1.0）")

            self.trainer = AngleRCSTrainer(
                model=self.angle_rcs_system['model'],
                device=device,
                checkpoint_dir='./angle_rcs_checkpoints',
                loss_normalization_factor=loss_normalization_factor
            )

            self._log("训练器配置:")
            self._log(f"  • 优化器: {self.angle_rcs_optimizer.get()}")
            self._log(f"  • 学习率: {self.angle_rcs_lr.get()}")
            self._log(f"  • 调度器: {self.angle_rcs_scheduler.get()}")
            self._log(f"  • Patience: {self.angle_rcs_patience.get()}")
            self._log("")

            # 训练（参数传给train方法）
            history = self.trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=self.angle_rcs_epochs.get(),
                lr=self.angle_rcs_lr.get(),
                optimizer_type=self.angle_rcs_optimizer.get(),
                scheduler_type=self.angle_rcs_scheduler.get(),
                patience=self.angle_rcs_patience.get(),
                weight_decay=self.angle_rcs_weight_decay.get(),
                log_callback=self._log,  # 传递日志回调
                # 学习率调度器详细参数（从AE复用）
                restart_period=self.angle_rcs_restart_period.get(),  # CosineRestart重启周期
                num_lr_stages=self.angle_rcs_num_lr_stages.get(),  # multi_stage阶段数
                lr_decay_factor=self.angle_rcs_lr_decay_factor.get()  # multi_stage LR衰减因子
            )

            # 保存训练历史
            self.training_history = history

            self._log("")
            self._log("=" * 60)
            self._log("✅ 训练完成！")
            self._log("=" * 60)

            # 绘制训练曲线
            self.main_gui.after(0, self._plot_training_curves)

        except KeyboardInterrupt:
            # 急停中断
            self._log("")
            self._log("=" * 60)
            self._log("🛑 训练被急停中断")
            self._log("=" * 60)
            self._log("提示：之前的最佳模型已保留")

            # 仍然绘制训练曲线（已完成的部分）
            if self.training_history is not None:
                self.main_gui.after(0, self._plot_training_curves)

        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            self._log(f"❌ {error_msg}")
            import traceback
            self._log(traceback.format_exc())

        finally:
            self.is_training = False

    def _stop_training(self):
        """停止训练（优雅停止）"""
        if not self.is_training:
            messagebox.showinfo("提示", "当前没有正在进行的训练")
            return

        if self.trainer is None:
            messagebox.showwarning("警告", "训练器未初始化")
            return

        # 请求用户确认
        result = messagebox.askyesno(
            "确认停止",
            "确定要停止训练吗？\n\n"
            "当前epoch将完成后停止，\n"
            "已完成的最佳模型会被保留。"
        )

        if result:
            self._log("⏹️ 用户请求停止训练...")
            self.trainer.stop()
            self._log("已发送停止信号，等待当前epoch完成...")

    def _immediate_stop_training(self):
        """急停训练（立即停止）"""
        if not self.is_training:
            messagebox.showinfo("提示", "当前没有正在进行的训练")
            return

        if self.trainer is None:
            messagebox.showwarning("警告", "训练器未初始化")
            return

        # 请求用户确认（警告更严厉）
        result = messagebox.askyesno(
            "⚠️ 确认急停",
            "确定要立即停止训练吗？\n\n"
            "⚠️ 警告：\n"
            "• 当前batch将立即中断\n"
            "• 当前epoch的训练进度会丢失\n"
            "• 之前的最佳模型会被保留\n\n"
            "建议使用'停止训练'等待epoch完成",
            icon='warning'
        )

        if result:
            self._log("🛑 用户请求急停训练...")
            self.trainer.immediate_stop()
            self._log("已发送急停信号，训练将立即中断...")

    def _save_model(self):
        """保存模型"""
        if self.angle_rcs_system is None:
            messagebox.showwarning("警告", "没有可保存的模型！")
            return

        try:
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                title="保存模型",
                defaultextension=".pth",
                filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
            )

            if not file_path:
                return

            # 准备保存内容
            save_dict = {
                'model_state_dict': self.angle_rcs_system['model'].state_dict(),
                'model_config': {
                    'num_frequencies': self.angle_rcs_system['num_frequencies'],
                    'angle_L': self.angle_rcs_L.get(),
                    'param_dim': 9,
                    'param_embed_dim': self.angle_rcs_param_embed_dim.get(),
                    'activation': self.angle_rcs_activation.get(),
                    'dropout_rate': self.angle_rcs_dropout.get()
                },
                'training_history': self.training_history
            }

            # 保存
            torch.save(save_dict, file_path)

            self._log(f"✅ 模型已保存到: {file_path}")
            messagebox.showinfo("成功", f"模型已保存到:\n{file_path}")

        except Exception as e:
            error_msg = f"保存失败: {str(e)}"
            self._log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _load_model(self):
        """加载模型"""
        try:
            from angle_based_rcs.models.angle_rcs_network import AngleRCSNetwork

            # 选择模型文件
            file_path = filedialog.askopenfilename(
                title="加载模型",
                filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
            )

            if not file_path:
                return

            # 加载
            checkpoint = torch.load(file_path, map_location='cpu')

            # 创建模型
            config = checkpoint['model_config']
            model = AngleRCSNetwork(**config)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 保存系统
            self.angle_rcs_system = {
                'model': model,
                'num_frequencies': config['num_frequencies']
            }

            # 恢复训练历史
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']

            # 更新GUI配置
            self.angle_rcs_L.set(config['angle_L'])
            self.angle_rcs_param_embed_dim.set(config['param_embed_dim'])
            self.angle_rcs_activation.set(config['activation'])
            self.angle_rcs_dropout.set(config['dropout_rate'])

            self._log(f"✅ 模型已加载: {file_path}")
            self._log(f"  • 配置: L={config['angle_L']}, embed_dim={config['param_embed_dim']}")
            messagebox.showinfo("成功", f"模型已加载:\n{file_path}")

        except Exception as e:
            error_msg = f"加载失败: {str(e)}"
            self._log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _evaluate_model(self):
        """评估模型"""
        if self.angle_rcs_system is None:
            messagebox.showwarning("警告", "请先创建或加载模型！")
            return

        # TODO: 实现评估逻辑
        messagebox.showinfo("提示", "评估功能待实现")

    def _initialize_loss_normalization(self):
        """初始化Loss归一化系数（使第一个epoch的loss归一化为1.0）"""
        try:
            # 检查前置条件
            if self.angle_rcs_system is None:
                messagebox.showwarning("警告", "请先创建模型！")
                return

            if self.train_loader is None:
                messagebox.showwarning("警告", "请先加载数据！")
                return

            self._log("=" * 60)
            self._log("🔧 开始初始化Loss归一化...")
            self._log("=" * 60)

            import torch
            import torch.nn as nn

            # 获取模型和设备
            model = self.angle_rcs_system['model']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()

            # 创建损失函数（MSE）
            criterion = nn.MSELoss()

            # 计算第一个epoch的平均loss
            total_loss = 0.0
            total_samples = 0

            self._log("🔄 计算初始loss（在训练数据上）...")
            self._log(f"  • 设备: {device}")
            self._log(f"  • 训练批次数: {len(self.train_loader)}")

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.train_loader, 1):
                    # 提取数据
                    theta = batch['theta'].to(device)
                    phi = batch['phi'].to(device)
                    params = batch['params'].to(device)
                    freq_idx = batch['freq_idx'].to(device)
                    target_rcs = batch['target_rcs'].to(device)

                    batch_size = theta.size(0)

                    # 前向传播
                    rcs_pred = model(theta, phi, params, freq_idx).squeeze()

                    # 计算loss
                    loss = criterion(rcs_pred, target_rcs)

                    # sample-weighted累加
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

                    # 每100个batch打印进度
                    if batch_idx % 100 == 0:
                        self._log(f"  进度: {batch_idx}/{len(self.train_loader)} batches...")

            # 计算平均loss
            initial_loss = total_loss / total_samples

            # 计算归一化系数（使loss归一化为1.0）
            loss_normalization_factor = 1.0 / initial_loss

            # 保存到angle_rcs_system
            self.angle_rcs_system['loss_normalization_factor'] = loss_normalization_factor

            self._log("")
            self._log(f"✅ Loss归一化初始化完成！")
            self._log(f"  • 初始Loss: {initial_loss:.6f}")
            self._log(f"  • 归一化系数: {loss_normalization_factor:.6f}")
            self._log(f"  • 归一化后Loss: {initial_loss * loss_normalization_factor:.6f}")
            self._log("=" * 60)

            messagebox.showinfo("成功",
                f"Loss归一化初始化成功！\n\n"
                f"初始Loss: {initial_loss:.6f}\n"
                f"归一化系数: {loss_normalization_factor:.6f}\n"
                f"归一化后Loss: 1.0")

        except Exception as e:
            import traceback
            error_msg = f"初始化Loss归一化失败: {str(e)}\n{traceback.format_exc()}"
            self._log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _visualize_prediction(self):
        """可视化预测结果（已弃用，使用_visualize_test_data）"""
        messagebox.showinfo("提示", "请使用'可视化测试'按钮")

    def _visualize_test_data(self):
        """可视化测试数据（验证数据加载是否正确）"""
        if self.rcs_data is None or self.param_data is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        try:
            self._log("=" * 60)
            self._log("开始可视化测试数据（样本001）")
            self._log("=" * 60)

            # 提取001样本（sample_idx=0）的所有角度RCS数据
            sample_idx = 0
            rcs_sample = self.rcs_data[sample_idx]  # [91, 91, num_freq]
            num_frequencies = rcs_sample.shape[-1]

            self._log(f"样本索引: {sample_idx}")
            self._log(f"RCS数据形状: {rcs_sample.shape}")
            self._log(f"频率数量: {num_frequencies}")

            # 转换为dB坐标
            def to_db(rcs_linear):
                """将线性RCS转换为dB"""
                rcs_db = np.where(rcs_linear > 0, 10 * np.log10(rcs_linear), -100)
                return rcs_db

            # 清除旧图
            for widget in self.viz_canvas_frame.winfo_children():
                widget.destroy()

            # 创建图形：每个频率一个子图
            fig, axes = plt.subplots(1, num_frequencies, figsize=(5 * num_frequencies, 4))
            if num_frequencies == 1:
                axes = [axes]  # 统一为列表

            # 频率标签
            freq_labels = {
                0: '1.5 GHz',
                1: '3.0 GHz',
                2: '6.0 GHz'
            }

            # 角度范围
            theta_range = np.linspace(45, 135, 91)
            phi_range = np.linspace(-45, 45, 91)

            # 绘制每个频率的RCS热图
            for freq_idx in range(num_frequencies):
                ax = axes[freq_idx]

                # 提取当前频率的RCS数据并转换为dB
                rcs_freq = rcs_sample[:, :, freq_idx]  # [91, 91]
                rcs_db = to_db(rcs_freq)

                # 统计信息
                valid_mask = rcs_db > -100
                if valid_mask.sum() > 0:
                    min_val = rcs_db[valid_mask].min()
                    max_val = rcs_db[valid_mask].max()
                    mean_val = rcs_db[valid_mask].mean()
                else:
                    min_val = max_val = mean_val = -100

                self._log(f"  频率 {freq_labels.get(freq_idx, f'{freq_idx}')}:")
                self._log(f"    - 范围: [{min_val:.2f}, {max_val:.2f}] dB")
                self._log(f"    - 均值: {mean_val:.2f} dB")

                # 绘制热图
                im = ax.imshow(
                    rcs_db.T,  # 转置：theta为x轴，phi为y轴
                    extent=[theta_range[0], theta_range[-1], phi_range[0], phi_range[-1]],
                    origin='lower',
                    aspect='auto',
                    cmap='jet',
                    vmin=min_val,
                    vmax=max_val
                )

                ax.set_xlabel('θ (度)')
                ax.set_ylabel('φ (度)')
                ax.set_title(f'{freq_labels.get(freq_idx, f"Freq {freq_idx}")} - 样本001\n范围: [{min_val:.1f}, {max_val:.1f}] dB')
                ax.grid(True, alpha=0.3, linestyle='--')

                # 添加颜色条
                plt.colorbar(im, ax=ax, label='RCS (dB)')

            plt.tight_layout()

            # 嵌入到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self._log("")
            self._log("✅ 可视化完成！数据加载流程验证成功。")
            self._log("=" * 60)

            # 切换到可视化测试标签页
            self.result_notebook.select(2)  # 第3个标签页（索引2）

        except Exception as e:
            import traceback
            error_msg = f"可视化失败: {str(e)}\n{traceback.format_exc()}"
            self._log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _plot_training_curves(self):
        """绘制训练曲线"""
        if self.training_history is None:
            return

        try:
            # 清除旧图
            for widget in self.curve_canvas_frame.winfo_children():
                widget.destroy()

            # 创建图形
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # 绘制损失曲线
            axes[0].plot(self.training_history['train_loss'], label='Train Loss', linewidth=2)
            axes[0].plot(self.training_history['val_loss'], label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss (MSE)')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 绘制学习率曲线
            if 'learning_rates' in self.training_history:
                axes[1].plot(self.training_history['learning_rates'], linewidth=2, color='orange')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Learning Rate')
                axes[1].set_title('Learning Rate Schedule')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_yscale('log')

            plt.tight_layout()

            # 嵌入到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.curve_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            self._log(f"绘图失败: {str(e)}")
