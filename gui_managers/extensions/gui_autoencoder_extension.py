#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GUI AutoEncoder扩展模块
为现有GUI添加：
1. 双模式AutoEncoder支持（小波增强 vs 直接模式）
2. 模式对比分析功能
3. 小波变换可视化分析
4. 性能对比界面

集成到现有的RCSWaveletGUI中
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import torch

class AutoEncoderExtension:
    """AutoEncoder功能扩展类"""

    def __init__(self, main_gui):
        """
        初始化扩展模块

        Args:
            main_gui: 主GUI实例
        """
        self.main_gui = main_gui
        self.comparison_results = None
        self.wavelet_analysis_results = None

        # 扩展变量
        self._init_extension_vars()

    def _init_extension_vars(self):
        """初始化扩展变量"""
        # 模式选择
        self.main_gui.ae_mode = tk.StringVar(value="wavelet")

        # 数据预处理设置
        self.main_gui.ae_normalization_method = tk.StringVar(value="none")  # 标准化方法：none, zscore, minmax
        self.main_gui.ae_db_transform = tk.BooleanVar(value=False)  # 默认关闭dB变换

        # 向后兼容：保留 ae_normalize 变量（从 normalization_method 派生）
        self.main_gui.ae_normalize = tk.BooleanVar(value=False)  # 默认关闭标准化（与none对应）

        # RCS物理约束：强制非负
        self.main_gui.ae_enforce_nonnegative_rcs = tk.BooleanVar(value=False)  # 默认关闭RCS非负约束

        # 通道注意力设置
        self.main_gui.ae_use_channel_attention = tk.BooleanVar(value=False)  # 默认关闭通道注意力

        # 后处理设置（仅用于推理/可视化，不影响训练）
        self.main_gui.ae_postprocess_abs_db = tk.BooleanVar(value=False)  # 默认关闭后处理分贝转换

        # 对比分析设置
        self.comparison_batch_size = tk.IntVar(value=20)
        self.comparison_enable_visual = tk.BooleanVar(value=True)

        # 小波分析设置
        self.wavelet_analysis_wavelet = tk.StringVar(value="db4")
        self.wavelet_show_coeffs = tk.BooleanVar(value=True)
        self.wavelet_show_stats = tk.BooleanVar(value=True)

        # 双系统状态
        self.wavelet_system = None
        self.direct_system = None

    def extend_autoencoder_tab(self):
        """扩展现有的AutoEncoder标签页"""
        # 获取AutoEncoder框架
        autoencoder_frame = self.main_gui.autoencoder_frame

        # 清除现有内容并重新布局
        for widget in autoencoder_frame.winfo_children():
            widget.destroy()

        # 创建新的布局
        self._create_extended_autoencoder_layout(autoencoder_frame)

    def _create_extended_autoencoder_layout(self, parent):
        """创建扩展的AutoEncoder布局"""
        # 主容器
        main_container = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧面板：配置和控制
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)

        # 右侧面板：状态和结果
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)

        # 构建左侧面板
        self._create_left_panel(left_panel)

        # 构建右侧面板
        self._create_right_panel(right_panel)

    def _create_left_panel(self, parent):
        """创建左侧配置面板（两列布局）"""
        # 创建两列容器
        columns_frame = ttk.Frame(parent)
        columns_frame.pack(fill=tk.BOTH, expand=True)

        # 左列
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 右列
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # === 左列配置组 ===

        # 1. 模式选择组
        mode_group = ttk.LabelFrame(left_column, text="🔄 AutoEncoder模式")
        mode_group.pack(fill=tk.X, pady=(0, 10))

        mode_frame = ttk.Frame(mode_group)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        # 模式选择单选按钮
        ttk.Radiobutton(mode_frame, text="🌊 小波增强模式 (Wavelet)",
                       variable=self.main_gui.ae_mode, value="wavelet").pack(anchor=tk.W)
        ttk.Label(mode_frame, text="   • RCS → 小波变换 → AutoEncoder",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W)

        ttk.Radiobutton(mode_frame, text="🔄 直接模式 (Direct)",
                       variable=self.main_gui.ae_mode, value="direct").pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(mode_frame, text="   • RCS → 直接输入AutoEncoder",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W)

        ttk.Radiobutton(mode_frame, text="⚡ 可微分小波模式 (Differentiable Wavelet)",
                       variable=self.main_gui.ae_mode, value="differentiable_wavelet").pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(mode_frame, text="   • RCS → 可微分小波 → AutoEncoder → 逆小波 → RCS",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W)

        # 2. 模型架构配置组
        model_group = ttk.LabelFrame(left_column, text="🏗️ 模型架构")
        model_group.pack(fill=tk.X, pady=(0, 10))

        model_frame = ttk.Frame(model_group)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：隐空间维度和Dropout率
        ttk.Label(model_frame, text="隐空间维度:").grid(row=0, column=0, sticky="w")
        ttk.Entry(model_frame, textvariable=self.main_gui.ae_latent_dim, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(model_frame, text="Dropout率:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Entry(model_frame, textvariable=self.main_gui.ae_dropout_rate, width=8).grid(row=0, column=3, sticky="w")

        # 第二行：架构类型
        # CNN: 标准4层卷积，平衡速度与性能 | Enhanced_CNN: 多尺度+注意力，大感受野 | Deep_CNN: 双卷积块，最强表达 | MLP: 全连接，参数敏感性分析
        # Dual_Branch_CNN: 双分支CNN V2，正确对称架构（推荐） | Dual_Branch_MLP: 双分支MLP V2，正确对称架构（推荐）
        # Additive_Dual_Branch: 叠加式双分支CNN，高频Sin+低频Tanh | Additive_Dual_Branch_MLP: 叠加式双分支MLP
        ttk.Label(model_frame, text="架构类型:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        architecture_combo = ttk.Combobox(model_frame, textvariable=self.main_gui.ae_architecture_type,
                                         values=["CNN", "Enhanced_CNN", "Deep_CNN", "MLP",
                                                "Dual_Branch_CNN", "Dual_Branch_MLP",
                                                "Additive_Dual_Branch", "Additive_Dual_Branch_MLP"], state="readonly", width=25)
        architecture_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(5, 0))

        # 第三行：小波类型（仅Wavelet模式可用）
        ttk.Label(model_frame, text="小波类型:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.wavelet_combo = ttk.Combobox(model_frame, textvariable=self.main_gui.ae_wavelet_type,
                                         values=["db4", "db8", "haar", "bior2.2"], state="readonly", width=12)
        self.wavelet_combo.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(5, 0))

        # 第四行：激活函数类型（AutoEncoder）
        ttk.Label(model_frame, text="AE激活函数:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        activation_combo = ttk.Combobox(model_frame, textvariable=self.main_gui.ae_activation,
                                       values=["relu", "sin", "gelu", "swish", "tanh", "sigmoid", "mish", "elu",
                                              "leaky_relu", "prelu"], state="readonly", width=12)
        activation_combo.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(5, 0))

        # 第五行：参数映射器配置
        ttk.Label(model_frame, text="映射器激活:").grid(row=4, column=0, sticky="w", pady=(5, 0))
        mapper_activation_combo = ttk.Combobox(model_frame, textvariable=self.main_gui.ae_mapper_activation,
                                              values=["auto", "relu", "sin", "gelu", "swish", "tanh", "mish"],
                                              state="readonly", width=12)
        mapper_activation_combo.grid(row=4, column=1, columnspan=3, sticky="ew", pady=(5, 0))

        # 第六行：自适应层选项
        mapper_adaptive_cb = ttk.Checkbutton(model_frame, text="映射器使用自适应层",
                                            variable=self.main_gui.ae_mapper_use_adaptive)
        mapper_adaptive_cb.grid(row=5, column=0, columnspan=4, sticky="w", pady=(5, 0))

        # === Additive Dual-Branch专用控件 ===
        self.additive_frame = ttk.LabelFrame(model_frame, text="叠加双分支配置")
        self.additive_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        # 为了配合 _hide_additive_controls，必须创建所有引用
        
        # 行1：编码器 & 高频解码器
        self.additive_encoder_label = ttk.Label(self.additive_frame, text="编码器:")
        self.additive_encoder_label.grid(row=0, column=0, sticky="w")
        
        self.additive_encoder_combo = ttk.Combobox(self.additive_frame, textvariable=self.main_gui.ae_activation_encoder,
                                                   values=["relu", "sin", "gelu", "swish", "tanh", "sigmoid", "mish", "elu", "leaky_relu", "prelu"], 
                                                   state="readonly", width=7)
        self.additive_encoder_combo.grid(row=0, column=1, sticky="ew", padx=2)
        
        self.additive_high_label = ttk.Label(self.additive_frame, text="高频解码:")
        self.additive_high_label.grid(row=0, column=2, sticky="w", padx=(5,0))
        
        self.additive_high_combo = ttk.Combobox(self.additive_frame, textvariable=self.main_gui.ae_activation_high,
                                                values=["relu", "sin", "gelu", "swish", "tanh", "sigmoid", "mish", "elu", "leaky_relu", "prelu"], 
                                                state="readonly", width=7)
        self.additive_high_combo.grid(row=0, column=3, sticky="ew", padx=2)

        # 行2：低频解码器 & 可学习权重
        self.additive_smooth_label = ttk.Label(self.additive_frame, text="低频解码:")
        self.additive_smooth_label.grid(row=1, column=0, sticky="w")
        
        self.additive_smooth_combo = ttk.Combobox(self.additive_frame, textvariable=self.main_gui.ae_activation_smooth,
                                                  values=["relu", "sin", "gelu", "swish", "tanh", "sigmoid", "mish", "elu", "leaky_relu", "prelu"], 
                                                  state="readonly", width=7)
        self.additive_smooth_combo.grid(row=1, column=1, sticky="ew", padx=2)

        self.additive_learnable_cb = ttk.Checkbutton(self.additive_frame, text="可学习权重", 
                                                    variable=self.main_gui.ae_learnable_weights,
                                                    command=self._on_learnable_weights_change)
        self.additive_learnable_cb.grid(row=1, column=2, columnspan=2, sticky="w", padx=(5, 0))

        # 行3：固定权重配置
        # dummy label for compatibility
        self.additive_label = ttk.Label(self.additive_frame, text="") 
        
        self.additive_alpha_label = ttk.Label(self.additive_frame, text="固定权重:")
        self.additive_alpha_label.grid(row=2, column=0, sticky="w", pady=(2, 0))
        
        alpha_frame = ttk.Frame(self.additive_frame)
        alpha_frame.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(2, 0))
        
        ttk.Label(alpha_frame, text="α(高频):").pack(side=tk.LEFT)
        self.additive_alpha_high_entry = ttk.Entry(alpha_frame, textvariable=self.main_gui.ae_alpha_high, width=5)
        self.additive_alpha_high_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(alpha_frame, text="β(低频):").pack(side=tk.LEFT)
        self.additive_alpha_smooth_entry = ttk.Entry(alpha_frame, textvariable=self.main_gui.ae_alpha_smooth, width=5)
        self.additive_alpha_smooth_entry.pack(side=tk.LEFT)

        # 默认隐藏
        self._hide_additive_controls()

        # 绑定架构类型变化事件
        self.main_gui.ae_architecture_type.trace('w', self._on_architecture_change)

        # 绑定模式变化事件（根据模式启用/禁用小波设置）
        self.main_gui.ae_mode.trace('w', self._on_mode_change)
        # 绑定标准化选项变化事件（影响dB变换）
        self.main_gui.ae_normalize.trace('w', self._on_mode_change)

        # 3. 数据预处理配置组
        preprocess_group = ttk.LabelFrame(left_column, text="🔧 数据预处理")
        preprocess_group.pack(fill=tk.X, pady=(0, 10))

        preprocess_frame = ttk.Frame(preprocess_group)
        preprocess_frame.pack(fill=tk.X, padx=5, pady=5)

        # 标准化选项（下拉选择）
        norm_frame = ttk.Frame(preprocess_frame)
        norm_frame.pack(fill=tk.X, anchor=tk.W, pady=(0, 5))

        ttk.Label(norm_frame, text="📊 标准化方法:", font=self.main_gui.font_small).pack(side=tk.LEFT)
        norm_combo = ttk.Combobox(norm_frame,
                                 textvariable=self.main_gui.ae_normalization_method,
                                 values=["none", "zscore", "minmax"],
                                 state="readonly",
                                 width=12)
        norm_combo.pack(side=tk.LEFT, padx=(5, 0))

        # 标准化方法说明
        norm_help_frame = ttk.Frame(preprocess_frame)
        norm_help_frame.pack(fill=tk.X, anchor=tk.W, pady=(0, 5))
        ttk.Label(norm_help_frame, text="   • none: 不标准化",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W)
        ttk.Label(norm_help_frame, text="   • zscore: Z-score标准化 (均值0, 标准差1)",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W)
        ttk.Label(norm_help_frame, text="   • minmax: Min-Max标准化 [0, 1]",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W)

        # dB变换选项（手动控制）
        self.db_checkbox = ttk.Checkbutton(preprocess_frame, text="📊 dB变换 (10*log10)",
                       variable=self.main_gui.ae_db_transform)
        self.db_checkbox.pack(anchor=tk.W)
        ttk.Label(preprocess_frame, text="   • 将线性RCS转换为dB空间 (sign(x)*log10(abs(x)))",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W, pady=(0, 5))

        # RCS非负约束选项（物理约束）
        ttk.Checkbutton(preprocess_frame, text="🛡️ 强制RCS非负 (物理约束)",
                       variable=self.main_gui.ae_enforce_nonnegative_rcs).pack(anchor=tk.W)
        ttk.Label(preprocess_frame, text="   • 推荐开启：确保模型输出满足物理规律 (RCS >= 0)",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W, pady=(0, 5))

        # 后处理分贝转换选项（仅用于推理/可视化）
        self.postprocess_abs_db_checkbox = ttk.Checkbutton(
            preprocess_frame,
            text="🔄 后处理: 绝对值+分贝转换 (仅推理时)",
            variable=self.main_gui.ae_postprocess_abs_db,
            state='disabled'  # 初始禁用，加载模型时根据情况启用
        )
        self.postprocess_abs_db_checkbox.pack(anchor=tk.W)
        self.postprocess_help_label = ttk.Label(
            preprocess_frame,
            text="   • 仅在模型线性训练时可用，用于消除负值影响",
            font=self.main_gui.font_small,
            foreground="gray"
        )
        self.postprocess_help_label.pack(anchor=tk.W, pady=(0, 5))

        # 通道注意力选项
        ttk.Checkbutton(preprocess_frame, text="🔍 输入层通道注意力 (Channel Attention)",
                       variable=self.main_gui.ae_use_channel_attention).pack(anchor=tk.W)
        ttk.Label(preprocess_frame, text="   • 自适应学习通道重要性，对小波系数特别有效",
                 font=self.main_gui.font_small, foreground="gray").pack(anchor=tk.W, pady=(0, 5))

        # 警告提示
        ttk.Label(preprocess_frame, text="⚠️ 标准化强烈推荐开启",
                 font=self.main_gui.font_small, foreground="orange").pack(anchor=tk.W)

        # 4. 训练配置组
        training_group = ttk.LabelFrame(left_column, text="🎯 训练配置")
        training_group.pack(fill=tk.X, pady=(0, 10))

        training_frame = ttk.Frame(training_group)
        training_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：批次大小和阶段1训练轮数
        ttk.Label(training_frame, text="批次大小:").grid(row=0, column=0, sticky="w")
        ttk.Entry(training_frame, textvariable=self.main_gui.ae_batch_size, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(training_frame, text="阶段1(AE):").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Entry(training_frame, textvariable=self.main_gui.ae_epochs_stage1, width=8).grid(row=0, column=3, sticky="w")

        # 第二行：阶段2和阶段3训练轮数
        ttk.Label(training_frame, text="阶段2(映射):").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.main_gui.ae_epochs_stage2, width=8).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(training_frame, text="阶段3(E2E):").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.main_gui.ae_epochs_stage3, width=8).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # 第三行：联合训练轮数
        ttk.Label(training_frame, text="联合训练:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.main_gui.ae_epochs_joint, width=8).grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Label(training_frame, text="epochs").grid(row=2, column=2, sticky="w", padx=(5, 0), pady=(5, 0))

        # 5. 优化器配置组
        optimizer_group = ttk.LabelFrame(left_column, text="⚙️ 优化器配置")
        optimizer_group.pack(fill=tk.X, pady=(0, 10))

        optimizer_frame = ttk.Frame(optimizer_group)
        optimizer_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：优化器类型和权重衰减
        ttk.Label(optimizer_frame, text="优化器:").grid(row=0, column=0, sticky="w")
        optimizer_combo = ttk.Combobox(optimizer_frame, textvariable=self.main_gui.ae_optimizer_type,
                                      values=["adam", "adamw", "sgd", "lbfgs"], state="readonly", width=10)
        optimizer_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(optimizer_frame, text="权重衰减:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Entry(optimizer_frame, textvariable=self.main_gui.ae_weight_decay, width=8).grid(row=0, column=3, sticky="w")

        # 第二行：动量和验证集比例
        ttk.Label(optimizer_frame, text="动量(SGD):").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(optimizer_frame, textvariable=self.main_gui.ae_momentum, width=8).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(optimizer_frame, text="验证集比例:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(optimizer_frame, textvariable=self.main_gui.ae_validation_split, width=8).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # 参数说明（简短注释）
        ttk.Label(optimizer_frame, text="提示: 权重衰减=L2正则化，防止过拟合；验证集用于早停和模型选择",
                 font=self.main_gui.font_small, foreground="gray", wraplength=280).grid(row=2, column=0, columnspan=4, sticky="w", pady=(5, 0))

        # === 右列配置组 ===

        # 5. 学习率调度配置组
        lr_group = ttk.LabelFrame(right_column, text="📈 学习率调度")
        lr_group.pack(fill=tk.X, pady=(0, 10))

        lr_frame = ttk.Frame(lr_group)
        lr_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：调度策略选择
        # constant: 固定学习率 | cosine_restart: 余弦退火+周期重启 | cosine_simple: 简单余弦退火 | adaptive: 自适应调整(ReduceLROnPlateau)
        # multi_stage: 多阶段学习率衰减 | adaptive_multi_stage: 自适应多阶段学习率衰减
        ttk.Label(lr_frame, text="调度策略:").grid(row=0, column=0, sticky="w")
        lr_scheduler_combo = ttk.Combobox(lr_frame, textvariable=self.main_gui.ae_lr_scheduler,
                                        values=['constant', 'cosine_restart', 'cosine_simple', 'adaptive',
                                                'multi_stage', 'adaptive_multi_stage'],
                                        state="readonly", width=18)
        lr_scheduler_combo.grid(row=0, column=1, columnspan=3, sticky="ew")

        # 第二行：初始学习率和最小学习率
        ttk.Label(lr_frame, text="初始学习率:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.main_gui.ae_learning_rate, width=8).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(lr_frame, text="最小学习率:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.main_gui.ae_min_lr, width=8).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # 第三行：重启周期（用于cosine_restart策略）和多阶段参数
        ttk.Label(lr_frame, text="重启周期:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.main_gui.ae_restart_period, width=8).grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Label(lr_frame, text="阶段数:").grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.main_gui.ae_num_lr_stages, width=8).grid(row=2, column=3, sticky="w", pady=(5, 0))

        # 第四行：学习率衰减因子（用于multi_stage策略）
        ttk.Label(lr_frame, text="LR衰减因子:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(lr_frame, textvariable=self.main_gui.ae_lr_decay_factor, width=8).grid(row=3, column=1, sticky="w", pady=(5, 0))
        ttk.Label(lr_frame, text="(multi_stage专用)", font=("", 8)).grid(row=3, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=(5, 0))

        # 第五行：multi_stage说明
        info_label = ttk.Label(lr_frame, text="💡 multi_stage: patience耗尽时降低LR，patience线性增长(30→60→90)",
                               font=("", 8), foreground="gray")
        info_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(5, 0))

        # 6. 早停配置组
        patience_group = ttk.LabelFrame(right_column, text="⏹️ 早停配置")
        patience_group.pack(fill=tk.X, pady=(0, 10))

        patience_frame = ttk.Frame(patience_group)
        patience_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：阶段1和阶段2的早停耐心值
        ttk.Label(patience_frame, text="阶段1耐心:").grid(row=0, column=0, sticky="w")
        ttk.Entry(patience_frame, textvariable=self.main_gui.ae_patience_stage1, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(patience_frame, text="阶段2耐心:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Entry(patience_frame, textvariable=self.main_gui.ae_patience_stage2, width=8).grid(row=0, column=3, sticky="w")

        # 第二行：阶段3和端到端的早停耐心值
        ttk.Label(patience_frame, text="阶段3耐心:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(patience_frame, textvariable=self.main_gui.ae_patience_stage3, width=8).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(patience_frame, text="端到端耐心:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(patience_frame, textvariable=self.main_gui.ae_patience_e2e, width=8).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # 第三行：联合训练的早停耐心值
        ttk.Label(patience_frame, text="联合训练耐心:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(patience_frame, textvariable=self.main_gui.ae_patience_joint, width=8).grid(row=2, column=1, sticky="w", pady=(5, 0))

        # 7. 损失函数配置组
        loss_group = ttk.LabelFrame(right_column, text="🔧 损失函数")
        loss_group.pack(fill=tk.X, pady=(0, 10))

        loss_frame = ttk.Frame(loss_group)
        loss_frame.pack(fill=tk.X, padx=5, pady=5)

        # 使用自定义损失函数选项
        ttk.Checkbutton(loss_frame, text="使用自定义损失函数", variable=self.main_gui.ae_use_custom_loss).pack(anchor=tk.W)
        # 打开损失函数配置对话框（包含联合训练权重配置）
        ttk.Button(loss_frame, text="配置损失函数 (包含联合训练权重)", command=self.main_gui._open_loss_config_for_ae).pack(fill=tk.X, pady=(5, 0))

        # 8. 训练控制组
        training_control_group = ttk.LabelFrame(right_column, text="⚙️ 训练控制")
        training_control_group.pack(fill=tk.X, pady=(0, 10))

        training_control_frame = ttk.Frame(training_control_group)
        training_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 训练模式选择（三阶段/端到端/仅Stage 1/仅Stage 2/联合训练）
        ttk.Label(training_control_frame, text="训练模式:").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(training_control_frame, textvariable=self.main_gui.ae_training_mode,
                                values=["三阶段训练", "端到端训练", "仅Stage 1", "仅Stage 2", "联合训练"], state="readonly", width=12)
        mode_combo.grid(row=0, column=1, columnspan=3, sticky="ew")

        # 按钮组（紧凑排列）
        button_frame = ttk.Frame(training_control_frame)
        button_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        # 训练控制按钮
        ttk.Button(button_frame, text="开始训练", command=self.main_gui.start_ae_training, width=7).pack(side=tk.LEFT, padx=(0, 2))
        self.continue_training_btn = ttk.Button(button_frame, text="继续训练", command=self.main_gui.resume_ae_training, width=7, state='disabled')
        self.continue_training_btn.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(button_frame, text="停止训练", command=self.main_gui.stop_ae_training, width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(button_frame, text="保存模型", command=self.main_gui.save_ae_model, width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(button_frame, text="加载模型", command=self.main_gui.load_ae_model, width=7).pack(side=tk.LEFT)

        # 9. 系统操作组（扩展功能）
        ops_group = ttk.LabelFrame(right_column, text="🔧 系统操作")
        ops_group.pack(fill=tk.X, pady=(0, 10))

        ops_frame = ttk.Frame(ops_group)
        ops_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建单个系统（当前选择的模式）
        ttk.Button(ops_frame, text="创建当前模式系统", command=self.create_current_system).pack(fill=tk.X, pady=(0, 3))
        # 初始化Loss归一化
        ttk.Button(ops_frame, text="初始化Loss归一化", command=self.initialize_loss_normalization).pack(fill=tk.X, pady=(0, 3))
        # 变更模式和参数（支持跨模式继续训练）
        ttk.Button(ops_frame, text="🔧 变更模式/参数", command=self.change_system_config).pack(fill=tk.X, pady=(0, 3))

        # 配置文件管理（两按钮并排）
        config_button_frame = ttk.Frame(ops_frame)
        config_button_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(config_button_frame, text="💾 保存配置", command=self.save_ae_config, width=11).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(config_button_frame, text="📂 加载配置", command=self.load_ae_config, width=11).pack(side=tk.LEFT)

        # 10. 小波分析组
        wavelet_group = ttk.LabelFrame(right_column, text="🌊 小波变换分析")
        wavelet_group.pack(fill=tk.X, pady=(0, 10))

        wavelet_frame = ttk.Frame(wavelet_group)
        wavelet_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：模型选择和频率选择
        sel_frame1 = ttk.Frame(wavelet_frame)
        sel_frame1.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(sel_frame1, text="分析模型:").pack(side=tk.LEFT)
        self.wavelet_model_selection = ttk.Combobox(sel_frame1, values=["001"], width=6, state="readonly")
        self.wavelet_model_selection.pack(side=tk.LEFT, padx=(5, 10))
        self.wavelet_model_selection.set("001")
        ttk.Label(sel_frame1, text="频率:").pack(side=tk.LEFT)
        self.wavelet_freq_selection = ttk.Combobox(sel_frame1, values=["1.5G", "3G", "6G"], width=6, state="readonly")
        self.wavelet_freq_selection.pack(side=tk.LEFT, padx=5)
        self.wavelet_freq_selection.set("1.5G")

        # 第二行：数据类型和小波类型
        sel_frame2 = ttk.Frame(wavelet_frame)
        sel_frame2.pack(fill=tk.X, pady=(0, 3))
        self.wavelet_data_type = tk.StringVar(value="dB")
        ttk.Radiobutton(sel_frame2, text="分贝(dB)", variable=self.wavelet_data_type, value="dB").pack(side=tk.LEFT)
        ttk.Radiobutton(sel_frame2, text="线性", variable=self.wavelet_data_type, value="linear").pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(sel_frame2, text="小波类型:").pack(side=tk.LEFT)
        wavelet_combo = ttk.Combobox(sel_frame2, textvariable=self.wavelet_analysis_wavelet,
                                     values=["db4", "db8", "haar"], state="readonly", width=6)
        wavelet_combo.pack(side=tk.LEFT, padx=5)

        # 第三行：小波变换模式选择
        sel_frame3 = ttk.Frame(wavelet_frame)
        sel_frame3.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(sel_frame3, text="变换模式:").pack(side=tk.LEFT)
        self.wavelet_transform_mode = tk.StringVar(value="numpy")
        ttk.Radiobutton(sel_frame3, text="NumPy (传统)", variable=self.wavelet_transform_mode, value="numpy").pack(side=tk.LEFT, padx=(5, 5))
        ttk.Radiobutton(sel_frame3, text="Differentiable (可微分)", variable=self.wavelet_transform_mode, value="differentiable").pack(side=tk.LEFT)

        # 运行小波分析按钮
        ttk.Button(wavelet_frame, text="🔬 运行小波分析", command=self.run_wavelet_analysis).pack(fill=tk.X)

        # 11. 梯度监控组 (新增)
        gradient_group = ttk.LabelFrame(right_column, text="📈 梯度监控")
        gradient_group.pack(fill=tk.X, pady=(0, 10))

        gradient_frame = ttk.Frame(gradient_group)
        gradient_frame.pack(fill=tk.X, padx=5, pady=5)

        # 梯度监控开关
        ttk.Checkbutton(gradient_frame, text="启用实时梯度监控 (影响性能)",
                       variable=self.main_gui.ae_gradient_monitoring).pack(anchor=tk.W, pady=(0, 5))

        # 梯度监控按钮
        ttk.Button(gradient_frame, text="📊 查看梯度历史曲线",
                  command=lambda: self.main_gui.visualization_manager._plot_gradient_history()).pack(fill=tk.X, pady=(0, 3))
        ttk.Button(gradient_frame, text="📋 梯度监控报告",
                  command=lambda: self.main_gui.visualization_manager._show_gradient_report()).pack(fill=tk.X)


    def _hide_additive_controls(self):
        """隐藏Additive Dual-Branch专用控件"""
        if hasattr(self, 'additive_frame'):
            self.additive_frame.grid_remove()

    def _show_additive_controls(self):
        """显示Additive Dual-Branch专用控件"""
        if hasattr(self, 'additive_frame'):
            self.additive_frame.grid()
        self._on_learnable_weights_change()

    def _on_architecture_change(self, *args):
        """架构类型变化回调"""
        architecture = self.main_gui.ae_architecture_type.get()
        if architecture.startswith("Additive_Dual_Branch"):
            self._show_additive_controls()
        else:
            self._hide_additive_controls()

    def _on_learnable_weights_change(self):
        """可学习权重变化回调"""
        learnable = self.main_gui.ae_learnable_weights.get()
        state = "disabled" if learnable else "normal"
        if hasattr(self, 'additive_alpha_high_entry'):
            self.additive_alpha_high_entry.configure(state=state)
        if hasattr(self, 'additive_alpha_smooth_entry'):
            self.additive_alpha_smooth_entry.configure(state=state)

    def _create_right_panel(self, parent):
        """创建右侧状态和结果面板"""
        # 创建标签页管理器
        self.result_notebook = ttk.Notebook(parent)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)

        # 1. 日志和状态标签页 (默认显示)
        log_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(log_frame, text="日志与状态")

        # 顶部：系统状态显示区域 (精简版)
        status_group = ttk.LabelFrame(log_frame, text="系统状态")
        status_group.pack(fill=tk.X, padx=5, pady=5)

        # 状态文本（显示5行，紧凑显示完整参数信息）
        self.status_text = tk.Text(status_group, wrap=tk.NONE, height=5, font=self.main_gui.font_small)
        status_scrollbar = ttk.Scrollbar(status_group, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)

        # 底部：完整日志显示区域
        log_group = ttk.LabelFrame(log_frame, text="完整日志")
        log_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # 重新创建ae_log_text组件 (显示所有输出)
        import tkinter.scrolledtext as scrolledtext
        self.main_gui.ae_log_text = scrolledtext.ScrolledText(log_group, wrap=tk.WORD, font=self.main_gui.font_small)
        self.main_gui.ae_log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 2. 对比分析标签页
        comparison_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(comparison_frame, text="性能对比")

        # 对比结果显示区域
        self.comparison_canvas_frame = ttk.Frame(comparison_frame)
        self.comparison_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 3. 小波分析标签页
        wavelet_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(wavelet_frame, text="小波分析")

        # 小波分析结果显示区域
        self.wavelet_canvas_frame = ttk.Frame(wavelet_frame)
        self.wavelet_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 重新创建ae_status_text组件 (系统状态) - 保持向后兼容
        self.main_gui.ae_status_text = self.status_text

        # 初始状态更新
        self._update_status_display()

    def _on_mode_change(self, *args):
        """模式变化回调"""
        mode = self.main_gui.ae_mode.get()

        # 更新小波设置可用性
        if mode in ("wavelet", "differentiable_wavelet"):
            # 小波模式和可微分小波模式都需要小波类型设置
            self.wavelet_combo.configure(state="readonly")
        else:
            self.wavelet_combo.configure(state="disabled")

        # dB变换现在改为手动控制，用户可根据需要自行选择
        # 注释掉原来的自动设置逻辑，允许用户手动控制
        # normalize = self.main_gui.ae_normalize.get()
        # if mode == "direct" and normalize:
        #     # Direct模式 + 标准化 → 自动启用dB
        #     self.main_gui.ae_db_transform.set(True)
        # else:
        #     # Wavelet/Differentiable模式或未启用标准化 → 不使用dB
        #     self.main_gui.ae_db_transform.set(False)

        # 更新状态显示
        self._update_status_display()

    def _log_model_structure(self, model, mode_name="AutoEncoder"):
        """
        将模型结构信息输出到日志

        Args:
            model: AutoEncoder模型实例
            mode_name: 模式名称（用于日志标题）
        """
        try:
            # 获取模型信息
            if not hasattr(model, 'get_model_info'):
                self.main_gui.ae_log("  ⚠️ 模型不支持get_model_info方法")
                return

            model_info = model.get_model_info()

            # 输出基本信息
            self.main_gui.ae_log(f"\n📊 【{mode_name}模式】模型结构信息:")
            self.main_gui.ae_log(f"  • 模型类: {type(model).__name__}")
            self.main_gui.ae_log(f"  • 架构: {model_info.get('architecture', 'Unknown')}")
            self.main_gui.ae_log(f"  • 隐空间维度: {model_info.get('latent_dim', 'Unknown')}")

            # 输入/输出形状
            if 'input_shape' in model_info:
                self.main_gui.ae_log(f"  • 输入形状: {model_info['input_shape']}")
            if 'output_shape' in model_info:
                self.main_gui.ae_log(f"  • 输出形状: {model_info['output_shape']}")

            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.main_gui.ae_log(f"  • 总参数量: {total_params:,}")
            self.main_gui.ae_log(f"  • 可训练参数: {trainable_params:,}")

            # 全连接层结构（自适应调整）
            if 'fc_structure' in model_info:
                self.main_gui.ae_log(f"\n  🔗 全连接层结构（自适应）:")
                self.main_gui.ae_log(f"     {model_info['fc_structure']}")

                if 'num_fc_layers' in model_info:
                    self.main_gui.ae_log(f"     层数: {model_info['num_fc_layers']}")

                # 压缩比
                if 'compression_ratios' in model_info:
                    self.main_gui.ae_log(f"     压缩比:")
                    for i, ratio in enumerate(model_info['compression_ratios'][:3], 1):  # 只显示前3个
                        self.main_gui.ae_log(f"       阶段{i}: {ratio}")
                    if len(model_info['compression_ratios']) > 3:
                        self.main_gui.ae_log(f"       ...")

            self.main_gui.ae_log("")  # 空行分隔

        except Exception as e:
            self.main_gui.ae_log(f"  ⚠️ 获取模型结构信息失败: {e}")

    def _update_status_display(self):
        """更新状态显示（包含完整网络参数和训练配置）"""
        self.status_text.delete(1.0, tk.END)

        # 更新模型选择列表
        self._update_model_selection()

        status_info = []

        # 如果主系统已创建，显示详细参数
        if hasattr(self.main_gui, 'ae_system') and self.main_gui.ae_system:
            try:
                config = self.main_gui.ae_system.get('config_info', {})
                model = self.main_gui.ae_system['autoencoder']

                # 提取关键参数
                mode = config.get('mode', 'N/A')
                architecture = config.get('architecture', 'N/A')
                activation = config.get('activation', 'relu')
                latent_dim = config.get('latent_dim', 'N/A')
                num_freq = config.get('num_frequencies', 'N/A')
                wavelet = config.get('wavelet', 'N/A')
                dropout = config.get('dropout_rate', 'N/A')

                # 获取参数量
                param_count = model.get_parameter_count()
                total_params = param_count.get('total', 0)
                if total_params >= 1_000_000:
                    params_str = f"{total_params/1_000_000:.2f}M"
                elif total_params >= 1_000:
                    params_str = f"{total_params/1_000:.1f}K"
                else:
                    params_str = f"{total_params}"

                # 数据预处理参数
                data_adapter = self.main_gui.ae_system.get('data_adapter')
                if data_adapter:
                    normalize = getattr(data_adapter, 'normalize', False)
                    use_db = getattr(data_adapter, 'use_db', False)
                    norm_method = getattr(data_adapter, 'normalization_method', 'z_score')
                    preprocess_str = []
                    if normalize:
                        preprocess_str.append(f"标准化({norm_method})")
                    if use_db:
                        preprocess_str.append("dB变换")
                    preprocess = " + ".join(preprocess_str) if preprocess_str else "无预处理"
                else:
                    preprocess = "N/A"

                # 训练模式
                training_mode = self.main_gui.ae_system.get('training_mode', 'N/A')
                mode_display = {
                    'stage1_only': 'Stage1',
                    'three_stage': '3Stage',
                    'stage2_only': 'Stage2',
                    'joint_training': '联合',
                    'end_to_end': '端到端'
                }.get(training_mode, training_mode)

                # 通道注意力（如果有）
                channel_attn = config.get('channel_attention', False)
                attn_str = "✓" if channel_attn else "✗"

                # 双分支参数（如果是dual_branch架构）
                ll_ratio = config.get('ll_ratio', None)
                dual_branch_str = f" | LL比例:{ll_ratio:.1f}" if ll_ratio else ""

                # 第1行：网络架构 + 频率配置（合并）
                status_info.append(f"【网络】{mode.upper()}-{architecture.upper()} | 激活:{activation} | 隐空间:{latent_dim}D | 参数量:{params_str} | 频率:{num_freq}freq | 小波:{wavelet} | Dropout:{dropout} | 注意力:{attn_str}{dual_branch_str}")

                # 第2行：数据预处理 + 系统状态（合并）
                main_sys = "✓" if (hasattr(self.main_gui, 'ae_system') and self.main_gui.ae_system) else "✗"
                dual_sys = "✓" if (self.wavelet_system and self.direct_system) else "✗"
                data_loaded = "✓" if (hasattr(self.main_gui, 'rcs_data') and self.main_gui.rcs_data is not None) else "✗"
                status_info.append(f"【数据】预处理:{preprocess} | 状态: 主系统{main_sys} 双系统{dual_sys} 数据{data_loaded}")

                # 第3行：训练配置 + 性能（合并）
                if hasattr(self.main_gui, 'ae_training_history') and self.main_gui.ae_training_history:
                    history = self.main_gui.ae_training_history
                    training_config = history.get('training_config', {})

                    # 提取训练参数
                    optimizer = training_config.get('optimizer_type', 'adam')
                    lr = training_config.get('learning_rate', 0.001)
                    batch_size = training_config.get('batch_size', 32)
                    lr_scheduler = training_config.get('lr_scheduler', 'adaptive')

                    # 训练历史（如果有Stage 1）
                    stage_histories = history.get('stage_histories', {})
                    perf_str = ""
                    if 'stage1' in stage_histories:
                        stage1 = stage_histories['stage1']
                        best_loss = stage1.get('best_val_loss', 'N/A')
                        best_epoch = stage1.get('best_epoch', 'N/A')
                        if isinstance(best_loss, float):
                            perf_str = f" | Stage1最佳:Loss={best_loss:.6f}@Epoch{best_epoch}"
                        else:
                            perf_str = f" | Stage1最佳:Loss={best_loss}@Epoch{best_epoch}"

                    # 显示训练配置 + 性能
                    status_info.append(f"【训练】模式:{mode_display} | 优化器:{optimizer} | LR:{lr} | Batch:{batch_size} | 调度:{lr_scheduler}{perf_str}")
                else:
                    status_info.append(f"【训练】模式:{mode_display} | 未训练")

            except Exception as e:
                # 如果获取详细信息失败，显示基本配置
                mode = self.main_gui.ae_mode.get()
                freq_config = self.main_gui.ae_freq_config.get()
                latent_dim = self.main_gui.ae_latent_dim.get()
                status_info.append(f"模式: {mode} | 频率: {freq_config} | 隐空间: {latent_dim}")
                status_info.append(f"⚠️ 无法获取详细参数: {str(e)}")
        else:
            # 系统未创建，显示配置信息（单行）
            mode = self.main_gui.ae_mode.get()
            freq_config = self.main_gui.ae_freq_config.get()
            latent_dim = self.main_gui.ae_latent_dim.get()
            architecture = self.main_gui.ae_architecture_type.get()
            activation = self.main_gui.ae_activation.get()
            main_sys = "✓" if (hasattr(self.main_gui, 'ae_system') and self.main_gui.ae_system) else "✗"
            dual_sys = "✓" if (self.wavelet_system and self.direct_system) else "✗"
            data_loaded = "✓" if (hasattr(self.main_gui, 'rcs_data') and self.main_gui.rcs_data is not None) else "✗"
            status_info.append(f"【配置】模式:{mode} | 架构:{architecture} | 激活:{activation} | 频率:{freq_config} | 隐空间:{latent_dim}D | 状态: 主系统{main_sys} 双系统{dual_sys} 数据{data_loaded}")
            status_info.append("⚠️ 主系统未创建，请点击'创建AutoEncoder系统'")

        # 对比结果（如果有）
        if self.comparison_results:
            timestamp = self.comparison_results.get('timestamp', '未知')
            sample_count = self.comparison_results.get('sample_count', 0)
            status_info.append(f"【对比】时间:{timestamp} | 样本:{sample_count}个")

        for line in status_info:
            self.status_text.insert(tk.END, line + "\n")

    def _update_model_selection(self):
        """更新模型选择列表"""
        # ✅ 防御性检查：确保控件存在
        if not hasattr(self, 'wavelet_model_selection'):
            # 控件未初始化，静默跳过（可能是界面初始化未完成）
            return

        if hasattr(self.main_gui, 'rcs_data') and self.main_gui.rcs_data is not None:
            num_models = len(self.main_gui.rcs_data)
            model_options = [f"{i+1:03d}" for i in range(num_models)]
            self.wavelet_model_selection['values'] = model_options

            # 如果当前选择不在列表中，重置为第一个模型
            current = self.wavelet_model_selection.get()
            if not current or current not in model_options:
                self.wavelet_model_selection.set(model_options[0] if model_options else "001")
        else:
            self.wavelet_model_selection['values'] = ["001"]
            self.wavelet_model_selection.set("001")

    def create_current_system(self):
        """创建当前选择模式的系统"""
        try:
            mode = self.main_gui.ae_mode.get()

            if not self.main_gui.data_loaded:
                messagebox.showwarning("警告", "请先加载数据！")
                return

            self.main_gui.ae_log(f"🚀 创建{mode}模式AutoEncoder系统...")

            # 导入所需模块
            import sys
            sys.path.append('autoencoder')
            from autoencoder.utils.frequency_config import create_autoencoder_system

            # 获取配置参数 (确保类型正确)
            freq_config = self.main_gui.ae_freq_config.get()
            latent_dim = int(self.main_gui.ae_latent_dim.get())
            dropout_rate = float(self.main_gui.ae_dropout_rate.get())
            wavelet_type = self.main_gui.ae_wavelet_type.get()
            architecture_type = self.main_gui.ae_architecture_type.get().lower()
            normalization_method = self.main_gui.ae_normalization_method.get()  # 从GUI读取标准化方法
            use_channel_attention = self.main_gui.ae_use_channel_attention.get()  # 通道注意力开关
            db_transform = self.main_gui.ae_db_transform.get()  # 从GUI读取dB变换开关（手动控制）
            activation = self.main_gui.ae_activation.get()  # 从GUI读取激活函数
            enforce_nonnegative_rcs = self.main_gui.ae_enforce_nonnegative_rcs.get()  # RCS非负约束

            # 参数映射器配置
            mapper_activation = self.main_gui.ae_mapper_activation.get()
            if mapper_activation == 'auto':
                mapper_activation = None  # None表示使用与AutoEncoder相同的激活函数
            mapper_use_adaptive = self.main_gui.ae_mapper_use_adaptive.get()

            # 向后兼容：更新 ae_normalize 变量
            self.main_gui.ae_normalize.set(normalization_method != 'none')

            # 读取Additive Dual-Branch专用参数
            activation_encoder = self.main_gui.ae_activation_encoder.get()
            activation_high = self.main_gui.ae_activation_high.get()
            activation_smooth = self.main_gui.ae_activation_smooth.get()
            learnable_weights = self.main_gui.ae_learnable_weights.get()
            alpha_high = float(self.main_gui.ae_alpha_high.get())
            alpha_smooth = float(self.main_gui.ae_alpha_smooth.get())

            # 创建系统（使用frequency_config的扩展参数）
            self.main_gui.ae_system = create_autoencoder_system(
                config_name=freq_config,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                wavelet=wavelet_type,
                normalize=(normalization_method != 'none'),
                mode=mode,
                architecture=architecture_type,
                use_channel_attention=use_channel_attention,
                activation=activation,
                db_transform=db_transform,
                normalization_method=normalization_method,
                mapper_activation=mapper_activation,
                mapper_use_adaptive=mapper_use_adaptive,
                # Additive Dual-Branch专用参数
                activation_encoder=activation_encoder,
                activation_high=activation_high,
                activation_smooth=activation_smooth,
                learnable_weights=learnable_weights,
                alpha_high=alpha_high,
                alpha_smooth=alpha_smooth,
                # RCS物理约束
                enforce_nonnegative_rcs=enforce_nonnegative_rcs
            )

            # 添加数据
            self.main_gui.ae_system['rcs_data'] = self.main_gui.rcs_data
            self.main_gui.ae_system['param_data'] = self.main_gui.param_data

            self.main_gui.ae_log(f"✅ {mode}模式系统创建成功!")

            # 输出模型结构信息
            self._log_model_structure(self.main_gui.ae_system['autoencoder'], mode)

            # 更新原有GUI状态
            self.main_gui.update_ae_status()
            self._update_status_display()

            # 更新状态栏显示网络参数
            self.main_gui._update_status_bar_with_model_info()

            messagebox.showinfo("成功", f"{mode}模式AutoEncoder系统创建成功！")

        except Exception as e:
            error_msg = f"创建系统失败: {e}"
            self.main_gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def initialize_loss_normalization(self):
        """初始化Loss归一化系数（使第一个epoch的loss归一化为1.0）"""
        try:
            if not self.main_gui.data_loaded:
                messagebox.showwarning("警告", "请先加载数据！")
                return

            if self.main_gui.ae_system is None:
                messagebox.showwarning("警告", "请先创建AutoEncoder系统！")
                return

            self.main_gui.ae_log("🔧 开始初始化Loss归一化...")

            import torch
            from torch.utils.data import DataLoader, TensorDataset

            # 获取系统组件
            ae_system = self.main_gui.ae_system
            autoencoder = ae_system['autoencoder']
            data_adapter = ae_system.get('data_adapter', None)
            mode = ae_system.get('mode', 'wavelet')

            # 获取训练配置和训练模式
            training_config = self.main_gui.training_config
            batch_size = training_config.get('batch_size', 32)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            training_mode = self.main_gui.ae_training_mode.get()

            self.main_gui.ae_log(f"🎯 检测到训练模式: {training_mode}")

            # 根据训练模式选择初始化方式
            if training_mode == "联合训练":
                self._initialize_loss_normalization_joint_mode(ae_system, training_config, device)
                return

            # 其他模式使用Stage 1的loss初始化（三阶段/端到端/仅Stage 1/仅Stage 2）
            self.main_gui.ae_log("📊 使用Stage 1 AE重建loss进行归一化初始化")

            # 准备数据（使用与训练相同的预处理）
            rcs_data = self.main_gui.rcs_data

            if mode == 'wavelet':
                # 小波模式：原始RCS → 小波变换 → 标准化
                wavelet_transform = ae_system['wavelet_transform']
                wavelet_coeffs = wavelet_transform.forward_transform(rcs_data)
                
                # 转换为numpy以便适配器处理
                if isinstance(wavelet_coeffs, torch.Tensor):
                    wavelet_coeffs_np = wavelet_coeffs.detach().cpu().numpy()
                else:
                    wavelet_coeffs_np = wavelet_coeffs
                    
                if data_adapter:
                    input_data = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
                else:
                    input_data = wavelet_coeffs_np
            else:
                # Direct模式：原始RCS → 标准化
                if data_adapter:
                    input_data = data_adapter.adapt_rcs_data(rcs_data)
                else:
                    input_data = rcs_data

            # 创建DataLoader
            input_tensor = torch.FloatTensor(input_data)
            dataset = TensorDataset(input_tensor, input_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # 创建损失函数（使用Stage 1的配置）
            criterion = self.main_gui.training_manager.ae_trainer._create_stage_loss_function(training_config, stage='stage1')

            # 将模型移到设备
            autoencoder.to(device)
            autoencoder.eval()

            # 计算第一个epoch的平均loss
            total_loss = 0.0
            total_samples = 0

            self.main_gui.ae_log("🔄 计算初始loss...")

            with torch.no_grad():
                for batch_data, batch_target in dataloader:
                    batch_data = batch_data.to(device)
                    batch_target = batch_target.to(device)

                    # 前向传播
                    recon, latent = autoencoder(batch_data)

                    # 计算loss
                    loss = criterion(recon, batch_target)

                    # sample-weighted累加
                    batch_size_actual = batch_data.size(0)
                    total_loss += loss.item() * batch_size_actual
                    total_samples += batch_size_actual

            # 计算平均loss
            initial_loss = total_loss / total_samples

            # 计算归一化系数（使loss归一化为1.0）
            loss_normalization_factor = 1.0 / initial_loss

            # 保存到ae_system
            ae_system['loss_normalization_factor'] = loss_normalization_factor

            self.main_gui.ae_log(f"✅ Loss归一化初始化完成！")
            self.main_gui.ae_log(f"  初始Loss: {initial_loss:.6f}")
            self.main_gui.ae_log(f"  归一化系数: {loss_normalization_factor:.6f}")
            self.main_gui.ae_log(f"  归一化后Loss: {initial_loss * loss_normalization_factor:.6f}")

            messagebox.showinfo("成功",
                f"Loss归一化初始化成功！\n\n"
                f"初始Loss: {initial_loss:.6f}\n"
                f"归一化系数: {loss_normalization_factor:.6f}\n"
                f"归一化后Loss: 1.0")

        except Exception as e:
            import traceback
            error_msg = f"初始化Loss归一化失败: {e}\n{traceback.format_exc()}"
            self.main_gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _initialize_loss_normalization_joint_mode(self, ae_system, training_config, device):
        """
        联合训练模式的损失归一化初始化

        计算 total_loss = α*L_recon_rcs + β*L_consistency + γ*L_param_recon 的初始值
        并生成归一化系数使其归一化为1.0

        Args:
            ae_system: AutoEncoder系统字典
            training_config: 训练配置
            device: 训练设备
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from tkinter import messagebox

        self.main_gui.ae_log("📊 使用联合训练loss进行归一化初始化")

        # 获取组件
        autoencoder = ae_system['autoencoder']
        parameter_mapper = ae_system['parameter_mapper']
        wavelet_transform = ae_system.get('wavelet_transform', None)
        data_adapter = ae_system.get('data_adapter', None)
        mode = ae_system.get('mode', 'wavelet')

        # 获取损失权重
        alpha = training_config.get('alpha_recon', 0.3)
        beta = training_config.get('beta_consistency', 0.5)
        gamma = training_config.get('gamma_param_recon', 1.0)

        self.main_gui.ae_log(f"  ⚖️ 损失权重: α={alpha}, β={beta}, γ={gamma}")

        # 准备RCS数据
        rcs_data = self.main_gui.rcs_data

        if mode in ['wavelet', 'differentiable_wavelet']:
            # 小波模式：原始RCS → 小波变换 → 标准化
            self.main_gui.ae_log("🔄 预处理RCS数据（小波变换 + 标准化）...")
            rcs_tensor = torch.FloatTensor(rcs_data)
            wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)

            if isinstance(wavelet_coeffs, torch.Tensor):
                wavelet_coeffs_np = wavelet_coeffs.detach().cpu().numpy()
            else:
                wavelet_coeffs_np = wavelet_coeffs

            if data_adapter:
                input_data = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
            else:
                input_data = wavelet_coeffs_np
        else:
            # Direct模式：原始RCS → 标准化
            self.main_gui.ae_log("🔄 预处理RCS数据（标准化）...")
            if data_adapter:
                input_data = data_adapter.adapt_rcs_data(rcs_data)
            else:
                input_data = rcs_data

        # 准备参数数据
        param_data = self.main_gui.param_data

        from sklearn.preprocessing import StandardScaler
        self.main_gui.ae_log("🔧 标准化设计参数...")
        param_scaler = StandardScaler()
        param_normalized = param_scaler.fit_transform(param_data)
        ae_system['param_scaler'] = param_scaler

        # 创建数据集
        input_tensor = torch.FloatTensor(input_data)
        param_tensor = torch.FloatTensor(param_normalized)
        dataset = TensorDataset(input_tensor, param_tensor)

        # 创建DataLoader
        batch_size = training_config.get('batch_size', 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 创建三个损失函数
        ae_trainer = self.main_gui.training_manager.ae_trainer
        criterion_recon = ae_trainer._create_joint_loss_function(training_config, 'recon_rcs')
        criterion_consistency = ae_trainer._create_joint_loss_function(training_config, 'consistency')
        criterion_param_recon = ae_trainer._create_joint_loss_function(training_config, 'param_recon')

        self.main_gui.ae_log(f"  L_recon_rcs: {type(criterion_recon).__name__}")
        self.main_gui.ae_log(f"  L_consistency: {type(criterion_consistency).__name__}")
        self.main_gui.ae_log(f"  L_param_recon: {type(criterion_param_recon).__name__}")

        # 将模型移到设备
        autoencoder.to(device)
        parameter_mapper.to(device)
        autoencoder.eval()
        parameter_mapper.eval()

        # 计算初始总损失
        total_loss_sum = 0.0
        total_samples = 0

        self.main_gui.ae_log("🔄 计算初始联合训练loss...")

        with torch.no_grad():
            for rcs_batch, param_batch in dataloader:
                rcs_batch = rcs_batch.to(device)
                param_batch = param_batch.to(device)
                batch_size_actual = rcs_batch.size(0)

                # 三条路径前向传播
                # 路径1: RCS → Encoder → Decoder → RCS
                recon_rcs, latent_from_rcs = autoencoder(rcs_batch)

                # 路径2: Params → Mapper → Latent
                latent_from_params = parameter_mapper(param_batch)

                # 路径3: Params → Mapper → Decoder → RCS
                recon_from_params = autoencoder.decode(latent_from_params)

                # 计算三个损失
                L_recon_rcs = criterion_recon(recon_rcs, rcs_batch)
                L_consistency = criterion_consistency(latent_from_rcs, latent_from_params)
                L_param_recon = criterion_param_recon(recon_from_params, rcs_batch)

                # 总损失（加权）
                total_loss = alpha * L_recon_rcs + beta * L_consistency + gamma * L_param_recon

                # sample-weighted累加
                total_loss_sum += total_loss.item() * batch_size_actual
                total_samples += batch_size_actual

        # 计算平均总损失
        initial_loss = total_loss_sum / total_samples

        # 计算归一化系数
        loss_normalization_factor = 1.0 / initial_loss

        # 保存到ae_system
        ae_system['loss_normalization_factor'] = loss_normalization_factor

        self.main_gui.ae_log(f"✅ Loss归一化初始化完成（联合训练模式）！")
        self.main_gui.ae_log(f"  初始总Loss: {initial_loss:.6f}")
        self.main_gui.ae_log(f"  归一化系数: {loss_normalization_factor:.6f}")
        self.main_gui.ae_log(f"  归一化后Loss: {initial_loss * loss_normalization_factor:.6f}")

        messagebox.showinfo("成功",
            f"Loss归一化初始化成功（联合训练模式）！\n\n"
            f"初始总Loss: {initial_loss:.6f}\n"
            f"归一化系数: {loss_normalization_factor:.6f}\n"
            f"归一化后Loss: 1.0")

    def save_ae_config(self):
        """保存AE训练配置到JSON文件"""
        try:
            import json
            from datetime import datetime

            # 收集所有AE配置参数
            config_data = {
                "metadata": {
                    "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0",
                    "description": "AutoEncoder训练配置"
                },
                "model": {
                    "mode": self.main_gui.ae_mode.get(),
                    "freq_config": self.main_gui.ae_freq_config.get(),
                    "latent_dim": int(self.main_gui.ae_latent_dim.get()),
                    "dropout_rate": float(self.main_gui.ae_dropout_rate.get()),
                    "wavelet_type": self.main_gui.ae_wavelet_type.get(),
                    "architecture_type": self.main_gui.ae_architecture_type.get()
                },
                "preprocessing": {
                    "normalize": self.main_gui.ae_normalize.get(),
                    "normalization_method": self.main_gui.ae_normalization_method.get(),
                    "db_transform": self.main_gui.ae_db_transform.get(),
                    "enforce_nonnegative_rcs": self.main_gui.ae_enforce_nonnegative_rcs.get()
                },
                "training": {
                    "batch_size": int(self.main_gui.ae_batch_size.get()),
                    "learning_rate": float(self.main_gui.ae_learning_rate.get()),
                    "epochs_stage1": int(self.main_gui.ae_epochs_stage1.get()),
                    "epochs_stage2": int(self.main_gui.ae_epochs_stage2.get()),
                    "epochs_stage3": int(self.main_gui.ae_epochs_stage3.get()),
                    "training_mode": self.main_gui.ae_training_mode.get()
                },
                "optimizer": {
                    "optimizer_type": self.main_gui.ae_optimizer_type.get(),
                    "weight_decay": float(self.main_gui.ae_weight_decay.get()),
                    "momentum": float(self.main_gui.ae_momentum.get())
                },
                "data_split": {
                    "validation_split": float(self.main_gui.ae_validation_split.get())
                },
                "learning_rate_schedule": {
                    "scheduler": self.main_gui.ae_lr_scheduler.get(),
                    "min_lr": float(self.main_gui.ae_min_lr.get()),
                    "restart_period": int(self.main_gui.ae_restart_period.get())
                },
                "early_stopping": {
                    "patience_stage1": int(self.main_gui.ae_patience_stage1.get()),
                    "patience_stage2": int(self.main_gui.ae_patience_stage2.get()),
                    "patience_stage3": int(self.main_gui.ae_patience_stage3.get()),
                    "patience_e2e": int(self.main_gui.ae_patience_e2e.get())
                },
                "loss": {
                    "use_custom_loss": self.main_gui.ae_use_custom_loss.get()
                }
            }

            # 弹出保存文件对话框
            file_path = filedialog.asksaveasfilename(
                title="保存AE配置",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"ae_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                self.main_gui.ae_log(f"✅ 配置已保存到: {file_path}")
                messagebox.showinfo("保存成功", f"配置已保存到:\n{file_path}")

        except Exception as e:
            error_msg = f"保存配置失败: {e}"
            self.main_gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("保存失败", error_msg)

    def load_ae_config(self):
        """从JSON文件加载AE训练配置"""
        try:
            import json

            # 弹出打开文件对话框
            file_path = filedialog.askopenfilename(
                title="加载AE配置",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if not file_path:
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 加载模型配置
            model_config = config_data.get("model", {})
            if "mode" in model_config:
                self.main_gui.ae_mode.set(model_config["mode"])
            if "freq_config" in model_config:
                self.main_gui.ae_freq_config.set(model_config["freq_config"])
            if "latent_dim" in model_config:
                self.main_gui.ae_latent_dim.set(str(model_config["latent_dim"]))
            if "dropout_rate" in model_config:
                self.main_gui.ae_dropout_rate.set(str(model_config["dropout_rate"]))
            if "wavelet_type" in model_config:
                self.main_gui.ae_wavelet_type.set(model_config["wavelet_type"])
            if "architecture_type" in model_config:
                self.main_gui.ae_architecture_type.set(model_config["architecture_type"])

            # 加载预处理配置
            preprocess_config = config_data.get("preprocessing", {})
            if "normalize" in preprocess_config:
                self.main_gui.ae_normalize.set(preprocess_config["normalize"])
            if "normalization_method" in preprocess_config:
                # 加载标准化方法（新版本）
                self.main_gui.ae_normalization_method.set(preprocess_config["normalization_method"])
                # 同步更新 ae_normalize（向后兼容）
                self.main_gui.ae_normalize.set(preprocess_config["normalization_method"] != 'none')
            elif "normalize" in preprocess_config:
                # 向后兼容旧配置：没有normalization_method字段时根据normalize推断
                if preprocess_config["normalize"]:
                    self.main_gui.ae_normalization_method.set("zscore")  # 旧版默认是zscore
                else:
                    self.main_gui.ae_normalization_method.set("none")
            if "db_transform" in preprocess_config:
                self.main_gui.ae_db_transform.set(preprocess_config["db_transform"])
            if "enforce_nonnegative_rcs" in preprocess_config:
                self.main_gui.ae_enforce_nonnegative_rcs.set(preprocess_config["enforce_nonnegative_rcs"])
            else:
                self.main_gui.ae_enforce_nonnegative_rcs.set(False)  # 默认关闭（保持兼容性）

            # 加载训练配置
            training_config = config_data.get("training", {})
            if "batch_size" in training_config:
                self.main_gui.ae_batch_size.set(str(training_config["batch_size"]))
            if "learning_rate" in training_config:
                self.main_gui.ae_learning_rate.set(str(training_config["learning_rate"]))
            if "epochs_stage1" in training_config:
                self.main_gui.ae_epochs_stage1.set(str(training_config["epochs_stage1"]))
            if "epochs_stage2" in training_config:
                self.main_gui.ae_epochs_stage2.set(str(training_config["epochs_stage2"]))
            if "epochs_stage3" in training_config:
                self.main_gui.ae_epochs_stage3.set(str(training_config["epochs_stage3"]))
            if "training_mode" in training_config:
                self.main_gui.ae_training_mode.set(training_config["training_mode"])

            # 加载优化器配置
            optimizer_config = config_data.get("optimizer", {})
            if "optimizer_type" in optimizer_config:
                self.main_gui.ae_optimizer_type.set(optimizer_config["optimizer_type"])
            if "weight_decay" in optimizer_config:
                self.main_gui.ae_weight_decay.set(str(optimizer_config["weight_decay"]))
            if "momentum" in optimizer_config:
                self.main_gui.ae_momentum.set(str(optimizer_config["momentum"]))

            # 加载数据划分配置
            data_split_config = config_data.get("data_split", {})
            if "validation_split" in data_split_config:
                self.main_gui.ae_validation_split.set(str(data_split_config["validation_split"]))

            # 加载学习率调度配置
            lr_config = config_data.get("learning_rate_schedule", {})
            if "scheduler" in lr_config:
                self.main_gui.ae_lr_scheduler.set(lr_config["scheduler"])
            if "min_lr" in lr_config:
                self.main_gui.ae_min_lr.set(str(lr_config["min_lr"]))
            if "restart_period" in lr_config:
                self.main_gui.ae_restart_period.set(str(lr_config["restart_period"]))

            # 加载早停配置
            es_config = config_data.get("early_stopping", {})
            if "patience_stage1" in es_config:
                self.main_gui.ae_patience_stage1.set(str(es_config["patience_stage1"]))
            if "patience_stage2" in es_config:
                self.main_gui.ae_patience_stage2.set(str(es_config["patience_stage2"]))
            if "patience_stage3" in es_config:
                self.main_gui.ae_patience_stage3.set(str(es_config["patience_stage3"]))
            if "patience_e2e" in es_config:
                self.main_gui.ae_patience_e2e.set(str(es_config["patience_e2e"]))

            # 加载损失配置
            loss_config = config_data.get("loss", {})
            if "use_custom_loss" in loss_config:
                self.main_gui.ae_use_custom_loss.set(loss_config["use_custom_loss"])

            # 显示元数据信息
            metadata = config_data.get("metadata", {})
            saved_time = metadata.get("saved_time", "未知")
            description = metadata.get("description", "")

            self.main_gui.ae_log(f"✅ 配置已加载: {file_path}")
            self.main_gui.ae_log(f"   保存时间: {saved_time}")
            messagebox.showinfo("加载成功", f"配置已加载:\n{file_path}\n\n保存时间: {saved_time}\n{description}")

            # 更新状态显示
            self._update_status_display()

        except Exception as e:
            error_msg = f"加载配置失败: {e}"
            self.main_gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("加载失败", error_msg)

    def change_system_config(self):
        """变更模式和参数（支持跨模式继续训练）"""
        try:
            # 1. 检查是否已加载模型
            if not hasattr(self.main_gui, 'ae_system') or self.main_gui.ae_system is None:
                messagebox.showwarning("警告", "请先加载模型或创建系统!")
                return

            # 2. 读取当前配置
            current_system = self.main_gui.ae_system
            current_config = {
                'mode': current_system.get('mode', 'wavelet'),
                'architecture': current_system.get('architecture', 'cnn'),
                'latent_dim': current_system.get('latent_dim', 256),
                'activation': current_system.get('activation', 'relu'),
                'training_mode': current_system.get('training_mode', 'three_stage'),
                'dropout_rate': current_system.get('dropout_rate', 0.2),
                'wavelet': current_system.get('wavelet_type', 'db4'),
            }

            # 读取数据预处理配置
            data_adapter = current_system.get('data_adapter', None)
            if data_adapter:
                current_config['normalize'] = getattr(data_adapter, 'normalize', True)
                current_config['db_transform'] = getattr(data_adapter, 'db_transform', False)
                current_config['normalization_method'] = getattr(data_adapter, 'normalization_method', 'zscore')
            else:
                current_config['normalize'] = True
                current_config['db_transform'] = False
                current_config['normalization_method'] = 'zscore'

            self.main_gui.ae_log("📋 当前配置:")
            for key, value in current_config.items():
                self.main_gui.ae_log(f"  {key}: {value}")

            # 3. 打开配置变更对话框
            self._show_config_change_dialog(current_config)

        except Exception as e:
            error_msg = f"变更配置失败: {e}"
            self.main_gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _show_config_change_dialog(self, current_config):
        """显示配置变更对话框"""
        dialog = tk.Toplevel(self.main_gui.root)
        dialog.title("变更模式和参数")
        dialog.geometry("600x700")
        dialog.transient(self.main_gui.root)
        dialog.grab_set()

        # 主容器
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 说明文本
        info_label = ttk.Label(main_frame, text="📝 修改配置后将重建AutoEncoder系统\n✅ 已加载的数据和权重将保留",
                              justify=tk.LEFT, foreground='blue')
        info_label.pack(pady=(0, 10))

        # 创建滚动容器
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 配置项容器
        config_vars = {}

        # === 1. 训练模式 ===
        mode_frame = ttk.LabelFrame(scrollable_frame, text="训练模式", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mode_frame, text=f"当前: {current_config['training_mode']}", foreground='gray').pack(anchor='w')

        config_vars['training_mode'] = tk.StringVar(value=current_config['training_mode'])
        mode_options = ['stage1_only', 'stage2_only', 'three_stage', 'joint_training']
        mode_display = {
            'stage1_only': '仅Stage 1（AE预训练）',
            'stage2_only': '仅Stage 2（Mapper训练）',
            'three_stage': '三阶段训练',
            'joint_training': '联合训练'
        }

        for mode in mode_options:
            ttk.Radiobutton(mode_frame, text=mode_display[mode],
                          variable=config_vars['training_mode'], value=mode).pack(anchor='w')

        # === 2. 模式 ===
        ae_mode_frame = ttk.LabelFrame(scrollable_frame, text="AE模式", padding=10)
        ae_mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ae_mode_frame, text=f"当前: {current_config['mode']}", foreground='gray').pack(anchor='w')

        config_vars['mode'] = tk.StringVar(value=current_config['mode'])
        mode_opts = ['wavelet', 'direct', 'differentiable_wavelet']
        for m in mode_opts:
            ttk.Radiobutton(ae_mode_frame, text=m, variable=config_vars['mode'], value=m).pack(anchor='w')

        # === 3. 架构 ===
        arch_frame = ttk.LabelFrame(scrollable_frame, text="架构", padding=10)
        arch_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(arch_frame, text=f"当前: {current_config['architecture']}", foreground='gray').pack(anchor='w')

        config_vars['architecture'] = tk.StringVar(value=current_config['architecture'])
        arch_opts = ['cnn', 'mlp', 'enhanced_cnn', 'deep_cnn',
                    'dual_branch_cnn', 'dual_branch_mlp',
                    'additive_dual_branch_cnn', 'additive_dual_branch_mlp']
        ttk.Combobox(arch_frame, textvariable=config_vars['architecture'],
                    values=arch_opts, state='readonly', width=30).pack(fill=tk.X)

        # === 4. 激活函数 ===
        act_frame = ttk.LabelFrame(scrollable_frame, text="激活函数", padding=10)
        act_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(act_frame, text=f"当前: {current_config['activation']}", foreground='gray').pack(anchor='w')

        config_vars['activation'] = tk.StringVar(value=current_config['activation'])
        act_opts = ['relu', 'sin', 'gelu', 'swish', 'tanh', 'mish', 'leaky_relu', 'elu', 'selu']
        ttk.Combobox(act_frame, textvariable=config_vars['activation'],
                    values=act_opts, state='readonly', width=30).pack(fill=tk.X)

        # === 5. 隐空间维度 ===
        latent_frame = ttk.LabelFrame(scrollable_frame, text="隐空间维度", padding=10)
        latent_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(latent_frame, text=f"当前: {current_config['latent_dim']}", foreground='gray').pack(anchor='w')

        config_vars['latent_dim'] = tk.IntVar(value=current_config['latent_dim'])
        latent_opts = [32, 64, 128, 256, 512]
        ttk.Combobox(latent_frame, textvariable=config_vars['latent_dim'],
                    values=latent_opts, state='readonly', width=30).pack(fill=tk.X)

        # === 6. Dropout率 ===
        dropout_frame = ttk.LabelFrame(scrollable_frame, text="Dropout率", padding=10)
        dropout_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(dropout_frame, text=f"当前: {current_config['dropout_rate']}", foreground='gray').pack(anchor='w')

        config_vars['dropout_rate'] = tk.DoubleVar(value=current_config['dropout_rate'])
        ttk.Entry(dropout_frame, textvariable=config_vars['dropout_rate'], width=10).pack(anchor='w')

        # === 7. 数据预处理 ===
        prep_frame = ttk.LabelFrame(scrollable_frame, text="数据预处理", padding=10)
        prep_frame.pack(fill=tk.X, pady=(0, 10))

        config_vars['normalize'] = tk.BooleanVar(value=current_config['normalize'])
        ttk.Checkbutton(prep_frame, text="标准化", variable=config_vars['normalize']).pack(anchor='w')

        config_vars['db_transform'] = tk.BooleanVar(value=current_config['db_transform'])
        ttk.Checkbutton(prep_frame, text="dB变换", variable=config_vars['db_transform']).pack(anchor='w')

        ttk.Label(prep_frame, text="标准化方法:", foreground='gray').pack(anchor='w', pady=(5, 0))
        config_vars['normalization_method'] = tk.StringVar(value=current_config['normalization_method'])
        norm_opts = ['zscore', 'minmax', 'none']
        ttk.Combobox(prep_frame, textvariable=config_vars['normalization_method'],
                    values=norm_opts, state='readonly', width=15).pack(anchor='w')

        # 布局滚动容器
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        def on_confirm():
            try:
                # 收集新配置
                new_config = {
                    'training_mode': config_vars['training_mode'].get(),
                    'mode': config_vars['mode'].get(),
                    'architecture': config_vars['architecture'].get(),
                    'activation': config_vars['activation'].get(),
                    'latent_dim': config_vars['latent_dim'].get(),
                    'dropout_rate': config_vars['dropout_rate'].get(),
                    'normalize': config_vars['normalize'].get(),
                    'db_transform': config_vars['db_transform'].get(),
                    'normalization_method': config_vars['normalization_method'].get(),
                }

                # 关闭对话框
                dialog.destroy()

                # 执行配置变更
                self._rebuild_system_with_new_config(new_config, current_config)

            except Exception as e:
                messagebox.showerror("错误", f"应用配置失败: {e}")

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="✅ 确认变更", command=on_confirm).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="❌ 取消", command=on_cancel).pack(side=tk.LEFT)

    def _rebuild_system_with_new_config(self, new_config, old_config):
        """使用新配置重建AE系统"""
        try:
            self.main_gui.ae_log("🔧 开始重建AutoEncoder系统...")

            # 显示变更对比
            self.main_gui.ae_log("\n📋 配置变更对比:")
            for key in new_config.keys():
                if key in old_config:
                    old_val = old_config[key]
                    new_val = new_config[key]
                    if old_val != new_val:
                        self.main_gui.ae_log(f"  {key}: {old_val} → {new_val}")

            # 保存当前数据和权重
            old_system = self.main_gui.ae_system
            rcs_data = old_system.get('rcs_data', None)
            param_data = old_system.get('param_data', None)
            old_weights = {
                'autoencoder': old_system['autoencoder'].state_dict() if 'autoencoder' in old_system else None,
                'parameter_mapper': old_system['parameter_mapper'].state_dict() if 'parameter_mapper' in old_system else None
            }

            # 获取频率配置（从旧系统）
            config_name = old_system.get('config_name', '2freq')

            # 创建新系统
            from autoencoder.utils.frequency_config import create_autoencoder_system

            self.main_gui.ae_log(f"🔨 创建新系统: mode={new_config['mode']}, arch={new_config['architecture']}")

            new_system = create_autoencoder_system(
                config_name=config_name,
                mode=new_config['mode'],
                architecture=new_config['architecture'],
                latent_dim=new_config['latent_dim'],
                dropout_rate=new_config['dropout_rate'],
                activation=new_config['activation'],
                normalize=new_config['normalize'],
                db_transform=new_config['db_transform'],
                normalization_method=new_config['normalization_method']
            )

            # 尝试加载旧权重（如果架构兼容）
            if old_weights['autoencoder'] is not None:
                try:
                    new_system['autoencoder'].load_state_dict(old_weights['autoencoder'])
                    self.main_gui.ae_log("✅ AutoEncoder权重加载成功")
                except Exception as e:
                    self.main_gui.ae_log(f"⚠️ AutoEncoder权重不兼容（架构变化），使用随机初始化: {e}")

            if old_weights['parameter_mapper'] is not None:
                try:
                    new_system['parameter_mapper'].load_state_dict(old_weights['parameter_mapper'])
                    self.main_gui.ae_log("✅ ParameterMapper权重加载成功")
                except Exception as e:
                    self.main_gui.ae_log(f"⚠️ ParameterMapper权重不兼容（架构变化），使用随机初始化: {e}")

            # 恢复数据
            if rcs_data is not None:
                new_system['rcs_data'] = rcs_data
                self.main_gui.ae_log(f"✅ RCS数据已恢复 (shape: {rcs_data.shape})")
            if param_data is not None:
                new_system['param_data'] = param_data
                self.main_gui.ae_log(f"✅ 参数数据已恢复 (shape: {param_data.shape})")

            # 设置训练模式
            new_system['training_mode'] = new_config['training_mode']

            # 更新GUI系统
            self.main_gui.ae_system = new_system

            # 更新GUI控件（让GUI显示新配置）
            self.ae_mode_var.set(new_config['mode'])
            self.ae_arch_var.set(new_config['architecture'])
            self.ae_activation_var.set(new_config['activation'])
            self.ae_latent_dim.set(new_config['latent_dim'])

            # 更新训练模式下拉框
            mode_display_reverse = {
                'stage1_only': '仅Stage 1',
                'stage2_only': '仅Stage 2',
                'three_stage': '三阶段训练',
                'joint_training': '联合训练'
            }
            if new_config['training_mode'] in mode_display_reverse:
                self.ae_training_mode.set(mode_display_reverse[new_config['training_mode']])

            self.main_gui.ae_log("\n✅ 系统重建完成！可以点击【继续训练】开始跨模式训练")
            self.main_gui.ae_log("💡 提示: 训练将从当前权重继续（如果架构兼容），否则从随机初始化开始")

            messagebox.showinfo("成功", "配置变更完成！\n可以开始继续训练了。")

        except Exception as e:
            error_msg = f"重建系统失败: {e}"
            self.main_gui.ae_log(f"❌ {error_msg}")
            import traceback
            self.main_gui.ae_log(traceback.format_exc())
            messagebox.showerror("错误", error_msg)

    def run_performance_comparison(self):
        """运行性能对比分析"""
        if not self.wavelet_system or not self.direct_system:
            messagebox.showwarning("警告", "请先创建双系统！")
            return

        # 在后台线程中运行对比分析
        def comparison_thread():
            try:
                self.main_gui.ae_log("📊 开始性能对比分析...")

                # 使用简化的对比分析
                from wavelet_gui_helper import simple_performance_comparison

                # 准备测试数据
                batch_size = self.comparison_batch_size.get()

                self.main_gui.ae_log(f"📈 执行性能对比 (批次大小: {batch_size})...")

                # 执行对比分析
                comparison_results = simple_performance_comparison(
                    self.main_gui.rcs_data,
                    self.main_gui.param_data,
                    self.wavelet_system,
                    self.direct_system,
                    batch_size
                )

                # 保存结果
                from datetime import datetime
                self.comparison_results = {
                    'performance': {
                        'wavelet_mode': {
                            'reconstruction_mse': comparison_results['wavelet']['mse'],
                            'inference_time': comparison_results['wavelet']['time']
                        },
                        'direct_mode': {
                            'reconstruction_mse': comparison_results['direct']['mse'],
                            'inference_time': comparison_results['direct']['time']
                        }
                    },
                    'efficiency': {
                        'model_complexity': {
                            'wavelet_total_params': comparison_results['wavelet']['params'],
                            'direct_total_params': comparison_results['direct']['params']
                        }
                    },
                    'comparison': comparison_results['comparison'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sample_count': len(self.main_gui.rcs_data)
                }

                self.main_gui.ae_log("📋 生成对比可视化...")

                # 在主线程中更新界面
                self.main_gui.root.after(0, self._display_comparison_results)

                self.main_gui.ae_log("✅ 性能对比分析完成!")

            except Exception as e:
                error_msg = f"性能对比分析失败: {e}"
                self.main_gui.ae_log(f"❌ {error_msg}")
                self.main_gui.root.after(0, lambda: messagebox.showerror("错误", error_msg))

        # 启动后台线程
        threading.Thread(target=comparison_thread, daemon=True).start()

    def _display_comparison_results(self):
        """显示对比分析结果"""
        try:
            # 清除之前的显示
            for widget in self.comparison_canvas_frame.winfo_children():
                widget.destroy()

            # 创建对比结果图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('AutoEncoder模式性能对比分析', fontsize=14, fontweight='bold')

            results = self.comparison_results
            perf = results['performance']
            eff = results['efficiency']

            # 1. 重建精度对比
            ax1 = axes[0, 0]
            modes = ['小波增强', '直接模式']

            if 'wavelet_mode' in perf and 'direct_mode' in perf:
                mse_values = [perf['wavelet_mode']['reconstruction_mse'],
                             perf['direct_mode']['reconstruction_mse']]

                bars = ax1.bar(modes, mse_values, color=['skyblue', 'lightcoral'], alpha=0.8)
                ax1.set_ylabel('重建MSE')
                ax1.set_title('重建精度对比')
                ax1.set_yscale('log')

                # 添加数值标签
                for bar, val in zip(bars, mse_values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.2e}', ha='center', va='bottom')

            # 2. 推理时间对比
            ax2 = axes[0, 1]
            if 'wavelet_mode' in perf and 'direct_mode' in perf:
                time_values = [perf['wavelet_mode']['inference_time'],
                              perf['direct_mode']['inference_time']]

                bars = ax2.bar(modes, time_values, color=['lightgreen', 'orange'], alpha=0.8)
                ax2.set_ylabel('推理时间 (秒)')
                ax2.set_title('推理速度对比')

                # 添加数值标签
                for bar, val in zip(bars, time_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.4f}s', ha='center', va='bottom')

            # 3. 模型复杂度对比
            ax3 = axes[1, 0]
            if 'model_complexity' in eff:
                complexity = eff['model_complexity']
                param_counts = [complexity.get('wavelet_total_params', 0),
                               complexity.get('direct_total_params', 0)]

                bars = ax3.bar(modes, param_counts, color=['purple', 'pink'], alpha=0.8)
                ax3.set_ylabel('参数数量')
                ax3.set_title('模型复杂度对比')

                # 添加数值标签
                for bar, val in zip(bars, param_counts):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val/1e6:.1f}M', ha='center', va='bottom')

            # 4. 综合评分对比
            ax4 = axes[1, 1]

            # 计算综合评分
            if 'wavelet_mode' in perf and 'direct_mode' in perf:
                # 精度得分 (MSE越小越好)
                wavelet_mse = perf['wavelet_mode']['reconstruction_mse']
                direct_mse = perf['direct_mode']['reconstruction_mse']
                accuracy_score_w = 100 * (direct_mse / (wavelet_mse + direct_mse))
                accuracy_score_d = 100 * (wavelet_mse / (wavelet_mse + direct_mse))

                # 速度得分 (时间越短越好)
                wavelet_time = perf['wavelet_mode']['inference_time']
                direct_time = perf['direct_mode']['inference_time']
                speed_score_w = 100 * (direct_time / (wavelet_time + direct_time))
                speed_score_d = 100 * (wavelet_time / (wavelet_time + direct_time))

                # 综合得分
                overall_w = (accuracy_score_w + speed_score_w) / 2
                overall_d = (accuracy_score_d + speed_score_d) / 2

                categories = ['精度得分', '速度得分', '综合得分']
                wavelet_scores = [accuracy_score_w, speed_score_w, overall_w]
                direct_scores = [accuracy_score_d, speed_score_d, overall_d]

                x = np.arange(len(categories))
                width = 0.35

                ax4.bar(x - width/2, wavelet_scores, width, label='小波增强', alpha=0.8, color='skyblue')
                ax4.bar(x + width/2, direct_scores, width, label='直接模式', alpha=0.8, color='lightcoral')

                ax4.set_ylabel('得分')
                ax4.set_title('综合性能对比')
                ax4.set_xticks(x)
                ax4.set_xticklabels(categories)
                ax4.legend()
                ax4.set_ylim(0, 100)

            plt.tight_layout()

            # 显示图表
            canvas = FigureCanvasTkAgg(fig, self.comparison_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 切换到对比分析标签页
            self.result_notebook.select(1)

            # 更新状态
            self._update_status_display()

        except Exception as e:
            messagebox.showerror("错误", f"显示对比结果失败: {e}")

    def run_wavelet_analysis(self):
        """运行小波变换分析"""
        if not self.main_gui.data_loaded:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        def analysis_thread():
            try:
                self.main_gui.ae_log("🌊 开始小波变换分析...")

                # 获取用户选择
                selected_model = self.wavelet_model_selection.get()
                selected_freq = self.wavelet_freq_selection.get()
                data_type = self.wavelet_data_type.get()

                # 解析模型选择
                if selected_model:
                    try:
                        # 直接解析模型ID (格式如 "001", "002", 等)
                        model_idx = int(selected_model) - 1  # 转换为0索引
                        if model_idx >= len(self.main_gui.rcs_data):
                            model_idx = 0
                    except:
                        model_idx = 0
                else:
                    model_idx = 0

                # 解析频率选择
                freq_map = {"1.5G": 0, "3G": 1, "6G": 2}
                freq_idx = freq_map.get(selected_freq, 0)

                # 检查频率索引是否在数据范围内
                num_freqs = self.main_gui.rcs_data.shape[3]
                if freq_idx >= num_freqs:
                    self.main_gui.ae_log(f"⚠️ 警告: 选择的频率 {selected_freq} 不在数据中，使用第一个频率")
                    freq_idx = 0
                    selected_freq = ["1.5G", "3G", "6G"][0] if num_freqs > 0 else "1.5G"

                # 执行小波分析
                from wavelet_gui_helper import simple_wavelet_analysis
                import numpy as np

                # 选择分析数据
                sample_data = self.main_gui.rcs_data[model_idx, :, :, freq_idx]

                # 如果选择分贝模式，转换数据用于显示
                if data_type == 'dB':
                    epsilon = 1e-10
                    # 转换为分贝：dB = 10 * log10(RCS)
                    sample_data_db = 10 * np.log10(np.maximum(sample_data, epsilon))
                    analysis_data = sample_data_db
                else:
                    analysis_data = sample_data

                self.main_gui.ae_log(f"📊 执行小波分解和重建 (模型: {selected_model}, 频率: {selected_freq}, 数据类型: {data_type})...")
                analysis_result = simple_wavelet_analysis(
                    analysis_data,
                    wavelet=self.wavelet_analysis_wavelet.get(),
                    data_type=data_type,
                    transform_mode=self.wavelet_transform_mode.get()
                )

                self.main_gui.ae_log("📈 生成可视化结果...")
                self.wavelet_analysis_results = analysis_result
                self.current_analysis_model = selected_model
                self.current_analysis_freq = selected_freq
                self.current_analysis_data_type = data_type

                # 在主线程中更新界面
                self.main_gui.root.after(0, self._display_wavelet_results)

                self.main_gui.ae_log("✅ 小波分析完成!")

            except Exception as e:
                error_msg = f"小波分析失败: {e}"
                self.main_gui.ae_log(f"❌ {error_msg}")
                self.main_gui.root.after(0, lambda: messagebox.showerror("错误", error_msg))

        # 启动后台线程
        threading.Thread(target=analysis_thread, daemon=True).start()

    def _display_wavelet_results(self):
        """显示小波分析结果"""
        try:
            # 清除之前的显示
            for widget in self.wavelet_canvas_frame.winfo_children():
                widget.destroy()

            # 创建小波分析图表
            from wavelet_gui_helper import create_wavelet_plot
            model_name = getattr(self, 'current_analysis_model', '001')
            frequency = getattr(self, 'current_analysis_freq', '1.5G')
            data_type = getattr(self, 'current_analysis_data_type', 'dB')
            fig = create_wavelet_plot(self.wavelet_analysis_results, data_type=data_type, model_name=model_name, frequency=frequency)

            # 显示图表
            canvas = FigureCanvasTkAgg(fig, self.wavelet_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 切换到小波分析标签页
            self.result_notebook.select(2)

        except Exception as e:
            messagebox.showerror("错误", f"显示小波分析结果失败: {e}")


def integrate_extension_to_gui(main_gui):
    """
    将扩展功能集成到主GUI中

    Args:
        main_gui: RCSWaveletGUI实例
    """
    # 创建扩展实例
    extension = AutoEncoderExtension(main_gui)

    # 将扩展实例绑定到主GUI
    main_gui.ae_extension = extension

    # 扩展AutoEncoder标签页
    extension.extend_autoencoder_tab()

    # 保存原始的ae_log方法
    original_ae_log = main_gui.ae_log

    # 添加扩展的日志方法
    def extended_ae_log(message):
        """扩展的AE日志方法"""
        original_ae_log(message)  # 调用原始方法
        extension._update_status_display()

    main_gui.ae_log = extended_ae_log

    print("✅ AutoEncoder扩展功能已成功集成到GUI中!")

    return extension
