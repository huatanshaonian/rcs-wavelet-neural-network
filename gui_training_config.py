"""
统一训练配置GUI组件
支持AutoEncoder和其他网络的训练参数配置
与项目主配置文件兼容
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from typing import Dict, Any, Optional

try:
    from autoencoder.utils.training_config import ConfigManager, AutoEncoderTrainingConfig
    from autoencoder.utils.frequency_config import create_training_compatible_system
    CONFIG_SYSTEM_AVAILABLE = True
except ImportError:
    CONFIG_SYSTEM_AVAILABLE = False
    print("警告: AutoEncoder配置系统不可用")


class UnifiedTrainingConfigGUI:
    """统一训练配置GUI类"""

    def __init__(self, parent, main_gui=None):
        """
        初始化统一训练配置GUI

        Args:
            parent: 父窗口
            main_gui: 主GUI实例
        """
        self.parent = parent
        self.main_gui = main_gui

        # 配置管理器
        self.config_manager = None
        if CONFIG_SYSTEM_AVAILABLE:
            try:
                self.config_manager = ConfigManager('config.json')
            except Exception as e:
                print(f"配置管理器初始化失败: {e}")

        # 配置变量 - 基础训练参数
        self.batch_size = tk.StringVar(value="8")
        self.learning_rate = tk.StringVar(value="0.001")
        self.min_learning_rate = tk.StringVar(value="2e-5")
        self.weight_decay = tk.StringVar(value="0.0001")
        self.epochs = tk.StringVar(value="200")
        self.early_stopping_patience = tk.StringVar(value="20")

        # 交叉验证
        self.use_cross_validation = tk.BooleanVar(value=True)
        self.n_folds = tk.StringVar(value="5")
        self.test_split = tk.StringVar(value="0.2")

        # 优化器和调度器
        self.optimizer_type = tk.StringVar(value="adam")
        self.scheduler_type = tk.StringVar(value="cosine")
        self.momentum = tk.StringVar(value="0.9")

        # 损失函数权重
        self.loss_mse = tk.StringVar(value="1.0")
        self.loss_symmetry = tk.StringVar(value="0.1")
        self.loss_frequency = tk.StringVar(value="0.05")
        self.loss_multiscale = tk.StringVar(value="0.2")

        # AutoEncoder特定参数
        self.ae_latent_dim = tk.StringVar(value="256")
        self.ae_dropout_rate = tk.StringVar(value="0.2")
        self.ae_mode = tk.StringVar(value="wavelet")  # wavelet 或 direct
        self.ae_freq_config = tk.StringVar(value="2freq")  # 2freq 或 3freq

        # 数据增强
        self.use_data_augmentation = tk.BooleanVar(value=False)
        self.augmentation_prob = tk.StringVar(value="0.3")

        # 当前配置类型
        self.config_type = tk.StringVar(value="autoencoder")  # autoencoder 或 wavelet_network

        # 加载配置
        self._load_config()

        # 创建界面
        self._create_widgets()

    def _load_config(self):
        """从配置管理器加载配置"""
        if self.config_manager:
            try:
                config = self.config_manager.get_ae_training_config()
                self.batch_size.set(str(config.batch_size))
                self.learning_rate.set(str(config.learning_rate))
                self.min_learning_rate.set(str(config.min_learning_rate))
                self.weight_decay.set(str(config.weight_decay))
                self.epochs.set(str(config.epochs))
                self.early_stopping_patience.set(str(config.early_stopping_patience))
                self.use_cross_validation.set(config.use_cross_validation)
                self.n_folds.set(str(config.n_folds))
                self.test_split.set(str(config.test_split))
                self.optimizer_type.set(config.optimizer_type)
                self.scheduler_type.set(config.scheduler_type)

                # 损失权重
                if config.loss_weights:
                    self.loss_mse.set(str(config.loss_weights.get('mse', 1.0)))
                    self.loss_symmetry.set(str(config.loss_weights.get('symmetry', 0.1)))
                    self.loss_frequency.set(str(config.loss_weights.get('frequency_consistency', 0.05)))
                    self.loss_multiscale.set(str(config.loss_weights.get('multiscale', 0.2)))

                print("✅ 已从config.json加载配置")
            except Exception as e:
                print(f"配置加载失败: {e}")

    def _create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建Notebook分页
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 基础配置页
        self._create_basic_config_tab()

        # AutoEncoder配置页
        self._create_autoencoder_config_tab()

        # 优化器配置页
        self._create_optimizer_config_tab()

        # 损失函数配置页
        self._create_loss_config_tab()

        # 配置管理页
        self._create_config_management_tab()

    def _create_basic_config_tab(self):
        """创建基础配置标签页"""
        basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(basic_frame, text="基础训练参数")

        # 滚动框架
        canvas = tk.Canvas(basic_frame)
        scrollbar = ttk.Scrollbar(basic_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 配置类型选择
        type_group = ttk.LabelFrame(scrollable_frame, text="配置类型")
        type_group.pack(fill=tk.X, pady=(0, 10))

        type_frame = ttk.Frame(type_group)
        type_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(type_frame, text="训练类型:").pack(side=tk.LEFT)
        type_combo = ttk.Combobox(type_frame, textvariable=self.config_type,
                                values=['autoencoder', 'wavelet_network'],
                                state='readonly', width=15)
        type_combo.pack(side=tk.LEFT, padx=10)
        type_combo.bind('<<ComboboxSelected>>', self._on_config_type_changed)

        # 基础训练参数
        basic_group = ttk.LabelFrame(scrollable_frame, text="基础训练参数")
        basic_group.pack(fill=tk.X, pady=(0, 10))

        basic_params_frame = ttk.Frame(basic_group)
        basic_params_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行
        row1 = ttk.Frame(basic_params_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="批次大小:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row1, textvariable=self.batch_size, width=10).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row1, text="训练轮数:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row1, textvariable=self.epochs, width=10).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row1, text="早停耐心:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row1, textvariable=self.early_stopping_patience, width=10).pack(side=tk.LEFT)

        # 第二行
        row2 = ttk.Frame(basic_params_frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="学习率:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2, textvariable=self.learning_rate, width=12).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row2, text="最小学习率:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2, textvariable=self.min_learning_rate, width=12).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row2, text="权重衰减:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2, textvariable=self.weight_decay, width=12).pack(side=tk.LEFT)

        # 交叉验证参数
        cv_group = ttk.LabelFrame(scrollable_frame, text="交叉验证")
        cv_group.pack(fill=tk.X, pady=(0, 10))

        cv_frame = ttk.Frame(cv_group)
        cv_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(cv_frame, text="使用交叉验证",
                       variable=self.use_cross_validation).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(cv_frame, text="折数:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(cv_frame, textvariable=self.n_folds, width=5).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(cv_frame, text="测试比例:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(cv_frame, textvariable=self.test_split, width=8).pack(side=tk.LEFT)

        # 数据增强
        aug_group = ttk.LabelFrame(scrollable_frame, text="数据增强")
        aug_group.pack(fill=tk.X, pady=(0, 10))

        aug_frame = ttk.Frame(aug_group)
        aug_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(aug_frame, text="使用数据增强",
                       variable=self.use_data_augmentation).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(aug_frame, text="增强概率:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(aug_frame, textvariable=self.augmentation_prob, width=8).pack(side=tk.LEFT)

        # 配置滚动
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_autoencoder_config_tab(self):
        """创建AutoEncoder配置标签页"""
        ae_frame = ttk.Frame(self.notebook)
        self.notebook.add(ae_frame, text="AutoEncoder配置")

        # 主框架
        main_ae_frame = ttk.Frame(ae_frame)
        main_ae_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # AutoEncoder架构参数
        arch_group = ttk.LabelFrame(main_ae_frame, text="架构参数")
        arch_group.pack(fill=tk.X, pady=(0, 10))

        arch_frame = ttk.Frame(arch_group)
        arch_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行
        arch_row1 = ttk.Frame(arch_frame)
        arch_row1.pack(fill=tk.X, pady=2)

        ttk.Label(arch_row1, text="模式:").pack(side=tk.LEFT, padx=(0, 5))
        mode_combo = ttk.Combobox(arch_row1, textvariable=self.ae_mode,
                                values=['wavelet', 'direct'], state='readonly', width=10)
        mode_combo.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(arch_row1, text="频率配置:").pack(side=tk.LEFT, padx=(0, 5))
        freq_combo = ttk.Combobox(arch_row1, textvariable=self.ae_freq_config,
                                values=['2freq', '3freq'], state='readonly', width=8)
        freq_combo.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(arch_row1, text="隐空间维度:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(arch_row1, textvariable=self.ae_latent_dim, width=8).pack(side=tk.LEFT)

        # 第二行
        arch_row2 = ttk.Frame(arch_frame)
        arch_row2.pack(fill=tk.X, pady=2)

        ttk.Label(arch_row2, text="Dropout率:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(arch_row2, textvariable=self.ae_dropout_rate, width=8).pack(side=tk.LEFT)

        # 模式说明
        info_group = ttk.LabelFrame(main_ae_frame, text="模式说明")
        info_group.pack(fill=tk.X, pady=(0, 10))

        info_text = tk.Text(info_group, height=6, width=80, wrap=tk.WORD)
        info_text.pack(fill=tk.X, padx=5, pady=5)

        mode_info = """小波增强模式 (wavelet):
• 使用小波变换预处理，98.9%能量集中在低频分量
• 训练稳定性更好，特征分离效果佳
• 推荐用于高精度要求的应用

直接模式 (direct):
• 直接处理原始数据，推理速度快56.4%
• 模型参数少28%，部署简单
• 推荐用于实时应用和生产环境"""

        info_text.insert(tk.END, mode_info)
        info_text.config(state=tk.DISABLED)

        # 配置预设
        preset_group = ttk.LabelFrame(main_ae_frame, text="快速配置")
        preset_group.pack(fill=tk.X, pady=(0, 10))

        preset_frame = ttk.Frame(preset_group)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(preset_frame, text="高精度配置",
                  command=self._set_high_precision_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="高速度配置",
                  command=self._set_high_speed_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="默认配置",
                  command=self._set_default_preset).pack(side=tk.LEFT, padx=5)

    def _create_optimizer_config_tab(self):
        """创建优化器配置标签页"""
        opt_frame = ttk.Frame(self.notebook)
        self.notebook.add(opt_frame, text="优化器/调度器")

        main_opt_frame = ttk.Frame(opt_frame)
        main_opt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 优化器配置
        opt_group = ttk.LabelFrame(main_opt_frame, text="优化器配置")
        opt_group.pack(fill=tk.X, pady=(0, 10))

        opt_config_frame = ttk.Frame(opt_group)
        opt_config_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(opt_config_frame, text="优化器类型:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        opt_combo = ttk.Combobox(opt_config_frame, textvariable=self.optimizer_type,
                               values=['adam', 'adamw', 'sgd'], state='readonly', width=10)
        opt_combo.grid(row=0, column=1, padx=(0, 20))

        ttk.Label(opt_config_frame, text="动量 (SGD):").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Entry(opt_config_frame, textvariable=self.momentum, width=8).grid(row=0, column=3)

        # 学习率调度器配置
        sched_group = ttk.LabelFrame(main_opt_frame, text="学习率调度器")
        sched_group.pack(fill=tk.X, pady=(0, 10))

        sched_frame = ttk.Frame(sched_group)
        sched_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sched_frame, text="调度器类型:").pack(side=tk.LEFT, padx=(0, 5))
        sched_combo = ttk.Combobox(sched_frame, textvariable=self.scheduler_type,
                                 values=['cosine', 'step', 'plateau'], state='readonly', width=12)
        sched_combo.pack(side=tk.LEFT, padx=(0, 20))
        sched_combo.bind('<<ComboboxSelected>>', self._on_scheduler_changed)

        # 调度器说明
        self.scheduler_info = tk.StringVar()
        ttk.Label(sched_frame, textvariable=self.scheduler_info,
                 font=("Arial", 8), foreground="gray").pack(side=tk.LEFT, padx=10)

        self._on_scheduler_changed()  # 初始化说明

    def _create_loss_config_tab(self):
        """创建损失函数配置标签页"""
        loss_frame = ttk.Frame(self.notebook)
        self.notebook.add(loss_frame, text="损失函数权重")

        main_loss_frame = ttk.Frame(loss_frame)
        main_loss_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 损失函数权重
        loss_group = ttk.LabelFrame(main_loss_frame, text="损失函数权重")
        loss_group.pack(fill=tk.X, pady=(0, 10))

        loss_weights_frame = ttk.Frame(loss_group)
        loss_weights_frame.pack(fill=tk.X, padx=5, pady=5)

        # MSE损失
        ttk.Label(loss_weights_frame, text="MSE损失:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        ttk.Entry(loss_weights_frame, textvariable=self.loss_mse, width=8).grid(row=0, column=1, padx=(0, 20), pady=2)

        # 对称性损失
        ttk.Label(loss_weights_frame, text="对称性损失:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5), pady=2)
        ttk.Entry(loss_weights_frame, textvariable=self.loss_symmetry, width=8).grid(row=0, column=3, padx=(0, 20), pady=2)

        # 频率一致性损失
        ttk.Label(loss_weights_frame, text="频率一致性:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        ttk.Entry(loss_weights_frame, textvariable=self.loss_frequency, width=8).grid(row=1, column=1, padx=(0, 20), pady=2)

        # 多尺度损失
        ttk.Label(loss_weights_frame, text="多尺度损失:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=2)
        ttk.Entry(loss_weights_frame, textvariable=self.loss_multiscale, width=8).grid(row=1, column=3, padx=(0, 20), pady=2)

        # 权重预设
        preset_loss_group = ttk.LabelFrame(main_loss_frame, text="权重预设")
        preset_loss_group.pack(fill=tk.X, pady=(0, 10))

        preset_loss_frame = ttk.Frame(preset_loss_group)
        preset_loss_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(preset_loss_frame, text="标准权重",
                  command=self._set_standard_loss_weights).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_loss_frame, text="物理约束增强",
                  command=self._set_physics_enhanced_weights).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_loss_frame, text="MSE优先",
                  command=self._set_mse_priority_weights).pack(side=tk.LEFT, padx=5)

        # 损失函数说明
        desc_group = ttk.LabelFrame(main_loss_frame, text="损失函数说明")
        desc_group.pack(fill=tk.X, pady=(0, 10))

        desc_text = tk.Text(desc_group, height=8, wrap=tk.WORD)
        desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        loss_desc = """损失函数说明:

• MSE损失: 主要的回归损失，控制重建精度
• 对称性损失: 强制φ=0°平面对称性约束，确保物理合理性
• 频率一致性: 建模双频间的物理关系，保持频率域特性
• 多尺度损失: 不同分辨率下的一致性约束，提高细节保持

推荐权重:
- 标准配置: MSE=1.0, 对称性=0.1, 频率=0.05, 多尺度=0.2
- 物理增强: 增加约束权重，提高物理合理性
- MSE优先: 专注重建精度，降低约束权重"""

        desc_text.insert(tk.END, loss_desc)
        desc_text.config(state=tk.DISABLED)

    def _create_config_management_tab(self):
        """创建配置管理标签页"""
        mgmt_frame = ttk.Frame(self.notebook)
        self.notebook.add(mgmt_frame, text="配置管理")

        main_mgmt_frame = ttk.Frame(mgmt_frame)
        main_mgmt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 配置文件操作
        file_group = ttk.LabelFrame(main_mgmt_frame, text="配置文件操作")
        file_group.pack(fill=tk.X, pady=(0, 10))

        file_frame = ttk.Frame(file_group)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(file_frame, text="加载配置",
                  command=self._load_config_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="保存配置",
                  command=self._save_config_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="重置为默认",
                  command=self._reset_to_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="应用到项目",
                  command=self._apply_to_project).pack(side=tk.LEFT, padx=5)

        # 配置同步
        sync_group = ttk.LabelFrame(main_mgmt_frame, text="配置同步")
        sync_group.pack(fill=tk.X, pady=(0, 10))

        sync_frame = ttk.Frame(sync_group)
        sync_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(sync_frame, text="从config.json读取",
                  command=self._sync_from_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(sync_frame, text="保存到config.json",
                  command=self._sync_to_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(sync_frame, text="创建AutoEncoder系统",
                  command=self._create_ae_system).pack(side=tk.LEFT, padx=5)

        # 配置预览
        preview_group = ttk.LabelFrame(main_mgmt_frame, text="当前配置预览")
        preview_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.config_preview = tk.Text(preview_group, height=15, wrap=tk.WORD)
        scrollbar_preview = ttk.Scrollbar(preview_group, orient="vertical",
                                        command=self.config_preview.yview)
        self.config_preview.configure(yscrollcommand=scrollbar_preview.set)

        self.config_preview.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
        scrollbar_preview.pack(side="right", fill="y", pady=5)

        # 按钮框架
        button_frame = ttk.Frame(main_mgmt_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="刷新预览",
                  command=self._update_config_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="验证配置",
                  command=self._validate_config).pack(side=tk.LEFT, padx=5)

        # 初始化预览
        self._update_config_preview()

    def _on_config_type_changed(self, event=None):
        """配置类型改变时的回调"""
        config_type = self.config_type.get()
        if config_type == "autoencoder":
            # 切换到AutoEncoder配置页
            self.notebook.select(1)
        else:
            # 切换到基础配置页
            self.notebook.select(0)

    def _on_scheduler_changed(self, event=None):
        """调度器改变时的回调"""
        scheduler = self.scheduler_type.get()
        info_map = {
            'cosine': '余弦退火调度，平滑降低学习率',
            'step': '阶梯式调度，按步长降低学习率',
            'plateau': '平台调度，验证停滞时降低学习率'
        }
        self.scheduler_info.set(info_map.get(scheduler, ''))

    def _set_high_precision_preset(self):
        """设置高精度预设"""
        self.ae_mode.set("wavelet")
        self.ae_latent_dim.set("512")
        self.ae_dropout_rate.set("0.15")
        self.learning_rate.set("0.001")
        self.epochs.set("300")
        self.batch_size.set("8")
        messagebox.showinfo("预设应用", "已应用高精度配置预设")

    def _set_high_speed_preset(self):
        """设置高速度预设"""
        self.ae_mode.set("direct")
        self.ae_latent_dim.set("256")
        self.ae_dropout_rate.set("0.2")
        self.learning_rate.set("0.003")
        self.epochs.set("150")
        self.batch_size.set("16")
        messagebox.showinfo("预设应用", "已应用高速度配置预设")

    def _set_default_preset(self):
        """设置默认预设"""
        self.ae_mode.set("wavelet")
        self.ae_latent_dim.set("256")
        self.ae_dropout_rate.set("0.2")
        self.learning_rate.set("0.001")
        self.epochs.set("200")
        self.batch_size.set("8")
        messagebox.showinfo("预设应用", "已应用默认配置预设")

    def _set_standard_loss_weights(self):
        """设置标准损失权重"""
        self.loss_mse.set("1.0")
        self.loss_symmetry.set("0.1")
        self.loss_frequency.set("0.05")
        self.loss_multiscale.set("0.2")

    def _set_physics_enhanced_weights(self):
        """设置物理约束增强权重"""
        self.loss_mse.set("1.0")
        self.loss_symmetry.set("0.3")
        self.loss_frequency.set("0.15")
        self.loss_multiscale.set("0.4")

    def _set_mse_priority_weights(self):
        """设置MSE优先权重"""
        self.loss_mse.set("1.0")
        self.loss_symmetry.set("0.05")
        self.loss_frequency.set("0.02")
        self.loss_multiscale.set("0.1")

    def _load_config_file(self):
        """加载配置文件"""
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self._apply_config_data(config_data)
                messagebox.showinfo("加载成功", f"已从 {file_path} 加载配置")
            except Exception as e:
                messagebox.showerror("加载失败", f"无法加载配置文件: {e}")

    def _save_config_file(self):
        """保存配置文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                config_data = self._get_current_config()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("保存成功", f"配置已保存到 {file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存配置文件: {e}")

    def _reset_to_default(self):
        """重置为默认配置"""
        if messagebox.askyesno("重置配置", "确定要重置为默认配置吗？"):
            self._set_default_preset()
            self._set_standard_loss_weights()
            self.optimizer_type.set("adam")
            self.scheduler_type.set("cosine")
            self.use_cross_validation.set(True)
            self.n_folds.set("5")
            messagebox.showinfo("重置完成", "已重置为默认配置")

    def _apply_to_project(self):
        """应用配置到项目主配置"""
        if not CONFIG_SYSTEM_AVAILABLE:
            messagebox.showerror("错误", "AutoEncoder配置系统不可用")
            return

        try:
            # 更新配置管理器
            config_data = self._get_ae_config_dict()
            self.config_manager.update_ae_config(**config_data)

            # 保存到config.json
            main_config = self._get_current_config()
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(main_config, f, indent=2, ensure_ascii=False)

            # 通知主GUI更新
            if self.main_gui and hasattr(self.main_gui, 'reload_config'):
                self.main_gui.reload_config()

            messagebox.showinfo("应用成功", "配置已应用到项目主配置文件")
        except Exception as e:
            messagebox.showerror("应用失败", f"无法应用配置: {e}")

    def _sync_from_config(self):
        """从config.json同步配置"""
        if self.config_manager:
            try:
                self.config_manager = ConfigManager('config.json')  # 重新加载
                self._load_config()
                self._update_config_preview()
                messagebox.showinfo("同步成功", "已从config.json同步配置")
            except Exception as e:
                messagebox.showerror("同步失败", f"无法从config.json同步: {e}")

    def _sync_to_config(self):
        """同步配置到config.json"""
        try:
            config_data = self._get_current_config()
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("同步成功", "配置已同步到config.json")
        except Exception as e:
            messagebox.showerror("同步失败", f"无法同步到config.json: {e}")

    def _create_ae_system(self):
        """创建AutoEncoder系统"""
        if not CONFIG_SYSTEM_AVAILABLE:
            messagebox.showerror("错误", "AutoEncoder配置系统不可用")
            return

        try:
            # 应用当前配置
            self._apply_to_project()

            # 创建AutoEncoder系统
            system = create_training_compatible_system(
                config_name=self.ae_freq_config.get(),
                mode=self.ae_mode.get(),
                latent_dim=int(self.ae_latent_dim.get()),
                learning_rate=float(self.learning_rate.get()),
                epochs=int(self.epochs.get()),
                batch_size=int(self.batch_size.get())
            )

            messagebox.showinfo("创建成功",
                              f"AutoEncoder系统创建成功！\\n"
                              f"模式: {self.ae_mode.get()}\\n"
                              f"频率配置: {self.ae_freq_config.get()}\\n"
                              f"隐空间维度: {self.ae_latent_dim.get()}")

            # 存储系统到主GUI
            if self.main_gui:
                self.main_gui.ae_system = system

        except Exception as e:
            messagebox.showerror("创建失败", f"无法创建AutoEncoder系统: {e}")

    def _update_config_preview(self):
        """更新配置预览"""
        try:
            config_data = self._get_current_config()
            config_text = json.dumps(config_data, indent=2, ensure_ascii=False)

            self.config_preview.config(state=tk.NORMAL)
            self.config_preview.delete(1.0, tk.END)
            self.config_preview.insert(tk.END, config_text)
            self.config_preview.config(state=tk.DISABLED)
        except Exception as e:
            self.config_preview.config(state=tk.NORMAL)
            self.config_preview.delete(1.0, tk.END)
            self.config_preview.insert(tk.END, f"配置预览错误: {e}")
            self.config_preview.config(state=tk.DISABLED)

    def _validate_config(self):
        """验证配置"""
        try:
            config_data = self._get_current_config()

            # 基础验证
            errors = []

            # 检查数值范围
            if float(self.learning_rate.get()) <= 0:
                errors.append("学习率必须大于0")
            if float(self.min_learning_rate.get()) >= float(self.learning_rate.get()):
                errors.append("最小学习率必须小于初始学习率")
            if int(self.batch_size.get()) <= 0:
                errors.append("批次大小必须大于0")
            if int(self.epochs.get()) <= 0:
                errors.append("训练轮数必须大于0")

            # 检查AutoEncoder参数
            if int(self.ae_latent_dim.get()) <= 0:
                errors.append("隐空间维度必须大于0")
            if float(self.ae_dropout_rate.get()) < 0 or float(self.ae_dropout_rate.get()) >= 1:
                errors.append("Dropout率必须在[0, 1)范围内")

            if errors:
                messagebox.showerror("配置验证失败", "\\n".join(errors))
            else:
                messagebox.showinfo("配置验证成功", "所有配置参数都有效！")

        except ValueError as e:
            messagebox.showerror("配置验证失败", f"参数类型错误: {e}")
        except Exception as e:
            messagebox.showerror("配置验证失败", f"验证过程出错: {e}")

    def _get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            "training": {
                "batch_size": int(self.batch_size.get()),
                "learning_rate": float(self.learning_rate.get()),
                "min_learning_rate": float(self.min_learning_rate.get()),
                "weight_decay": float(self.weight_decay.get()),
                "epochs": int(self.epochs.get()),
                "early_stopping_patience": int(self.early_stopping_patience.get()),
                "use_cross_validation": self.use_cross_validation.get(),
                "n_folds": int(self.n_folds.get()),
                "optimizer_type": self.optimizer_type.get(),
                "scheduler_type": self.scheduler_type.get(),
                "momentum": float(self.momentum.get()),
                "use_data_augmentation": self.use_data_augmentation.get(),
                "augmentation_prob": float(self.augmentation_prob.get()),
                "loss_weights": {
                    "mse": float(self.loss_mse.get()),
                    "symmetry": float(self.loss_symmetry.get()),
                    "frequency_consistency": float(self.loss_frequency.get()),
                    "multiscale": float(self.loss_multiscale.get())
                }
            },
            "autoencoder": {
                "mode": self.ae_mode.get(),
                "freq_config": self.ae_freq_config.get(),
                "latent_dim": int(self.ae_latent_dim.get()),
                "dropout_rate": float(self.ae_dropout_rate.get())
            },
            "evaluation": {
                "test_split": float(self.test_split.get())
            }
        }

    def _get_ae_config_dict(self) -> Dict[str, Any]:
        """获取AutoEncoder配置字典"""
        return {
            "batch_size": int(self.batch_size.get()),
            "learning_rate": float(self.learning_rate.get()),
            "min_learning_rate": float(self.min_learning_rate.get()),
            "weight_decay": float(self.weight_decay.get()),
            "epochs": int(self.epochs.get()),
            "early_stopping_patience": int(self.early_stopping_patience.get()),
            "use_cross_validation": self.use_cross_validation.get(),
            "n_folds": int(self.n_folds.get()),
            "optimizer_type": self.optimizer_type.get(),
            "scheduler_type": self.scheduler_type.get(),
            "use_data_augmentation": self.use_data_augmentation.get(),
            "augmentation_prob": float(self.augmentation_prob.get())
        }

    def _apply_config_data(self, config_data: Dict[str, Any]):
        """应用配置数据"""
        try:
            training = config_data.get("training", {})

            # 基础训练参数
            if "batch_size" in training:
                self.batch_size.set(str(training["batch_size"]))
            if "learning_rate" in training:
                self.learning_rate.set(str(training["learning_rate"]))
            if "min_learning_rate" in training:
                self.min_learning_rate.set(str(training["min_learning_rate"]))
            if "weight_decay" in training:
                self.weight_decay.set(str(training["weight_decay"]))
            if "epochs" in training:
                self.epochs.set(str(training["epochs"]))
            if "early_stopping_patience" in training:
                self.early_stopping_patience.set(str(training["early_stopping_patience"]))

            # 交叉验证
            if "use_cross_validation" in training:
                self.use_cross_validation.set(training["use_cross_validation"])
            if "n_folds" in training:
                self.n_folds.set(str(training["n_folds"]))

            # 优化器和调度器
            if "optimizer_type" in training:
                self.optimizer_type.set(training["optimizer_type"])
            if "scheduler_type" in training:
                self.scheduler_type.set(training["scheduler_type"])

            # 损失权重
            loss_weights = training.get("loss_weights", {})
            if "mse" in loss_weights:
                self.loss_mse.set(str(loss_weights["mse"]))
            if "symmetry" in loss_weights:
                self.loss_symmetry.set(str(loss_weights["symmetry"]))
            if "frequency_consistency" in loss_weights:
                self.loss_frequency.set(str(loss_weights["frequency_consistency"]))
            if "multiscale" in loss_weights:
                self.loss_multiscale.set(str(loss_weights["multiscale"]))

            # AutoEncoder配置
            ae_config = config_data.get("autoencoder", {})
            if "mode" in ae_config:
                self.ae_mode.set(ae_config["mode"])
            if "freq_config" in ae_config:
                self.ae_freq_config.set(ae_config["freq_config"])
            if "latent_dim" in ae_config:
                self.ae_latent_dim.set(str(ae_config["latent_dim"]))
            if "dropout_rate" in ae_config:
                self.ae_dropout_rate.set(str(ae_config["dropout_rate"]))

            # 评估配置
            eval_config = config_data.get("evaluation", {})
            if "test_split" in eval_config:
                self.test_split.set(str(eval_config["test_split"]))

        except Exception as e:
            messagebox.showerror("配置应用失败", f"无法应用配置数据: {e}")


def create_training_config_window(parent_gui=None):
    """创建统一训练配置窗口"""
    config_window = tk.Toplevel()
    config_window.title("统一训练参数配置")
    config_window.geometry("800x600")

    # 设置窗口图标（如果有的话）
    try:
        config_window.iconbitmap("icon.ico")
    except:
        pass

    # 创建配置GUI
    config_gui = UnifiedTrainingConfigGUI(config_window, parent_gui)

    # 窗口关闭协议
    def on_closing():
        if messagebox.askokcancel("退出", "确定要关闭配置窗口吗？"):
            config_window.destroy()

    config_window.protocol("WM_DELETE_WINDOW", on_closing)

    return config_window, config_gui


if __name__ == "__main__":
    # 测试GUI
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    config_window, config_gui = create_training_config_window()
    config_window.mainloop()