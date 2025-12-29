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

        # 数据配置
        self.angle_rcs_train_split = tk.DoubleVar(value=0.8)  # 训练集比例
        self.angle_rcs_use_subset = tk.BooleanVar(value=False)  # 是否使用子集
        self.angle_rcs_subset_size = tk.IntVar(value=300000)  # 子集大小
        self.angle_rcs_normalize_params = tk.BooleanVar(value=True)  # 是否标准化参数

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

        # 学习率
        ttk.Label(training_frame, text="学习率:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.angle_rcs_lr, width=8).grid(row=1, column=1, sticky="w", pady=(5, 0))

        # 权重衰减
        ttk.Label(training_frame, text="权重衰减:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.angle_rcs_weight_decay, width=8).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # 优化器
        ttk.Label(training_frame, text="优化器:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        optimizer_combo = ttk.Combobox(training_frame, textvariable=self.angle_rcs_optimizer,
                                      values=["adam", "adamw", "sgd", "lbfgs"],
                                      state="readonly", width=12)
        optimizer_combo.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(5, 0))

        # 学习率调度器
        ttk.Label(training_frame, text="调度器:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        scheduler_combo = ttk.Combobox(training_frame, textvariable=self.angle_rcs_scheduler,
                                      values=["constant", "cosine", "cosine_restart", "adaptive"],
                                      state="readonly", width=12)
        scheduler_combo.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(5, 0))

        # Patience
        ttk.Label(training_frame, text="Patience:").grid(row=4, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(training_frame, textvariable=self.angle_rcs_patience, width=8).grid(row=4, column=1, sticky="w", pady=(5, 0))
        ttk.Label(training_frame, text="(Early Stopping)").grid(row=4, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=(5, 0))

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

        # 使用子集
        ttk.Checkbutton(data_frame, text="使用训练子集",
                       variable=self.angle_rcs_use_subset,
                       command=self._on_subset_toggle).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 子集大小
        ttk.Label(data_frame, text="子集大小:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.subset_entry = ttk.Entry(data_frame, textvariable=self.angle_rcs_subset_size, width=12)
        self.subset_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(5, 0))
        self.subset_entry.config(state='disabled')  # 初始禁用

        # 数据说明
        info_text = """
数据点数量：
• 200样本 × 91θ × 91φ × 3频率
• 总计: 4,968,600个数据点

采样策略：
• 全局混合采样（80-20划分）
• 支持子集训练（快速验证）
"""
        ttk.Label(data_frame, text=info_text, justify=tk.LEFT,
                 font=('Courier', 8), foreground="gray").grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 4. 控制按钮组
        control_group = ttk.LabelFrame(right_column, text="🎮 控制")
        control_group.pack(fill=tk.X, pady=(0, 10))

        button_frame = ttk.Frame(control_group)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行按钮
        ttk.Button(button_frame, text="创建模型", command=self._create_model).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="开始训练", command=self._start_training).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        # 第二行按钮
        ttk.Button(button_frame, text="停止训练", command=self._stop_training).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="保存模型", command=self._save_model).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        # 第三行按钮
        ttk.Button(button_frame, text="加载模型", command=self._load_model).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="评估模型", command=self._evaluate_model).grid(row=2, column=1, sticky="ew", padx=2, pady=2)

        # 第四行按钮
        ttk.Button(button_frame, text="可视化预测", command=self._visualize_prediction).grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

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

        # 3. 预测可视化标签页
        viz_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(viz_frame, text="预测可视化")

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

            messagebox.showinfo("成功", f"模型创建成功！\n参数量: {param_count:,}")

        except Exception as e:
            error_msg = f"模型创建失败: {str(e)}"
            self._log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _start_training(self):
        """开始训练"""
        if self.angle_rcs_system is None:
            messagebox.showwarning("警告", "请先创建模型！")
            return

        if self.is_training:
            messagebox.showwarning("警告", "训练正在进行中！")
            return

        # 检查数据
        if not hasattr(self.main_gui, 'rcs_data') or self.main_gui.rcs_data is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        # 在新线程中启动训练
        self.is_training = True
        self.training_thread = threading.Thread(target=self._train_model_thread, daemon=True)
        self.training_thread.start()

        self._log("🚀 训练已启动（后台线程）")

    def _train_model_thread(self):
        """训练线程（后台执行）"""
        try:
            from angle_based_rcs.data.angle_dataset import create_dataloaders
            from angle_based_rcs.training.angle_trainer import AngleRCSTrainer

            self._log("=" * 60)
            self._log("开始训练Angle-based RCS模型")
            self._log("=" * 60)

            # 获取数据
            rcs_data = self.main_gui.rcs_data  # [N, 91, 91, num_freq]
            param_data = self.main_gui.param_data  # [N, 9]

            self._log(f"数据形状: RCS {rcs_data.shape}, 参数 {param_data.shape}")

            # 创建DataLoader
            train_subset_size = self.angle_rcs_subset_size.get() if self.angle_rcs_use_subset.get() else None

            train_loader, val_loader, sampler = create_dataloaders(
                rcs_data=rcs_data,
                param_data=param_data,
                batch_size=self.angle_rcs_batch_size.get(),
                num_frequencies=self.angle_rcs_system['num_frequencies'],
                train_split=self.angle_rcs_train_split.get(),
                random_seed=42,
                train_subset_size=train_subset_size,
                normalize_params=self.angle_rcs_normalize_params.get(),
                num_workers=0
            )

            self._log(f"训练集: {len(train_loader)} batches")
            self._log(f"验证集: {len(val_loader)} batches")
            self._log("")

            # 创建训练器
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._log(f"使用设备: {device}")

            trainer = AngleRCSTrainer(
                model=self.angle_rcs_system['model'],
                device=device,
                learning_rate=self.angle_rcs_lr.get(),
                weight_decay=self.angle_rcs_weight_decay.get(),
                optimizer_type=self.angle_rcs_optimizer.get(),
                scheduler_type=self.angle_rcs_scheduler.get(),
                patience=self.angle_rcs_patience.get()
            )

            self._log("训练器配置:")
            self._log(f"  • 优化器: {self.angle_rcs_optimizer.get()}")
            self._log(f"  • 学习率: {self.angle_rcs_lr.get()}")
            self._log(f"  • 调度器: {self.angle_rcs_scheduler.get()}")
            self._log(f"  • Patience: {self.angle_rcs_patience.get()}")
            self._log("")

            # 训练
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.angle_rcs_epochs.get(),
                log_callback=self._log  # 传递日志回调
            )

            # 保存训练历史
            self.training_history = history

            self._log("")
            self._log("=" * 60)
            self._log("✅ 训练完成！")
            self._log("=" * 60)

            # 绘制训练曲线
            self.main_gui.after(0, self._plot_training_curves)

        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            self._log(f"❌ {error_msg}")
            import traceback
            self._log(traceback.format_exc())

        finally:
            self.is_training = False

    def _stop_training(self):
        """停止训练"""
        if not self.is_training:
            messagebox.showinfo("提示", "当前没有正在进行的训练")
            return

        # TODO: 实现训练停止逻辑
        messagebox.showinfo("提示", "训练停止功能待实现")

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

    def _visualize_prediction(self):
        """可视化预测结果"""
        if self.angle_rcs_system is None:
            messagebox.showwarning("警告", "请先创建或加载模型！")
            return

        # TODO: 实现可视化逻辑
        messagebox.showinfo("提示", "可视化功能待实现")

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
