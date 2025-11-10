#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验扩展模块

为GUI添加批量对比训练功能：
1. 从AE标签页读取基准配置
2. 选择要对比的维度和值
3. 自动批量训练
4. 生成对比图表和报告

使用方式：
    extension = BatchExperimentExtension(main_gui)
    extension.extend_batch_experiment_tab()
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, List, Any
import numpy as np

from autoencoder.utils.batch_experiment import BatchExperimentManager
from autoencoder.utils.frequency_config import create_autoencoder_system


class BatchExperimentExtension:
    """批量实验功能扩展类"""

    def __init__(self, main_gui):
        """
        初始化扩展模块

        Args:
            main_gui: 主GUI实例
        """
        self.main_gui = main_gui
        self.batch_manager = None
        self.is_training = False

        # 初始化变量
        self._init_extension_vars()

    def _init_extension_vars(self):
        """初始化扩展变量"""
        # 实验设置
        self.batch_experiment_name = tk.StringVar(value="batch_experiment")
        self.batch_save_models = tk.BooleanVar(value=True)
        self.batch_generate_plots = tk.BooleanVar(value=True)

        # 对比维度选择（布尔型，表示是否启用该维度）
        self.compare_mode = tk.BooleanVar(value=False)
        self.compare_architecture = tk.BooleanVar(value=False)
        self.compare_activation = tk.BooleanVar(value=False)
        self.compare_preprocessing = tk.BooleanVar(value=False)
        self.compare_wavelet_type = tk.BooleanVar(value=False)

        # 各维度的值选择（字典：值名 -> BooleanVar）
        self.mode_values = {}
        self.architecture_values = {}
        self.activation_values = {}
        self.preprocessing_values = {}
        self.wavelet_type_values = {}

        # 训练进度
        self.batch_current_experiment = tk.StringVar(value="未开始")
        self.batch_progress = tk.StringVar(value="0/0")

    def extend_batch_experiment_tab(self):
        """扩展批量实验标签页"""
        # 获取批量实验框架
        batch_frame = self.main_gui.batch_experiment_frame

        # 清除现有内容
        for widget in batch_frame.winfo_children():
            widget.destroy()

        # 创建新的布局
        self._create_batch_experiment_layout(batch_frame)

    def _create_batch_experiment_layout(self, parent):
        """创建批量实验布局"""
        # 主容器：左右分栏
        main_container = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧面板：配置
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)

        # 右侧面板：进度和结果
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)

        # 构建左侧面板
        self._create_config_panel(left_panel)

        # 构建右侧面板
        self._create_progress_panel(right_panel)

    def _create_config_panel(self, parent):
        """创建配置面板"""
        # 使用Canvas实现滚动
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

        # 1. 基准配置显示区
        self._create_base_config_section(scrollable_frame)

        # 2. 对比维度选择区
        self._create_compare_dimensions_section(scrollable_frame)

        # 3. 实验设置区
        self._create_experiment_settings_section(scrollable_frame)

        # 4. 控制按钮区
        self._create_control_buttons_section(scrollable_frame)

    def _create_base_config_section(self, parent):
        """创建基准配置显示区"""
        group = ttk.LabelFrame(parent, text="📋 基准配置（从AE页面读取）")
        group.pack(fill=tk.X, padx=5, pady=5)

        frame = ttk.Frame(group)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 配置显示文本框（只读）
        self.base_config_text = tk.Text(frame, height=12, width=60, state='disabled',
                                        font=('Consolas', 9), bg='#f0f0f0')
        self.base_config_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(frame, command=self.base_config_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.base_config_text.config(yscrollcommand=scrollbar.set)

        # 读取按钮
        btn_frame = ttk.Frame(group)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Button(btn_frame, text="🔄 从AE页面读取配置",
                  command=self._load_base_config_from_ae).pack(side=tk.LEFT, padx=5)

        ttk.Label(btn_frame, text="提示: 点击按钮读取当前AE页面的所有配置作为基准",
                 foreground="gray", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=5)

    def _create_compare_dimensions_section(self, parent):
        """创建对比维度选择区"""
        group = ttk.LabelFrame(parent, text="🔍 批量对比维度（可多选）")
        group.pack(fill=tk.X, padx=5, pady=5)

        main_frame = ttk.Frame(group)
        main_frame.pack(fill=tk.X, padx=5, pady=5)

        # 1. AE模式
        self._create_dimension_selector(
            main_frame,
            dimension_name="AE模式",
            dimension_var=self.compare_mode,
            values=['Wavelet模式', 'Direct模式', '可微分小波模式'],
            value_vars=self.mode_values,
            row=0,
            help_text="wavelet: 小波变换, direct: 直接输入, differentiable: 可微分小波"
        )

        # 2. 架构类型
        self._create_dimension_selector(
            main_frame,
            dimension_name="架构类型",
            dimension_var=self.compare_architecture,
            values=['CNN', 'Enhanced_CNN', 'Deep_CNN', 'MLP', 'Dual_Branch_CNN', 'Dual_Branch_MLP'],
            value_vars=self.architecture_values,
            row=1
        )

        # 3. 激活函数
        self._create_dimension_selector(
            main_frame,
            dimension_name="激活函数",
            dimension_var=self.compare_activation,
            values=['relu', 'sin', 'gelu', 'swish', 'tanh', 'mish', 'elu', 'leaky_relu', 'prelu'],
            value_vars=self.activation_values,
            row=2
        )

        # 4. 数据预处理
        self._create_dimension_selector(
            main_frame,
            dimension_name="数据预处理",
            dimension_var=self.compare_preprocessing,
            values=['none', 'zscore', 'minmax', 'zscore+db', 'minmax+db'],
            value_vars=self.preprocessing_values,
            row=3,
            help_text="格式: 标准化方法 或 标准化方法+db（含dB变换）"
        )

        # 5. 小波类型（仅Wavelet模式）
        self._create_dimension_selector(
            main_frame,
            dimension_name="小波类型",
            dimension_var=self.compare_wavelet_type,
            values=['db4', 'db8', 'haar', 'bior2.2'],
            value_vars=self.wavelet_type_values,
            row=4,
            help_text="仅在Wavelet模式下有效"
        )

    def _create_dimension_selector(self, parent, dimension_name: str, dimension_var: tk.BooleanVar,
                                   values: List[str], value_vars: Dict[str, tk.BooleanVar],
                                   row: int, help_text: str = ""):
        """创建单个维度选择器"""
        frame = ttk.LabelFrame(parent, text=f"☐ {dimension_name}")
        frame.grid(row=row, column=0, sticky='ew', pady=5, padx=2)
        parent.columnconfigure(0, weight=1)

        # 启用复选框
        enable_cb = ttk.Checkbutton(frame, text=f"启用{dimension_name}对比",
                                    variable=dimension_var,
                                    command=lambda: self._on_dimension_toggle(frame, dimension_var, value_vars))
        enable_cb.pack(anchor=tk.W, padx=5, pady=2)

        # 帮助文本
        if help_text:
            ttk.Label(frame, text=f"   {help_text}", foreground="gray",
                     font=('TkDefaultFont', 8)).pack(anchor=tk.W, padx=5)

        # 值选择区（初始禁用）
        values_frame = ttk.Frame(frame)
        values_frame.pack(fill=tk.X, padx=15, pady=5)

        # 创建复选框（3列布局）
        for idx, value in enumerate(values):
            if value not in value_vars:
                value_vars[value] = tk.BooleanVar(value=False)

            cb = ttk.Checkbutton(values_frame, text=value, variable=value_vars[value], state='disabled')
            cb.grid(row=idx // 3, column=idx % 3, sticky='w', padx=5, pady=2)

        # 保存引用以便后续启用/禁用
        frame.values_checkbuttons = values_frame.winfo_children()

    def _on_dimension_toggle(self, frame: ttk.Frame, dimension_var: tk.BooleanVar,
                            value_vars: Dict[str, tk.BooleanVar]):
        """维度启用/禁用切换"""
        enabled = dimension_var.get()
        state = 'normal' if enabled else 'disabled'

        # 更新所有值复选框状态
        for cb in frame.values_checkbuttons:
            cb.config(state=state)

        # 如果禁用，清除所有选择
        if not enabled:
            for var in value_vars.values():
                var.set(False)

    def _create_experiment_settings_section(self, parent):
        """创建实验设置区"""
        group = ttk.LabelFrame(parent, text="⚙️ 实验设置")
        group.pack(fill=tk.X, padx=5, pady=5)

        frame = ttk.Frame(group)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 实验名称
        ttk.Label(frame, text="实验名称:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(frame, textvariable=self.batch_experiment_name, width=40).grid(
            row=0, column=1, columnspan=2, sticky='ew', pady=2, padx=5)

        # 选项
        ttk.Checkbutton(frame, text="保存所有模型", variable=self.batch_save_models).grid(
            row=1, column=0, sticky='w', pady=2)
        ttk.Checkbutton(frame, text="生成对比图表", variable=self.batch_generate_plots).grid(
            row=1, column=1, sticky='w', pady=2)

        frame.columnconfigure(1, weight=1)

        # 预计信息（动态更新）
        info_frame = ttk.Frame(group)
        info_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.experiment_count_label = ttk.Label(info_frame, text="预计实验数量: 0",
                                               foreground="blue", font=('TkDefaultFont', 9, 'bold'))
        self.experiment_count_label.pack(anchor=tk.W, padx=5)

        ttk.Button(info_frame, text="🔢 计算实验数量",
                  command=self._update_experiment_count).pack(anchor=tk.W, padx=5, pady=2)

    def _create_control_buttons_section(self, parent):
        """创建控制按钮区"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(frame, text="▶ 开始批量训练", command=self._start_batch_training,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)

        ttk.Button(frame, text="⏸ 停止训练", command=self._stop_batch_training).pack(side=tk.LEFT, padx=5)

        ttk.Button(frame, text="📊 查看结果", command=self._view_results).pack(side=tk.LEFT, padx=5)

    def _create_progress_panel(self, parent):
        """创建进度面板"""
        # 训练进度
        progress_group = ttk.LabelFrame(parent, text="📊 训练进度")
        progress_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame = ttk.Frame(progress_group)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 当前实验
        ttk.Label(frame, text="当前实验:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=2)
        ttk.Label(frame, textvariable=self.batch_current_experiment,
                 foreground="blue").pack(anchor=tk.W, padx=10)

        # 进度条
        ttk.Label(frame, text="总体进度:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(10, 2))
        self.batch_progress_bar = ttk.Progressbar(frame, mode='determinate', length=300)
        self.batch_progress_bar.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame, textvariable=self.batch_progress).pack(anchor=tk.W, padx=10)

        # 日志显示
        log_group = ttk.LabelFrame(parent, text="📝 训练日志")
        log_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        log_frame = ttk.Frame(log_group)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.batch_log_text = tk.Text(log_frame, height=20, width=50, state='disabled',
                                      font=('Consolas', 8), wrap=tk.WORD)
        self.batch_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_scrollbar = ttk.Scrollbar(log_frame, command=self.batch_log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_log_text.config(yscrollcommand=log_scrollbar.set)

    def _load_base_config_from_ae(self):
        """从AE页面读取基准配置"""
        try:
            gui = self.main_gui

            # 读取所有AE配置
            base_config = {
                'mode': gui.ae_mode.get(),
                'architecture': self._map_architecture(gui.ae_architecture_type.get()),
                'activation': gui.ae_activation.get(),
                'wavelet_type': gui.ae_wavelet_type.get(),
                'normalization_method': gui.ae_normalization_method.get(),
                'db_transform': gui.ae_db_transform.get(),
                'latent_dim': int(gui.ae_latent_dim.get()),
                'batch_size': int(gui.ae_batch_size.get()),
                'learning_rate': float(gui.ae_learning_rate.get()),
                'training_mode': self._map_training_mode(gui.ae_training_mode.get()),
                'epochs': {
                    'stage1': int(gui.ae_epochs_stage1.get()),
                    'stage2': int(gui.ae_epochs_stage2.get()),
                    'stage3': int(gui.ae_epochs_stage3.get()),
                }
            }

            # 显示配置
            self.base_config_text.config(state='normal')
            self.base_config_text.delete('1.0', tk.END)

            config_str = "基准配置:\n" + "="*50 + "\n"
            for key, value in base_config.items():
                if isinstance(value, dict):
                    config_str += f"{key}:\n"
                    for k, v in value.items():
                        config_str += f"  {k}: {v}\n"
                else:
                    config_str += f"{key}: {value}\n"

            self.base_config_text.insert('1.0', config_str)
            self.base_config_text.config(state='disabled')

            # 保存配置
            self.cached_base_config = base_config

            self._batch_log("✓ 已从AE页面读取基准配置")

        except Exception as e:
            messagebox.showerror("错误", f"读取配置失败: {str(e)}")

    def _map_architecture(self, gui_architecture: str) -> str:
        """映射GUI架构名称到内部名称"""
        mapping = {
            'CNN': 'cnn',
            'Enhanced_CNN': 'enhanced_cnn',
            'Deep_CNN': 'deep_cnn',
            'MLP': 'mlp',
            'Dual_Branch_CNN': 'dual_branch_cnn',
            'Dual_Branch_MLP': 'dual_branch_mlp'
        }
        return mapping.get(gui_architecture, gui_architecture.lower())

    def _map_mode(self, gui_mode: str) -> str:
        """映射GUI模式名称到内部名称"""
        mapping = {
            'Wavelet模式': 'wavelet',
            'Direct模式': 'direct',
            '可微分小波模式': 'differentiable_wavelet'
        }
        return mapping.get(gui_mode, gui_mode.lower())

    def _map_training_mode(self, gui_mode: str) -> str:
        """映射GUI训练模式到内部名称"""
        mapping = {
            '三阶段训练': 'three_stage',
            '仅Stage 1': 'stage1_only'
        }
        return mapping.get(gui_mode, 'three_stage')

    def _update_experiment_count(self):
        """计算并显示预计实验数量"""
        try:
            count = 1

            # 统计各维度选择的值数量
            if self.compare_mode.get():
                selected = sum(1 for var in self.mode_values.values() if var.get())
                if selected > 0:
                    count *= selected

            if self.compare_architecture.get():
                selected = sum(1 for var in self.architecture_values.values() if var.get())
                if selected > 0:
                    count *= selected

            if self.compare_activation.get():
                selected = sum(1 for var in self.activation_values.values() if var.get())
                if selected > 0:
                    count *= selected

            if self.compare_preprocessing.get():
                selected = sum(1 for var in self.preprocessing_values.values() if var.get())
                if selected > 0:
                    count *= selected

            if self.compare_wavelet_type.get():
                selected = sum(1 for var in self.wavelet_type_values.values() if var.get())
                if selected > 0:
                    count *= selected

            self.experiment_count_label.config(text=f"预计实验数量: {count}")

            # 如果count == 1说明没有选择任何对比维度
            if count == 1:
                if not any([self.compare_mode.get(), self.compare_architecture.get(),
                           self.compare_activation.get(), self.compare_preprocessing.get(),
                           self.compare_wavelet_type.get()]):
                    messagebox.showwarning("警告", "请至少选择一个对比维度！")

        except Exception as e:
            messagebox.showerror("错误", f"计算失败: {str(e)}")

    def _start_batch_training(self):
        """开始批量训练"""
        if self.is_training:
            messagebox.showinfo("提示", "训练正在进行中...")
            return

        # 检查基准配置
        if not hasattr(self, 'cached_base_config'):
            messagebox.showwarning("警告", "请先从AE页面读取基准配置！")
            return

        # 检查数据
        if self.main_gui.rcs_data is None or self.main_gui.param_data is None:
            messagebox.showerror("错误", "请先加载训练数据！")
            return

        # 收集对比维度
        compare_dimensions = self._collect_compare_dimensions()
        if not compare_dimensions:
            messagebox.showwarning("警告", "请至少选择一个对比维度并勾选对应的值！")
            return

        # 确认开始
        total_experiments = 1
        for values in compare_dimensions.values():
            total_experiments *= len(values)

        msg = f"将要运行 {total_experiments} 个实验，可能需要较长时间。\n\n确认开始？"
        if not messagebox.askyesno("确认", msg):
            return

        # 在后台线程执行
        self.is_training = True
        thread = threading.Thread(target=self._run_batch_training_thread,
                                 args=(compare_dimensions,))
        thread.daemon = True
        thread.start()

    def _collect_compare_dimensions(self) -> Dict[str, List[Any]]:
        """收集启用的对比维度和选择的值"""
        dimensions = {}

        if self.compare_mode.get():
            selected = [name for name, var in self.mode_values.items() if var.get()]
            if selected:
                # 映射到内部名称
                selected = [self._map_mode(name) for name in selected]
                dimensions['mode'] = selected

        if self.compare_architecture.get():
            selected = [name for name, var in self.architecture_values.items() if var.get()]
            if selected:
                # 映射到内部名称
                selected = [self._map_architecture(name) for name in selected]
                dimensions['architecture'] = selected

        if self.compare_activation.get():
            selected = [name for name, var in self.activation_values.items() if var.get()]
            if selected:
                dimensions['activation'] = selected

        if self.compare_preprocessing.get():
            selected = [name for name, var in self.preprocessing_values.items() if var.get()]
            if selected:
                # 解析预处理配置
                parsed = []
                for item in selected:
                    if '+db' in item:
                        norm_method = item.replace('+db', '')
                        parsed.append({'normalization_method': norm_method, 'db_transform': True})
                    else:
                        parsed.append({'normalization_method': item, 'db_transform': False})
                dimensions['preprocessing'] = parsed

        if self.compare_wavelet_type.get():
            selected = [name for name, var in self.wavelet_type_values.items() if var.get()]
            if selected:
                dimensions['wavelet_type'] = selected

        return dimensions

    def _run_batch_training_thread(self, compare_dimensions: Dict):
        """批量训练线程"""
        try:
            self._batch_log("="*80)
            self._batch_log("开始批量实验")
            self._batch_log("="*80)

            # 创建BatchExperimentManager
            self.batch_manager = BatchExperimentManager(
                base_config=self.cached_base_config,
                compare_dimensions=compare_dimensions,
                experiment_name=self.batch_experiment_name.get(),
                save_dir="batch_experiments"
            )

            # 打开批量实验日志文件
            import os
            from datetime import datetime
            log_filename = f"batch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_path = os.path.join(self.batch_manager.experiment_dir, log_filename)
            self.main_gui.batch_experiment_log_file = open(log_path, 'w', encoding='utf-8')

            # 同时设置给 batch_manager，让它的所有输出也写入日志
            self.batch_manager.log_file = self.main_gui.batch_experiment_log_file

            # 写入日志头部
            self.main_gui.batch_experiment_log_file.write("="*80 + "\n")
            self.main_gui.batch_experiment_log_file.write(f"批量实验训练日志\n")
            self.main_gui.batch_experiment_log_file.write(f"实验名称: {self.batch_experiment_name.get()}\n")
            self.main_gui.batch_experiment_log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.main_gui.batch_experiment_log_file.write("="*80 + "\n\n")
            self.main_gui.batch_experiment_log_file.flush()

            # 设置回调
            self.batch_manager.on_experiment_start = self._on_experiment_start
            self.batch_manager.on_experiment_complete = self._on_experiment_complete

            # 执行批量训练
            results = self.batch_manager.run_batch_experiments(
                training_func=self._training_wrapper,
                rcs_data=self.main_gui.rcs_data,
                param_data=self.main_gui.param_data,
                ae_system_creator=self._create_ae_system,
                save_models=self.batch_save_models.get(),
                evaluator_func=self._evaluate_model,
                visualizer_func=self._visualize_model,
                n_samples_visualize=3
            )

            self._batch_log("="*80)
            self._batch_log(f"批量实验完成！共 {len(results)} 个实验")
            self._batch_log(f"实验目录: {self.batch_manager.experiment_dir}")
            self._batch_log(f"训练日志: {log_filename}")
            self._batch_log("="*80)

            messagebox.showinfo("完成", f"批量实验完成！\n实验目录: {self.batch_manager.experiment_dir}\n训练日志: {log_filename}")

        except Exception as e:
            self._batch_log(f"✗ 批量实验失败: {str(e)}")
            messagebox.showerror("错误", f"批量实验失败: {str(e)}")

        finally:
            # 关闭批量实验日志文件
            if hasattr(self.main_gui, 'batch_experiment_log_file') and self.main_gui.batch_experiment_log_file is not None:
                try:
                    self.main_gui.batch_experiment_log_file.write("\n" + "="*80 + "\n")
                    self.main_gui.batch_experiment_log_file.write(f"批量实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    self.main_gui.batch_experiment_log_file.write("="*80 + "\n")
                    self.main_gui.batch_experiment_log_file.close()
                    self.main_gui.batch_experiment_log_file = None
                except Exception as e:
                    print(f"关闭日志文件失败: {e}")

            self.is_training = False
            self.batch_current_experiment.set("已完成")

    def _create_ae_system(self, config: Dict) -> Dict:
        """创建AutoEncoder系统"""
        # 处理预处理配置（可能是字典）
        if 'preprocessing' in config and isinstance(config['preprocessing'], dict):
            normalization_method = config['preprocessing']['normalization_method']
            db_transform = config['preprocessing']['db_transform']
        else:
            normalization_method = config.get('normalization_method', 'zscore')
            db_transform = config.get('db_transform', False)

        # 创建系统
        ae_system = create_autoencoder_system(
            config_name='2freq',  # 固定使用2freq
            mode=config['mode'],
            architecture=config['architecture'],
            latent_dim=config['latent_dim'],
            wavelet=config.get('wavelet_type', 'db4'),  # 修正参数名: wavelet而非wavelet_type
            dropout_rate=0.2,
            activation=config.get('activation', 'relu'),
            normalization_method=normalization_method,
            db_transform=db_transform  # 修正参数名: db_transform而非log_transform
        )

        return ae_system

    def _training_wrapper(self, rcs_data, param_data, training_config, ae_system):
        """训练包装器（调用主GUI的训练函数）"""
        # ⚠️ 关键: 训练函数期望通过self.gui.ae_system访问系统
        # 需要临时设置ae_system到主GUI中
        original_ae_system = self.main_gui.ae_system

        try:
            # ⚠️ Bug修复2: data_adapter通过adapt_rcs_data自动计算统计信息
            # RCS_DataAdapter没有fit方法，而是在adapt_rcs_data时自动计算mean/std
            # 这里需要先调用一次adapt_rcs_data来初始化统计信息
            data_adapter = ae_system.get('data_adapter', None)
            if data_adapter is not None:
                mode = ae_system.get('mode', 'direct')

                # 根据模式决定预处理的数据
                if mode == 'wavelet':
                    # Wavelet模式: 先小波变换，再调用adapt_rcs_data计算统计
                    import torch
                    wavelet_transform = ae_system.get('wavelet_transform', None)
                    if wavelet_transform:
                        rcs_tensor = torch.FloatTensor(rcs_data)
                        wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
                        # 调用adapt_rcs_data会自动计算并保存统计信息到data_stats
                        _ = data_adapter.adapt_rcs_data(wavelet_coeffs.numpy())
                    else:
                        # 如果没有wavelet_transform，直接处理RCS
                        _ = data_adapter.adapt_rcs_data(rcs_data)
                else:
                    # Direct模式: 直接处理RCS数据，自动计算统计
                    _ = data_adapter.adapt_rcs_data(rcs_data)

            # 设置当前实验的ae_system
            self.main_gui.ae_system = ae_system

            # 将数据集成到ae_system中（训练函数期望）
            self.main_gui.ae_system['rcs_data'] = rcs_data
            self.main_gui.ae_system['param_data'] = param_data

            # ⚠️ Bug修复1: training_config是experiment config，需要构建完整的training_config
            # 从主GUI读取基础训练配置（包含lr_scheduler等必需字段）
            full_training_config = self.main_gui._create_ae_training_config()

            # 用experiment config覆盖training_mode（如果存在）
            if 'training_mode' in training_config:
                full_training_config['training_mode'] = training_config['training_mode']

            # 启用批量实验模式（禁用弹窗）
            from gui_managers.managers.training_manager import TrainingManager
            if hasattr(self.main_gui, 'training_manager') and isinstance(self.main_gui.training_manager, TrainingManager):
                original_batch_mode = self.main_gui.training_manager.batch_experiment_mode
                self.main_gui.training_manager.batch_experiment_mode = True
            else:
                original_batch_mode = False

            try:
                # 调用主GUI的三阶段训练函数（只传3个参数）
                self.main_gui._run_three_stage_training_v2(
                    rcs_data=rcs_data,
                    param_data=param_data,
                    training_config=full_training_config
                )

                # 训练完成后，从主GUI获取训练历史
                training_history = getattr(self.main_gui, 'ae_training_history', None)
                return training_history

            finally:
                # 恢复批量实验模式标志
                if hasattr(self.main_gui, 'training_manager') and isinstance(self.main_gui.training_manager, TrainingManager):
                    self.main_gui.training_manager.batch_experiment_mode = original_batch_mode

        finally:
            # 恢复原始ae_system
            self.main_gui.ae_system = original_ae_system

    def _on_experiment_start(self, index: int, total: int, config: Dict):
        """实验开始回调"""
        experiment_id = config.get('experiment_id', f'experiment_{index}')
        self.batch_current_experiment.set(f"{experiment_id} ({index}/{total})")
        self.batch_progress.set(f"{index}/{total}")

        progress_percent = (index - 1) / total * 100
        self.batch_progress_bar['value'] = progress_percent

        self._batch_log(f"\n[{index}/{total}] 开始实验: {experiment_id}")

    def _on_experiment_complete(self, index: int, total: int, result):
        """实验完成回调"""
        progress_percent = index / total * 100
        self.batch_progress_bar['value'] = progress_percent

        status = "✓" if result.status == "completed" else "✗"
        self._batch_log(f"{status} 实验完成: {result.experiment_id} ({result.training_time:.1f}秒)")

        if result.final_metrics:
            self._batch_log(f"  指标: {result.final_metrics}")

    def _batch_log(self, message: str):
        """添加日志"""
        self.batch_log_text.config(state='normal')
        self.batch_log_text.insert(tk.END, message + "\n")
        self.batch_log_text.see(tk.END)
        self.batch_log_text.config(state='disabled')

    def _stop_batch_training(self):
        """停止批量训练"""
        if not self.is_training:
            messagebox.showinfo("提示", "当前没有训练在进行")
            return

        # TODO: 实现优雅停止（需要在训练循环中检查标志）
        messagebox.showinfo("提示", "暂不支持中途停止，请等待当前实验完成")

    def _evaluate_model(self, ae_system: Dict, rcs_data: np.ndarray,
                       param_data: np.ndarray, indices: List[int]) -> Dict[str, float]:
        """
        评估模型性能（使用统一重建和评估接口）

        Args:
            ae_system: AutoEncoder系统
            rcs_data: RCS数据
            param_data: 参数数据
            indices: 评估样本索引

        Returns:
            评估指标字典 {'mse': ..., 'rmse': ..., 'mae': ...}
        """
        # ===== 使用统一的重建和评估接口 =====
        from autoencoder.utils.reconstruction import reconstruct_from_params, get_device_from_model
        from autoencoder.utils.evaluation import compute_reconstruction_metrics

        # 提取评估样本
        eval_rcs = rcs_data[indices]
        eval_params = param_data[indices]

        # 获取设备
        device = get_device_from_model(ae_system)

        # 从参数预测RCS（使用统一重建函数）
        predicted_rcs = reconstruct_from_params(ae_system, eval_params, device=device)

        # 计算评估指标（使用统一评估函数）
        metrics = compute_reconstruction_metrics(eval_rcs, predicted_rcs, per_frequency=False)

        # 返回兼容的指标格式（只包含基本指标）
        return {
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae']
        }
        # ===== 统一接口调用结束 =====

    def _visualize_model(self, ae_system: Dict, rcs_data: np.ndarray,
                        param_data: np.ndarray, indices: List[int],
                        save_dir: str, experiment_id: str):
        """
        生成模型可视化图表（使用统一重建接口）

        Args:
            ae_system: AutoEncoder系统
            rcs_data: RCS数据
            param_data: 参数数据
            indices: 可视化样本索引
            save_dir: 保存目录
            experiment_id: 实验ID（用于文件名前缀）
        """
        # ===== 使用统一的重建接口 =====
        from autoencoder.utils.reconstruction import reconstruct_from_params, get_device_from_model
        import torch

        # 获取设备和模式
        device = get_device_from_model(ae_system)
        mode = ae_system.get('mode', 'direct')
        is_wavelet_mode = mode in ('wavelet', 'differentiable_wavelet')

        # 逐个样本可视化
        for idx, sample_idx in enumerate(indices):
            # 获取单个样本
            params = param_data[sample_idx:sample_idx+1]
            true_rcs = rcs_data[sample_idx]

            # 从参数预测RCS（使用统一重建函数）
            predicted_rcs = reconstruct_from_params(ae_system, params, device=device)
            predicted_rcs_np = predicted_rcs[0]  # [91, 91, num_freq]

            # 生成RCS对比图（包含真实、预测、残差）
            self._plot_rcs_heatmap_comparison(
                true_rcs, predicted_rcs_np,
                save_dir, f"{experiment_id}_sample{idx}.png",
                experiment_id=experiment_id
            )

            # 如果是Wavelet模式，额外生成小波系数对比图
            if is_wavelet_mode:
                try:
                    autoencoder = ae_system['autoencoder']
                    parameter_mapper = ae_system['parameter_mapper']
                    wavelet_transform = ae_system.get('wavelet_transform', None)
                    data_adapter = ae_system.get('data_adapter', None)

                    if wavelet_transform is not None:
                        autoencoder.eval()
                        parameter_mapper.eval()

                        with torch.no_grad():
                            # 1. 计算原始小波系数
                            rcs_tensor = torch.FloatTensor(true_rcs).unsqueeze(0).to(device)  # [1, 91, 91, num_freq]
                            original_wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)  # [1, 49, 49, num_freq*4]

                            # 2. 标准化（如果有data_adapter）
                            if data_adapter:
                                original_wavelet_coeffs_np = original_wavelet_coeffs.cpu().numpy()[0]
                                original_wavelet_coeffs_adapted = data_adapter.adapt_rcs_data(original_wavelet_coeffs_np)
                                original_wavelet_coeffs = torch.FloatTensor(original_wavelet_coeffs_adapted).unsqueeze(0).to(device)

                            # 3. 通过AutoEncoder重建小波系数
                            latent = autoencoder.encode(original_wavelet_coeffs)
                            reconstructed_wavelet_coeffs = autoencoder.decode(latent)  # [1, 49, 49, num_freq*4]

                            # 4. 逆标准化（如果有data_adapter）
                            if data_adapter:
                                reconstructed_wavelet_coeffs_np = reconstructed_wavelet_coeffs.cpu().numpy()[0]
                                reconstructed_wavelet_coeffs_np = data_adapter.inverse_adapt(reconstructed_wavelet_coeffs_np)
                                original_wavelet_coeffs_np = data_adapter.inverse_adapt(original_wavelet_coeffs.cpu().numpy()[0])
                            else:
                                reconstructed_wavelet_coeffs_np = reconstructed_wavelet_coeffs.cpu().numpy()[0]
                                original_wavelet_coeffs_np = original_wavelet_coeffs.cpu().numpy()[0]

                            # 5. 为每个频率生成小波系数对比图
                            self._plot_wavelet_coeffs_comparison(
                                original_wavelet_coeffs_np,
                                reconstructed_wavelet_coeffs_np,
                                save_dir,
                                f"{experiment_id}_sample{idx}_wavelet.png",
                                experiment_id=experiment_id
                            )

                except Exception as e:
                    print(f"  ⚠ 生成小波系数对比图失败: {str(e)}")

        # ===== 统一接口调用结束 =====

    def _plot_rcs_heatmap_comparison(self, true_rcs, pred_rcs, save_dir, filename, experiment_id="N/A"):
        """绘制RCS热图对比（真实vs预测）- 使用统一绘图接口"""
        import os
        from autoencoder.utils.plotting import plot_rcs_comparison

        freq_labels = ['1.5 GHz', '3.0 GHz']
        num_freq = true_rcs.shape[-1]

        # 为每个频率生成对比图
        for freq_idx in range(num_freq):
            freq_label = freq_labels[freq_idx] if freq_idx < len(freq_labels) else f"Freq {freq_idx}"

            # 提取该频率的数据
            true_rcs_2d = true_rcs[:, :, freq_idx]
            pred_rcs_2d = pred_rcs[:, :, freq_idx]

            # 使用统一绘图函数（批量实验使用2倍尺寸）
            save_path = os.path.join(save_dir, filename.replace('.png', f'_freq{freq_idx}.png'))
            plot_rcs_comparison(
                true_rcs=true_rcs_2d,
                pred_rcs=pred_rcs_2d,
                freq_label=freq_label,
                model_id=experiment_id,
                figsize=(30, 10),  # 批量实验图片尺寸翻倍
                save_path=save_path
            )

    def _plot_wavelet_coeffs_comparison(self, original_coeffs, reconstructed_coeffs,
                                       save_dir, filename, experiment_id="N/A"):
        """绘制小波系数对比图（原始vs重建）- 使用统一绘图接口"""
        import os
        from autoencoder.utils.plotting import plot_wavelet_coefficients_comparison

        freq_labels = ['1.5 GHz', '3.0 GHz', '6.0 GHz']

        # 推断频率数量（小波系数格式: [H, W, num_freq*4]）
        num_freq = original_coeffs.shape[-1] // 4

        # 为每个频率生成小波系数对比图
        for freq_idx in range(num_freq):
            freq_label = freq_labels[freq_idx] if freq_idx < len(freq_labels) else f"Freq {freq_idx}"

            # 使用统一绘图函数（批量实验使用大尺寸）
            save_path = os.path.join(save_dir, filename.replace('.png', f'_freq{freq_idx}.png'))
            plot_wavelet_coefficients_comparison(
                original_coeffs=original_coeffs,
                reconstructed_coeffs=reconstructed_coeffs,
                freq_idx=freq_idx,
                freq_label=freq_label,
                model_id=experiment_id,
                figsize=(20, 25),  # 批量实验使用大尺寸
                fontsize_scale=1.0,
                save_path=save_path
            )

    def _plot_residual(self, true_rcs, pred_rcs, save_dir, filename):
        """绘制残差分布图"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        freq_labels = ['1.5 GHz', '3.0 GHz']

        for freq_idx in range(2):
            ax = axes[freq_idx]
            residual = (true_rcs[:, :, freq_idx] - pred_rcs[:, :, freq_idx]).flatten()

            ax.hist(residual, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f'Residual Distribution - {freq_labels[freq_idx]}', fontsize=12)
            ax.set_xlabel('Residual (True - Predicted)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            mean_res = np.mean(residual)
            std_res = np.std(residual)
            ax.axvline(mean_res, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_res:.4f}')
            ax.axvline(mean_res + std_res, color='orange', linestyle=':', linewidth=1.5,
                      label=f'±Std: {std_res:.4f}')
            ax.axvline(mean_res - std_res, color='orange', linestyle=':', linewidth=1.5)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def _view_results(self):
        """查看结果"""
        if self.batch_manager is None:
            messagebox.showinfo("提示", "还没有实验结果")
            return

        # 打开实验目录
        import subprocess
        import sys
        try:
            if os.name == 'nt':  # Windows
                os.startfile(self.batch_manager.experiment_dir)
            elif os.name == 'posix':  # macOS/Linux
                subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open',
                                 self.batch_manager.experiment_dir])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录: {str(e)}")


if __name__ == "__main__":
    print("批量实验扩展模块")
    print("请在主GUI中使用此模块")
