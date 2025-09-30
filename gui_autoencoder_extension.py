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
        """创建左侧配置面板"""
        # 1. 模式选择组
        mode_group = ttk.LabelFrame(parent, text="🔄 AutoEncoder模式选择")
        mode_group.pack(fill=tk.X, pady=(0, 10))

        mode_frame = ttk.Frame(mode_group)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        # 模式选择单选按钮
        ttk.Radiobutton(mode_frame, text="🌊 小波增强模式 (推荐)",
                       variable=self.main_gui.ae_mode, value="wavelet").pack(anchor=tk.W)
        ttk.Label(mode_frame, text="   • 特点：小波预处理 + CNN-AE",
                 font=self.main_gui.font_small).pack(anchor=tk.W)
        ttk.Label(mode_frame, text="   • 优势：更好精度、特征分离、训练稳定",
                 font=self.main_gui.font_small).pack(anchor=tk.W)

        ttk.Radiobutton(mode_frame, text="🔄 直接模式 (高速)",
                       variable=self.main_gui.ae_mode, value="direct").pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(mode_frame, text="   • 特点：直接CNN端到端处理",
                 font=self.main_gui.font_small).pack(anchor=tk.W)
        ttk.Label(mode_frame, text="   • 优势：更快速度、更少参数、简单部署",
                 font=self.main_gui.font_small).pack(anchor=tk.W)

        # 2. 频率配置组（沿用原有）
        freq_group = ttk.LabelFrame(parent, text="📡 频率配置")
        freq_group.pack(fill=tk.X, pady=(0, 10))

        freq_frame = ttk.Frame(freq_group)
        freq_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(freq_frame, text="频率配置:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        freq_combo = ttk.Combobox(freq_frame, textvariable=self.main_gui.ae_freq_config,
                                 values=["2freq", "3freq"], state="readonly", width=10)
        freq_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(freq_frame, text="(2freq: 1.5+3GHz, 3freq: +6GHz)",
                 font=self.main_gui.font_small).grid(row=0, column=2, sticky="w", padx=(5, 0))

        # 3. 模型架构配置组
        model_group = ttk.LabelFrame(parent, text="🏗️ 模型架构配置")
        model_group.pack(fill=tk.X, pady=(0, 10))

        model_frame = ttk.Frame(model_group)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行
        ttk.Label(model_frame, text="隐空间维度:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(model_frame, textvariable=self.main_gui.ae_latent_dim, width=8).grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Label(model_frame, text="Dropout率:").grid(row=0, column=2, sticky="w", padx=(0, 5))
        ttk.Entry(model_frame, textvariable=self.main_gui.ae_dropout_rate, width=8).grid(row=0, column=3, sticky="w")

        # 第二行（小波设置，仅在小波模式下启用）
        ttk.Label(model_frame, text="小波类型:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.wavelet_combo = ttk.Combobox(model_frame, textvariable=self.main_gui.ae_wavelet_type,
                                         values=["db4", "db8", "haar", "bior2.2"], state="readonly", width=8)
        self.wavelet_combo.grid(row=1, column=1, sticky="w", pady=(5, 0))

        # 绑定模式变化事件
        self.main_gui.ae_mode.trace('w', self._on_mode_change)

        # 4. 系统操作组
        ops_group = ttk.LabelFrame(parent, text="🔧 系统操作")
        ops_group.pack(fill=tk.X, pady=(0, 10))

        ops_frame = ttk.Frame(ops_group)
        ops_frame.pack(fill=tk.X, padx=5, pady=5)

        # 单个系统创建
        ttk.Button(ops_frame, text="创建当前模式系统",
                  command=self.create_current_system).pack(fill=tk.X, pady=(0, 5))

        # 双系统创建和对比
        ttk.Button(ops_frame, text="🔄 创建双系统 (对比分析)",
                  command=self.create_dual_systems).pack(fill=tk.X, pady=(0, 5))

        ttk.Button(ops_frame, text="📊 性能对比分析",
                  command=self.run_performance_comparison).pack(fill=tk.X, pady=(0, 5))

        # 5. 小波分析组
        wavelet_group = ttk.LabelFrame(parent, text="🌊 小波变换分析")
        wavelet_group.pack(fill=tk.X, pady=(0, 10))

        wavelet_frame = ttk.Frame(wavelet_group)
        wavelet_frame.pack(fill=tk.X, padx=5, pady=5)

        # 模型选择
        model_select_frame = ttk.Frame(wavelet_frame)
        model_select_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(model_select_frame, text="分析模型:").pack(side=tk.LEFT)
        self.wavelet_model_selection = ttk.Combobox(model_select_frame,
                                                   values=[], width=15, state="readonly")
        self.wavelet_model_selection.pack(side=tk.LEFT, padx=(5, 0))
        self.wavelet_model_selection.set("001 (默认)")

        # 数据类型选择
        data_type_frame = ttk.Frame(wavelet_frame)
        data_type_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(data_type_frame, text="数据类型:").pack(side=tk.LEFT)
        self.wavelet_data_type = tk.StringVar(value="dB")
        ttk.Radiobutton(data_type_frame, text="分贝(dB)", variable=self.wavelet_data_type,
                       value="dB").pack(side=tk.LEFT, padx=(5, 10))
        ttk.Radiobutton(data_type_frame, text="原始数据", variable=self.wavelet_data_type,
                       value="linear").pack(side=tk.LEFT)

        # 分析选项
        ttk.Checkbutton(wavelet_frame, text="显示小波系数",
                       variable=self.wavelet_show_coeffs).pack(anchor=tk.W)
        ttk.Checkbutton(wavelet_frame, text="显示统计分析",
                       variable=self.wavelet_show_stats).pack(anchor=tk.W)

        ttk.Button(wavelet_frame, text="🔬 运行小波分析",
                  command=self.run_wavelet_analysis).pack(fill=tk.X, pady=(5, 0))

        # 6. 模型操作组（沿用原有功能）
        model_ops_group = ttk.LabelFrame(parent, text="💾 模型操作")
        model_ops_group.pack(fill=tk.X, pady=(0, 10))

        model_ops_frame = ttk.Frame(model_ops_group)
        model_ops_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(model_ops_frame, text="保存模型", command=self.main_gui.save_ae_model).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(model_ops_frame, text="加载模型", command=self.main_gui.load_ae_model).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(model_ops_frame, text="开始训练", command=self.main_gui.start_ae_training).pack(fill=tk.X, pady=(0, 2))

    def _create_right_panel(self, parent):
        """创建右侧状态和结果面板"""
        # 创建标签页管理器
        self.result_notebook = ttk.Notebook(parent)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)

        # 1. 系统状态标签页
        status_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(status_frame, text="系统状态")

        # 状态文本
        self.status_text = tk.Text(status_frame, wrap=tk.WORD, height=15, font=self.main_gui.font_small)
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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

        # 初始状态更新
        self._update_status_display()

    def _on_mode_change(self, *args):
        """模式变化回调"""
        mode = self.main_gui.ae_mode.get()

        # 更新小波设置可用性
        if mode == "wavelet":
            self.wavelet_combo.configure(state="readonly")
        else:
            self.wavelet_combo.configure(state="disabled")

        # 更新状态显示
        self._update_status_display()

    def _update_status_display(self):
        """更新状态显示"""
        self.status_text.delete(1.0, tk.END)

        # 更新模型选择列表
        self._update_model_selection()

        status_info = []
        status_info.append("=== AutoEncoder 扩展状态 ===")
        status_info.append(f"当前模式: {self.main_gui.ae_mode.get()}")
        status_info.append(f"频率配置: {self.main_gui.ae_freq_config.get()}")
        status_info.append(f"隐空间维度: {self.main_gui.ae_latent_dim.get()}")

        if hasattr(self.main_gui, 'ae_system') and self.main_gui.ae_system:
            status_info.append(f"主系统状态: 已创建")
            mode = self.main_gui.ae_system.get('mode', '未知')
            status_info.append(f"主系统模式: {mode}")

        status_info.append("")
        status_info.append("=== 双系统状态 ===")
        status_info.append(f"小波系统: {'已创建' if self.wavelet_system else '未创建'}")
        status_info.append(f"直接系统: {'已创建' if self.direct_system else '未创建'}")

        if self.comparison_results:
            status_info.append("")
            status_info.append("=== 对比分析结果 ===")
            status_info.append(f"测试完成时间: {self.comparison_results.get('timestamp', '未知')}")
            status_info.append(f"测试样本数: {self.comparison_results.get('sample_count', 0)}")

        status_info.append("")
        status_info.append("=== 操作提示 ===")
        status_info.append("1. 选择模式并创建系统")
        status_info.append("2. 加载数据后可运行对比分析")
        status_info.append("3. 小波分析可独立运行")

        for line in status_info:
            self.status_text.insert(tk.END, line + "\\n")

    def _update_model_selection(self):
        """更新模型选择列表"""
        if hasattr(self.main_gui, 'rcs_data') and self.main_gui.rcs_data is not None:
            num_models = len(self.main_gui.rcs_data)
            model_options = [f"{i+1:03d}" for i in range(num_models)]
            self.wavelet_model_selection['values'] = model_options

            # 如果当前选择不在列表中，重置为默认
            current = self.wavelet_model_selection.get()
            if not current or current == "001 (默认)":
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

            # 获取配置参数
            freq_config = self.main_gui.ae_freq_config.get()
            latent_dim = self.main_gui.ae_latent_dim.get()
            dropout_rate = self.main_gui.ae_dropout_rate.get()
            wavelet_type = self.main_gui.ae_wavelet_type.get()
            normalize = True

            # 创建系统
            self.main_gui.ae_system = create_autoencoder_system(
                config_name=freq_config,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                wavelet=wavelet_type,
                normalize=normalize,
                mode=mode
            )

            # 添加数据
            self.main_gui.ae_system['rcs_data'] = self.main_gui.rcs_data
            self.main_gui.ae_system['param_data'] = self.main_gui.param_data

            self.main_gui.ae_log(f"✅ {mode}模式系统创建成功!")

            # 更新原有GUI状态
            self.main_gui.update_ae_status()
            self._update_status_display()

            messagebox.showinfo("成功", f"{mode}模式AutoEncoder系统创建成功！")

        except Exception as e:
            error_msg = f"创建系统失败: {e}"
            self.main_gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def create_dual_systems(self):
        """创建双系统用于对比分析"""
        try:
            if not self.main_gui.data_loaded:
                messagebox.showwarning("警告", "请先加载数据！")
                return

            self.main_gui.ae_log("🔄 开始创建双系统...")

            # 导入所需模块
            import sys
            sys.path.append('autoencoder')
            from autoencoder.utils.frequency_config import create_autoencoder_system

            # 获取配置参数
            freq_config = self.main_gui.ae_freq_config.get()
            latent_dim = self.main_gui.ae_latent_dim.get()
            dropout_rate = self.main_gui.ae_dropout_rate.get()
            wavelet_type = self.main_gui.ae_wavelet_type.get()
            normalize = True

            # 创建小波增强系统
            self.main_gui.ae_log("🌊 创建小波增强系统...")
            self.wavelet_system = create_autoencoder_system(
                config_name=freq_config,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                wavelet=wavelet_type,
                normalize=normalize,
                mode='wavelet'
            )

            # 创建直接系统
            self.main_gui.ae_log("🔄 创建直接系统...")
            self.direct_system = create_autoencoder_system(
                config_name=freq_config,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                wavelet=wavelet_type,
                normalize=normalize,
                mode='direct'
            )

            # 添加数据到两个系统
            for system in [self.wavelet_system, self.direct_system]:
                system['rcs_data'] = self.main_gui.rcs_data
                system['param_data'] = self.main_gui.param_data

            self.main_gui.ae_log("✅ 双系统创建成功!")
            self._update_status_display()

            messagebox.showinfo("成功", "双系统创建成功！现在可以进行性能对比分析。")

        except Exception as e:
            error_msg = f"创建双系统失败: {e}"
            self.main_gui.ae_log(f"❌ {error_msg}")
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
                data_type = self.wavelet_data_type.get()

                # 解析模型选择
                if selected_model and selected_model != "001 (默认)":
                    try:
                        model_idx = int(selected_model.split()[0]) - 1  # 转换为0索引
                        if model_idx >= len(self.main_gui.rcs_data):
                            model_idx = 0
                    except:
                        model_idx = 0
                else:
                    model_idx = 0

                # 执行小波分析
                from wavelet_gui_helper import simple_wavelet_analysis
                import numpy as np

                # 选择分析数据
                sample_data = self.main_gui.rcs_data[model_idx, :, :, 0]  # 取选择的模型的第一个频率

                # 如果选择分贝模式，转换数据用于显示
                if data_type == 'dB':
                    epsilon = 1e-10
                    # 转换为分贝：dB = 10 * log10(RCS)
                    sample_data_db = 10 * np.log10(np.maximum(sample_data, epsilon))
                    analysis_data = sample_data_db
                else:
                    analysis_data = sample_data

                self.main_gui.ae_log(f"📊 执行小波分解和重建 (模型: {selected_model}, 数据类型: {data_type})...")
                analysis_result = simple_wavelet_analysis(
                    analysis_data,
                    wavelet=self.wavelet_analysis_wavelet.get(),
                    data_type=data_type
                )

                self.main_gui.ae_log("📈 生成可视化结果...")
                self.wavelet_analysis_results = analysis_result
                self.current_analysis_model = selected_model
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
            data_type = getattr(self, 'current_analysis_data_type', 'dB')
            fig = create_wavelet_plot(self.wavelet_analysis_results, data_type=data_type, model_name=model_name)

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