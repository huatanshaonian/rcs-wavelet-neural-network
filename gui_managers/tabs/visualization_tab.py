import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sys
import os
import json # 用于生成报告

# 从app中获取字体配置
# from gui import setup_matplotlib_font # 字体设置在app启动时完成

class VisualizationTab(ttk.Frame):
    """
    可视化标签页
    负责各种数据和模型结果的可视化展示。
    """

    def __init__(self, notebook, app):
        """
        初始化可视化标签页。

        参数:
            notebook: 父容器 (ttk.Notebook)
            app: 主应用程序实例 (RCSWaveletGUI)，用于访问共享状态和配置。
        """
        super().__init__(notebook)
        self.app = app

        # 引用主应用的字体
        self.font_small = getattr(app, 'font_small', None)
        self.font_medium = getattr(app, 'font_medium', None)
        self.font_large = getattr(app, 'font_large', None)

        # 可视化UI变量
        self.vis_model_var = tk.StringVar(value="001")
        self.vis_aux_model_var = tk.StringVar(value="002")  # 辅助样本ID（用于插值）
        self.vis_freq_var = tk.StringVar(value="1.5G")
        self.fontsize_scale_var = tk.DoubleVar(value=1.0)
        self.vis_type_var = tk.StringVar(value="2D热图")
        self.color_param_var = tk.StringVar(value="参数1(kw)")  # 隐空间分布着色参数
        self.ae_reconstruction_mode_var = tk.StringVar(value="auto")  # AutoEncoder重建模式

        # Matplotlib图形初始化 (作为属性，以便在各种方法中访问和更新)
        self.vis_fig = Figure(figsize=(12, 8), dpi=80)
        
        self.create_widgets()
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 可视化控制组
        control_group = ttk.LabelFrame(main_frame, text="可视化控制")
        control_group.pack(fill=tk.X, pady=(0, 10))

        control_frame = ttk.Frame(control_group)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 模型选择
        ttk.Label(control_frame, text="模型ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(control_frame, textvariable=self.vis_model_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # 频率选择
        ttk.Label(control_frame, text="频率:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        freq_combo = ttk.Combobox(control_frame, textvariable=self.vis_freq_var,
                                 values=["1.5G", "3G", "6G"], state="readonly", width=8)
        freq_combo.grid(row=0, column=3, padx=5, pady=2)

        # 保存图片按钮
        ttk.Button(control_frame, text="💾 保存图片", command=self.save_current_visualization,
                  width=12).grid(row=0, column=4, padx=5, pady=2)

        # 字号缩放因子
        ttk.Label(control_frame, text="字号缩放:").grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        fontsize_entry = ttk.Entry(control_frame, textvariable=self.fontsize_scale_var, width=6)
        fontsize_entry.grid(row=0, column=6, padx=5, pady=2)
        ttk.Label(control_frame, text="(0.5-3.0)").grid(row=0, column=7, sticky=tk.W, padx=0, pady=2)

        # 可视化类型选择
        ttk.Label(control_frame, text="图表类型:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        type_combo = ttk.Combobox(control_frame, textvariable=self.vis_type_var,
                                 values=["2D热图", "3D表面图", "球坐标图", "对比图", "小波系数对比", "差值分析", "相关性分析",
                                        "训练历史", "统计对比", "AE隐空间分析", "AE隐空间分布", "AE重建质量", "AE参数映射", "AE训练进度", "AE注意力权重", "AE隐空间插值"],
                                 state="readonly", width=12)
        type_combo.grid(row=1, column=1, padx=5, pady=2)

        # 辅助样本ID（用于隐空间插值）
        ttk.Label(control_frame, text="辅助样本ID:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(control_frame, textvariable=self.vis_aux_model_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        # AE重建模式选择（用于对比图等）
        ttk.Label(control_frame, text="AE重建模式:").grid(row=1, column=4, sticky=tk.W, padx=5, pady=2)
        ae_mode_combo = ttk.Combobox(control_frame, textvariable=self.ae_reconstruction_mode_var,
                                     values=["auto", "ae_only", "end_to_end"],
                                     state="readonly", width=12)
        ae_mode_combo.grid(row=1, column=5, padx=5, pady=2)

        # 着色参数选择（用于隐空间分布）
        ttk.Label(control_frame, text="着色参数:").grid(row=1, column=6, sticky=tk.W, padx=5, pady=2)
        param_combo = ttk.Combobox(control_frame, textvariable=self.color_param_var,
                                   values=["参数1(kw)", "参数2(phi)", "参数3(yita)", "参数4(lam)",
                                          "参数5(Ht)", "参数6(Nc)", "参数7(Theta)", "参数8(R)", "参数9(Beta)"],
                                   state="readonly", width=12)
        param_combo.grid(row=1, column=7, padx=5, pady=2)

        # 第三行：生成按钮和提示
        # 生成按钮（左下角）
        ttk.Button(control_frame, text="生成图表", command=self.generate_visualization,
                  style="Accent.TButton").grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # AE重建模式提示
        ttk.Label(control_frame, text="AE重建模式说明: auto(自动) | ae_only(纯AE重建) | end_to_end(参数→RCS)",
                 font=self.font_small, foreground="gray").grid(row=2, column=2, columnspan=6, sticky=tk.W, padx=5, pady=0)

        # 图表显示区域
        chart_group = ttk.LabelFrame(main_frame, text="图表显示")
        chart_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 创建matplotlib图形 (已经在__init__中)
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, chart_group)
        self.vis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加工具栏
        self.vis_toolbar = NavigationToolbar2Tk(self.vis_canvas, chart_group)
        self.vis_toolbar.update()

    def generate_visualization(self):
        """生成可视化图表（支持AutoEncoder和传统网络）"""
        try:
            chart_type = self.vis_type_var.get()

            # 检查模型可用性
            has_traditional_model = self.app.model_trained and self.app.current_model is not None
            has_ae_model = hasattr(self.app, 'ae_system') and self.app.ae_system is not None

            # 分类处理：需要model_id的图表 vs 全局统计图表 vs AutoEncoder特定图表
            if chart_type in ["训练历史", "统计对比"]:
                # 全局统计图表 - 不需要model_id
                if chart_type == "训练历史":
                    self._plot_training_history()
                elif chart_type == "统计对比":
                    self._plot_global_statistics_comparison()
            elif chart_type in ["AE隐空间分析", "AE隐空间分布", "AE重建质量", "AE参数映射", "AE训练进度", "AE注意力权重", "AE隐空间插值"]:
                # AutoEncoder特定图表
                if not has_ae_model:
                    messagebox.showwarning("警告", "AutoEncoder图表需要先训练或加载AutoEncoder模型")
                    return
                if chart_type == "AE注意力权重":
                    self._plot_attention_weights()
                elif chart_type == "AE隐空间插值":
                    self._plot_ae_latent_interpolation()
                elif chart_type == "AE隐空间分布":
                    self._plot_ae_latent_distribution()
                else:
                    self._plot_autoencoder_visualization(chart_type)
            elif chart_type in ["2D热图", "3D表面图", "球坐标图"]:
                # 这些图表始终显示原始RCS数据，不使用模型预测
                model_id = self.vis_model_var.get()
                if not model_id:
                    messagebox.showwarning("警告", "请输入模型ID")
                    return

                freq = self.vis_freq_var.get()

                if chart_type == "2D热图":
                    self._plot_2d_heatmap(model_id, freq)
                elif chart_type == "3D表面图":
                    self._plot_3d_surface(model_id, freq)
                elif chart_type == "球坐标图":
                    self._plot_spherical(model_id, freq)
            else:
                # 对比图、差值分析、相关性分析 - 需要模型预测
                if not has_traditional_model and not has_ae_model:
                    messagebox.showwarning("警告", "请先训练或加载模型")
                    return

                model_id = self.vis_model_var.get()
                if not model_id:
                    messagebox.showwarning("警告", "请输入模型ID")
                    return

                freq = self.vis_freq_var.get()

                if chart_type == "对比图":
                    if has_ae_model:
                        self._plot_ae_comparison()
                    else:
                        self._plot_comparison(model_id)
                elif chart_type == "小波系数对比":
                    if has_ae_model:
                        self._plot_wavelet_coefficients_comparison()
                    else:
                        messagebox.showwarning("警告", "小波系数对比功能需要AutoEncoder模型")
                elif chart_type == "差值分析":
                    self._plot_difference_analysis(model_id)
                elif chart_type == "相关性分析":
                    self._plot_correlation_analysis(model_id)

        except Exception as e:
            messagebox.showerror("错误", f"图表生成失败: {str(e)}")
            self.app.log_message(f"图表生成失败: {str(e)}")
            import traceback
            self.app.log_message(traceback.format_exc())

    def _plot_2d_heatmap(self, model_id, freq):
        """绘制2D热图"""
        return self.app.visualization_manager._plot_2d_heatmap(model_id, freq, self.vis_fig, self.vis_canvas)

    def _plot_3d_surface(self, model_id, freq):
        """绘制3D表面图"""
        return self.app.visualization_manager._plot_3d_surface(model_id, freq, self.vis_fig, self.vis_canvas)

    def _plot_spherical(self, model_id, freq):
        """绘制球坐标图"""
        return self.app.visualization_manager._plot_spherical(model_id, freq, self.vis_fig, self.vis_canvas)

    def _plot_comparison(self, model_id):
        """绘制原始RCS vs 神经网络预测RCS对比图"""
        return self.app.visualization_manager._plot_comparison(model_id, self.app.current_model, self.vis_fig, self.vis_canvas, self.app.rcs_data, self.app.param_data, self.app.log_message)

    def _plot_difference_analysis(self, model_id):
        """绘制差值分析图（原始RCS - 预测RCS）"""
        return self.app.visualization_manager._plot_difference_analysis(model_id, self.app.current_model, self.vis_fig, self.vis_canvas, self.app.rcs_data, self.app.param_data, self.app.log_message)

    def _plot_correlation_analysis(self, model_id):
        """绘制相关性分析图"""
        return self.app.visualization_manager._plot_correlation_analysis(model_id, self.app.current_model, self.vis_fig, self.vis_canvas, self.app.rcs_data, self.app.param_data, self.app.log_message)

    def _plot_training_history(self):
        """绘制训练历史图（对交叉验证，分别保存每折到results文件夹，GUI显示最佳折）"""
        # Training history for traditional network is in app.training_history
        # For AE, it's in app.ae_training_history
        if self.app.training_history:
            return self.app.visualization_manager._plot_training_history(self.app.training_history, self.vis_fig, self.vis_canvas, self.app.log_message)
        elif self.app.ae_training_history:
            # For AE training history, use a specific plot for AE if available, or adapt generic
            return self._plot_ae_training_progress_vis() # Fallback to AE specific plot
        else:
            messagebox.showwarning("警告", "没有可用的训练历史数据。" )


    def _save_fold_plot(self, fold_data, fold_idx, results_dir):
        """保存单个折的训练历史图表"""
        return self.app.visualization_manager._save_fold_plot(fold_data, fold_idx, results_dir)

    def _display_fold_in_gui(self, fold_data, fold_idx):
        """在GUI中显示指定折的训练历史"""
        return self.app.visualization_manager._display_fold_in_gui(fold_data, fold_idx, self.vis_fig, self.vis_canvas, self.app.log_message)

    def _display_simple_training_history(self):
        """显示简单训练模式的历史（非交叉验证）"""
        return self.app.visualization_manager._display_simple_training_history(self.app.training_history, self.vis_fig, self.vis_canvas, self.app.log_message)

    def _plot_global_statistics_comparison(self):
        """改进的全局统计对比分析 - 委托给StatisticsManager处理"""
        # This function belongs to the StatisticsManager, VisualizationTab only calls it.
        # Ensure StatisticsManager is available through app.
        if hasattr(self.app, 'statistics_manager') and self.app.statistics_manager is not None:
            self.app.statistics_manager.plot_global_statistics_comparison(self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data, self.app.param_data, self.app.current_model)
        else:
            self.app.log_message("警告: StatisticsManager未初始化，无法进行全局统计对比分析。" )

    def _plot_autoencoder_visualization(self, chart_type):
        """绘制AutoEncoder特定可视化图表"""
        # This function belongs to the VisualizationManager, VisualizationTab only calls it.
        # Ensure VisualizationManager is available through app.
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_autoencoder_visualization(chart_type, self.app.ae_system, self.app.ae_training_history, self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data, self.app.param_data)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制AutoEncoder特定可视化图表。" )

    def _plot_ae_latent_space(self):
        """绘制AutoEncoder隐空间分析"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_ae_latent_space(self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data, self.app.param_data)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制AutoEncoder隐空间分析。" )

    def _plot_ae_reconstruction_quality(self):
        """绘制AutoEncoder重建质量分析 - 使用统一重建函数"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_ae_reconstruction_quality(self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data, self.app.param_data)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制AutoEncoder重建质量分析。" )

    def _plot_ae_latent_interpolation(self):
        """绘制AutoEncoder隐空间插值"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            sample_id1 = self.vis_model_var.get()
            sample_id2 = self.vis_aux_model_var.get()
            return self.app.visualization_manager._plot_ae_latent_interpolation(
                self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message,
                self.app.rcs_data, sample_id1, sample_id2)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制隐空间插值。" )

    def _plot_ae_latent_distribution(self):
        """绘制AutoEncoder隐空间分布（PCA, t-SNE, UMAP）"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            # 将参数选择转换为索引
            color_param_str = self.color_param_var.get()
            param_map = {
                "参数1(kw)": 0, "参数2(phi)": 1, "参数3(yita)": 2,
                "参数4(lam)": 3, "参数5(Ht)": 4, "参数6(Nc)": 5,
                "参数7(Theta)": 6, "参数8(R)": 7, "参数9(Beta)": 8
            }
            color_param_idx = param_map.get(color_param_str, 0)

            return self.app.visualization_manager._plot_ae_latent_distribution(
                self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message,
                self.app.rcs_data, self.app.param_data, color_param_idx)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制隐空间分布。" )

    def _plot_ae_parameter_mapping(self):
        """绘制AutoEncoder参数映射分析"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_ae_parameter_mapping(self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data, self.app.param_data)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制AutoEncoder参数映射分析。" )

    def _plot_ae_training_progress_vis(self):
        """绘制AutoEncoder训练进度可视化"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_ae_training_progress_vis(self.app.ae_training_history, self.vis_fig, self.vis_canvas, self.app.log_message)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制AutoEncoder训练进度可视化。" )


    def _plot_autoencoder_prediction_visualization(self, chart_type, freq):
        """使用AutoEncoder进行预测可视化"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_autoencoder_prediction_visualization(chart_type, freq, self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data, self.app.param_data)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法使用AutoEncoder进行预测可视化。" )

    def _plot_ae_2d_heatmap(self, freq):
        """绘制AutoEncoder预测的2D热图 - 支持模型未加载时显示原始数据"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_ae_2d_heatmap(freq, self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制AutoEncoder预测的2D热图。" )

    def _plot_original_rcs_fallback(self, freq):
        """当AutoEncoder模型未加载时，显示原始RCS数据作为替代"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_original_rcs_fallback(freq, self.app.rcs_data, self.vis_fig, self.vis_canvas, self.app.log_message)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法显示原始RCS数据。" )

    def _plot_ae_comparison(self):
        """绘制AutoEncoder对比图：原图、重构图、残差图 - 使用统一重建函数"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            # 获取用户选择的重建模式
            reconstruction_mode = self.ae_reconstruction_mode_var.get()
            return self.app.visualization_manager._plot_ae_comparison(
                self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message,
                self.app.rcs_data, self.app.param_data, reconstruction_mode=reconstruction_mode
            )
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制AutoEncoder对比图。" )

    def _plot_attention_weights(self):
        """绘制通道注意力权重历史折线图"""
        import traceback
        try:
            # 检查是否有训练历史
            if not hasattr(self.app, 'ae_training_history') or self.app.ae_training_history is None:
                messagebox.showwarning("警告", "没有训练历史数据\n\n请先训练模型")
                return

            # 获取注意力权重历史
            stage_histories = self.app.ae_training_history.get('stage_histories', {})
            stage1_history = stage_histories.get('stage1', {})
            attention_history = stage1_history.get('attention_history', None)

            if attention_history is None or not attention_history.get('epochs'):
                messagebox.showwarning("警告", "没有注意力权重历史记录\n\n可能原因：\n1. 未启用通道注意力\n2. 训练epoch数不足（需要≥100）")
                return

            epochs = attention_history['epochs']
            weights_list = attention_history['weights']
            channel_names = attention_history['channel_names']

            if not epochs or not weights_list or channel_names is None:
                messagebox.showwarning("警告", "注意力权重数据不完整")
                return

            self.app.log_message(f"📈 绘制注意力权重历史折线图 ({len(epochs)} 个记录点)...")

            # 清除之前的图表
            self.vis_fig.clear()

            # 转换为numpy数组便于处理
            weights_array = np.array(weights_list)  # [n_records, n_channels]
            mode = self.app.ae_system.get('mode', 'wavelet')

            # 获取字号缩放因子
            try:
                fontsize_scale = self.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0


            # 创建子图
            if mode == 'wavelet':
                # Wavelet模式：LL和高频分开绘制
                ax1 = self.vis_fig.add_subplot(2, 1, 1)
                ax2 = self.vis_fig.add_subplot(2, 1, 2)

                # === 子图1: LL通道 ===
                ll_indices = [i for i, name in enumerate(channel_names) if name.startswith('LL')]
                for idx in ll_indices:
                    ax1.plot(epochs, weights_array[:, idx], marker='o', linewidth=2,
                            label=channel_names[idx], alpha=0.8)

                ax1.set_xlabel('Epoch', fontsize=int(11*fontsize_scale), fontweight='bold')
                ax1.set_ylabel('注意力权重', fontsize=int(11*fontsize_scale), fontweight='bold')
                ax1.set_title('LL通道权重演化', fontsize=int(13*fontsize_scale), fontweight='bold')
                ax1.legend(loc='best', fontsize=int(9*fontsize_scale))
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.set_ylim(0, 1.0)

                # === 子图2: 高频通道 ===
                hf_indices = [i for i, name in enumerate(channel_names) if not name.startswith('LL')] 
                for idx in hf_indices:
                    ax2.plot(epochs, weights_array[:, idx], marker='s', linewidth=2,
                            label=channel_names[idx], alpha=0.8, markersize=4)

                ax2.set_xlabel('Epoch', fontsize=int(11*fontsize_scale), fontweight='bold')
                ax2.set_ylabel('注意力权重', fontsize=int(11*fontsize_scale), fontweight='bold')
                ax2.set_title('高频通道权重演化 (LH/HL/HH)', fontsize=int(13*fontsize_scale), fontweight='bold')
                ax2.legend(loc='best', fontsize=int(9*fontsize_scale), ncol=2)
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.set_ylim(0, 1.0)

            else:
                # Direct模式：所有通道在一张图
                ax = self.vis_fig.add_subplot(1, 1, 1)

                for i, name in enumerate(channel_names):
                    ax.plot(epochs, weights_array[:, i], marker='o', linewidth=2.5,
                           label=name, alpha=0.8)

                ax.set_xlabel('Epoch', fontsize=int(12*fontsize_scale), fontweight='bold')
                ax.set_ylabel('注意力权重', fontsize=int(12*fontsize_scale), fontweight='bold')
                ax.set_title('通道注意力权重演化', fontsize=int(14*fontsize_scale), fontweight='bold')
                ax.legend(loc='best', fontsize=int(10*fontsize_scale))
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_ylim(0, 1.0)

            self.vis_fig.tight_layout()
            self.vis_canvas.draw()

            # 打印统计信息
            self.app.log_message("\n" + "="*60)
            self.app.log_message("📊 注意力权重演化统计")
            self.app.log_message("="*60)
            self.app.log_message(f"记录点数: {len(epochs)}")
            self.app.log_message(f"Epoch范围: {epochs[0]} - {epochs[-1]}")

            # 显示初始和最终权重对比
            self.app.log_message(f"\n初始权重 (Epoch {epochs[0]}):")
            for name, w in zip(channel_names, weights_list[0]):
                self.app.log_message(f"  {name:12s}: {w:.4f}")

            self.app.log_message(f"\n最终权重 (Epoch {epochs[-1]}):")
            for name, w in zip(channel_names, weights_list[-1]):
                self.app.log_message(f"  {name:12s}: {w:.4f}")

            # 计算权重变化
            self.app.log_message(f"\n权重变化量:")
            for i, name in enumerate(channel_names):
                change = weights_list[-1][i] - weights_list[0][i]
                self.app.log_message(f"  {name:12s}: {change:+.4f}")

            self.app.log_message("="*60 + "\n")
            self.app.log_message("✅ 注意力权重历史可视化完成！")

        except Exception as e:
            error_msg = f"可视化失败: {str(e)}"
            self.app.log_message(f"❌ {error_msg}")
            traceback.print_exc()
            messagebox.showerror("错误", error_msg)

    def _plot_wavelet_coefficients_comparison(self):
        """绘制小波系数对比图：原始vs重建的4个通道（LL, LH, HL, HH）"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager._plot_wavelet_coefficients_comparison(self.app.ae_system, self.vis_fig, self.vis_canvas, self.app.log_message, self.app.rcs_data, self.app.param_data)
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法绘制小波系数对比图。" )

    def save_current_visualization(self):
        """保存当前显示的可视化图表到results文件夹"""
        if hasattr(self.app, 'visualization_manager') and self.app.visualization_manager is not None:
            return self.app.visualization_manager.save_current_visualization(self.vis_fig, self.vis_type_var.get())
        else:
            self.app.log_message("警告: VisualizationManager未初始化，无法保存当前可视化图表。" )
