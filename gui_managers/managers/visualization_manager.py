"""
可视化管理器
处理所有绘图和可视化相关功能 - Refactored with Strategy Pattern
"""

import tkinter as tk
from tkinter import messagebox
from .plotters.rcs_plotter import RCSPlotter
from .plotters.comparison_plotter import ComparisonPlotter
from .plotters.ae_plotter import AEPlotter
from .plotters.history_plotter import HistoryPlotter

class VisualizationManager:
    """可视化管理器 - 负责所有绘图和可视化功能"""

    def __init__(self, parent_gui):
        """
        初始化可视化管理器
        
        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui
        
        # 初始化各个绘图器
        self.rcs_plotter = RCSPlotter(parent_gui)
        self.comparison_plotter = ComparisonPlotter(parent_gui)
        self.ae_plotter = AEPlotter(parent_gui)
        self.history_plotter = HistoryPlotter(parent_gui)

    def log_message(self, message):
        """代理日志记录"""
        if hasattr(self.gui, 'log_message'):
            self.gui.log_message(message)
        else:
            print(message)

    # --- RCS Plotter Delegates ---

    def _plot_2d_heatmap(self, model_id, freq, fig=None, canvas=None):
        self.rcs_plotter.plot_2d_heatmap(model_id, freq, fig, canvas)

    def _plot_3d_surface(self, model_id, freq, fig=None, canvas=None):
        self.rcs_plotter.plot_3d_surface(model_id, freq, fig, canvas)

    def _plot_spherical(self, model_id, freq, fig=None, canvas=None):
        self.rcs_plotter.plot_spherical(model_id, freq, fig, canvas)
        
    def _plot_original_rcs_fallback(self, freq, rcs_data=None, fig=None, canvas=None, log_callback=None):
        self.rcs_plotter.plot_original_rcs_fallback(freq, rcs_data, fig, canvas, log_callback)

    # --- Comparison Plotter Delegates ---

    def _plot_comparison(self, model_id, current_model=None, fig=None, canvas=None, rcs_data=None, param_data=None, log_callback=None):
        self.comparison_plotter.plot_comparison(model_id, current_model, fig, canvas, log_callback)

    def _plot_difference_analysis(self, model_id, current_model=None, fig=None, canvas=None, rcs_data=None, param_data=None, log_callback=None):
        self.comparison_plotter.plot_difference_analysis(model_id, current_model, fig, canvas, log_callback)

    def _plot_correlation_analysis(self, model_id, current_model=None, fig=None, canvas=None, rcs_data=None, param_data=None, log_callback=None):
        self.comparison_plotter.plot_correlation_analysis(model_id, current_model, fig, canvas, log_callback)
        
    def _plot_angle_rcs_comparison(self, angle_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        self.comparison_plotter.plot_angle_rcs_comparison(angle_system, fig, canvas, log_callback, rcs_data, param_data)

    # --- AE Plotter Delegates ---

    def _plot_autoencoder_visualization(self, chart_type, ae_system=None, training_history=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        self.ae_plotter.plot_autoencoder_visualization(chart_type, ae_system, training_history, fig, canvas, log_callback, rcs_data, param_data)

    def _plot_ae_latent_space(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        self.ae_plotter.plot_ae_latent_space(ae_system, fig, canvas, rcs_data, param_data)

    def _plot_ae_latent_distribution(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None, color_param_idx=0):
        self.ae_plotter.plot_ae_latent_distribution(ae_system, fig, canvas, log_callback, rcs_data, param_data, color_param_idx)

    def _plot_ae_latent_interpolation(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, sample_id1='001', sample_id2='002'):
        self.ae_plotter.plot_ae_latent_interpolation(ae_system, fig, canvas, log_callback, rcs_data, sample_id1, sample_id2)

    def _plot_ae_reconstruction_quality(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        self.ae_plotter.plot_ae_reconstruction_quality(ae_system, fig, canvas, rcs_data, param_data)

    def _plot_ae_parameter_mapping(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        self.ae_plotter.plot_ae_parameter_mapping(ae_system, fig, canvas, rcs_data, param_data)

    def _plot_ae_training_progress_vis(self, training_history=None, fig=None, canvas=None, log_callback=None):
        self.ae_plotter.plot_ae_training_progress_vis(training_history, fig, canvas)

    def _plot_autoencoder_prediction_visualization(self, chart_type, freq, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        self.ae_plotter.plot_autoencoder_prediction_visualization(chart_type, freq, ae_system, fig, canvas, log_callback, rcs_data, param_data)

    def _plot_ae_2d_heatmap(self, freq, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None):
        self.ae_plotter.plot_ae_2d_heatmap(freq, ae_system, fig, canvas, rcs_data)

    def _plot_ae_comparison(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None, reconstruction_mode='auto'):
        self.ae_plotter.plot_ae_comparison(ae_system, fig, canvas, rcs_data, param_data, reconstruction_mode)

    def _plot_wavelet_coefficients_comparison(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        self.ae_plotter.plot_wavelet_coefficients_comparison(ae_system, fig, canvas, rcs_data)
        
    def _plot_ae_branch_comparison(self, ae_system=None, fig=None, canvas=None, log_callback=None, model_id=None, freq_str=None):
        self.ae_plotter.plot_ae_branch_comparison(ae_system, fig, canvas, log_callback, model_id, freq_str)

    # --- History Plotter Delegates ---

    def _plot_training_history(self):
        self.history_plotter.plot_training_history()
        
    def _save_fold_plot(self, fold_data, fold_idx, results_dir, fontsize_scale=None):
        # 注意: 这里的fontsize_scale参数如果传入None，history_plotter内部会自动处理
        self.history_plotter._save_fold_plot(fold_data, fold_idx, results_dir, fontsize_scale)
        
    def _display_fold_in_gui(self, fold_data, fold_idx, fontsize_scale=None):
        self.history_plotter._display_fold_in_gui(fold_data, fold_idx, fontsize_scale)
        
    def _display_simple_training_history(self, fontsize_scale=None):
        self.history_plotter._display_simple_training_history(fontsize_scale)

    def _plot_gradient_history(self):
        self.history_plotter.plot_gradient_history()
        
    def _show_gradient_report(self):
        self.history_plotter.show_gradient_report()

    # --- Shared Utility ---

    def save_current_visualization(self):
        """保存当前显示的可视化图表到results文件夹"""
        import os
        from datetime import datetime

        try:
            # 检查是否有图表可以保存
            if not hasattr(self.gui, 'vis_fig') or self.gui.vis_fig is None:
                messagebox.showwarning("警告", "没有可保存的图表！请先生成图表。" )
                return

            # 创建结果保存目录（使用会话时间戳，与统计对比保持一致）
            timestamp = self.gui.get_ae_session_timestamp() if hasattr(self.gui, 'ae_session_timestamp') else datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join("results", f"visualization_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)

            # 获取当前图表信息用于生成文件名
            # 尝试从visualization_tab获取
            if hasattr(self.gui, 'visualization_tab'):
                model_id = self.gui.visualization_tab.vis_model_var.get()
                freq = self.gui.visualization_tab.vis_freq_var.get()
                chart_type = self.gui.visualization_tab.vis_type_var.get()
            else:
                model_id = self.gui.vis_model_var.get()
                freq = self.gui.vis_freq_var.get()
                chart_type = self.gui.vis_type_var.get()

            # 生成文件名（将图表类型中的空格替换为下划线，不包含时间戳）
            chart_type_safe = chart_type.replace(' ', '_')
            filename = f"vis_{chart_type_safe}_model{model_id}_{freq}.png"
            filepath = os.path.join(results_dir, filename)

            # 保存图表
            self.gui.vis_fig.savefig(filepath, dpi=300, bbox_inches='tight')

            # 显示成功消息
            self.log_message(f"✅ 图表已保存到: {filepath}")
            messagebox.showinfo("保存成功", f"图表已保存到:\n{filepath}")

        except Exception as e:
            error_msg = f"保存图表失败: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("保存失败", error_msg)
            import traceback
            traceback.print_exc()