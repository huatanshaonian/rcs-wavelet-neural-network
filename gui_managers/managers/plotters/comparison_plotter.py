import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tkinter import messagebox
import rcs_visual as rv
from angle_based_rcs.utils.reconstruction import reconstruct_rcs_grid
from autoencoder.utils.plotting import plot_rcs_comparison
from .base_plotter import BasePlotter

class ComparisonPlotter(BasePlotter):
    """负责模型预测与真实值的对比分析"""

    def plot_comparison(self, model_id, current_model=None, fig=None, canvas=None, log_callback=None):
        """绘制原始RCS vs 神经网络预测RCS对比图"""
        # 获取依赖项
        model = current_model if current_model is not None else getattr(self.gui, 'current_model', None)
        is_trained = (model is not None) if current_model is not None else getattr(self.gui, 'model_trained', False)
        
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        # 临时日志覆盖
        original_log = self.log
        if log_callback: self.log = log_callback

        if not is_trained or model is None:
            messagebox.showwarning("警告", "请先训练模型")
            if log_callback: self.log = original_log
            return

        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            if log_callback: self.log = original_log
            return

        try:
            # 清除当前图形
            target_fig.clear()
            fontsize_scale = self.get_fontsize_scale()

            # 获取原始RCS数据
            self.log(f"加载模型 {model_id} 的原始RCS数据...")
            data_1_5g = rv.get_rcs_matrix(model_id, "1.5G", self.gui.data_config['rcs_data_dir'])
            data_3g = rv.get_rcs_matrix(model_id, "3G", self.gui.data_config['rcs_data_dir'])

            # 提取线性值数据
            original_rcs_1_5g = data_1_5g['rcs_linear']
            original_rcs_3g = data_3g['rcs_linear']

            # 获取对应的参数
            params_df = pd.read_csv(self.gui.data_config['params_file'])
            model_params = params_df.iloc[int(model_id) - 1].values.astype(np.float32)

            # 使用神经网络进行预测
            self.log(f"使用神经网络预测模型 {model_id} 的RCS...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            model.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(model_params).unsqueeze(0).to(device)
                predicted_rcs = model(params_tensor).cpu().numpy().squeeze()

            # predicted_rcs shape: [91, 91, 2]
            predicted_rcs_1_5g = predicted_rcs[:, :, 0]  # 1.5GHz
            predicted_rcs_3g = predicted_rcs[:, :, 1]    # 3GHz

            # 原始RCS转换为分贝 (dB = 10 * log10(RCS))
            epsilon = 1e-10
            original_rcs_1_5g_db = 10 * np.log10(np.maximum(original_rcs_1_5g, epsilon))
            original_rcs_3g_db = 10 * np.log10(np.maximum(original_rcs_3g, epsilon))

            # 预测RCS转换为dB
            preprocessing_stats = getattr(self.gui, 'preprocessing_stats', None)
            
            if preprocessing_stats:
                mean = preprocessing_stats['mean']
                std = preprocessing_stats['std']
                predicted_rcs_1_5g_db = predicted_rcs_1_5g * std + mean
                predicted_rcs_3g_db = predicted_rcs_3g * std + mean
                self.log(f"使用preprocessing_stats反标准化: mean={mean:.2f}, std={std:.2f}")
            else:
                predicted_rcs_1_5g_db = 10 * np.log10(np.maximum(predicted_rcs_1_5g, epsilon))
                predicted_rcs_3g_db = 10 * np.log10(np.maximum(predicted_rcs_3g, epsilon))
                self.log("警告: 无preprocessing_stats，假设网络输出为线性值")

            # 计算统一的colorbar范围
            vmin_1_5g = min(original_rcs_1_5g_db.min(), predicted_rcs_1_5g_db.min())
            vmax_1_5g = max(original_rcs_1_5g_db.max(), predicted_rcs_1_5g_db.max())
            vmin_3g = min(original_rcs_3g_db.min(), predicted_rcs_3g_db.min())
            vmax_3g = max(original_rcs_3g_db.max(), predicted_rcs_3g_db.max())

            self.log(f"1.5GHz dB范围: {vmin_1_5g:.1f} ~ {vmax_1_5g:.1f}")
            self.log(f"3GHz dB范围: {vmin_3g:.1f} ~ {vmax_3g:.1f}")

            # 创建2x2子图布局
            phi_range = (-45.0, 45.0)
            theta_range = (45.0, 135.0)
            extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

            # 1.5GHz对比
            ax1 = target_fig.add_subplot(2, 2, 1)
            im1 = ax1.imshow(original_rcs_1_5g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_1_5g, vmax=vmax_1_5g)
            ax1.set_title(f'原始RCS - 1.5GHz (模型{model_id})',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1.ax.tick_params(labelsize=int(16*fontsize_scale))

            ax2 = target_fig.add_subplot(2, 2, 2)
            im2 = ax2.imshow(predicted_rcs_1_5g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_1_5g, vmax=vmax_1_5g)
            ax2.set_title(f'神经网络预测RCS - 1.5GHz',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 3GHz对比
            ax3 = target_fig.add_subplot(2, 2, 3)
            im3 = ax3.imshow(original_rcs_3g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_3g, vmax=vmax_3g)
            ax3.set_title(f'原始RCS - 3GHz (模型{model_id})',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
            cbar3.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar3.ax.tick_params(labelsize=int(16*fontsize_scale))

            ax4 = target_fig.add_subplot(2, 2, 4)
            im4 = ax4.imshow(predicted_rcs_3g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_3g, vmax=vmax_3g)
            ax4.set_title(f'神经网络预测RCS - 3GHz',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
            cbar4.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar4.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 误差统计
            mse_db_1_5g = np.mean((original_rcs_1_5g_db - predicted_rcs_1_5g_db) ** 2)
            mse_db_3g = np.mean((original_rcs_3g_db - predicted_rcs_3g_db) ** 2)
            rmse_db_1_5g = np.sqrt(mse_db_1_5g)
            rmse_db_3g = np.sqrt(mse_db_3g)

            target_fig.suptitle(f'RCS对比分析 (dB) - 模型{model_id}\n1.5GHz RMSE: {rmse_db_1_5g:.2f} dB, 3GHz RMSE: {rmse_db_3g:.2f} dB',
                        fontsize=int(24*fontsize_scale), fontweight='bold', y=0.95)

            for axis in (ax1, ax2, ax3, ax4):
                axis.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            plt.tight_layout()
            target_canvas.draw()

            self.log(f"对比图生成完成")
            self.log(f"1.5GHz预测误差(MSE): {mse_db_1_5g:.6f} dB²")
            self.log(f"3GHz预测误差(MSE): {mse_db_3g:.6f} dB²")

        except Exception as e:
            self.handle_error("对比图生成失败", e)
        finally:
            if log_callback: self.log = original_log

    def plot_difference_analysis(self, model_id, current_model=None, fig=None, canvas=None, log_callback=None):
        """绘制差值分析图（原始RCS - 预测RCS）"""
        # 获取依赖项
        model = current_model if current_model is not None else getattr(self.gui, 'current_model', None)
        is_trained = (model is not None) if current_model is not None else getattr(self.gui, 'model_trained', False)
        
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        original_log = self.log
        if log_callback: self.log = log_callback

        if not is_trained or model is None:
            messagebox.showwarning("警告", "请先训练模型")
            if log_callback: self.log = original_log
            return

        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            if log_callback: self.log = original_log
            return

        try:
            target_fig.clear()
            self.log(f"加载模型 {model_id} 进行差值分析...")
            fontsize_scale = self.get_fontsize_scale()

            data_1_5g = rv.get_rcs_matrix(model_id, "1.5G", self.gui.data_config['rcs_data_dir'])
            data_3g = rv.get_rcs_matrix(model_id, "3G", self.gui.data_config['rcs_data_dir'])

            original_rcs_1_5g = data_1_5g['rcs_linear']
            original_rcs_3g = data_3g['rcs_linear']

            params_df = pd.read_csv(self.gui.data_config['params_file'])
            model_params = params_df.iloc[int(model_id) - 1].values.astype(np.float32)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            model.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(model_params).unsqueeze(0).to(device)
                predicted_rcs = model(params_tensor).cpu().numpy().squeeze()

            epsilon = 1e-10
            original_rcs_1_5g_db = 10 * np.log10(np.maximum(original_rcs_1_5g, epsilon))
            original_rcs_3g_db = 10 * np.log10(np.maximum(original_rcs_3g, epsilon))

            preprocessing_stats = getattr(self.gui, 'preprocessing_stats', None)
            
            if preprocessing_stats:
                mean = preprocessing_stats['mean']
                std = preprocessing_stats['std']
                predicted_rcs_1_5g_db = predicted_rcs[:, :, 0] * std + mean
                predicted_rcs_3g_db = predicted_rcs[:, :, 1] * std + mean
            else:
                predicted_rcs_1_5g_db = 10 * np.log10(np.maximum(predicted_rcs[:, :, 0], epsilon))
                predicted_rcs_3g_db = 10 * np.log10(np.maximum(predicted_rcs[:, :, 1], epsilon))

            diff_1_5g_db = original_rcs_1_5g_db - predicted_rcs_1_5g_db
            diff_3g_db = original_rcs_3g_db - predicted_rcs_3g_db

            max_diff_1_5g = max(abs(diff_1_5g_db.min()), abs(diff_1_5g_db.max()))
            max_diff_3g = max(abs(diff_3g_db.min()), abs(diff_3g_db.max()))

            # 1.5GHz差值
            ax1 = target_fig.add_subplot(2, 2, 1)
            im1 = ax1.imshow(diff_1_5g_db, cmap='RdBu_r', aspect='equal',
                            vmin=-max_diff_1_5g, vmax=max_diff_1_5g)
            ax1.set_title(f'差值图 - 1.5GHz (原始-预测)',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('差值 (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 3GHz差值
            ax2 = target_fig.add_subplot(2, 2, 2)
            im2 = ax2.imshow(diff_3g_db, cmap='RdBu_r', aspect='equal',
                            vmin=-max_diff_3g, vmax=max_diff_3g)
            ax2.set_title(f'差值图 - 3GHz (原始-预测)',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('差值 (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 误差直方图
            ax3 = target_fig.add_subplot(2, 2, 3)
            ax3.hist(np.abs(diff_1_5g_db).flatten(), bins=30, alpha=0.7, label='1.5GHz', density=True)
            ax3.hist(np.abs(diff_3g_db).flatten(), bins=30, alpha=0.7, label='3GHz', density=True)
            ax3.set_xlabel('绝对误差 (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('频率密度', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_title('误差分布', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.legend(fontsize=int(14*fontsize_scale))
            ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 统计信息
            ax4 = target_fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            stats_text = f"""误差统计 (dB) - 模型{model_id}:

1.5GHz:
  MSE: {np.mean(diff_1_5g_db**2):.6f} dB²
  RMSE: {np.sqrt(np.mean(diff_1_5g_db**2)):.6f} dB
  MAE: {np.mean(np.abs(diff_1_5g_db)):.6f} dB

3GHz:
  MSE: {np.mean(diff_3g_db**2):.6f} dB²
  RMSE: {np.sqrt(np.mean(diff_3g_db**2)):.6f} dB
  MAE: {np.mean(np.abs(diff_3g_db)):.6f} dB"""

            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                     fontsize=int(20*fontsize_scale), verticalalignment='top')

            for axis in (ax1, ax2):
                axis.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            plt.tight_layout()
            target_canvas.draw()
            self.log("差值分析图生成完成")

        except Exception as e:
            self.handle_error("差值分析失败", e)
        finally:
            if log_callback: self.log = original_log

    def plot_correlation_analysis(self, model_id, current_model=None, fig=None, canvas=None, log_callback=None):
        """绘制相关性分析图"""
        # 获取依赖项
        model = current_model if current_model is not None else getattr(self.gui, 'current_model', None)
        is_trained = (model is not None) if current_model is not None else getattr(self.gui, 'model_trained', False)
        
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        original_log = self.log
        if log_callback: self.log = log_callback

        if not is_trained or model is None:
            messagebox.showwarning("警告", "请先训练模型")
            if log_callback: self.log = original_log
            return

        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            if log_callback: self.log = original_log
            return

        try:
            target_fig.clear()
            self.log(f"加载模型 {model_id} 进行相关性分析...")
            fontsize_scale = self.get_fontsize_scale()

            data_1_5g = rv.get_rcs_matrix(model_id, "1.5G", self.gui.data_config['rcs_data_dir'])
            data_3g = rv.get_rcs_matrix(model_id, "3G", self.gui.data_config['rcs_data_dir'])

            original_rcs_1_5g = data_1_5g['rcs_linear']
            original_rcs_3g = data_3g['rcs_linear']

            params_df = pd.read_csv(self.gui.data_config['params_file'])
            model_params = params_df.iloc[int(model_id) - 1].values.astype(np.float32)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            model.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(model_params).unsqueeze(0).to(device)
                predicted_rcs = model(params_tensor).cpu().numpy().squeeze()

            x1, y1 = original_rcs_1_5g.flatten(), predicted_rcs[:, :, 0].flatten()
            x2, y2 = original_rcs_3g.flatten(), predicted_rcs[:, :, 1].flatten()

            # 1.5GHz散点图
            ax1 = target_fig.add_subplot(2, 2, 1)
            ax1.scatter(x1, y1, alpha=0.5, s=1)
            r1, p1 = stats.pearsonr(x1, y1)
            ax1.plot([x1.min(), x1.max()], [x1.min(), x1.max()], 'k-', alpha=0.5)
            ax1.set_xlabel('原始RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_ylabel('预测RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_title(f'1.5GHz 相关性\nR={r1:.4f}',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 3GHz散点图
            ax2 = target_fig.add_subplot(2, 2, 2)
            ax2.scatter(x2, y2, alpha=0.5, s=1)
            r2, p2 = stats.pearsonr(x2, y2)
            ax2.plot([x2.min(), x2.max()], [x2.min(), x2.max()], 'k-', alpha=0.5)
            ax2.set_xlabel('原始RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_ylabel('预测RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_title(f'3GHz 相关性\nR={r2:.4f}',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 残差分析
            ax3 = target_fig.add_subplot(2, 2, 3)
            residuals1, residuals2 = y1 - x1, y2 - x2
            ax3.scatter(x1, residuals1, alpha=0.5, s=1, label='1.5GHz')
            ax3.scatter(x2, residuals2, alpha=0.5, s=1, label='3GHz')
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax3.set_xlabel('原始RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('残差', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_title('残差分析', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.legend(fontsize=int(14*fontsize_scale))
            ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 统计摘要
            ax4 = target_fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            summary = f"""相关性报告 - 模型{model_id}:

1.5GHz:
  相关系数: {r1:.6f}
  P值: {p1:.6f}
  R²: {r1**2:.6f}

3GHz:
  相关系数: {r2:.6f}
  P值: {p2:.6f}
  R²: {r2**2:.6f}

质量评估: {'优秀' if min(r1, r2) > 0.9 else '良好' if min(r1, r2) > 0.8 else '一般'}"""

            ax4.text(0.1, 0.9, summary, transform=ax4.transAxes,
                     fontsize=int(20*fontsize_scale), verticalalignment='top')

            plt.tight_layout()
            target_canvas.draw()
            self.log("相关性分析完成")
            self.log(f"相关系数 - 1.5GHz: {r1:.6f}, 3GHz: {r2:.6f}")

        except Exception as e:
            self.handle_error("相关性分析失败", e)
        finally:
            if log_callback: self.log = original_log

    def plot_angle_rcs_comparison(self, angle_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        """绘制Angle-based RCS对比图"""
        # 参数获取与回退
        sys = angle_system if angle_system is not None else None
        if sys is None and hasattr(self.gui, 'angle_rcs_extension'):
            sys = self.gui.angle_rcs_extension.angle_rcs_system

        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        original_log = self.log
        if log_callback: self.log = log_callback

        if sys is None:
            self.log("错误: Angle-based RCS系统未初始化")
            messagebox.showerror("错误", "Angle-based RCS模型未加载！\n请先训练或加载模型。 সন")
            if log_callback: self.log = original_log
            return
        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            if log_callback: self.log = original_log
            return

        fontsize_scale = self.get_fontsize_scale()

        # 获取数据（统一从GUI读取）
        current_rcs_data = rcs_data if rcs_data is not None else getattr(self.gui, 'rcs_data', None)
        current_param_data = param_data if param_data is not None else getattr(self.gui, 'param_data', None)

        if current_rcs_data is None or current_param_data is None:
            self.log("错误: RCS数据或参数数据未加载")
            messagebox.showerror("错误", "请先加载RCS数据和参数数据！")
            if log_callback: self.log = original_log
            return

        # 获取用户输入的模型ID和频率
        try:
            if hasattr(self.gui, 'visualization_tab'):
                model_id_str = self.gui.visualization_tab.vis_model_var.get()
                # angle-based使用独立的频率选择框
                if hasattr(self.gui.visualization_tab, 'vis_ab_freq_var'):
                    freq_str = self.gui.visualization_tab.vis_ab_freq_var.get()
                else:
                    freq_str = self.gui.visualization_tab.vis_freq_var.get()
            else:
                model_id_str = self.gui.vis_model_var.get()
                freq_str = self.gui.vis_freq_var.get()
        except:
            model_id_str = "001"
            freq_str = "1.5G"

        try:
            # 从文件读取原始RCS数据
            data = rv.get_rcs_matrix(model_id_str, freq_str, self.gui.data_config['rcs_data_dir'])
            true_rcs_linear = data['rcs_linear']
            phi_values = data['phi_values']
            theta_values = data['theta_values']

            # 重建
            print(f"\n【Angle-based RCS重建 - 模型{model_id_str}】")
            print(f"  频率: {freq_str}")

            model = sys.get('model', None)
            if model is None:
                raise ValueError("angle_rcs_system中未找到model")

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            freq_map = {"1.5G": 0, "3G": 1, "6G": 2}
            freq_idx = freq_map.get(freq_str, 0)
            sample_idx = int(model_id_str) - 1

            pred_rcs = reconstruct_rcs_grid(
                model=model,
                sample_idx=sample_idx,
                freq_idx=freq_idx,
                param_data=current_param_data,
                device=device,
                theta_range=(theta_values.min(), theta_values.max()),
                phi_range=(phi_values.min(), phi_values.max()),
                grid_size=91
            )

            # 使用统一绘图函数
            phi_range = (phi_values.min(), phi_values.max())
            theta_range = (theta_values.min(), theta_values.max())

            plot_rcs_comparison(
                true_rcs=true_rcs_linear,
                pred_rcs=pred_rcs,
                freq_label=freq_str,
                model_id=model_id_str,
                phi_range=phi_range,
                theta_range=theta_range,
                fontsize_scale=fontsize_scale,
                fig=target_fig
            )

            if hasattr(target_fig, '_suptitle') and target_fig._suptitle:
                current_title = target_fig._suptitle.get_text()
                target_fig.suptitle(f'{current_title}\n(Angle-based RCS预测)',
                                   fontsize=int(24*fontsize_scale), fontweight='bold')

            target_canvas.draw()

        except Exception as e:
            self.handle_error("Angle-based对比图生成失败", e)
            import traceback
            traceback.print_exc()
        finally:
            if log_callback: self.log = original_log
