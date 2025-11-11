"""
可视化管理器
处理所有绘图和可视化相关功能
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
import os
from datetime import datetime
from tkinter import messagebox
import rcs_visual as rv
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class VisualizationManager:
    """可视化管理器 - 负责所有绘图和可视化功能"""

    def __init__(self, parent_gui):
        """
        初始化可视化管理器

        Args:
            parent_gui: 父GUI窗口实例，用于访问GUI状态和数据
        """
        self.gui = parent_gui

    def _plot_2d_heatmap(self, model_id, freq):
        """绘制2D热图"""
        self.gui.vis_fig.clear()

        try:
            # 使用现有的可视化函数
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])

            ax = self.gui.vis_fig.add_subplot(1, 1, 1)

            # 获取实际的角度范围
            phi_values = data['phi_values']
            theta_values = data['theta_values']

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                # 限制范围在0.5-3.0之间
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            im = ax.imshow(data['rcs_db'], cmap='jet', aspect='equal',
                          extent=[phi_values.min(), phi_values.max(),
                                 theta_values.max(), theta_values.min()])
            ax.set_title(f'模型 {model_id} - {freq} RCS分布',
                        fontsize=int(24*fontsize_scale), fontweight='bold')
            ax.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')

            # 设置刻度标签字号
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 添加colorbar并设置字号
            cbar = self.gui.vis_fig.colorbar(im, ax=ax, label='RCS (dB)')
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            self.gui.vis_fig.tight_layout()
            self.gui.vis_canvas.draw()

        except Exception as e:
            self.gui.log_message(f"无法生成2D热图: {str(e)}")

    def _plot_3d_surface(self, model_id, freq):
        """绘制3D表面图"""
        try:
            import numpy as np
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            self.gui.vis_fig.clear()
            self.gui.log_message(f"绘制模型 {model_id} - {freq} 的3D表面图...")

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 获取RCS数据
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])
            rcs_data = data['rcs_db']  # dB值

            # 创建坐标网格
            theta_range = np.linspace(45, 135, rcs_data.shape[0])  # 俯仰角
            phi_range = np.linspace(-45, 45, rcs_data.shape[1])    # 偏航角
            Theta, Phi = np.meshgrid(theta_range, phi_range, indexing='ij')

            # 创建3D子图
            ax = self.gui.vis_fig.add_subplot(1, 1, 1, projection='3d')

            # 绘制表面图
            surf = ax.plot_surface(Theta, Phi, rcs_data,
                                 cmap='jet', alpha=0.8,
                                 linewidth=0, antialiased=True)

            # 设置标签和标题
            ax.set_xlabel('θ (俯仰角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_ylabel('φ (偏航角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_zlabel('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_title(f'模型 {model_id} - {freq} RCS 3D表面图',
                         fontsize=int(24*fontsize_scale), fontweight='bold')

            # 添加颜色条
            cbar = self.gui.vis_fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='RCS (dB)')
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 设置视角
            ax.view_init(elev=30, azim=45)
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax.zaxis.set_tick_params(labelsize=int(16*fontsize_scale))

            self.gui.vis_canvas.draw()
            self.gui.log_message("3D表面图绘制完成")

        except Exception as e:
            error_msg = f"3D表面图绘制失败: {str(e)}"
            self.gui.log_message(error_msg)
            messagebox.showerror("错误", error_msg)

    def _plot_spherical(self, model_id, freq):
        """绘制球坐标图"""
        try:
            import numpy as np
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            self.gui.vis_fig.clear()
            self.gui.log_message(f"绘制模型 {model_id} - {freq} 的球坐标图...")

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 获取RCS数据
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])
            rcs_linear = data['rcs_linear']  # 线性值用于径向距离

            # 创建角度网格
            theta_deg = np.linspace(45, 135, rcs_linear.shape[0])  # 俯仰角
            phi_deg = np.linspace(-45, 45, rcs_linear.shape[1])    # 偏航角

            # 转换为弧度
            theta_rad = np.deg2rad(theta_deg)
            phi_rad = np.deg2rad(phi_deg)

            Theta, Phi = np.meshgrid(theta_rad, phi_rad, indexing='ij')

            # 球坐标转换为笛卡尔坐标
            # 使用RCS值的对数作为径向距离（避免过大的动态范围）
            R = np.log10(rcs_linear + 1e-10)  # 添加小值避免log(0)
            R = np.maximum(R, -6)  # 限制最小值为-60dB

            # 球坐标到笛卡尔坐标转换
            X = R * np.sin(Theta) * np.cos(Phi)
            Y = R * np.sin(Theta) * np.sin(Phi)
            Z = R * np.cos(Theta)

            # 创建3D子图
            ax = self.gui.vis_fig.add_subplot(1, 1, 1, projection='3d')

            # 绘制球面图
            surf = ax.plot_surface(X, Y, Z,
                                 facecolors=plt.cm.jet((rcs_linear - rcs_linear.min()) /
                                                      (rcs_linear.max() - rcs_linear.min())),
                                 alpha=0.8, linewidth=0, antialiased=True)

            # 设置坐标轴
            ax.set_xlabel('X', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_ylabel('Y', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_zlabel('Z', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_title(f'模型 {model_id} - {freq} RCS 球坐标图',
                         fontsize=int(24*fontsize_scale), fontweight='bold')

            # 设置等比例坐标轴
            max_range = np.max([np.max(np.abs(X)), np.max(np.abs(Y)), np.max(np.abs(Z))])
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

            # 添加颜色映射说明
            sm = plt.cm.ScalarMappable(cmap='jet')
            sm.set_array(data['rcs_db'])
            cbar = self.gui.vis_fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 设置视角
            ax.view_init(elev=20, azim=30)
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax.zaxis.set_tick_params(labelsize=int(16*fontsize_scale))

            self.gui.vis_canvas.draw()
            self.gui.log_message("球坐标图绘制完成")

        except Exception as e:
            error_msg = f"球坐标图绘制失败: {str(e)}"
            self.gui.log_message(error_msg)
            messagebox.showerror("错误", error_msg)

    def _plot_comparison(self, model_id):
        """绘制原始RCS vs 神经网络预测RCS对比图"""
        if not self.gui.model_trained or self.gui.current_model is None:
            messagebox.showwarning("警告", "请先训练模型")
            return

        try:
            import numpy as np
            from matplotlib import pyplot as plt

            # 清除当前图形
            self.gui.vis_fig.clear()

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 获取原始RCS数据
            print(f"加载模型 {model_id} 的原始RCS数据...")
            data_1_5g = rv.get_rcs_matrix(model_id, "1.5G", self.gui.data_config['rcs_data_dir'])
            data_3g = rv.get_rcs_matrix(model_id, "3G", self.gui.data_config['rcs_data_dir'])

            # 提取线性值数据
            original_rcs_1_5g = data_1_5g['rcs_linear']
            original_rcs_3g = data_3g['rcs_linear']

            # 获取对应的参数
            params_df = pd.read_csv(self.gui.data_config['params_file'])
            model_params = params_df.iloc[int(model_id) - 1].values.astype(np.float32)

            # 使用神经网络进行预测
            print(f"使用神经网络预测模型 {model_id} 的RCS...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.gui.current_model.to(device)
            self.gui.current_model.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(model_params).unsqueeze(0).to(device)
                predicted_rcs = self.gui.current_model(params_tensor).cpu().numpy().squeeze()

            # predicted_rcs shape: [91, 91, 2]
            predicted_rcs_1_5g = predicted_rcs[:, :, 0]  # 1.5GHz
            predicted_rcs_3g = predicted_rcs[:, :, 1]    # 3GHz

            # 原始RCS转换为分贝 (dB = 10 * log10(RCS))
            epsilon = 1e-10
            original_rcs_1_5g_db = 10 * np.log10(np.maximum(original_rcs_1_5g, epsilon))
            original_rcs_3g_db = 10 * np.log10(np.maximum(original_rcs_3g, epsilon))

            # 预测RCS转换为dB：检查是否为对数域输出
            if hasattr(self.gui, 'preprocessing_stats') and self.gui.preprocessing_stats:
                # 新格式：网络输出是标准化的dB值，需要反标准化
                mean = self.gui.preprocessing_stats['mean']
                std = self.gui.preprocessing_stats['std']
                predicted_rcs_1_5g_db = predicted_rcs_1_5g * std + mean
                predicted_rcs_3g_db = predicted_rcs_3g * std + mean
                print(f"使用preprocessing_stats反标准化: mean={mean:.2f}, std={std:.2f}")
            else:
                # 旧格式或无preprocessing_stats：假设是线性值，转dB
                predicted_rcs_1_5g_db = 10 * np.log10(np.maximum(predicted_rcs_1_5g, epsilon))
                predicted_rcs_3g_db = 10 * np.log10(np.maximum(predicted_rcs_3g, epsilon))
                print("警告: 无preprocessing_stats，假设网络输出为线性值")

            # 计算统一的colorbar范围（对于每个频率）
            vmin_1_5g = min(original_rcs_1_5g_db.min(), predicted_rcs_1_5g_db.min())
            vmax_1_5g = max(original_rcs_1_5g_db.max(), predicted_rcs_1_5g_db.max())
            vmin_3g = min(original_rcs_3g_db.min(), predicted_rcs_3g_db.min())
            vmax_3g = max(original_rcs_3g_db.max(), predicted_rcs_3g_db.max())

            print(f"1.5GHz dB范围: {vmin_1_5g:.1f} ~ {vmax_1_5g:.1f}")
            print(f"3GHz dB范围: {vmin_3g:.1f} ~ {vmax_3g:.1f}")

            # 创建2x2子图布局
            fig = self.gui.vis_fig

            # 定义角度范围 (基于实际数据)
            phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
            theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°
            extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

            # 1.5GHz频率对比 (dB显示) - 使用统一的colorbar范围
            ax1 = fig.add_subplot(2, 2, 1)
            im1 = ax1.imshow(original_rcs_1_5g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_1_5g, vmax=vmax_1_5g)
            ax1.set_title(f'原始RCS - 1.5GHz (模型{model_id})',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1.ax.tick_params(labelsize=int(16*fontsize_scale))

            ax2 = fig.add_subplot(2, 2, 2)
            im2 = ax2.imshow(predicted_rcs_1_5g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_1_5g, vmax=vmax_1_5g)
            ax2.set_title(f'神经网络预测RCS - 1.5GHz',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 3GHz频率对比 (dB显示) - 使用统一的colorbar范围
            ax3 = fig.add_subplot(2, 2, 3)
            im3 = ax3.imshow(original_rcs_3g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_3g, vmax=vmax_3g)
            ax3.set_title(f'原始RCS - 3GHz (模型{model_id})',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
            cbar3.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar3.ax.tick_params(labelsize=int(16*fontsize_scale))

            ax4 = fig.add_subplot(2, 2, 4)
            im4 = ax4.imshow(predicted_rcs_3g_db, cmap='jet', aspect='equal', extent=extent,
                            vmin=vmin_3g, vmax=vmax_3g)
            ax4.set_title(f'神经网络预测RCS - 3GHz',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
            cbar4.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar4.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 计算并显示误差统计 (dB域)
            mse_db_1_5g = np.mean((original_rcs_1_5g_db - predicted_rcs_1_5g_db) ** 2)
            mse_db_3g = np.mean((original_rcs_3g_db - predicted_rcs_3g_db) ** 2)
            rmse_db_1_5g = np.sqrt(mse_db_1_5g)
            rmse_db_3g = np.sqrt(mse_db_3g)

            # 在图上添加误差信息
            fig.suptitle(f'RCS对比分析 (dB) - 模型{model_id}\n1.5GHz RMSE: {rmse_db_1_5g:.2f} dB, 3GHz RMSE: {rmse_db_3g:.2f} dB',
                        fontsize=int(24*fontsize_scale), fontweight='bold', y=0.95)

            for axis in (ax1, ax2, ax3, ax4):
                axis.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            plt.tight_layout()
            self.gui.vis_canvas.draw()

            print(f"对比图生成完成")
            print(f"1.5GHz预测误差(MSE): {mse_db_1_5g:.6f} dB²")
            print(f"3GHz预测误差(MSE): {mse_db_3g:.6f} dB²")

        except Exception as e:
            print(f"对比图生成失败: {str(e)}")
            messagebox.showerror("错误", f"对比图生成失败: {str(e)}")

    def _plot_difference_analysis(self, model_id):
        """绘制差值分析图（原始RCS - 预测RCS）"""
        if not self.gui.model_trained or self.gui.current_model is None:
            messagebox.showwarning("警告", "请先训练模型")
            return

        try:
            import numpy as np
            from matplotlib import pyplot as plt

            self.gui.vis_fig.clear()
            print(f"加载模型 {model_id} 进行差值分析...")

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 获取原始和预测数据
            data_1_5g = rv.get_rcs_matrix(model_id, "1.5G", self.gui.data_config['rcs_data_dir'])
            data_3g = rv.get_rcs_matrix(model_id, "3G", self.gui.data_config['rcs_data_dir'])

            original_rcs_1_5g = data_1_5g['rcs_linear']
            original_rcs_3g = data_3g['rcs_linear']

            params_df = pd.read_csv(self.gui.data_config['params_file'])
            model_params = params_df.iloc[int(model_id) - 1].values.astype(np.float32)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.gui.current_model.to(device)
            self.gui.current_model.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(model_params).unsqueeze(0).to(device)
                predicted_rcs = self.gui.current_model(params_tensor).cpu().numpy().squeeze()

            # 原始RCS转换为分贝
            epsilon = 1e-10
            original_rcs_1_5g_db = 10 * np.log10(np.maximum(original_rcs_1_5g, epsilon))
            original_rcs_3g_db = 10 * np.log10(np.maximum(original_rcs_3g, epsilon))

            # 预测RCS转换为dB：检查是否为对数域输出
            if hasattr(self.gui, 'preprocessing_stats') and self.gui.preprocessing_stats:
                # 新格式：网络输出是标准化的dB值，需要反标准化
                mean = self.gui.preprocessing_stats['mean']
                std = self.gui.preprocessing_stats['std']
                predicted_rcs_1_5g_db = predicted_rcs[:, :, 0] * std + mean
                predicted_rcs_3g_db = predicted_rcs[:, :, 1] * std + mean
            else:
                # 旧格式或无preprocessing_stats：假设是线性值，转dB
                predicted_rcs_1_5g_db = 10 * np.log10(np.maximum(predicted_rcs[:, :, 0], epsilon))
                predicted_rcs_3g_db = 10 * np.log10(np.maximum(predicted_rcs[:, :, 1], epsilon))

            # 计算分贝差值
            diff_1_5g_db = original_rcs_1_5g_db - predicted_rcs_1_5g_db
            diff_3g_db = original_rcs_3g_db - predicted_rcs_3g_db

            # 计算统一的差值范围（使用对称范围）
            max_diff_1_5g = max(abs(diff_1_5g_db.min()), abs(diff_1_5g_db.max()))
            max_diff_3g = max(abs(diff_3g_db.min()), abs(diff_3g_db.max()))

            # 创建子图
            ax1 = self.gui.vis_fig.add_subplot(2, 2, 1)
            im1 = ax1.imshow(diff_1_5g_db, cmap='RdBu_r', aspect='equal',
                            vmin=-max_diff_1_5g, vmax=max_diff_1_5g)
            ax1.set_title(f'差值图 - 1.5GHz (原始-预测)',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('差值 (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar1.ax.tick_params(labelsize=int(16*fontsize_scale))

            ax2 = self.gui.vis_fig.add_subplot(2, 2, 2)
            im2 = ax2.imshow(diff_3g_db, cmap='RdBu_r', aspect='equal',
                            vmin=-max_diff_3g, vmax=max_diff_3g)
            ax2.set_title(f'差值图 - 3GHz (原始-预测)',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('差值 (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar2.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 误差统计
            ax3 = self.gui.vis_fig.add_subplot(2, 2, 3)
            ax3.hist(np.abs(diff_1_5g_db).flatten(), bins=30, alpha=0.7, label='1.5GHz', density=True)
            ax3.hist(np.abs(diff_3g_db).flatten(), bins=30, alpha=0.7, label='3GHz', density=True)
            ax3.set_xlabel('绝对误差 (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('频率密度', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_title('误差分布', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.legend(fontsize=int(14*fontsize_scale))
            ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 统计信息
            ax4 = self.gui.vis_fig.add_subplot(2, 2, 4)
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
            self.gui.vis_canvas.draw()
            print("差值分析图生成完成")

        except Exception as e:
            print(f"差值分析失败: {str(e)}")
            messagebox.showerror("错误", f"差值分析失败: {str(e)}")

    def _plot_correlation_analysis(self, model_id):
        """绘制相关性分析图"""
        if not self.gui.model_trained or self.gui.current_model is None:
            messagebox.showwarning("警告", "请先训练模型")
            return

        try:
            import numpy as np
            from matplotlib import pyplot as plt
            from scipy import stats

            self.gui.vis_fig.clear()
            print(f"加载模型 {model_id} 进行相关性分析...")

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 获取数据
            data_1_5g = rv.get_rcs_matrix(model_id, "1.5G", self.gui.data_config['rcs_data_dir'])
            data_3g = rv.get_rcs_matrix(model_id, "3G", self.gui.data_config['rcs_data_dir'])

            original_rcs_1_5g = data_1_5g['rcs_linear']
            original_rcs_3g = data_3g['rcs_linear']

            params_df = pd.read_csv(self.gui.data_config['params_file'])
            model_params = params_df.iloc[int(model_id) - 1].values.astype(np.float32)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.gui.current_model.to(device)
            self.gui.current_model.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(model_params).unsqueeze(0).to(device)
                predicted_rcs = self.gui.current_model(params_tensor).cpu().numpy().squeeze()

            # 相关性分析
            x1, y1 = original_rcs_1_5g.flatten(), predicted_rcs[:, :, 0].flatten()
            x2, y2 = original_rcs_3g.flatten(), predicted_rcs[:, :, 1].flatten()

            # 1.5GHz散点图
            ax1 = self.gui.vis_fig.add_subplot(2, 2, 1)
            ax1.scatter(x1, y1, alpha=0.5, s=1)
            r1, p1 = stats.pearsonr(x1, y1)
            ax1.plot([x1.min(), x1.max()], [x1.min(), x1.max()], 'k-', alpha=0.5)
            ax1.set_xlabel('原始RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_ylabel('预测RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_title(f'1.5GHz 相关性\\nR={r1:.4f}',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 3GHz散点图
            ax2 = self.gui.vis_fig.add_subplot(2, 2, 2)
            ax2.scatter(x2, y2, alpha=0.5, s=1)
            r2, p2 = stats.pearsonr(x2, y2)
            ax2.plot([x2.min(), x2.max()], [x2.min(), x2.max()], 'k-', alpha=0.5)
            ax2.set_xlabel('原始RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_ylabel('预测RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_title(f'3GHz 相关性\\nR={r2:.4f}',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 残差分析
            ax3 = self.gui.vis_fig.add_subplot(2, 2, 3)
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
            ax4 = self.gui.vis_fig.add_subplot(2, 2, 4)
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
            self.gui.vis_canvas.draw()
            print("相关性分析完成")
            print(f"相关系数 - 1.5GHz: {r1:.6f}, 3GHz: {r2:.6f}")

        except Exception as e:
            print(f"相关性分析失败: {str(e)}")
            messagebox.showerror("错误", f"相关性分析失败: {str(e)}")

    def _plot_training_history(self):
        """绘制训练历史图（对交叉验证，分别保存每折到results文件夹，GUI显示最佳折）"""
        if not hasattr(self.gui, 'training_history') or not self.gui.training_history:
            messagebox.showwarning("警告", "没有训练历史数据，请先进行训练")
            return

        try:
            import numpy as np
            from matplotlib import pyplot as plt
            import os
            from datetime import datetime

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 确保results目录存在
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            print("绘制并保存训练历史图...")

            # 检查是否有交叉验证的fold_details
            if 'fold_details' in self.gui.training_history and self.gui.training_history['fold_details']:
                # 交叉验证模式：分别保存每折的图
                fold_details = self.gui.training_history['fold_details']
                fold_scores = self.gui.training_history.get('fold_scores', [])

                # 找到最佳折用于GUI显示
                best_fold_idx = np.argmin(fold_scores) if fold_scores else 0

                # 为每折创建单独的图表
                for fold_idx, fold_data in enumerate(fold_details):
                    self._save_fold_plot(fold_data, fold_idx, results_dir, fontsize_scale=fontsize_scale)

                # 在GUI显示最佳折
                best_fold_data = fold_details[best_fold_idx]
                self._display_fold_in_gui(best_fold_data, best_fold_idx, fontsize_scale=fontsize_scale)

                self.gui.log_message(f"已保存{len(fold_details)}折训练图表到{results_dir}目录")
                self.gui.log_message(f"GUI显示最佳折 {best_fold_idx + 1} 的训练历史")

            else:
                # 单次训练模式：直接显示
                self._display_simple_training_history(fontsize_scale=fontsize_scale)

        except Exception as e:
            error_msg = f"绘制训练历史失败: {str(e)}"
            self.gui.log_message(error_msg)
            messagebox.showerror("错误", error_msg)

    def _save_fold_plot(self, fold_data, fold_idx, results_dir, fontsize_scale=None):
        """保存单个折的训练历史图表"""
        import matplotlib.pyplot as plt
        from datetime import datetime

        # 获取字号缩放因子
        if fontsize_scale is None:
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建独立的图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'交叉验证第{fold_idx + 1}折 - 训练历史',
                     fontsize=int(24*fontsize_scale), fontweight='bold')

        epochs = fold_data.get('epochs', [])
        train_losses = fold_data.get('train_losses', [])
        val_losses = fold_data.get('val_losses', [])

        if not epochs or not train_losses:
            return

        # 主损失曲线
        axes[0, 0].semilogy(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        if val_losses:
            axes[0, 0].semilogy(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
        axes[0, 0].set_ylabel('Loss (对数坐标)', fontsize=int(20*fontsize_scale), fontweight='bold')
        axes[0, 0].set_title('训练和验证损失', fontsize=int(20*fontsize_scale), fontweight='bold')
        axes[0, 0].legend(fontsize=int(14*fontsize_scale))
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 分量损失
        axes[0, 1].set_title('损失组件分析', fontsize=int(20*fontsize_scale), fontweight='bold')
        if fold_data.get('train_mse'):
            axes[0, 1].semilogy(epochs, fold_data['train_mse'], 'g-', label='MSE', alpha=0.8)
        if fold_data.get('train_symmetry'):
            axes[0, 1].semilogy(epochs, fold_data['train_symmetry'], 'm-', label='对称性', alpha=0.8)
        if fold_data.get('train_multiscale'):
            axes[0, 1].semilogy(epochs, fold_data['train_multiscale'], 'c-', label='多尺度', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
        axes[0, 1].set_ylabel('损失分量 (对数坐标)', fontsize=int(20*fontsize_scale), fontweight='bold')
        axes[0, 1].legend(fontsize=int(14*fontsize_scale))
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 学习率曲线
        axes[1, 0].set_title('学习率变化', fontsize=int(20*fontsize_scale), fontweight='bold')
        if fold_data.get('learning_rates'):
            axes[1, 0].plot(epochs, fold_data['learning_rates'], 'purple', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
            axes[1, 0].set_ylabel('Learning Rate', fontsize=int(20*fontsize_scale), fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '学习率数据不可用', ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=int(20*fontsize_scale),
                            fontweight='bold')
        axes[1, 0].tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 统计摘要
        axes[1, 1].axis('off')
        total_epochs = len(epochs)
        final_train = train_losses[-1] if train_losses else 0
        final_val = val_losses[-1] if val_losses else 0
        min_val = min(val_losses) if val_losses else 0

        stats = f"""第{fold_idx + 1}折统计:

总轮数: {total_epochs}
最终训练损失: {final_train:.6f}
最终验证损失: {final_val:.6f}
最佳验证损失: {min_val:.6f}

训练完成时间:
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        axes[1, 1].text(0.1, 0.9, stats, transform=axes[1, 1].transAxes,
                        fontsize=int(20*fontsize_scale), verticalalignment='top',
                        fontfamily='monospace', fontweight='bold')

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"fold_{fold_idx + 1}_training_history_{timestamp}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"已保存第{fold_idx + 1}折训练历史到: {filepath}")

    def _display_fold_in_gui(self, fold_data, fold_idx, fontsize_scale=None):
        """在GUI中显示指定折的训练历史"""
        # 设置中文字体
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.gui.vis_fig.clear()

        # 获取字号缩放因子
        if fontsize_scale is None:
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

        epochs = fold_data.get('epochs', [])
        train_losses = fold_data.get('train_losses', [])
        val_losses = fold_data.get('val_losses', [])

        if not epochs or not train_losses:
            self.gui.vis_fig.text(0.5, 0.5, f'第{fold_idx + 1}折数据不完整',
                                  ha='center', va='center',
                                  fontsize=int(20*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()
            return

        # 主损失曲线
        ax1 = self.gui.vis_fig.add_subplot(2, 2, 1)
        ax1.semilogy(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        if val_losses:
            ax1.semilogy(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.set_ylabel('Loss (对数坐标)', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.set_title(f'第{fold_idx + 1}折 - 训练和验证损失',
                      fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.legend(fontsize=int(14*fontsize_scale))
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 分量损失
        ax2 = self.gui.vis_fig.add_subplot(2, 2, 2)
        if fold_data.get('train_mse'):
            ax2.semilogy(epochs, fold_data['train_mse'], 'g-', label='MSE', alpha=0.8)
        if fold_data.get('train_symmetry'):
            ax2.semilogy(epochs, fold_data['train_symmetry'], 'm-', label='对称性', alpha=0.8)
        if fold_data.get('train_multiscale'):
            ax2.semilogy(epochs, fold_data['train_multiscale'], 'c-', label='多尺度', alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax2.set_ylabel('损失分量 (对数坐标)', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax2.set_title('损失组件分析', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax2.legend(fontsize=int(14*fontsize_scale))
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 学习率
        ax3 = self.gui.vis_fig.add_subplot(2, 2, 3)
        if fold_data.get('learning_rates'):
            ax3.plot(epochs, fold_data['learning_rates'], 'purple', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('Learning Rate', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_title('学习率变化', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '学习率数据不可用', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_title('学习率监控', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 统计摘要
        ax4 = self.gui.vis_fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        total_epochs = len(epochs)
        final_train = train_losses[-1] if train_losses else 0
        final_val = val_losses[-1] if val_losses else 0
        min_val = min(val_losses) if val_losses else 0

        stats = f"""第{fold_idx + 1}折摘要:

总轮数: {total_epochs}
最终训练损失: {final_train:.6f}
最终验证损失: {final_val:.6f}
最佳验证损失: {min_val:.6f}

注: 其他折已保存到results/"""

        ax4.text(0.1, 0.9, stats, transform=ax4.transAxes,
                 fontsize=int(20*fontsize_scale), verticalalignment='top',
                 fontfamily='monospace', fontweight='bold')

        self.gui.vis_fig.tight_layout()
        self.gui.vis_canvas.draw()

    def _display_simple_training_history(self, fontsize_scale=None):
        """显示简单训练模式的历史（非交叉验证）"""
        # 设置中文字体
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.gui.vis_fig.clear()

        # 获取字号缩放因子
        if fontsize_scale is None:
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

        epochs = self.gui.training_history.get('epochs', [])
        train_loss = self.gui.training_history.get('train_loss', [])
        val_loss = self.gui.training_history.get('val_loss', [])

        if not epochs or not train_loss:
            self.gui.vis_fig.text(0.5, 0.5, '训练历史数据不完整',
                                  ha='center', va='center',
                                  fontsize=int(20*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()
            return

        # 主损失曲线
        ax1 = self.gui.vis_fig.add_subplot(2, 2, 1)
        ax1.semilogy(epochs, train_loss, 'b-', label='训练损失', linewidth=2)
        if val_loss:
            ax1.semilogy(epochs, val_loss, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.set_ylabel('Loss (对数坐标)', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.set_title('训练和验证损失', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax1.legend(fontsize=int(14*fontsize_scale))
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 分量损失
        ax2 = self.gui.vis_fig.add_subplot(2, 2, 2)
        if self.gui.training_history.get('train_mse'):
            ax2.semilogy(epochs, self.gui.training_history['train_mse'], 'g-', label='MSE', alpha=0.8)
        if self.gui.training_history.get('train_symmetry'):
            ax2.semilogy(epochs, self.gui.training_history['train_symmetry'], 'm-', label='对称性', alpha=0.8)
        if self.gui.training_history.get('train_multiscale'):
            ax2.semilogy(epochs, self.gui.training_history['train_multiscale'], 'c-', label='多尺度', alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax2.set_ylabel('损失分量 (对数坐标)', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax2.set_title('损失组件分析', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax2.legend(fontsize=int(14*fontsize_scale))
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # GPU显存监控
        ax3 = self.gui.vis_fig.add_subplot(2, 2, 3)
        if self.gui.training_history.get('gpu_memory') and any(x > 0 for x in self.gui.training_history['gpu_memory']):
            ax3.plot(epochs, self.gui.training_history['gpu_memory'], 'orange', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('GPU显存 (GB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_title('GPU显存监控', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'GPU显存监控不可用', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_title('GPU显存监控', fontsize=int(20*fontsize_scale), fontweight='bold')
        ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))

        # 统计摘要
        ax4 = self.gui.vis_fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        total_epochs = len(epochs)
        batch_size = self.gui.training_history.get('batch_sizes', [None])[0] or 'N/A'
        final_train = train_loss[-1] if train_loss else 0
        final_val = val_loss[-1] if val_loss else 0
        min_val = min(val_loss) if val_loss else 0
        gpu_peak = max(self.gui.training_history.get('gpu_memory', [0])) if self.gui.training_history.get('gpu_memory') else 0

        stats = f"""训练摘要:

总轮数: {total_epochs}
批次大小: {batch_size}
最终训练损失: {final_train:.6f}
最终验证损失: {final_val:.6f}
最佳验证损失: {min_val:.6f}
GPU显存峰值: {gpu_peak:.2f} GB"""

        ax4.text(0.1, 0.9, stats, transform=ax4.transAxes,
                 fontsize=int(20*fontsize_scale), verticalalignment='top',
                 fontfamily='monospace', fontweight='bold')

        self.gui.vis_fig.tight_layout()
        self.gui.vis_canvas.draw()

    def _plot_autoencoder_visualization(self, chart_type):
        """绘制AutoEncoder特定可视化图表"""
        try:
            fontsize_scale = self.gui.fontsize_scale_var.get()
            fontsize_scale = max(0.5, min(3.0, fontsize_scale))
        except:
            fontsize_scale = 1.0

        if chart_type == "AE隐空间分析":
            self._plot_ae_latent_space()
        elif chart_type == "AE重建质量":
            self._plot_ae_reconstruction_quality()
        elif chart_type == "AE参数映射":
            self._plot_ae_parameter_mapping()
        elif chart_type == "AE训练进度":
            self._plot_ae_training_progress_vis()
        else:
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'未知的可视化类型: {chart_type}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=int(20*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()

    def _plot_ae_latent_space(self):
        """绘制AutoEncoder隐空间分析"""
        import torch
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # 获取字号缩放因子
        try:
            fontsize_scale = self.gui.fontsize_scale_var.get()
            fontsize_scale = max(0.5, min(3.0, fontsize_scale))
        except:
            fontsize_scale = 1.0

        try:
            # 获取AutoEncoder组件
            autoencoder = self.gui.ae_system['autoencoder']
            wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)  # 直接模式时为None
            data_adapter = self.gui.ae_system.get('data_adapter', None)
            rcs_data = self.gui.ae_system['rcs_data']
            mode = self.gui.ae_system.get('mode', 'direct')

            # 设置设备和评估模式
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()

            # 编码所有数据到隐空间
            with torch.no_grad():
                # 取前50个样本避免内存问题
                sample_data = rcs_data[:50]

                # 根据模式准备输入数据
                if mode in ('wavelet', 'differentiable_wavelet'):
                    # Wavelet模式：RCS → 小波变换 → 标准化
                    rcs_tensor = torch.FloatTensor(sample_data).to(device)
                    wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)

                    if data_adapter:
                        # 标准化小波系数
                        wavelet_coeffs_np = wavelet_coeffs.cpu().numpy()
                        input_adapted = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
                        input_tensor = torch.FloatTensor(input_adapted).to(device)
                    else:
                        input_tensor = wavelet_coeffs
                else:
                    # Direct模式：RCS → 标准化
                    if data_adapter:
                        input_adapted = data_adapter.adapt_rcs_data(sample_data)
                        input_tensor = torch.FloatTensor(input_adapted).to(device)
                    else:
                        input_tensor = torch.FloatTensor(sample_data).to(device)

                # 编码到隐空间
                latent_vectors = autoencoder.encode(input_tensor)
                latent_vectors = latent_vectors.cpu().numpy()

            # 降维可视化
            self.gui.vis_fig.clear()

            # 子图1: PCA降维
            ax1 = self.gui.vis_fig.add_subplot(2, 2, 1)
            pca = PCA(n_components=2)
            latent_2d_pca = pca.fit_transform(latent_vectors)
            scatter = ax1.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1],
                                c=range(len(latent_2d_pca)), cmap='viridis', alpha=0.6)
            ax1.set_title('隐空间分布 - PCA',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                           fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                           fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 子图2: t-SNE降维
            ax2 = self.gui.vis_fig.add_subplot(2, 2, 2)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
            latent_2d_tsne = tsne.fit_transform(latent_vectors)
            ax2.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1],
                       c=range(len(latent_2d_tsne)), cmap='viridis', alpha=0.6)
            ax2.set_title('隐空间分布 - t-SNE',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_xlabel('t-SNE1', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_ylabel('t-SNE2', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 子图3: 隐空间维度分布
            ax3 = self.gui.vis_fig.add_subplot(2, 2, 3)
            latent_means = np.mean(latent_vectors, axis=0)
            latent_stds = np.std(latent_vectors, axis=0)
            dims = range(len(latent_means[:20]))  # 只显示前20个维度
            ax3.errorbar(dims, latent_means[:20], yerr=latent_stds[:20],
                        capsize=3, marker='o', markersize=4)
            ax3.set_title('隐空间维度统计 (前20维)',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_xlabel('隐空间维度', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('数值', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax3.grid(True, alpha=0.3)

            # 子图4: 隐空间激活热图
            ax4 = self.gui.vis_fig.add_subplot(2, 2, 4)
            im = ax4.imshow(latent_vectors[:10, :20].T, cmap='RdYlBu', aspect='auto')
            ax4.set_title('隐空间激活模式 (前10样本×前20维)',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_xlabel('样本索引', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_ylabel('隐空间维度', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar = self.gui.vis_fig.colorbar(im, ax=ax4)
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))
            cbar.set_label('激活值', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            self.gui.vis_fig.tight_layout()
            self.gui.vis_canvas.draw()

        except Exception as e:
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'隐空间分析失败:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=int(20*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()

    def _plot_ae_reconstruction_quality(self):
        """绘制AutoEncoder重建质量分析 - 使用统一重建接口"""
        import numpy as np
        import torch

        # 获取字号缩放因子
        try:
            fontsize_scale = self.gui.fontsize_scale_var.get()
            fontsize_scale = max(0.5, min(3.0, fontsize_scale))
        except:
            fontsize_scale = 1.0

        try:
            # 获取AutoEncoder组件
            autoencoder = self.gui.ae_system['autoencoder']
            wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)
            data_adapter = self.gui.ae_system.get('data_adapter', None)
            rcs_data = self.gui.ae_system['rcs_data']
            mode = self.gui.ae_system.get('mode', 'direct')

            # 设置设备和评估模式
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()

            # 选择测试样本
            test_indices = [0, 10, 20, 30]  # 选择几个代表性样本
            test_samples = rcs_data[test_indices]

            # 使用AutoEncoder重建RCS（模拟Stage1-Only的行为）
            with torch.no_grad():
                # 根据模式准备输入数据
                if mode in ('wavelet', 'differentiable_wavelet'):
                    # Wavelet模式：RCS → 小波变换 → 标准化
                    rcs_tensor = torch.FloatTensor(test_samples).to(device)
                    wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)

                    if data_adapter:
                        wavelet_coeffs_np = wavelet_coeffs.cpu().numpy()
                        input_adapted = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
                        input_tensor = torch.FloatTensor(input_adapted).to(device)
                    else:
                        input_tensor = wavelet_coeffs
                else:
                    # Direct模式：RCS → 标准化
                    if data_adapter:
                        input_adapted = data_adapter.adapt_rcs_data(test_samples)
                        input_tensor = torch.FloatTensor(input_adapted).to(device)
                    else:
                        input_tensor = torch.FloatTensor(test_samples).to(device)

                # AutoEncoder重建
                latent = autoencoder.encode(input_tensor)
                reconstructed_output = autoencoder.decode(latent)

                # 逆变换到RCS空间
                if mode in ('wavelet', 'differentiable_wavelet'):
                    # 小波模式：逆标准化 → 逆小波变换 → RCS
                    if data_adapter:
                        reconstructed_coeffs_np = data_adapter.inverse_adapt(reconstructed_output)
                        reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
                    else:
                        reconstructed_coeffs = reconstructed_output

                    reconstructed_samples = wavelet_transform.inverse_transform(reconstructed_coeffs).cpu().numpy()
                else:
                    # 直接模式：逆标准化 → RCS
                    if data_adapter:
                        reconstructed_samples = data_adapter.inverse_adapt(reconstructed_output)
                    else:
                        reconstructed_samples = reconstructed_output.cpu().numpy()

            self.gui.vis_fig.clear()

            for i, sample_idx in enumerate(test_indices):
                # 原始数据和重建数据
                original_rcs = test_samples[i]
                reconstructed_rcs = reconstructed_samples[i]

                # 绘制对比图
                ax = self.gui.vis_fig.add_subplot(2, 2, i+1)

                # 只显示第一个频率的数据
                freq_idx = 0
                original_2d = original_rcs[:, :, freq_idx]
                reconstructed_2d = reconstructed_rcs[:, :, freq_idx]

                # 计算重建误差
                mse = np.mean((original_2d - reconstructed_2d)**2)

                # 并排显示原始和重建
                combined = np.hstack([original_2d, reconstructed_2d])
                im = ax.imshow(combined, cmap='jet', aspect='equal')

                # 添加分割线
                ax.axvline(x=original_2d.shape[1]-0.5, color='white', linewidth=2)

                ax.set_title(f'样本{sample_idx+1} (MSE={mse:.4e})\n左:原始 右:重建',
                             fontsize=int(20*fontsize_scale), fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

                cbar = self.gui.vis_fig.colorbar(im, ax=ax, shrink=0.6)
                cbar.ax.tick_params(labelsize=int(16*fontsize_scale))
                cbar.set_label('RCS 值', fontsize=int(20*fontsize_scale), fontweight='bold')

            self.gui.vis_fig.suptitle('AutoEncoder重建质量对比',
                                      fontsize=int(24*fontsize_scale), fontweight='bold')
            self.gui.vis_fig.tight_layout()
            self.gui.vis_canvas.draw()

        except Exception as e:
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'重建质量分析失败:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=int(20*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()

    def _plot_ae_parameter_mapping(self):
        """绘制AutoEncoder参数映射分析"""
        import torch
        import numpy as np
        from sklearn.decomposition import PCA

        # 获取字号缩放因子
        try:
            fontsize_scale = self.gui.fontsize_scale_var.get()
            fontsize_scale = max(0.5, min(3.0, fontsize_scale))
        except:
            fontsize_scale = 1.0

        try:
            # 获取组件
            parameter_mapper = self.gui.ae_system['parameter_mapper']
            param_data = self.gui.ae_system['param_data']

            # 设置设备和评估模式
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            parameter_mapper.to(device).eval()

            # 获取参数映射结果
            with torch.no_grad():
                param_tensor = torch.FloatTensor(param_data[:50])  # 前50个样本
                mapped_latents = parameter_mapper(param_tensor.to(device))
                mapped_latents = mapped_latents.cpu().numpy()

            self.gui.vis_fig.clear()

            # 子图1: 参数空间分布
            ax1 = self.gui.vis_fig.add_subplot(2, 2, 1)
            # 假设前两个参数最重要
            ax1.scatter(param_data[:50, 0], param_data[:50, 1],
                       c=range(50), cmap='viridis', alpha=0.6)
            ax1.set_title('参数空间分布', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_xlabel('参数1', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_ylabel('参数2', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 子图2: 映射后隐空间分布
            ax2 = self.gui.vis_fig.add_subplot(2, 2, 2)
            pca = PCA(n_components=2)
            mapped_2d = pca.fit_transform(mapped_latents)
            ax2.scatter(mapped_2d[:, 0], mapped_2d[:, 1],
                       c=range(50), cmap='viridis', alpha=0.6)
            ax2.set_title('映射后隐空间分布', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_xlabel('隐空间PC1', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_ylabel('隐空间PC2', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 子图3: 参数-隐空间相关性
            ax3 = self.gui.vis_fig.add_subplot(2, 2, 3)
            # 计算每个参数与隐空间主成分的相关性
            correlations = []
            for param_idx in range(min(param_data.shape[1], 5)):  # 最多5个参数
                corr = np.corrcoef(param_data[:50, param_idx], mapped_2d[:, 0])[0, 1]
                correlations.append(abs(corr))

            param_names = [f'参数{i+1}' for i in range(len(correlations))]
            ax3.bar(param_names, correlations)
            ax3.set_title('参数与隐空间PC1相关性',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('绝对相关系数', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.tick_params(axis='x', rotation=45, labelsize=int(16*fontsize_scale))
            ax3.tick_params(axis='y', labelsize=int(16*fontsize_scale))

            # 子图4: 隐空间维度激活强度
            ax4 = self.gui.vis_fig.add_subplot(2, 2, 4)
            latent_means = np.mean(np.abs(mapped_latents), axis=0)
            dims = range(len(latent_means[:20]))  # 前20维
            ax4.bar(dims, latent_means[:20])
            ax4.set_title('隐空间维度激活强度 (前20维)',
                          fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_xlabel('隐空间维度', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_ylabel('平均激活强度', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            self.gui.vis_fig.tight_layout()
            self.gui.vis_canvas.draw()

        except Exception as e:
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'参数映射分析失败:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=int(20*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()

    def _plot_ae_training_progress_vis(self):
        """绘制AutoEncoder训练进度可视化（使用统一绘图函数）"""
        try:
            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 获取训练历史数据
            ae_training_history = getattr(self.gui, 'ae_training_history', None)

            # 使用统一绘图函数
            from autoencoder.utils.plotting import plot_ae_training_progress

            plot_ae_training_progress(
                ae_training_history=ae_training_history,
                fontsize_scale=fontsize_scale,
                fig=self.gui.vis_fig,
                use_log_scale=True,
                show_best_epoch=True
            )

            # 刷新画布
            self.gui.vis_canvas.draw()

        except Exception as e:
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'AutoEncoder训练进度可视化失败:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_title('AutoEncoder训练进度',
                         fontsize=int(24*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()

    def _plot_autoencoder_prediction_visualization(self, chart_type, freq):
        """使用AutoEncoder进行预测可视化"""
        import torch
        import numpy as np

        try:
            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            if chart_type == "2D热图":
                self._plot_ae_2d_heatmap(freq)
            elif chart_type == "对比图":
                self._plot_ae_comparison()
            elif chart_type == "小波系数对比":
                self._plot_wavelet_coefficients_comparison()
            else:
                # 对其他图表类型，显示提示信息
                self.gui.vis_fig.clear()
                ax = self.gui.vis_fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, f'AutoEncoder暂不支持"{chart_type}"类型\n请选择其他图表类型',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=int(20*fontsize_scale), fontweight='bold')
                self.gui.vis_canvas.draw()

        except Exception as e:
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'AutoEncoder预测可视化失败:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=int(20*fontsize_scale), fontweight='bold')
            self.gui.vis_canvas.draw()

    def _plot_ae_2d_heatmap(self, freq):
        """绘制AutoEncoder预测的2D热图 - 支持模型未加载时显示原始数据"""
        import torch
        import numpy as np

        # 检查是否有加载的AutoEncoder系统
        if (not hasattr(self.gui, 'ae_system') or
            self.gui.ae_system is None or
            'autoencoder' not in self.gui.ae_system or
            self.gui.ae_system['autoencoder'] is None):

            # 如果没有加载模型，显示原始RCS数据作为替代
            self._plot_original_rcs_fallback(freq)
            return

        try:
            # 获取组件
            autoencoder = self.gui.ae_system['autoencoder']
            parameter_mapper = self.gui.ae_system['parameter_mapper']
            wavelet_transform = self.gui.ae_system.get('wavelet_transform', None)  # 直接模式时为None
            param_data = self.gui.ae_system['param_data']

            # 设置设备和评估模式
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()
            parameter_mapper.to(device).eval()

            # 选择一个测试样本
            sample_idx = 0
            test_params = param_data[sample_idx:sample_idx+1]

            # 进行端到端预测
            with torch.no_grad():
                param_tensor = torch.FloatTensor(test_params).to(device)
                predicted_latents = parameter_mapper(param_tensor)
                predicted_output = autoencoder.decode(predicted_latents)

                # ⚠️ 关键修复：decoder输出在标准化空间，必须逆变换回原始RCS空间
                # 获取data_adapter用于逆标准化
                data_adapter = self.gui.ae_system.get('data_adapter', None)

                # 根据模式处理输出
                if wavelet_transform is not None:
                    # 小波模式：标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
                    if data_adapter:
                        # inverse_adapt期望tensor输入，返回numpy
                        predicted_coeffs_np = data_adapter.inverse_adapt(predicted_output)
                        predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
                    else:
                        predicted_coeffs = predicted_output

                    predicted_rcs = wavelet_transform.inverse_transform(predicted_coeffs)
                else:
                    # 直接模式：标准化RCS → 逆标准化（逆dB + 逆Z-score） → RCS
                    if data_adapter:
                        # inverse_adapt期望tensor输入，返回numpy
                        predicted_rcs_np = data_adapter.inverse_adapt(predicted_output)
                        predicted_rcs = torch.FloatTensor(predicted_rcs_np).to(device)
                    else:
                        predicted_rcs = predicted_output

                predicted_rcs = predicted_rcs.cpu().numpy()[0]

            # 绘制热图
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)

            # 选择频率索引
            freq_idx = 0 if freq == "1.5G" else 1
            rcs_2d = predicted_rcs[:, :, freq_idx]

            # 创建角度网格（实际角度范围：theta 45-135°, phi -45-45°）
            theta_values = np.linspace(45, 135, rcs_2d.shape[0])
            phi_values = np.linspace(-45, 45, rcs_2d.shape[1])

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                # 限制范围在0.5-3.0之间
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            im = ax.imshow(rcs_2d, cmap='jet', aspect='equal',
                          extent=[phi_values.min(), phi_values.max(),
                                 theta_values.max(), theta_values.min()])
            ax.set_title(f'AutoEncoder预测 - 样本{sample_idx+1} - {freq}Hz RCS分布',
                        fontsize=int(24*fontsize_scale), fontweight='bold')
            ax.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')

            # 设置刻度标签字号
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 添加colorbar并设置字号
            cbar = self.gui.vis_fig.colorbar(im, ax=ax, label='RCS')
            cbar.set_label('RCS', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            self.gui.vis_fig.tight_layout()
            self.gui.vis_canvas.draw()

        except Exception as e:
            # 如果模型预测失败，回退到原始数据显示
            print(f"AutoEncoder预测失败，回退到原始数据显示: {str(e)}")
            self._plot_original_rcs_fallback(freq)

    def _plot_original_rcs_fallback(self, freq):
        """当AutoEncoder模型未加载时，显示原始RCS数据作为替代"""
        import rcs_visual as rv

        try:
            # 使用第一个可用的模型数据
            model_id = "001"  # 默认使用模型001

            # 从文件读取原始RCS数据
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])

            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)

            # 定义角度范围 (基于实际数据)
            phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
            theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°
            extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

            # 获取字号缩放因子
            try:
                fontsize_scale = self.gui.fontsize_scale_var.get()
                # 限制范围在0.5-3.0之间
                fontsize_scale = max(0.5, min(3.0, fontsize_scale))
            except:
                fontsize_scale = 1.0

            # 绘制原始数据热图
            im = ax.imshow(data, cmap='jet', aspect='equal', extent=extent)
            ax.set_title(f'原始RCS数据 - 模型 {model_id} - {freq}Hz\n(AutoEncoder模型未加载，显示原始数据)',
                        fontsize=int(24*fontsize_scale), fontweight='bold')
            ax.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')

            # 设置刻度标签字号
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 添加colorbar并设置字号
            cbar = self.gui.vis_fig.colorbar(im, ax=ax, label='RCS (dB)')
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            self.gui.vis_fig.tight_layout()
            self.gui.vis_canvas.draw()

        except Exception as e:
            # 如果连原始数据也读取失败
            self.gui.vis_fig.clear()
            ax = self.gui.vis_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'无法显示数据:\nAutoEncoder模型未加载\n且原始数据读取失败\n\n错误: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
            self.gui.vis_canvas.draw()

    def _plot_ae_comparison(self):
        """绘制AutoEncoder对比图：原图、重构图、残差图 - 使用统一重建函数"""
        import numpy as np

        # 获取字号缩放因子
        try:
            fontsize_scale = self.gui.fontsize_scale_var.get()
            fontsize_scale = max(0.5, min(3.0, fontsize_scale))
        except:
            fontsize_scale = 1.0

        # 检查模型配置
        config_info = self.gui.ae_system.get('config_info', {})
        model_num_freq = config_info.get('num_frequencies', 'unknown')
        model_freq_labels = config_info.get('frequency_labels', 'unknown')
        print(f"\n【模型配置检查】")
        print(f"模型频率数: {model_num_freq}")
        print(f"模型频率标签: {model_freq_labels}")

        # 验证频率配置匹配
        data_num_freq = self.gui.ae_system['rcs_data'].shape[-1]
        if model_num_freq != data_num_freq:
            error_msg = (
                f"频率配置不匹配！无法生成对比图。\n\n"
                f"模型频率: {model_num_freq}频 {model_freq_labels}\n"
                f"数据频率: {data_num_freq}频\n\n"
                f"请重新加载匹配的数据或模型！"
            )
            print(f"❌ {error_msg}")
            messagebox.showerror("频率配置不匹配", error_msg)
            return

        # 获取用户输入的模型ID和频率
        model_id_str = self.gui.vis_model_var.get()
        freq_str = self.gui.vis_freq_var.get()

        try:
            # 从文件读取原始RCS数据 (与2D热图使用相同的数据源)
            data = rv.get_rcs_matrix(model_id_str, freq_str, self.gui.data_config['rcs_data_dir'])
            true_rcs_linear = data['rcs_linear']  # 线性值 [91, 91]
            true_rcs_db = data['rcs_db']  # dB值 [91, 91]
            phi_values = data['phi_values']
            theta_values = data['theta_values']

            # 使用统一重建函数重建RCS
            print(f"\n【使用统一重建函数 - 模型{model_id_str}】")
            result = self.gui._reconstruct_rcs(
                input_data=None,
                input_type='model_ids',
                model_ids=[model_id_str],
                return_latents=False
            )

            predicted_rcs = result['reconstructed_rcs'][0]  # [91, 91, num_freq]
            training_mode = result['training_mode']

            print(f"训练模式: {training_mode}")
            print(f"重建RCS形状: {predicted_rcs.shape}")
            print(f"重建RCS范围: [{predicted_rcs.min():.6e}, {predicted_rcs.max():.6e}]")

            # 获取频率索引
            freq_map = {"1.5G": 0, "3G": 1, "6G": 2}
            freq_idx = freq_map.get(freq_str, 0)

            # 提取该频率的数据
            pred_2d = predicted_rcs[:, :, freq_idx]
            print(f"提取频率{freq_str} (索引{freq_idx}): {pred_2d.shape}")

            # 使用统一绘图函数
            from autoencoder.utils.plotting import plot_rcs_comparison

            # 计算角度范围
            phi_range = (phi_values.min(), phi_values.max())
            theta_range = (theta_values.min(), theta_values.max())

            # 调用统一绘图函数（复用GUI的figure）
            plot_rcs_comparison(
                true_rcs=true_rcs_linear,  # 线性值
                pred_rcs=pred_2d,           # 线性值
                freq_label=freq_str,
                model_id=model_id_str,
                phi_range=phi_range,
                theta_range=theta_range,
                fontsize_scale=fontsize_scale,
                fig=self.gui.vis_fig  # 复用GUI的figure
            )

            # 添加训练模式信息到总标题
            mode_display = {
                'stage1_only': 'Stage 1 Only (RCS重建)',
                'three_stage': 'Three-Stage (参数预测)'
            }.get(training_mode, training_mode)

            self.gui.vis_fig.suptitle(
                f'AutoEncoder对比分析 - 模型{model_id_str} @ {freq_str}\n({mode_display})',
                fontsize=int(24*fontsize_scale),
                fontweight='bold'
            )
            self.gui.vis_fig.tight_layout()
            self.gui.vis_canvas.draw()

        except Exception as e:
            messagebox.showerror("错误", f"无法生成对比图: {str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_wavelet_coefficients_comparison(self):
        """绘制小波系数对比图：原始vs重建的4个通道（LL, LH, HL, HH） - 使用统一绘图函数"""
        import numpy as np

        # 获取字号缩放因子
        try:
            fontsize_scale = self.gui.fontsize_scale_var.get()
            fontsize_scale = max(0.5, min(3.0, fontsize_scale))
        except:
            fontsize_scale = 1.0

        # 检查是否是Wavelet或Differentiable Wavelet模式
        mode = self.gui.ae_system.get('mode', 'wavelet')
        if mode not in ('wavelet', 'differentiable_wavelet'):
            messagebox.showwarning("警告", "此功能仅适用于Wavelet和Differentiable Wavelet模式！")
            return

        # 获取用户输入的模型ID和频率
        model_id_str = self.gui.vis_model_var.get()
        freq_str = self.gui.vis_freq_var.get()

        try:
            # 使用统一重建函数重建RCS，同时获取小波系数
            print(f"\n【小波系数可视化 - 模型{model_id_str}】")
            result = self.gui._reconstruct_rcs(
                input_data=None,
                input_type='model_ids',
                model_ids=[model_id_str],
                return_latents=False,
                return_wavelet_coeffs=True  # 关键：获取小波系数
            )

            # 检查是否成功获取小波系数
            if 'original_wavelet_coeffs' not in result or 'reconstructed_wavelet_coeffs' not in result:
                messagebox.showerror("错误", "无法获取小波系数，请确保模型处于Wavelet模式")
                return

            original_coeffs = result['original_wavelet_coeffs'][0]  # [49, 49, 8]
            reconstructed_coeffs = result['reconstructed_wavelet_coeffs'][0]  # [49, 49, 8]
            training_mode = result['training_mode']

            print(f"原始小波系数形状: {original_coeffs.shape}")
            print(f"重建小波系数形状: {reconstructed_coeffs.shape}")

            # 获取频率索引
            freq_map = {"1.5G": 0, "3G": 1, "6G": 2}
            freq_idx = freq_map.get(freq_str, 0)

            # 频率标签映射
            freq_label_map = {"1.5G": "1.5 GHz", "3G": "3.0 GHz", "6G": "6.0 GHz"}
            freq_label = freq_label_map.get(freq_str, freq_str)

            # 使用统一绘图函数
            from autoencoder.utils.plotting import plot_wavelet_coefficients_comparison

            plot_wavelet_coefficients_comparison(
                original_coeffs=original_coeffs,
                reconstructed_coeffs=reconstructed_coeffs,
                freq_idx=freq_idx,
                freq_label=freq_label,
                model_id=model_id_str,
                fontsize_scale=fontsize_scale,
                fig=self.gui.vis_fig
            )

            # 添加训练模式信息到总标题
            mode_display = {
                'stage1_only': 'Stage 1 Only (RCS重建)',
                'three_stage': 'Three-Stage (参数预测)'
            }.get(training_mode, training_mode)

            # 更新标题以包含训练模式
            current_title = self.gui.vis_fig._suptitle.get_text()
            self.gui.vis_fig.suptitle(f'{current_title}\n({mode_display})',
                                     fontsize=int(24*fontsize_scale), fontweight='bold')

            self.gui.vis_canvas.draw()

        except Exception as e:
            messagebox.showerror("错误", f"无法生成小波系数对比图: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_current_visualization(self):
        """保存当前显示的可视化图表到results文件夹"""
        import os
        from datetime import datetime

        try:
            # 检查是否有图表可以保存
            if not hasattr(self.gui, 'vis_fig') or self.gui.vis_fig is None:
                messagebox.showwarning("警告", "没有可保存的图表！请先生成图表。")
                return

            # 创建结果保存目录（使用会话时间戳，与统计对比保持一致）
            timestamp = self.gui.get_ae_session_timestamp() if hasattr(self.gui, 'ae_session_timestamp') else datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join("results", f"visualization_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)

            # 获取当前图表信息用于生成文件名
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
            self.gui.log_message(f"✅ 图表已保存到: {filepath}")
            messagebox.showinfo("保存成功", f"图表已保存到:\n{filepath}")

        except Exception as e:
            error_msg = f"保存图表失败: {str(e)}"
            self.gui.log_message(f"❌ {error_msg}")
            messagebox.showerror("保存失败", error_msg)
            import traceback
            traceback.print_exc()

    def _plot_gradient_history(self):
        """绘制梯度历史曲线 (三阶段综合视图)"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            # 检查是否有训练历史
            if not hasattr(self.gui, 'ae_training_history') or self.gui.ae_training_history is None:
                messagebox.showwarning("警告", "没有找到训练历史数据！\n请先完成训练。")
                return

            training_history = self.gui.ae_training_history
            stage_histories = training_history.get('stage_histories', {})

            # 检查是否有梯度历史数据
            has_gradient_data = False
            for stage_name in ['stage1', 'stage2', 'stage3']:
                if stage_name in stage_histories:
                    gradient_history = stage_histories[stage_name].get('gradient_history', {})
                    if gradient_history and len(gradient_history.get('epochs', [])) > 0:
                        has_gradient_data = True
                        break

            if not has_gradient_data:
                messagebox.showwarning("警告", "没有找到梯度监控数据！\n可能是使用旧版本训练的模型。")
                return

            # 创建3x2子图布局
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

            # 定义阶段信息
            stages = [
                ('stage1', '阶段1: AutoEncoder预训练', 0),
                ('stage2', '阶段2: 参数映射训练', 1),
                ('stage3', '阶段3: 端到端微调', 2)
            ]

            # 绘制每个阶段的梯度范数和梯度分布
            for stage_name, stage_title, row in stages:
                if stage_name not in stage_histories:
                    continue

                gradient_history = stage_histories[stage_name].get('gradient_history', {})
                if not gradient_history or len(gradient_history.get('epochs', [])) == 0:
                    # 该阶段没有梯度数据
                    ax1 = fig.add_subplot(gs[row, 0])
                    ax1.text(0.5, 0.5, f'{stage_title}\n无梯度数据',
                            ha='center', va='center', fontsize=12, color='gray')
                    ax1.axis('off')

                    ax2 = fig.add_subplot(gs[row, 1])
                    ax2.text(0.5, 0.5, f'{stage_title}\n无梯度数据',
                            ha='center', va='center', fontsize=12, color='gray')
                    ax2.axis('off')
                    continue

                epochs = gradient_history['epochs']
                grad_norm = gradient_history['grad_norm']
                grad_mean = gradient_history['grad_mean']
                grad_std = gradient_history['grad_std']

                # 左侧: 梯度范数历史
                ax1 = fig.add_subplot(gs[row, 0])
                ax1.plot(epochs, grad_norm, linewidth=2, color='blue', marker='o', markersize=4)
                ax1.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                           label='Explosion Threshold (10.0)')
                ax1.axhline(y=1e-5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                           label='Vanishing Threshold (1e-5)')
                ax1.set_yscale('log')
                ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Gradient Norm (L2)', fontsize=11, fontweight='bold')
                ax1.set_title(f'{stage_title} - Gradient Norm', fontsize=12, fontweight='bold')
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)

                # 右侧: 梯度分布 (均值和标准差)
                ax2 = fig.add_subplot(gs[row, 1])
                ax2.plot(epochs, grad_mean, linewidth=2, color='green', marker='s', markersize=4,
                        label='Mean')
                ax2.plot(epochs, grad_std, linewidth=2, color='purple', marker='^', markersize=4,
                        label='Std')
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Gradient Value', fontsize=11, fontweight='bold')
                ax2.set_title(f'{stage_title} - Gradient Distribution', fontsize=12, fontweight='bold')
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)

            # 总标题
            training_mode = training_history.get('training_mode', 'three_stage')
            if training_mode == 'stage1_only':
                fig.suptitle('Gradient Monitoring History (Stage 1 Only Mode)',
                            fontsize=14, fontweight='bold', y=0.995)
            else:
                fig.suptitle('Gradient Monitoring History (Three-Stage Training)',
                            fontsize=14, fontweight='bold', y=0.995)

            # 显示图表
            plt.show()

            self.gui.ae_log("✅ 梯度历史图表已生成")

        except Exception as e:
            error_msg = f"绘制梯度历史失败: {str(e)}"
            self.gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("绘图失败", error_msg)
            import traceback
            traceback.print_exc()

    def _show_gradient_report(self):
        """显示梯度监控总结报告"""
        try:
            import numpy as np

            # 检查是否有训练历史
            if not hasattr(self.gui, 'ae_training_history') or self.gui.ae_training_history is None:
                messagebox.showwarning("警告", "没有找到训练历史数据！\n请先完成训练。")
                return

            training_history = self.gui.ae_training_history
            stage_histories = training_history.get('stage_histories', {})

            # 收集所有阶段的梯度报告
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("梯度监控总结报告")
            report_lines.append("=" * 80)
            report_lines.append("")

            # 定义阶段信息
            stages = [
                ('stage1', '阶段1: AutoEncoder预训练'),
                ('stage2', '阶段2: 参数映射训练'),
                ('stage3', '阶段3: 端到端微调')
            ]

            has_any_data = False

            for stage_name, stage_title in stages:
                if stage_name not in stage_histories:
                    continue

                gradient_history = stage_histories[stage_name].get('gradient_history', {})
                if not gradient_history or len(gradient_history.get('epochs', [])) == 0:
                    continue

                has_any_data = True

                grad_norms = np.array(gradient_history['grad_norm'])
                grad_means = np.array(gradient_history['grad_mean'])
                grad_stds = np.array(gradient_history['grad_std'])

                report_lines.append(f"【{stage_title}】")
                report_lines.append(f"  记录步数: {len(grad_norms)}")
                report_lines.append(f"  Epochs: {gradient_history['epochs'][0]} ~ {gradient_history['epochs'][-1]}")
                report_lines.append("")
                report_lines.append("  梯度范数统计:")
                report_lines.append(f"    均值:   {np.mean(grad_norms):.2e}")
                report_lines.append(f"    中位数: {np.median(grad_norms):.2e}")
                report_lines.append(f"    标准差: {np.std(grad_norms):.2e}")
                report_lines.append(f"    最大值: {np.max(grad_norms):.2e}")
                report_lines.append(f"    最小值: {np.min(grad_norms):.2e}")
                report_lines.append("")
                report_lines.append("  健康度评估:")
                report_lines.append(f"    梯度爆炸次数 (>10.0):  {np.sum(grad_norms > 10.0)}")
                report_lines.append(f"    梯度消失次数 (<1e-5): {np.sum(grad_norms < 1e-5)}")
                healthy_count = np.sum((grad_norms >= 1e-5) & (grad_norms <= 10.0))
                healthy_ratio = healthy_count / len(grad_norms) * 100
                report_lines.append(f"    健康比例: {healthy_ratio:.1f}% ({healthy_count}/{len(grad_norms)})")
                report_lines.append("")
                report_lines.append("-" * 80)
                report_lines.append("")

            if not has_any_data:
                messagebox.showwarning("警告", "没有找到梯度监控数据！\n可能是使用旧版本训练的模型。")
                return

            # 打印报告到日志
            full_report = "\n".join(report_lines)
            self.gui.ae_log("\n" + full_report)

            # 同时弹窗显示
            from tkinter import scrolledtext, Toplevel
            report_window = Toplevel(self.gui.root)
            report_window.title("梯度监控报告")
            report_window.geometry("800x600")

            # 创建滚动文本框
            text_widget = scrolledtext.ScrolledText(report_window, wrap='word', font=('Courier New', 10))
            text_widget.pack(fill='both', expand=True, padx=10, pady=10)
            text_widget.insert('1.0', full_report)
            text_widget.configure(state='disabled')  # 只读

            self.gui.ae_log("✅ 梯度监控报告已生成")

        except Exception as e:
            error_msg = f"生成梯度报告失败: {str(e)}"
            self.gui.ae_log(f"❌ {error_msg}")
            messagebox.showerror("生成失败", error_msg)
            import traceback
            traceback.print_exc()
