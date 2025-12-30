import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rcs_visual as rv
from .base_plotter import BasePlotter

class RCSPlotter(BasePlotter):
    """负责标准RCS数据的可视化"""

    def plot_2d_heatmap(self, model_id, freq, fig=None, canvas=None):
        """绘制2D热图"""
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标 (Figure/Canvas)")
            return

        target_fig.clear()

        try:
            # 使用现有的可视化函数
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])

            ax = target_fig.add_subplot(1, 1, 1)

            # 获取实际的角度范围
            phi_values = data['phi_values']
            theta_values = data['theta_values']

            fontsize_scale = self.get_fontsize_scale()

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
            cbar = target_fig.colorbar(im, ax=ax, label='RCS (dB)')
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            target_fig.tight_layout()
            target_canvas.draw()

        except Exception as e:
            self.handle_error("无法生成2D热图", e)

    def plot_3d_surface(self, model_id, freq, fig=None, canvas=None):
        """绘制3D表面图"""
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            return

        try:
            target_fig.clear()
            self.log(f"绘制模型 {model_id} - {freq} 的3D表面图...")

            fontsize_scale = self.get_fontsize_scale()

            # 获取RCS数据
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])
            rcs_data = data['rcs_db']  # dB值

            # 创建坐标网格
            theta_range = np.linspace(45, 135, rcs_data.shape[0])  # 俯仰角
            phi_range = np.linspace(-45, 45, rcs_data.shape[1])    # 偏航角
            Theta, Phi = np.meshgrid(theta_range, phi_range, indexing='ij')

            # 创建3D子图
            ax = target_fig.add_subplot(1, 1, 1, projection='3d')

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
            cbar = target_fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='RCS (dB)')
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 设置视角
            ax.view_init(elev=30, azim=45)
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax.zaxis.set_tick_params(labelsize=int(16*fontsize_scale))

            target_canvas.draw()
            self.log("3D表面图绘制完成")

        except Exception as e:
            self.handle_error("3D表面图绘制失败", e)

    def plot_spherical(self, model_id, freq, fig=None, canvas=None):
        """绘制球坐标图"""
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            return

        try:
            target_fig.clear()
            self.log(f"绘制模型 {model_id} - {freq} 的球坐标图...")

            fontsize_scale = self.get_fontsize_scale()

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
            ax = target_fig.add_subplot(1, 1, 1, projection='3d')

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
            cbar = target_fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            # 设置视角
            ax.view_init(elev=20, azim=30)
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax.zaxis.set_tick_params(labelsize=int(16*fontsize_scale))

            target_canvas.draw()
            self.log("球坐标图绘制完成")

        except Exception as e:
            self.handle_error("球坐标图绘制失败", e)

    def plot_original_rcs_fallback(self, freq, rcs_data=None, fig=None, canvas=None, log_callback=None):
        """当AutoEncoder模型未加载时，显示原始RCS数据作为替代"""
        
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        if log_callback:
            # 临时覆盖log方法，如果提供了专门的回调
            self_log_backup = self.log
            self.log = log_callback
            
        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            if log_callback: self.log = self_log_backup
            return

        try:
            # 使用第一个可用的模型数据
            model_id = "001"  # 默认使用模型001

            # 从文件读取原始RCS数据
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])

            target_fig.clear()
            ax = target_fig.add_subplot(1, 1, 1)

            # 定义角度范围 (基于实际数据)
            phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
            theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°
            extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

            fontsize_scale = self.get_fontsize_scale()

            # 绘制原始数据热图
            im = ax.imshow(data['rcs_db'], cmap='jet', aspect='equal', extent=extent)
            ax.set_title(f'原始RCS数据 - 模型 {model_id} - {freq}Hz\n(AutoEncoder模型未加载，显示原始数据)',
                        fontsize=int(24*fontsize_scale), fontweight='bold')
            ax.set_xlabel('φ (方位角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_ylabel('θ (俯仰角, 度)', fontsize=int(20*fontsize_scale), fontweight='bold')

            # 设置刻度标签字号
            ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 添加colorbar并设置字号
            cbar = target_fig.colorbar(im, ax=ax, label='RCS (dB)')
            cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

            target_fig.tight_layout()
            target_canvas.draw()

        except Exception as e:
            # 如果连原始数据也读取失败
            target_fig.clear()
            ax = target_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'无法显示数据:\nAutoEncoder模型未加载\n且原始数据读取失败\n\n错误: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
            target_canvas.draw()
        finally:
            if log_callback: self.log = self_log_backup
