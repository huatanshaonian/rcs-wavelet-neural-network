import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tkinter import messagebox, Toplevel, scrolledtext
from matplotlib.gridspec import GridSpec
from .base_plotter import BasePlotter

class HistoryPlotter(BasePlotter):
    """负责训练历史和梯度监控的可视化"""

    def plot_training_history(self):
        """绘制训练历史图（对交叉验证，分别保存每折到results文件夹，GUI显示最佳折）"""
        if not hasattr(self.gui, 'training_history') or not self.gui.training_history:
            messagebox.showwarning("警告", "没有训练历史数据，请先进行训练")
            return

        try:
            fontsize_scale = self.get_fontsize_scale()

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

                self.log(f"已保存{len(fold_details)}折训练图表到{results_dir}目录")
                self.log(f"GUI显示最佳折 {best_fold_idx + 1} 的训练历史")

            else:
                # 单次训练模式：直接显示
                self._display_simple_training_history(fontsize_scale=fontsize_scale)

        except Exception as e:
            self.handle_error("绘制训练历史失败", e)

    def _save_fold_plot(self, fold_data, fold_idx, results_dir, fontsize_scale=1.0):
        """保存单个折的训练历史图表"""
        self.setup_chinese_font()

        # 创建独立的图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'交叉验证第{fold_idx + 1}折 - 训练历史',
                     fontsize=int(24*fontsize_scale), fontweight='bold')

        epochs = fold_data.get('epochs', [])
        train_losses = fold_data.get('train_losses', [])
        val_losses = fold_data.get('val_losses', [])

        if not epochs or not train_losses:
            plt.close(fig)
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

    def _display_fold_in_gui(self, fold_data, fold_idx, fontsize_scale=1.0):
        """在GUI中显示指定折的训练历史"""
        self.setup_chinese_font()
        target_fig, target_canvas = self.get_target_fig_canvas()
        if not target_fig: return

        target_fig.clear()

        epochs = fold_data.get('epochs', [])
        train_losses = fold_data.get('train_losses', [])
        val_losses = fold_data.get('val_losses', [])

        if not epochs or not train_losses:
            target_fig.text(0.5, 0.5, f'第{fold_idx + 1}折数据不完整',
                            ha='center', va='center',
                            fontsize=int(20*fontsize_scale), fontweight='bold')
            target_canvas.draw()
            return

        # 主损失曲线
        ax1 = target_fig.add_subplot(2, 2, 1)
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
        ax2 = target_fig.add_subplot(2, 2, 2)
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
        ax3 = target_fig.add_subplot(2, 2, 3)
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
        ax4 = target_fig.add_subplot(2, 2, 4)
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

        target_fig.tight_layout()
        target_canvas.draw()

    def _display_simple_training_history(self, fontsize_scale=1.0):
        """显示简单训练模式的历史（非交叉验证）"""
        self.setup_chinese_font()
        target_fig, target_canvas = self.get_target_fig_canvas()
        if not target_fig: return

        target_fig.clear()

        epochs = self.gui.training_history.get('epochs', [])
        train_loss = self.gui.training_history.get('train_loss', [])
        val_loss = self.gui.training_history.get('val_loss', [])

        if not epochs or not train_loss:
            target_fig.text(0.5, 0.5, '训练历史数据不完整',
                            ha='center', va='center',
                            fontsize=int(20*fontsize_scale), fontweight='bold')
            target_canvas.draw()
            return

        # 主损失曲线
        ax1 = target_fig.add_subplot(2, 2, 1)
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
        ax2 = target_fig.add_subplot(2, 2, 2)
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
        ax3 = target_fig.add_subplot(2, 2, 3)
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
        ax4 = target_fig.add_subplot(2, 2, 4)
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

        target_fig.tight_layout()
        target_canvas.draw()

    def plot_gradient_history(self):
        """绘制梯度历史曲线 (三阶段综合视图)"""
        try:
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
            # 注意：这里我们创建一个新的Figure，而不是使用GUI的main figure
            # 因为这个函数通常在弹窗中使用，或者需要独立的窗口
            # 但原代码似乎没有指明目标，所以我们假设它是弹出的
            # 然而，matplotlib.pyplot.figure() 会创建一个新窗口
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

            self.log("✅ 梯度历史图表已生成")

        except Exception as e:
            self.handle_error("绘制梯度历史失败", e)

    def show_gradient_report(self):
        """显示梯度监控总结报告"""
        try:
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
            if hasattr(self.gui, 'ae_log'):
                self.gui.ae_log("\n" + full_report)
            else:
                self.log("\n" + full_report)

            # 同时弹窗显示
            report_window = Toplevel(self.gui.root)
            report_window.title("梯度监控报告")
            report_window.geometry("800x600")

            # 创建滚动文本框
            text_widget = scrolledtext.ScrolledText(report_window, wrap='word', font=('Courier New', 10))
            text_widget.pack(fill='both', expand=True, padx=10, pady=10)
            text_widget.insert('1.0', full_report)
            text_widget.configure(state='disabled')  # 只读

            self.log("✅ 梯度监控报告已生成")

        except Exception as e:
            self.handle_error("生成梯度报告失败", e)
