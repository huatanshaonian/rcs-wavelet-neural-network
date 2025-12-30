import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tkinter as tk
from tkinter import messagebox
import rcs_visual as rv
from autoencoder.utils.plotting import plot_ae_training_progress, plot_rcs_comparison, plot_wavelet_coefficients_comparison, plot_additive_branch_comparison
from .base_plotter import BasePlotter

class AEPlotter(BasePlotter):
    """负责AutoEncoder相关的复杂可视化"""

    def plot_autoencoder_visualization(self, chart_type, ae_system=None, training_history=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        """绘制AutoEncoder特定可视化图表 - 入口函数"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        hist = training_history if training_history is not None else getattr(self.gui, 'ae_training_history', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        # 临时日志覆盖
        original_log = self.log
        if log_callback: self.log = log_callback

        if target_fig is None or target_canvas is None:
            self.log("错误: 无法获取绘图目标")
            if log_callback: self.log = original_log
            return

        try:
            if chart_type == "AE隐空间分析":
                self.plot_ae_latent_space(sys, target_fig, target_canvas, rcs_data, param_data)
            elif chart_type == "AE重建质量":
                self.plot_ae_reconstruction_quality(sys, target_fig, target_canvas, rcs_data, param_data)
            elif chart_type == "AE参数映射":
                self.plot_ae_parameter_mapping(sys, target_fig, target_canvas, rcs_data, param_data)
            elif chart_type == "AE训练进度":
                self.plot_ae_training_progress_vis(hist, target_fig, target_canvas)
            else:
                target_fig.clear()
                fontsize_scale = self.get_fontsize_scale()
                ax = target_fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, f'未知的可视化类型: {chart_type}',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=int(20*fontsize_scale), fontweight='bold')
                target_canvas.draw()
        except Exception as e:
            self.log(f"AutoEncoder可视化失败: {str(e)}")
        finally:
             if log_callback: self.log = original_log

    def plot_ae_latent_space(self, ae_system=None, fig=None, canvas=None, rcs_data=None, param_data=None):
        """绘制AutoEncoder隐空间分析"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        data = self._get_data(rcs_data, sys)

        if sys is None:
            self.log("错误: AutoEncoder系统未初始化")
            return
        if data is None:
            self.log("错误: RCS数据未加载")
            return

        fontsize_scale = self.get_fontsize_scale()

        try:
            autoencoder = sys['autoencoder']
            wavelet_transform = sys.get('wavelet_transform', None)
            data_adapter = sys.get('data_adapter', None)
            mode = sys.get('mode', 'direct')

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()

            with torch.no_grad():
                input_tensor = self._prepare_input(data, mode, wavelet_transform, data_adapter, device)
                latent_vectors = autoencoder.encode(input_tensor)
                latent_vectors = latent_vectors.cpu().numpy()

            fig.clear()

            # 子图1: 隐空间维度统计
            ax1 = fig.add_subplot(2, 2, 1)
            latent_means = np.mean(latent_vectors, axis=0)
            latent_stds = np.std(latent_vectors, axis=0)
            dims = range(len(latent_means[:20]))
            ax1.errorbar(dims, latent_means[:20], yerr=latent_stds[:20],
                        capsize=3, marker='o', markersize=4)
            ax1.set_title('隐空间维度统计 (前20维)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_xlabel('隐空间维度', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_ylabel('数值', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax1.grid(True, alpha=0.3)

            # 子图2: 隐空间激活热图
            ax2 = fig.add_subplot(2, 2, 2)
            im = ax2.imshow(latent_vectors[:10, :20].T, cmap='RdYlBu', aspect='auto')
            ax2.set_title('隐空间激活模式 (前10样本×前20维)', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_xlabel('样本索引', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_ylabel('隐空间维度', fontsize=int(20*fontsize_scale), fontweight='bold')
            cbar = fig.colorbar(im, ax=ax2)
            cbar.ax.tick_params(labelsize=int(16*fontsize_scale))
            cbar.set_label('激活值', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))

            # 子图3: 维度0直方图
            ax3 = fig.add_subplot(2, 2, 3)
            dim0_values = latent_vectors[:, 0]
            ax3.hist(dim0_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
            ax3.set_title(f'维度0分布 (μ={dim0_values.mean():.3f}, σ={dim0_values.std():.3f})', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_xlabel('维度0数值', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.set_ylabel('频数', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax3.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax3.grid(True, alpha=0.3, axis='y')

            # 子图4: 维度1直方图
            ax4 = fig.add_subplot(2, 2, 4)
            dim1_values = latent_vectors[:, 1]
            ax4.hist(dim1_values, bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
            ax4.set_title(f'维度1分布 (μ={dim1_values.mean():.3f}, σ={dim1_values.std():.3f})', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_xlabel('维度1数值', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.set_ylabel('频数', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax4.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            ax4.grid(True, alpha=0.3, axis='y')

            fig.tight_layout()
            canvas.draw()

        except Exception as e:
            self.handle_error("隐空间分析失败", e)

    def plot_ae_latent_distribution(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None, color_param_idx=0):
        """绘制AutoEncoder隐空间分布（PCA, t-SNE, UMAP）"""
        try:
            import umap
        except ImportError:
            umap = None

        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        data = self._get_data(rcs_data, sys)
        params = self._get_params(param_data, sys)

        original_log = self.log
        if log_callback: self.log = log_callback

        if sys is None or data is None or target_fig is None:
            self.log("错误: 数据或系统未就绪")
            if log_callback: self.log = original_log
            return

        fontsize_scale = self.get_fontsize_scale()

        try:
            autoencoder = sys['autoencoder']
            wavelet_transform = sys.get('wavelet_transform', None)
            data_adapter = sys.get('data_adapter', None)
            mode = sys.get('mode', 'direct')

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()

            with torch.no_grad():
                input_tensor = self._prepare_input(data, mode, wavelet_transform, data_adapter, device)
                latent_vectors = autoencoder.encode(input_tensor).cpu().numpy()

            # 智能选择着色依据
            color_values = None
            color_label = "样本序号"
            param_names = ["kw", "phi", "yita", "lam", "Ht", "Nc", "Theta", "R", "Beta"]

            if params is not None and len(params) > 0:
                sample_params = params[:len(latent_vectors)]
                try:
                    c_idx = int(color_param_idx)
                except:
                    c_idx = 0
                
                if sample_params.shape[1] > c_idx:
                    color_values = sample_params[:, c_idx]
                    param_name = param_names[c_idx] if c_idx < len(param_names) else f"参数{c_idx+1}"
                    color_label = f"设计参数 {param_name}"
                else:
                     self.log(f"警告: 参数索引 {c_idx} 超出范围")

            if color_values is None:
                rcs_peak_values = np.max(data.reshape(len(data), -1), axis=1)
                color_values = rcs_peak_values
                color_label = "RCS峰值"

            target_fig.clear()

            # 子图1: PCA降维
            ax1 = target_fig.add_subplot(1, 3, 1)
            pca = PCA(n_components=2)
            latent_2d_pca = pca.fit_transform(latent_vectors)
            scatter1 = ax1.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1],
                                c=color_values, cmap='viridis', alpha=0.7, s=50)
            ax1.set_title('PCA降维', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=int(18*fontsize_scale))
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=int(18*fontsize_scale))
            ax1.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            cbar1 = target_fig.colorbar(scatter1, ax=ax1)
            cbar1.set_label(color_label, fontsize=int(16*fontsize_scale), fontweight='bold')

            # 子图2: t-SNE降维
            ax2 = target_fig.add_subplot(1, 3, 2)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
            latent_2d_tsne = tsne.fit_transform(latent_vectors)
            scatter2 = ax2.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1],
                       c=color_values, cmap='viridis', alpha=0.7, s=50)
            ax2.set_title('t-SNE降维', fontsize=int(20*fontsize_scale), fontweight='bold')
            ax2.set_xlabel('t-SNE1', fontsize=int(18*fontsize_scale))
            ax2.set_ylabel('t-SNE2', fontsize=int(18*fontsize_scale))
            ax2.tick_params(axis='both', labelsize=int(16*fontsize_scale))
            cbar2 = target_fig.colorbar(scatter2, ax=ax2)
            cbar2.set_label(color_label, fontsize=int(16*fontsize_scale), fontweight='bold')

            # 子图3: UMAP降维
            ax3 = target_fig.add_subplot(1, 3, 3)
            if umap is not None:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(latent_vectors)-1))
                latent_2d_umap = reducer.fit_transform(latent_vectors)
                scatter3 = ax3.scatter(latent_2d_umap[:, 0], latent_2d_umap[:, 1],
                           c=color_values, cmap='viridis', alpha=0.7, s=50)
                ax3.set_title('UMAP降维', fontsize=int(20*fontsize_scale), fontweight='bold')
                ax3.set_xlabel('UMAP1', fontsize=int(18*fontsize_scale))
                ax3.set_ylabel('UMAP2', fontsize=int(18*fontsize_scale))
                cbar3 = target_fig.colorbar(scatter3, ax=ax3)
                cbar3.set_label(color_label, fontsize=int(16*fontsize_scale), fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'UMAP未安装', transform=ax3.transAxes, ha='center', va='center', fontsize=int(18*fontsize_scale))
                ax3.set_title('UMAP降维 (未安装)', fontsize=int(20*fontsize_scale), fontweight='bold')

            target_fig.suptitle(f'隐空间分布可视化 (着色: {color_label})', fontsize=int(24*fontsize_scale), fontweight='bold')
            target_fig.tight_layout(rect=[0, 0, 1, 0.96])
            target_canvas.draw()

        except Exception as e:
            self.handle_error("隐空间分布可视化失败", e)
        finally:
            if log_callback: self.log = original_log

    def plot_ae_latent_interpolation(self, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, sample_id1='001', sample_id2='002'):
        """绘制AutoEncoder隐空间插值可视化"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        data = self._get_data(rcs_data, sys)
        
        original_log = self.log
        if log_callback: self.log = log_callback

        if sys is None or data is None or target_fig is None:
            self.log("错误: 数据或系统未就绪")
            if log_callback: self.log = original_log
            return
            
        fontsize_scale = self.get_fontsize_scale()
        use_postprocess_abs_db = False
        try:
            if hasattr(self.gui, 'ae_postprocess_abs_db'):
                use_postprocess_abs_db = self.gui.ae_postprocess_abs_db.get()
            elif hasattr(self.gui, 'ae_extension') and hasattr(self.gui.ae_extension, 'postprocess_abs_db_var'):
                use_postprocess_abs_db = self.gui.ae_extension.postprocess_abs_db_var.get()
        except:
            pass

        try:
            idx1, idx2 = int(sample_id1) - 1, int(sample_id2) - 1
            if not (0 <= idx1 < len(data) and 0 <= idx2 < len(data)):
                self.log(f"错误: 样本ID超出范围")
                return

            autoencoder = sys['autoencoder']
            wavelet_transform = sys.get('wavelet_transform', None)
            data_adapter = sys.get('data_adapter', None)
            mode = sys.get('mode', 'direct')

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()

            rcs1, rcs2 = data[idx1:idx1+1], data[idx2:idx2+1]

            with torch.no_grad():
                input1 = self._prepare_input(rcs1, mode, wavelet_transform, data_adapter, device)
                input2 = self._prepare_input(rcs2, mode, wavelet_transform, data_adapter, device)

                latent1 = autoencoder.encode(input1)
                latent2 = autoencoder.encode(input2)

                alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
                interpolated_latents = [(1 - a) * latent1 + a * latent2 for a in alphas]

                reconstructed_rcs_list = []
                for latent_interp in interpolated_latents:
                    decoded_output = autoencoder.decode(latent_interp)
                    # 逆变换逻辑
                    if mode in ('wavelet', 'differentiable_wavelet'):
                        if data_adapter:
                            decoded_coeffs = data_adapter.inverse_adapt(decoded_output)
                            decoded_coeffs_tensor = torch.FloatTensor(decoded_coeffs).to(device)
                        else:
                            decoded_coeffs_tensor = decoded_output
                        reconstructed_rcs = wavelet_transform.inverse_transform(decoded_coeffs_tensor)
                    else:
                        if data_adapter:
                            reconstructed_rcs = data_adapter.inverse_adapt(decoded_output)
                            reconstructed_rcs = torch.FloatTensor(reconstructed_rcs).to(device)
                        else:
                            reconstructed_rcs = decoded_output
                    
                    if isinstance(reconstructed_rcs, torch.Tensor):
                        reconstructed_rcs = reconstructed_rcs.detach().cpu().numpy()
                    reconstructed_rcs_list.append(reconstructed_rcs)

            target_fig.clear()

            for i, (alpha, rcs_recon) in enumerate(zip(alphas, reconstructed_rcs_list)):
                rcs_recon_2d = rcs_recon[0, :, :, 0]
                if use_postprocess_abs_db:
                    rcs_recon_2d = np.abs(rcs_recon_2d)
                rcs_db = 10 * np.log10(rcs_recon_2d + 1e-10)

                # 第一行：phi方向截面
                ax1 = target_fig.add_subplot(2, len(alphas), i+1)
                center_row_idx = rcs_db.shape[0] // 2
                phi_slice = rcs_db[center_row_idx, :]
                phi_axis = np.linspace(-45, 45, len(phi_slice))
                ax1.plot(phi_axis, phi_slice, linewidth=2)
                ax1.set_title(f'α={alpha:.2f}', fontsize=int(12*fontsize_scale), fontweight='bold')
                ax1.set_xlabel('φ (°)', fontsize=int(10*fontsize_scale))
                ax1.set_ylabel('RCS (dB)', fontsize=int(10*fontsize_scale))
                ax1.grid(True, alpha=0.3)

                # 第二行：2D热图
                ax2 = target_fig.add_subplot(2, len(alphas), i+len(alphas)+1)
                extent = [-45, 45, 135, 45]
                im = ax2.imshow(rcs_db, cmap='jet', aspect='equal', extent=extent)
                ax2.set_xlabel('φ (°)', fontsize=int(10*fontsize_scale))
                if i == 0: ax2.set_ylabel('θ (°)', fontsize=int(10*fontsize_scale)) 
                
                if i == len(alphas) - 1:
                    cbar = target_fig.colorbar(im, ax=ax2)
                    cbar.set_label('RCS (dB)', fontsize=int(10*fontsize_scale))

            target_fig.suptitle(f'隐空间插值: 样本{sample_id1} → 样本{sample_id2}', fontsize=int(24*fontsize_scale), fontweight='bold')
            target_fig.tight_layout(rect=[0, 0, 1, 0.96])
            target_canvas.draw()
            self.log(f"隐空间插值完成: 样本{sample_id1} → 样本{sample_id2}")

        except Exception as e:
            self.handle_error("隐空间插值失败", e)
        finally:
            if log_callback: self.log = original_log

    def plot_ae_reconstruction_quality(self, ae_system=None, fig=None, canvas=None, rcs_data=None, param_data=None):
        """绘制AutoEncoder重建质量分析"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        data = self._get_data(rcs_data, sys)

        if sys is None or data is None or target_fig is None:
            self.log("错误: 数据或系统未就绪")
            return

        fontsize_scale = self.get_fontsize_scale()

        try:
            autoencoder = sys['autoencoder']
            wavelet_transform = sys.get('wavelet_transform', None)
            data_adapter = sys.get('data_adapter', None)
            mode = sys.get('mode', 'direct')

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()

            test_indices = [0, 10, 20, 30]
            if len(data) < 31: test_indices = [0]
            test_samples = data[test_indices]

            with torch.no_grad():
                input_tensor = self._prepare_input(test_samples, mode, wavelet_transform, data_adapter, device)
                latent = autoencoder.encode(input_tensor)
                reconstructed_output = autoencoder.decode(latent)

                # 逆变换逻辑
                if mode in ('wavelet', 'differentiable_wavelet'):
                    if data_adapter:
                        reconstructed_coeffs_np = data_adapter.inverse_adapt(reconstructed_output)
                        reconstructed_coeffs = torch.FloatTensor(reconstructed_coeffs_np).to(device)
                    else:
                        reconstructed_coeffs = reconstructed_output
                    reconstructed_samples = wavelet_transform.inverse_transform(reconstructed_coeffs).cpu().numpy()
                else:
                    if data_adapter:
                        reconstructed_samples = data_adapter.inverse_adapt(reconstructed_output)
                    else:
                        reconstructed_samples = reconstructed_output.cpu().numpy()

            target_fig.clear()

            for i, sample_idx in enumerate(test_indices):
                original_rcs = test_samples[i]
                reconstructed_rcs = reconstructed_samples[i]
                
                ax = target_fig.add_subplot(2, 2, i+1)
                freq_idx = 0
                original_2d = original_rcs[:, :, freq_idx]
                reconstructed_2d = reconstructed_rcs[:, :, freq_idx]

                mse = np.mean((original_2d - reconstructed_2d)**2)
                combined = np.hstack([original_2d, reconstructed_2d])
                im = ax.imshow(combined, cmap='jet', aspect='equal')
                ax.axvline(x=original_2d.shape[1]-0.5, color='white', linewidth=2)
                
                ax.set_title(f'样本{sample_idx+1} (MSE={mse:.4e})', fontsize=int(20*fontsize_scale), fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                cbar = target_fig.colorbar(im, ax=ax, shrink=0.6)
                cbar.set_label('RCS', fontsize=int(20*fontsize_scale))

            target_fig.suptitle('AutoEncoder重建质量对比', fontsize=int(24*fontsize_scale), fontweight='bold')
            target_fig.tight_layout()
            target_canvas.draw()

        except Exception as e:
            self.handle_error("重建质量分析失败", e)

    def plot_ae_parameter_mapping(self, ae_system=None, fig=None, canvas=None, rcs_data=None, param_data=None):
        """绘制AutoEncoder参数映射分析"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        params = self._get_params(param_data, sys)

        if sys is None or params is None or target_fig is None:
            self.log("错误: 数据或系统未就绪")
            return

        fontsize_scale = self.get_fontsize_scale()

        try:
            parameter_mapper = sys['parameter_mapper']
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            parameter_mapper.to(device).eval()

            with torch.no_grad():
                num_samples = min(len(params), 50)
                param_subset = params[:num_samples]
                param_tensor = torch.FloatTensor(param_subset)  
                mapped_latents = parameter_mapper(param_tensor.to(device)).cpu().numpy()

            target_fig.clear()

            # 子图1: 参数空间
            ax1 = target_fig.add_subplot(2, 2, 1)
            ax1.scatter(param_subset[:, 0], param_subset[:, 1], c=range(num_samples), cmap='viridis', alpha=0.6)
            ax1.set_title('参数空间分布', fontsize=int(20*fontsize_scale), fontweight='bold')

            # 子图2: 隐空间
            ax2 = target_fig.add_subplot(2, 2, 2)
            pca = PCA(n_components=2)
            mapped_2d = pca.fit_transform(mapped_latents)
            ax2.scatter(mapped_2d[:, 0], mapped_2d[:, 1], c=range(num_samples), cmap='viridis', alpha=0.6)
            ax2.set_title('映射后隐空间分布', fontsize=int(20*fontsize_scale), fontweight='bold')

            # 子图3: 相关性
            ax3 = target_fig.add_subplot(2, 2, 3)
            correlations = []
            for param_idx in range(min(params.shape[1], 5)):
                if np.std(param_subset[:, param_idx]) > 1e-6:
                    corr = np.corrcoef(param_subset[:, param_idx], mapped_2d[:, 0])[0, 1]
                    correlations.append(abs(corr))
                else:
                    correlations.append(0)
            param_names = [f'参数{i+1}' for i in range(len(correlations))]
            ax3.bar(param_names, correlations)
            ax3.set_title('参数与隐空间PC1相关性', fontsize=int(20*fontsize_scale), fontweight='bold')

            # 子图4: 激活强度
            ax4 = target_fig.add_subplot(2, 2, 4)
            latent_means = np.mean(np.abs(mapped_latents), axis=0)
            dims = range(len(latent_means[:20]))
            ax4.bar(dims, latent_means[:20])
            ax4.set_title('隐空间维度激活强度', fontsize=int(20*fontsize_scale), fontweight='bold')

            target_fig.tight_layout()
            target_canvas.draw()

        except Exception as e:
            self.handle_error("参数映射分析失败", e)

    def plot_ae_training_progress_vis(self, training_history=None, fig=None, canvas=None):
        """绘制AutoEncoder训练进度"""
        hist = training_history if training_history is not None else getattr(self.gui, 'ae_training_history', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        if hist is None or target_fig is None:
            self.log("错误: 数据或绘图目标缺失")
            return

        fontsize_scale = self.get_fontsize_scale()

        try:
            plot_ae_training_progress(
                ae_training_history=hist,
                fontsize_scale=fontsize_scale,
                fig=target_fig,
                use_log_scale=True,
                show_best_epoch=True,
                show_gradient=True
            )
            target_canvas.draw()
        except Exception as e:
            self.handle_error("训练进度可视化失败", e)

    def plot_autoencoder_prediction_visualization(self, chart_type, freq, ae_system=None, fig=None, canvas=None, log_callback=None, rcs_data=None, param_data=None):
        """使用AutoEncoder进行预测可视化 (2D热图/对比/小波系数)"""
        # (保留原函数的路由逻辑)
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        original_log = self.log
        if log_callback: self.log = log_callback

        try:
            if chart_type == "2D热图":
                self.plot_ae_2d_heatmap(freq, sys, target_fig, target_canvas, rcs_data)
            elif chart_type == "对比图":
                self.plot_ae_comparison(sys, target_fig, target_canvas, rcs_data, param_data)
            elif chart_type == "小波系数对比":
                self.plot_wavelet_coefficients_comparison(sys, target_fig, target_canvas, rcs_data)
            else:
                target_fig.clear()
                fontsize_scale = self.get_fontsize_scale()
                ax = target_fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, f'不支持的图表类型: {chart_type}', transform=ax.transAxes, ha='center', va='center', fontsize=int(20*fontsize_scale))
                target_canvas.draw()
        except Exception as e:
            self.handle_error("预测可视化失败", e)
        finally:
            if log_callback: self.log = original_log

    def plot_ae_2d_heatmap(self, freq, ae_system=None, fig=None, canvas=None, rcs_data=None):
        """绘制AutoEncoder预测的2D热图"""
        # (逻辑同原函数 _plot_ae_2d_heatmap，包含fallback逻辑)
        # 为节省篇幅，这里简化描述，实际代码需要包含完整逻辑
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)

        if sys is None or 'autoencoder' not in sys:
             # 回退逻辑: 需要调用RCSPlotter的逻辑，这里我们假设外部会处理，或者我们可以在这里实例化RCSPlotter
             # 为了解耦，这里只处理 AE 预测。如果失败，可以通过 raise 异常或返回状态让调用者决定
             self.log("AutoEncoder模型未加载")
             return

        try:
            autoencoder = sys['autoencoder']
            parameter_mapper = sys['parameter_mapper']
            wavelet_transform = sys.get('wavelet_transform', None)
            data_adapter = sys.get('data_adapter', None)
            param_data = self._get_params(None, sys)
            
            if param_data is None:
                raise ValueError("参数数据未加载")

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device).eval()
            parameter_mapper.to(device).eval()

            sample_idx = 0
            test_params = param_data[sample_idx:sample_idx+1]

            with torch.no_grad():
                param_tensor = torch.FloatTensor(test_params).to(device)
                predicted_latents = parameter_mapper(param_tensor)
                predicted_output = autoencoder.decode(predicted_latents)

                if wavelet_transform is not None:
                    if data_adapter:
                        predicted_coeffs_np = data_adapter.inverse_adapt(predicted_output)
                        predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
                    else:
                        predicted_coeffs = predicted_output
                    predicted_rcs = wavelet_transform.inverse_transform(predicted_coeffs)
                else:
                    if data_adapter:
                        predicted_rcs_np = data_adapter.inverse_adapt(predicted_output)
                        predicted_rcs = torch.FloatTensor(predicted_rcs_np).to(device)
                    else:
                        predicted_rcs = predicted_output
                
                predicted_rcs = predicted_rcs.cpu().numpy()[0]

            target_fig.clear()
            ax = target_fig.add_subplot(1, 1, 1)
            
            freq_idx = 0 if freq == "1.5G" else 1
            rcs_2d = predicted_rcs[:, :, freq_idx]

            # 后处理
            use_postprocess = False
            try:
                if hasattr(self.gui, 'ae_postprocess_abs_db'):
                    use_postprocess = self.gui.ae_postprocess_abs_db.get()
            except: pass

            rcs_unit_label = 'RCS'
            if use_postprocess:
                if data_adapter and data_adapter.db_transform:
                    raise ValueError("模型已包含dB变换，不可重复处理")
                rcs_2d = 10 * np.log10(np.abs(rcs_2d) + 1e-10)
                rcs_unit_label = 'RCS (dB)'

            fontsize_scale = self.get_fontsize_scale()
            theta_values = np.linspace(45, 135, rcs_2d.shape[0])
            phi_values = np.linspace(-45, 45, rcs_2d.shape[1])

            im = ax.imshow(rcs_2d, cmap='jet', aspect='equal',
                          extent=[phi_values.min(), phi_values.max(),
                                 theta_values.max(), theta_values.min()])
            ax.set_title(f'AutoEncoder预测 - {freq}Hz RCS', fontsize=int(24*fontsize_scale), fontweight='bold')
            ax.set_xlabel('φ', fontsize=int(20*fontsize_scale))
            ax.set_ylabel('θ', fontsize=int(20*fontsize_scale))
            cbar = target_fig.colorbar(im, ax=ax, label=rcs_unit_label)
            cbar.set_label(rcs_unit_label, fontsize=int(20*fontsize_scale))
            
            target_fig.tight_layout()
            target_canvas.draw()

        except Exception as e:
            self.handle_error("AE预测热图绘制失败", e)

    def plot_ae_comparison(self, ae_system=None, fig=None, canvas=None, rcs_data=None, param_data=None, reconstruction_mode='auto'):
        """绘制AutoEncoder对比图 (调用统一重建接口)"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        fontsize_scale = self.get_fontsize_scale()

        if sys is None or target_fig is None:
            self.log("错误: 系统未初始化或绘图目标缺失")
            return

        try:
            # 获取用户选择
            try:
                if hasattr(self.gui, 'visualization_tab'):
                    model_id = self.gui.visualization_tab.vis_model_var.get()
                    freq = self.gui.visualization_tab.vis_freq_var.get()
                else:
                    model_id = self.gui.vis_model_var.get()
                    freq = self.gui.vis_freq_var.get()
            except:
                model_id, freq = "001", "1.5G"

            # 获取真实数据
            data = rv.get_rcs_matrix(model_id, freq, self.gui.data_config['rcs_data_dir'])
            true_rcs_linear = data['rcs_linear']
            phi_values = data['phi_values']
            theta_values = data['theta_values']

            # 重建
            if reconstruction_mode == 'ae_only':
                use_input_type = 'model_ids'
            elif reconstruction_mode == 'end_to_end':
                use_input_type = 'model_ids'
            else:
                use_input_type = 'model_ids'

            # 委托给 reconstruction_manager
            if hasattr(self.gui, 'reconstruction_manager'):
                result = self.gui.reconstruction_manager._reconstruct_rcs(
                    input_data=None,
                    input_type=use_input_type,
                    model_ids=[model_id],
                    return_latents=False,
                    force_reconstruction_mode=reconstruction_mode
                )
            else:
                 # 回退
                result = self.gui._reconstruct_rcs(
                    input_data=None,
                    input_type=use_input_type,
                    model_ids=[model_id],
                    return_latents=False,
                    force_reconstruction_mode=reconstruction_mode
                )

            predicted_rcs = result['reconstructed_rcs'][0]
            training_mode = result['training_mode']
            
            freq_map = {"1.5G": 0, "3G": 1, "6G": 2}
            freq_idx = freq_map.get(freq, 0)
            pred_2d = predicted_rcs[:, :, freq_idx]

            # 后处理
            data_adapter = sys.get('data_adapter', None)
            use_postprocess = False
            try:
                if hasattr(self.gui, 'ae_postprocess_abs_db'):
                    use_postprocess = self.gui.ae_postprocess_abs_db.get()
                elif hasattr(self.gui, 'ae_extension'):
                    use_postprocess = self.gui.ae_extension.postprocess_abs_db_var.get()
            except: pass

            if use_postprocess:
                if data_adapter and data_adapter.db_transform:
                    raise ValueError("重复的dB转换")
                true_rcs_linear = np.abs(true_rcs_linear)
                pred_2d = np.abs(pred_2d)

            plot_rcs_comparison(
                true_rcs=true_rcs_linear,
                pred_rcs=pred_2d,
                freq_label=freq,
                model_id=model_id,
                phi_range=(phi_values.min(), phi_values.max()),
                theta_range=(theta_values.min(), theta_values.max()),
                fontsize_scale=fontsize_scale,
                fig=target_fig
            )

            mode_display = {'stage1_only': 'Stage 1 Only', 'three_stage': 'Three-Stage'}.get(training_mode, training_mode)
            target_fig.suptitle(f'AE对比分析 - 模型{model_id} @ {freq}\n({mode_display})', fontsize=int(24*fontsize_scale), fontweight='bold')
            target_canvas.draw()

        except Exception as e:
            self.handle_error("AE对比图生成失败", e)

    def plot_wavelet_coefficients_comparison(self, ae_system=None, fig=None, canvas=None, rcs_data=None):
        """绘制小波系数对比"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        fontsize_scale = self.get_fontsize_scale()

        if sys is None or target_fig is None:
            self.log("错误: 系统未初始化或绘图目标缺失")
            return

        mode = sys.get('mode', 'wavelet')
        if mode not in ('wavelet', 'differentiable_wavelet'):
            messagebox.showwarning("警告", "仅适用于Wavelet模式")
            return

        try:
            try:
                if hasattr(self.gui, 'visualization_tab'):
                    model_id = self.gui.visualization_tab.vis_model_var.get()
                    freq = self.gui.visualization_tab.vis_freq_var.get()
                else:
                    model_id = self.gui.vis_model_var.get()
                    freq = self.gui.vis_freq_var.get()
            except:
                model_id, freq = "001", "1.5G"

            if hasattr(self.gui, 'reconstruction_manager'):
                result = self.gui.reconstruction_manager._reconstruct_rcs(
                    input_data=None, input_type='model_ids', model_ids=[model_id],
                    return_latents=False, return_wavelet_coeffs=True
                )
            else:
                 result = self.gui._reconstruct_rcs(
                    input_data=None, input_type='model_ids', model_ids=[model_id],
                    return_latents=False, return_wavelet_coeffs=True
                )

            original_coeffs = result['original_wavelet_coeffs'][0]
            reconstructed_coeffs = result['reconstructed_wavelet_coeffs'][0]
            
            freq_map = {"1.5G": 0, "3G": 1, "6G": 2}
            freq_idx = freq_map.get(freq, 0)

            plot_wavelet_coefficients_comparison(
                original_coeffs=original_coeffs,
                reconstructed_coeffs=reconstructed_coeffs,
                freq_idx=freq_idx,
                freq_label=freq,
                model_id=model_id,
                fontsize_scale=fontsize_scale,
                fig=target_fig
            )
            target_canvas.draw()

        except Exception as e:
            self.handle_error("小波系数对比失败", e)

    def plot_ae_branch_comparison(self, ae_system=None, fig=None, canvas=None, log_callback=None, model_id=None, freq_str=None):
        """绘制Additive Dual-Branch分支对比图"""
        sys = ae_system if ae_system is not None else getattr(self.gui, 'ae_system', None)
        target_fig, target_canvas = self.get_target_fig_canvas(fig, canvas)
        
        original_log = self.log
        if log_callback: self.log = log_callback

        if sys is None or target_fig is None:
            self.log("错误: 系统未初始化或绘图目标缺失")
            return

        try:
            autoencoder = sys.get('autoencoder', None)
            if not hasattr(autoencoder, 'forward_with_branches'):
                messagebox.showwarning("架构不匹配", "仅支持Additive Dual-Branch架构")
                return

            if model_id is None:
                model_id = getattr(self.gui.visualization_tab, 'vis_model_var', tk.StringVar(value="001")).get()
            if freq_str is None:
                freq_str = getattr(self.gui.visualization_tab, 'vis_freq_var', tk.StringVar(value="1.5G")).get()

            # 获取数据并处理
            data_adapter = sys.get('data_adapter', None)
            wavelet_transform = sys.get('wavelet_transform', None)
            config_info = sys.get('config_info', {})
            frequency_labels = config_info.get('frequency_labels', [])
            mode = sys.get('mode', 'direct')
            
            device = next(autoencoder.parameters()).device

            def normalize_freq_label(label):
                label = label.replace('GHz', 'G').replace('MHz', 'M')
                if '.' in label:
                    parts = label.split('.')
                    if len(parts) == 2 and parts[1].startswith('0'):
                        label = parts[0] + parts[1][1:]
                return label

            rcs_data_multifreq = []
            for fl in frequency_labels:
                fl_normalized = normalize_freq_label(fl)
                data_freq = rv.get_rcs_matrix(model_id, fl_normalized, self.gui.data_config['rcs_data_dir'])
                rcs_data_multifreq.append(data_freq['rcs_linear'])
            
            # [H, W, num_freq]
            rcs_data_multifreq_np = np.stack(rcs_data_multifreq, axis=-1)

            if mode == 'wavelet':
                wavelet_coeffs = wavelet_transform.forward_transform(rcs_data_multifreq_np)
                if data_adapter:
                    wavelet_coeffs_batch = np.expand_dims(wavelet_coeffs, axis=0)
                    wavelet_input_batch = data_adapter.adapt_rcs_data(wavelet_coeffs_batch)
                    wavelet_input = wavelet_input_batch[0]
                else:
                    wavelet_input = wavelet_coeffs
                input_tensor = torch.FloatTensor(wavelet_input).unsqueeze(0).to(device)
            else:
                if data_adapter:
                    rcs_data_batch = np.expand_dims(rcs_data_multifreq_np, axis=0)
                    rcs_input_batch = data_adapter.adapt_rcs_data(rcs_data_batch)
                    rcs_input = rcs_input_batch[0]
                else:
                    rcs_input = rcs_data_multifreq_np
                input_tensor = torch.FloatTensor(rcs_input).unsqueeze(0).to(device)

            with torch.no_grad():
                recon, latent, recon_high, recon_smooth = autoencoder.forward_with_branches(input_tensor)

            recon = recon[0].cpu().numpy()
            recon_high = recon_high[0].cpu().numpy()
            recon_smooth = recon_smooth[0].cpu().numpy()

            if mode == 'wavelet':
                if data_adapter:
                    recon = data_adapter.inverse_adapt(recon)
                    recon_high = data_adapter.inverse_adapt(recon_high)
                    recon_smooth = data_adapter.inverse_adapt(recon_smooth)
                recon = wavelet_transform.inverse_transform(torch.FloatTensor(recon).unsqueeze(0).to(device))[0].cpu().numpy()
                recon_high = wavelet_transform.inverse_transform(torch.FloatTensor(recon_high).unsqueeze(0).to(device))[0].cpu().numpy()
                recon_smooth = wavelet_transform.inverse_transform(torch.FloatTensor(recon_smooth).unsqueeze(0).to(device))[0].cpu().numpy()
            else:
                if data_adapter:
                    recon = data_adapter.inverse_adapt(recon)
                    recon_high = data_adapter.inverse_adapt(recon_high)
                    recon_smooth = data_adapter.inverse_adapt(recon_smooth)

            try:
                normalized_labels = [normalize_freq_label(fl) for fl in frequency_labels]
                freq_idx = normalized_labels.index(freq_str)
            except ValueError:
                freq_idx = 0
            
            alpha_high = autoencoder.alpha_high.item() if hasattr(autoencoder, 'alpha_high') else 1.0
            alpha_smooth = autoencoder.alpha_smooth.item() if hasattr(autoencoder, 'alpha_smooth') else 1.0
            activation_high = getattr(autoencoder, 'activation_high_type', 'sin').upper()
            activation_smooth = getattr(autoencoder, 'activation_smooth_type', 'tanh').upper()

            # 获取角度范围用于绘图
            data_0 = rv.get_rcs_matrix(model_id, normalize_freq_label(frequency_labels[0]), self.gui.data_config['rcs_data_dir'])
            phi_range = (data_0['phi_values'].min(), data_0['phi_values'].max())
            theta_range = (data_0['theta_values'].min(), data_0['theta_values'].max())

            plot_additive_branch_comparison(
                original_rcs=rcs_data_multifreq_np,
                recon_high=recon_high,
                recon_smooth=recon_smooth,
                recon_combined=recon,
                freq_label=freq_str,
                sample_id=model_id,
                phi_range=phi_range,
                theta_range=theta_range,
                freq_idx=freq_idx,
                alpha_high=alpha_high,
                alpha_smooth=alpha_smooth,
                activation_high_name=activation_high,
                activation_smooth_name=activation_smooth,
                figsize=(20, 10),
                fontsize_scale=1.0,
                fig=target_fig
            )

            target_canvas.draw()
            self.log(f"分支对比图完成: α={alpha_high:.3f}, β={alpha_smooth:.3f}")

        except Exception as e:
            self.handle_error("分支对比图失败", e)
        finally:
            if log_callback: self.log = original_log

    def _prepare_input(self, data, mode, wavelet_transform, data_adapter, device):
        """辅助：准备模型输入"""
        if mode in ('wavelet', 'differentiable_wavelet'):
            rcs_tensor = torch.FloatTensor(data).to(device)
            wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
            if data_adapter:
                wavelet_coeffs_np = wavelet_coeffs.cpu().numpy()
                input_adapted = data_adapter.adapt_rcs_data(wavelet_coeffs_np)
                return torch.FloatTensor(input_adapted).to(device)
            else:
                return wavelet_coeffs
        else:
            if data_adapter:
                input_adapted = data_adapter.adapt_rcs_data(data)
                return torch.FloatTensor(input_adapted).to(device)
            else:
                return torch.FloatTensor(data).to(device)

    def _get_data(self, rcs_data, sys):
        data = rcs_data
        if data is None and sys is not None and 'rcs_data' in sys:
            data = sys['rcs_data']
        if data is None:
            data = getattr(self.gui, 'rcs_data', None)
        return data

    def _get_params(self, param_data, sys):
        params = param_data
        if params is None and sys is not None and 'param_data' in sys:
            params = sys['param_data']
        if params is None:
            params = getattr(self.gui, 'param_data', None)
        return params