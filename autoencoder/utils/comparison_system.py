"""
AutoEncoder对比分析系统
支持小波增强模式 vs 直接模式的全面对比
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# 导入两种AutoEncoder
from ..models.cnn_autoencoder import WaveletAutoEncoder
from ..models.direct_autoencoder import DirectAutoEncoder
from ..models.cnn_autoencoder import ParameterMapper
from .wavelet_transform import WaveletTransform


class AutoEncoderComparator:
    """AutoEncoder对比分析器"""

    def __init__(self,
                 wavelet_system: Dict[str, Any],
                 direct_system: Dict[str, Any]):
        """
        初始化对比分析器

        Args:
            wavelet_system: 小波增强AutoEncoder系统
            direct_system: 直接AutoEncoder系统
        """
        self.wavelet_system = wavelet_system
        self.direct_system = direct_system

        # 设备设置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 结果存储
        self.comparison_results = {}

    def compare_performance(self, test_data: Dict[str, np.ndarray],
                          batch_size: int = 10) -> Dict[str, Any]:
        """
        性能对比分析

        Args:
            test_data: 测试数据 {'rcs_data': [...], 'param_data': [...]}
            batch_size: 批次大小

        Returns:
            performance_results: 性能对比结果
        """
        print("🔬 开始性能对比分析...")

        results = {
            'wavelet_mode': {},
            'direct_mode': {},
            'comparison': {}
        }

        rcs_data = test_data['rcs_data']
        param_data = test_data['param_data']

        # 测试小波增强模式
        print("  📊 测试小波增强模式...")
        wavelet_results = self._evaluate_system(
            self.wavelet_system, rcs_data, param_data,
            batch_size, mode='wavelet'
        )
        results['wavelet_mode'] = wavelet_results

        # 测试直接模式
        print("  📊 测试直接模式...")
        direct_results = self._evaluate_system(
            self.direct_system, rcs_data, param_data,
            batch_size, mode='direct'
        )
        results['direct_mode'] = direct_results

        # 对比分析
        print("  📊 计算对比指标...")
        results['comparison'] = self._calculate_comparison_metrics(
            wavelet_results, direct_results
        )

        self.comparison_results['performance'] = results
        return results

    def _evaluate_system(self, system: Dict[str, Any],
                        rcs_data: np.ndarray, param_data: np.ndarray,
                        batch_size: int, mode: str) -> Dict[str, Any]:
        """评估单个系统"""
        autoencoder = system['autoencoder']
        parameter_mapper = system['parameter_mapper']
        wavelet_transform = system.get('wavelet_transform', None)

        # 移动到设备
        autoencoder.to(self.device).eval()
        parameter_mapper.to(self.device).eval()

        # 测试数据准备
        test_size = min(len(rcs_data), 50)  # 限制测试样本数
        test_rcs = rcs_data[:test_size]
        test_params = param_data[:test_size]

        # 性能指标
        reconstruction_errors = []
        prediction_errors = []
        latent_vectors = []
        inference_times = []

        # 批次测试
        with torch.no_grad():
            for i in range(0, test_size, batch_size):
                end_idx = min(i + batch_size, test_size)
                batch_rcs = torch.FloatTensor(test_rcs[i:end_idx]).to(self.device)
                batch_params = torch.FloatTensor(test_params[i:end_idx]).to(self.device)

                # 测试重建能力
                start_time = time.time()

                if mode == 'wavelet':
                    # 小波模式：RCS → 小波系数 → AE → 小波系数 → RCS
                    wavelet_coeffs = wavelet_transform.forward_transform(batch_rcs)
                    reconstructed_coeffs, latent = autoencoder(wavelet_coeffs)
                    reconstructed_rcs = wavelet_transform.inverse_transform(reconstructed_coeffs)
                else:
                    # 直接模式：RCS → AE → RCS
                    reconstructed_rcs, latent = autoencoder(batch_rcs)

                reconstruction_time = time.time() - start_time

                # 测试端到端预测能力
                start_time = time.time()

                if mode == 'wavelet':
                    # 参数 → 隐空间 → 小波系数 → RCS
                    predicted_latent = parameter_mapper(batch_params)
                    predicted_coeffs = autoencoder.decode(predicted_latent)
                    predicted_rcs = wavelet_transform.inverse_transform(predicted_coeffs)
                else:
                    # 参数 → 隐空间 → RCS
                    predicted_latent = parameter_mapper(batch_params)
                    predicted_rcs = autoencoder.decode(predicted_latent)

                prediction_time = time.time() - start_time

                # 计算误差
                recon_error = torch.mean((batch_rcs - reconstructed_rcs) ** 2).item()
                pred_error = torch.mean((batch_rcs - predicted_rcs) ** 2).item()

                reconstruction_errors.append(recon_error)
                prediction_errors.append(pred_error)
                latent_vectors.append(latent.cpu().numpy())
                inference_times.append(reconstruction_time + prediction_time)

        # 合并结果
        all_latents = np.concatenate(latent_vectors, axis=0)

        return {
            'reconstruction_mse': np.mean(reconstruction_errors),
            'reconstruction_std': np.std(reconstruction_errors),
            'prediction_mse': np.mean(prediction_errors),
            'prediction_std': np.std(prediction_errors),
            'latent_vectors': all_latents,
            'inference_time': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'mode': mode
        }

    def _calculate_comparison_metrics(self, wavelet_results: Dict[str, Any],
                                    direct_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算对比指标"""
        comparison = {}

        # 重建精度对比
        comparison['reconstruction_improvement'] = {
            'mse_ratio': wavelet_results['reconstruction_mse'] / direct_results['reconstruction_mse'],
            'better_mode': 'wavelet' if wavelet_results['reconstruction_mse'] < direct_results['reconstruction_mse'] else 'direct',
            'improvement_percent': abs(wavelet_results['reconstruction_mse'] - direct_results['reconstruction_mse']) /
                                 max(wavelet_results['reconstruction_mse'], direct_results['reconstruction_mse']) * 100
        }

        # 预测精度对比
        comparison['prediction_improvement'] = {
            'mse_ratio': wavelet_results['prediction_mse'] / direct_results['prediction_mse'],
            'better_mode': 'wavelet' if wavelet_results['prediction_mse'] < direct_results['prediction_mse'] else 'direct',
            'improvement_percent': abs(wavelet_results['prediction_mse'] - direct_results['prediction_mse']) /
                                 max(wavelet_results['prediction_mse'], direct_results['prediction_mse']) * 100
        }

        # 推理速度对比
        comparison['speed_comparison'] = {
            'time_ratio': wavelet_results['inference_time'] / direct_results['inference_time'],
            'faster_mode': 'wavelet' if wavelet_results['inference_time'] < direct_results['inference_time'] else 'direct',
            'speedup_percent': abs(wavelet_results['inference_time'] - direct_results['inference_time']) /
                             max(wavelet_results['inference_time'], direct_results['inference_time']) * 100
        }

        return comparison

    def compare_feature_learning(self, test_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        特征学习能力对比

        Args:
            test_data: 测试数据

        Returns:
            feature_comparison: 特征学习对比结果
        """
        print("🧠 开始特征学习对比分析...")

        # 获取隐空间表示
        wavelet_latents = self.comparison_results['performance']['wavelet_mode']['latent_vectors']
        direct_latents = self.comparison_results['performance']['direct_mode']['latent_vectors']

        results = {}

        # PCA分析
        print("  📊 PCA降维分析...")
        results['pca_analysis'] = self._compare_pca(wavelet_latents, direct_latents)

        # t-SNE分析
        print("  📊 t-SNE聚类分析...")
        results['tsne_analysis'] = self._compare_tsne(wavelet_latents, direct_latents)

        # 特征分布分析
        print("  📊 特征分布分析...")
        results['distribution_analysis'] = self._compare_distributions(wavelet_latents, direct_latents)

        self.comparison_results['feature_learning'] = results
        return results

    def _compare_pca(self, wavelet_latents: np.ndarray,
                    direct_latents: np.ndarray) -> Dict[str, Any]:
        """PCA对比分析"""
        # 小波模式PCA
        pca_wavelet = PCA(n_components=min(10, wavelet_latents.shape[1]))
        wavelet_pca = pca_wavelet.fit_transform(wavelet_latents)

        # 直接模式PCA
        pca_direct = PCA(n_components=min(10, direct_latents.shape[1]))
        direct_pca = pca_direct.fit_transform(direct_latents)

        return {
            'wavelet_explained_variance': pca_wavelet.explained_variance_ratio_,
            'direct_explained_variance': pca_direct.explained_variance_ratio_,
            'wavelet_cumulative_variance': np.cumsum(pca_wavelet.explained_variance_ratio_),
            'direct_cumulative_variance': np.cumsum(pca_direct.explained_variance_ratio_),
            'wavelet_pca_coords': wavelet_pca[:, :2],
            'direct_pca_coords': direct_pca[:, :2]
        }

    def _compare_tsne(self, wavelet_latents: np.ndarray,
                     direct_latents: np.ndarray) -> Dict[str, Any]:
        """t-SNE对比分析"""
        # 限制样本数量避免计算过慢
        max_samples = 50

        if len(wavelet_latents) > max_samples:
            indices = np.random.choice(len(wavelet_latents), max_samples, replace=False)
            wavelet_subset = wavelet_latents[indices]
            direct_subset = direct_latents[indices]
        else:
            wavelet_subset = wavelet_latents
            direct_subset = direct_latents

        # t-SNE降维
        perplexity = min(30, len(wavelet_subset) - 1)

        tsne_wavelet = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        wavelet_tsne = tsne_wavelet.fit_transform(wavelet_subset)

        tsne_direct = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        direct_tsne = tsne_direct.fit_transform(direct_subset)

        return {
            'wavelet_tsne_coords': wavelet_tsne,
            'direct_tsne_coords': direct_tsne
        }

    def _compare_distributions(self, wavelet_latents: np.ndarray,
                             direct_latents: np.ndarray) -> Dict[str, Any]:
        """特征分布对比"""
        return {
            'wavelet_mean': np.mean(wavelet_latents, axis=0),
            'wavelet_std': np.std(wavelet_latents, axis=0),
            'direct_mean': np.mean(direct_latents, axis=0),
            'direct_std': np.std(direct_latents, axis=0),
            'mean_difference': np.mean(np.abs(np.mean(wavelet_latents, axis=0) - np.mean(direct_latents, axis=0))),
            'std_difference': np.mean(np.abs(np.std(wavelet_latents, axis=0) - np.std(direct_latents, axis=0)))
        }

    def compare_computational_efficiency(self) -> Dict[str, Any]:
        """
        计算效率对比

        Returns:
            efficiency_comparison: 计算效率对比结果
        """
        print("⚡ 开始计算效率对比分析...")

        results = {}

        # 模型复杂度对比
        results['model_complexity'] = self._compare_model_complexity()

        # 内存使用对比
        results['memory_usage'] = self._compare_memory_usage()

        # 推理时间对比（从之前的结果中获取）
        if 'performance' in self.comparison_results:
            results['inference_time'] = {
                'wavelet_time': self.comparison_results['performance']['wavelet_mode']['inference_time'],
                'direct_time': self.comparison_results['performance']['direct_mode']['inference_time'],
                'speedup_ratio': self.comparison_results['performance']['comparison']['speed_comparison']['time_ratio']
            }

        self.comparison_results['computational_efficiency'] = results
        return results

    def _compare_model_complexity(self) -> Dict[str, Any]:
        """模型复杂度对比"""
        # 小波系统参数
        wavelet_ae_params = sum(p.numel() for p in self.wavelet_system['autoencoder'].parameters())
        wavelet_mapper_params = sum(p.numel() for p in self.wavelet_system['parameter_mapper'].parameters())
        wavelet_total = wavelet_ae_params + wavelet_mapper_params

        # 直接系统参数
        direct_ae_params = sum(p.numel() for p in self.direct_system['autoencoder'].parameters())
        direct_mapper_params = sum(p.numel() for p in self.direct_system['parameter_mapper'].parameters())
        direct_total = direct_ae_params + direct_mapper_params

        return {
            'wavelet_autoencoder_params': wavelet_ae_params,
            'wavelet_mapper_params': wavelet_mapper_params,
            'wavelet_total_params': wavelet_total,
            'direct_autoencoder_params': direct_ae_params,
            'direct_mapper_params': direct_mapper_params,
            'direct_total_params': direct_total,
            'parameter_ratio': wavelet_total / direct_total,
            'complexity_advantage': 'direct' if direct_total < wavelet_total else 'wavelet'
        }

    def _compare_memory_usage(self) -> Dict[str, Any]:
        """内存使用对比（估算）"""
        # 基于模型参数和中间变量估算

        # 小波模式：需要存储小波系数
        wavelet_intermediate = 91 * 91 * 8 * 4  # 小波系数占用（float32）
        wavelet_base = sum(p.numel() * 4 for p in self.wavelet_system['autoencoder'].parameters())
        wavelet_total = wavelet_base + wavelet_intermediate

        # 直接模式：直接处理RCS
        direct_intermediate = 91 * 91 * 2 * 4  # RCS数据占用（float32）
        direct_base = sum(p.numel() * 4 for p in self.direct_system['autoencoder'].parameters())
        direct_total = direct_base + direct_intermediate

        return {
            'wavelet_memory_mb': wavelet_total / (1024 * 1024),
            'direct_memory_mb': direct_total / (1024 * 1024),
            'memory_ratio': wavelet_total / direct_total,
            'memory_advantage': 'direct' if direct_total < wavelet_total else 'wavelet'
        }

    def generate_comparison_report(self) -> str:
        """
        生成详细的对比分析报告

        Returns:
            report: 格式化的对比报告
        """
        if not self.comparison_results:
            return "❌ 请先运行对比分析！"

        report = []
        report.append("=" * 60)
        report.append("🔬 AutoEncoder对比分析报告")
        report.append("=" * 60)

        # 性能对比
        if 'performance' in self.comparison_results:
            perf = self.comparison_results['performance']
            report.append("\n📊 性能对比:")
            report.append(f"  重建精度 (MSE):")
            report.append(f"    小波模式: {perf['wavelet_mode']['reconstruction_mse']:.6f}")
            report.append(f"    直接模式: {perf['direct_mode']['reconstruction_mse']:.6f}")
            report.append(f"    优势: {perf['comparison']['reconstruction_improvement']['better_mode']}")
            report.append(f"    改善: {perf['comparison']['reconstruction_improvement']['improvement_percent']:.1f}%")

            report.append(f"  预测精度 (MSE):")
            report.append(f"    小波模式: {perf['wavelet_mode']['prediction_mse']:.6f}")
            report.append(f"    直接模式: {perf['direct_mode']['prediction_mse']:.6f}")
            report.append(f"    优势: {perf['comparison']['prediction_improvement']['better_mode']}")
            report.append(f"    改善: {perf['comparison']['prediction_improvement']['improvement_percent']:.1f}%")

        # 计算效率对比
        if 'computational_efficiency' in self.comparison_results:
            eff = self.comparison_results['computational_efficiency']
            report.append("\n⚡ 计算效率对比:")
            report.append(f"  模型参数:")
            report.append(f"    小波模式: {eff['model_complexity']['wavelet_total_params']:,}")
            report.append(f"    直接模式: {eff['model_complexity']['direct_total_params']:,}")
            report.append(f"    优势: {eff['model_complexity']['complexity_advantage']}")

            if 'inference_time' in eff:
                report.append(f"  推理时间:")
                report.append(f"    小波模式: {eff['inference_time']['wavelet_time']:.4f}s")
                report.append(f"    直接模式: {eff['inference_time']['direct_time']:.4f}s")
                report.append(f"    加速比: {eff['inference_time']['speedup_ratio']:.2f}x")

        # 总结建议
        report.append("\n🎯 建议:")
        report.append("  高精度要求 → 选择表现更好的模式")
        report.append("  实时应用 → 选择速度更快的模式")
        report.append("  内存受限 → 选择参数更少的模式")

        return "\n".join(report)


def create_comparison_system(wavelet_system: Dict[str, Any],
                           direct_system: Dict[str, Any]) -> AutoEncoderComparator:
    """
    创建AutoEncoder对比分析系统

    Args:
        wavelet_system: 小波增强系统
        direct_system: 直接系统

    Returns:
        comparator: 对比分析器
    """
    return AutoEncoderComparator(wavelet_system, direct_system)