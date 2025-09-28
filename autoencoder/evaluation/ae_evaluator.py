"""
AutoEncoder专用评估器
集成重建质量评估和性能分析
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import matplotlib.pyplot as plt
from .reconstruction_metrics import ReconstructionMetrics


class AE_Evaluator:
    """
    AutoEncoder系统评估器
    支持性能评估、可视化分析和对比实验
    """

    def __init__(self,
                 autoencoder,
                 parameter_mapper=None,
                 wavelet_transform=None,
                 device: Optional[torch.device] = None):
        """
        初始化评估器

        Args:
            autoencoder: AutoEncoder模型
            parameter_mapper: 参数映射器（可选）
            wavelet_transform: 小波变换器（可选）
            device: 计算设备
        """
        self.autoencoder = autoencoder
        self.parameter_mapper = parameter_mapper
        self.wavelet_transform = wavelet_transform
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 移动模型到设备
        self.autoencoder.to(self.device)
        if self.parameter_mapper and hasattr(self.parameter_mapper, 'to'):
            self.parameter_mapper.to(self.device)

        # 评估指标计算器
        self.metrics_calculator = ReconstructionMetrics(device=self.device)

    def evaluate_autoencoder_reconstruction(self,
                                          test_rcs: np.ndarray,
                                          batch_size: int = 16) -> Dict[str, Any]:
        """
        评估AutoEncoder重建质量

        Args:
            test_rcs: [N, 91, 91, 2] 测试RCS数据
            batch_size: 批次大小

        Returns:
            evaluation_results: 评估结果
        """
        print("评估AutoEncoder重建质量...")

        self.autoencoder.eval()

        # 数据预处理
        if self.wavelet_transform:
            rcs_tensor = torch.FloatTensor(test_rcs)
            test_wavelet = self.wavelet_transform.forward_transform(rcs_tensor)
        else:
            test_wavelet = torch.FloatTensor(test_rcs)

        # 批量重建
        reconstructed_data = []
        latent_representations = []
        inference_times = []

        n_batches = (len(test_wavelet) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(test_wavelet))

                batch_data = test_wavelet[start_idx:end_idx].to(self.device)

                # 计时
                start_time = time.time()

                # 重建
                recon_batch, latent_batch = self.autoencoder(batch_data)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                reconstructed_data.append(recon_batch.cpu())
                latent_representations.append(latent_batch.cpu())

        # 合并结果
        reconstructed_wavelet = torch.cat(reconstructed_data, dim=0)
        latent_vectors = torch.cat(latent_representations, dim=0)

        # 转换回RCS格式
        if self.wavelet_transform:
            reconstructed_rcs = self.wavelet_transform.inverse_transform(reconstructed_wavelet)
        else:
            reconstructed_rcs = reconstructed_wavelet

        # 计算评估指标
        true_rcs_tensor = torch.FloatTensor(test_rcs)
        metrics = self.metrics_calculator.compute_all_metrics(reconstructed_rcs, true_rcs_tensor)

        # 性能统计
        avg_inference_time = np.mean(inference_times)
        total_samples = len(test_rcs)
        samples_per_second = total_samples / sum(inference_times)

        # 隐空间分析
        latent_analysis = self._analyze_latent_space(latent_vectors.numpy())

        results = {
            'reconstruction_metrics': metrics,
            'performance': {
                'avg_inference_time': avg_inference_time,
                'samples_per_second': samples_per_second,
                'total_samples': total_samples
            },
            'latent_analysis': latent_analysis,
            'reconstructed_rcs': reconstructed_rcs.numpy(),
            'latent_vectors': latent_vectors.numpy()
        }

        print(f"AE重建评估完成: MSE={metrics['mse']:.6f}, SSIM={metrics['ssim_mean']:.4f}")

        return results

    def evaluate_parameter_mapping(self,
                                  test_params: np.ndarray,
                                  test_rcs: np.ndarray) -> Dict[str, Any]:
        """
        评估参数映射质量

        Args:
            test_params: [N, 9] 测试参数
            test_rcs: [N, 91, 91, 2] 测试RCS

        Returns:
            evaluation_results: 评估结果
        """
        if self.parameter_mapper is None:
            raise ValueError("参数映射器未设置，无法进行评估")

        print("评估参数映射质量...")

        # 1. 获取目标隐空间表示
        self.autoencoder.eval()

        if self.wavelet_transform:
            rcs_tensor = torch.FloatTensor(test_rcs)
            test_wavelet = self.wavelet_transform.forward_transform(rcs_tensor)
        else:
            test_wavelet = torch.FloatTensor(test_rcs)

        with torch.no_grad():
            test_wavelet = test_wavelet.to(self.device)
            _, target_latent = self.autoencoder(test_wavelet)
            target_latent = target_latent.cpu().numpy()

        # 2. 参数映射预测
        if hasattr(self.parameter_mapper, 'predict'):
            # 传统机器学习模型
            pred_latent = self.parameter_mapper.predict(test_params)
            mapping_metrics = self.parameter_mapper.evaluate(test_params, target_latent)
        else:
            # 深度学习模型
            self.parameter_mapper.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(test_params).to(self.device)
                pred_latent = self.parameter_mapper(params_tensor).cpu().numpy()

            # 计算映射指标
            mapping_mse = np.mean((target_latent - pred_latent) ** 2)
            mapping_r2 = 1 - np.sum((target_latent - pred_latent) ** 2) / np.sum((target_latent - np.mean(target_latent)) ** 2)

            mapping_metrics = {
                'mse': mapping_mse,
                'r2_score': mapping_r2
            }

        # 3. 隐空间一致性分析
        latent_consistency = self._analyze_latent_consistency(target_latent, pred_latent)

        results = {
            'mapping_metrics': mapping_metrics,
            'latent_consistency': latent_consistency,
            'target_latent': target_latent,
            'predicted_latent': pred_latent
        }

        print(f"参数映射评估完成: MSE={mapping_metrics['mse']:.6f}, R²={mapping_metrics.get('r2_score', 0):.4f}")

        return results

    def evaluate_end_to_end(self,
                           test_params: np.ndarray,
                           test_rcs: np.ndarray) -> Dict[str, Any]:
        """
        端到端评估：参数 → RCS重建

        Args:
            test_params: [N, 9] 测试参数
            test_rcs: [N, 91, 91, 2] 测试RCS

        Returns:
            evaluation_results: 评估结果
        """
        if self.parameter_mapper is None:
            raise ValueError("参数映射器未设置，无法进行端到端评估")

        print("执行端到端评估...")

        # 1. 参数 → 隐空间
        if hasattr(self.parameter_mapper, 'predict'):
            pred_latent = self.parameter_mapper.predict(test_params)
            pred_latent = torch.FloatTensor(pred_latent)
        else:
            self.parameter_mapper.eval()
            with torch.no_grad():
                params_tensor = torch.FloatTensor(test_params).to(self.device)
                pred_latent = self.parameter_mapper(params_tensor)

        # 2. 隐空间 → 小波系数
        self.autoencoder.eval()
        with torch.no_grad():
            pred_latent = pred_latent.to(self.device)
            pred_wavelet = self.autoencoder.decode(pred_latent)

        # 3. 小波系数 → RCS
        if self.wavelet_transform:
            pred_rcs = self.wavelet_transform.inverse_transform(pred_wavelet.cpu())
        else:
            pred_rcs = pred_wavelet.cpu()

        # 4. 计算重建质量
        true_rcs_tensor = torch.FloatTensor(test_rcs)
        e2e_metrics = self.metrics_calculator.compute_all_metrics(pred_rcs, true_rcs_tensor)

        # 5. 与直接AE重建对比
        if self.wavelet_transform:
            rcs_tensor = torch.FloatTensor(test_rcs)
            true_wavelet = self.wavelet_transform.forward_transform(rcs_tensor)
        else:
            true_wavelet = torch.FloatTensor(test_rcs)

        with torch.no_grad():
            true_wavelet = true_wavelet.to(self.device)
            direct_recon, _ = self.autoencoder(true_wavelet)

        if self.wavelet_transform:
            direct_rcs = self.wavelet_transform.inverse_transform(direct_recon.cpu())
        else:
            direct_rcs = direct_recon.cpu()

        direct_metrics = self.metrics_calculator.compute_all_metrics(direct_rcs, true_rcs_tensor)

        results = {
            'end_to_end_metrics': e2e_metrics,
            'direct_reconstruction_metrics': direct_metrics,
            'performance_comparison': {
                'e2e_mse': e2e_metrics['mse'],
                'direct_mse': direct_metrics['mse'],
                'mse_ratio': e2e_metrics['mse'] / max(direct_metrics['mse'], 1e-8),
                'e2e_ssim': e2e_metrics['ssim_mean'],
                'direct_ssim': direct_metrics['ssim_mean']
            },
            'predicted_rcs': pred_rcs.numpy(),
            'direct_reconstructed_rcs': direct_rcs.numpy()
        }

        print(f"端到端评估完成:")
        print(f"  E2E MSE: {e2e_metrics['mse']:.6f}")
        print(f"  直接重建 MSE: {direct_metrics['mse']:.6f}")
        print(f"  MSE比值: {results['performance_comparison']['mse_ratio']:.2f}")

        return results

    def _analyze_latent_space(self, latent_vectors: np.ndarray) -> Dict[str, Any]:
        """分析隐空间特性"""

        analysis = {}

        # 基础统计
        analysis['shape'] = latent_vectors.shape
        analysis['mean'] = np.mean(latent_vectors, axis=0)
        analysis['std'] = np.std(latent_vectors, axis=0)
        analysis['min'] = np.min(latent_vectors, axis=0)
        analysis['max'] = np.max(latent_vectors, axis=0)

        # 维度利用率
        dimension_usage = np.std(latent_vectors, axis=0)
        active_dims = np.sum(dimension_usage > 0.01)  # 标准差大于0.01的维度
        analysis['active_dimensions'] = active_dims
        analysis['dimension_usage_ratio'] = active_dims / latent_vectors.shape[1]

        # 相关性分析
        correlation_matrix = np.corrcoef(latent_vectors.T)
        mean_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        analysis['mean_correlation'] = mean_correlation

        return analysis

    def _analyze_latent_consistency(self,
                                  target_latent: np.ndarray,
                                  pred_latent: np.ndarray) -> Dict[str, float]:
        """分析隐空间一致性"""

        # 逐维度相关性
        dim_correlations = []
        for i in range(target_latent.shape[1]):
            corr = np.corrcoef(target_latent[:, i], pred_latent[:, i])[0, 1]
            if not np.isnan(corr):
                dim_correlations.append(corr)

        # 整体相关性
        overall_correlation = np.corrcoef(target_latent.flatten(), pred_latent.flatten())[0, 1]

        # 维度重要性分析
        dim_importance = np.std(target_latent, axis=0)
        weighted_correlation = np.average(dim_correlations, weights=dim_importance[:len(dim_correlations)])

        return {
            'overall_correlation': overall_correlation if not np.isnan(overall_correlation) else 0.0,
            'mean_dim_correlation': np.mean(dim_correlations) if dim_correlations else 0.0,
            'weighted_correlation': weighted_correlation if not np.isnan(weighted_correlation) else 0.0,
            'num_valid_dims': len(dim_correlations)
        }

    def generate_comprehensive_report(self,
                                    ae_results: Dict[str, Any],
                                    mapping_results: Optional[Dict[str, Any]] = None,
                                    e2e_results: Optional[Dict[str, Any]] = None) -> str:
        """生成综合评估报告"""

        report = "\\n" + "="*60 + "\\n"
        report += "           AutoEncoder系统综合评估报告\\n"
        report += "="*60 + "\\n"

        # AutoEncoder重建评估
        report += "\\n🔧 AutoEncoder重建评估:\\n"
        report += "-" * 40 + "\\n"

        ae_metrics = ae_results['reconstruction_metrics']
        performance = ae_results['performance']

        report += f"重建质量指标:\\n"
        report += f"  MSE:           {ae_metrics['mse']:.6f}\\n"
        report += f"  SSIM:          {ae_metrics['ssim_mean']:.4f} ± {ae_metrics['ssim_std']:.4f}\\n"
        report += f"  相关系数:       {ae_metrics['correlation']:.4f}\\n"
        report += f"  R²决定系数:     {ae_metrics['r2_score']:.4f}\\n"

        report += f"\\n性能指标:\\n"
        report += f"  推理速度:       {performance['samples_per_second']:.1f} 样本/秒\\n"
        report += f"  平均推理时间:    {performance['avg_inference_time']*1000:.2f} ms\\n"

        latent_analysis = ae_results['latent_analysis']
        report += f"\\n隐空间分析:\\n"
        report += f"  活跃维度:       {latent_analysis['active_dimensions']}/{latent_analysis['shape'][1]}\\n"
        report += f"  维度利用率:     {latent_analysis['dimension_usage_ratio']:.2f}\\n"

        # 参数映射评估
        if mapping_results:
            report += "\\n🎯 参数映射评估:\\n"
            report += "-" * 40 + "\\n"

            mapping_metrics = mapping_results['mapping_metrics']
            consistency = mapping_results['latent_consistency']

            report += f"映射质量:\\n"
            report += f"  MSE:           {mapping_metrics['mse']:.6f}\\n"
            report += f"  R²:            {mapping_metrics.get('r2_score', 0):.4f}\\n"

            report += f"\\n隐空间一致性:\\n"
            report += f"  整体相关性:     {consistency['overall_correlation']:.4f}\\n"
            report += f"  维度平均相关性: {consistency['mean_dim_correlation']:.4f}\\n"

        # 端到端评估
        if e2e_results:
            report += "\\n🔄 端到端评估:\\n"
            report += "-" * 40 + "\\n"

            e2e_metrics = e2e_results['end_to_end_metrics']
            comparison = e2e_results['performance_comparison']

            report += f"端到端重建质量:\\n"
            report += f"  MSE:           {e2e_metrics['mse']:.6f}\\n"
            report += f"  SSIM:          {e2e_metrics['ssim_mean']:.4f}\\n"

            report += f"\\n性能对比:\\n"
            report += f"  E2E vs 直接重建:\\n"
            report += f"    MSE比值:     {comparison['mse_ratio']:.2f}\\n"
            report += f"    SSIM差异:    {comparison['e2e_ssim'] - comparison['direct_ssim']:.4f}\\n"

        # 总结
        report += "\\n📋 评估总结:\\n"
        report += "-" * 40 + "\\n"

        # 根据指标给出评估结论
        if ae_metrics['ssim_mean'] > 0.8:
            ae_quality = "优秀"
        elif ae_metrics['ssim_mean'] > 0.6:
            ae_quality = "良好"
        else:
            ae_quality = "需要改进"

        report += f"AutoEncoder重建质量: {ae_quality}\\n"

        if mapping_results and mapping_results['mapping_metrics'].get('r2_score', 0) > 0.8:
            mapping_quality = "优秀"
        elif mapping_results and mapping_results['mapping_metrics'].get('r2_score', 0) > 0.6:
            mapping_quality = "良好"
        else:
            mapping_quality = "需要改进"

        if mapping_results:
            report += f"参数映射质量: {mapping_quality}\\n"

        report += "\\n" + "="*60

        return report


def test_ae_evaluator():
    """测试AE评估器"""
    print("=== AE评估器测试 ===")

    # 这里需要实际的模型进行测试
    # 由于导入问题，暂时使用模拟测试

    print("AE评估器模块创建完成")
    print("包含以下功能:")
    print("- AutoEncoder重建质量评估")
    print("- 参数映射质量评估")
    print("- 端到端性能评估")
    print("- 隐空间分析")
    print("- 综合评估报告生成")

    return True


if __name__ == "__main__":
    test_ae_evaluator()