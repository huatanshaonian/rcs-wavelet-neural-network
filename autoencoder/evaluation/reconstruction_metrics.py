"""
重建质量评估指标
包含多种评估RCS重建质量的指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from skimage.metrics import structural_similarity as ssim
import warnings


class ReconstructionMetrics:
    """
    重建质量评估指标计算器
    """

    def __init__(self, device: torch.device = None):
        """
        初始化评估器

        Args:
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_all_metrics(self,
                           pred_rcs: torch.Tensor,
                           true_rcs: torch.Tensor) -> Dict[str, float]:
        """
        计算所有重建质量指标

        Args:
            pred_rcs: [B, 91, 91, 2] 预测RCS
            true_rcs: [B, 91, 91, 2] 真实RCS

        Returns:
            metrics: 所有评估指标
        """
        metrics = {}

        # 基础误差指标
        metrics.update(self.compute_basic_errors(pred_rcs, true_rcs))

        # 结构相似性指标
        metrics.update(self.compute_ssim_metrics(pred_rcs, true_rcs))

        # 频域一致性指标
        metrics.update(self.compute_frequency_metrics(pred_rcs, true_rcs))

        # 物理约束指标
        metrics.update(self.compute_physics_metrics(pred_rcs, true_rcs))

        # 统计指标
        metrics.update(self.compute_statistical_metrics(pred_rcs, true_rcs))

        return metrics

    def compute_basic_errors(self,
                           pred_rcs: torch.Tensor,
                           true_rcs: torch.Tensor) -> Dict[str, float]:
        """计算基础误差指标"""

        # 确保在同一设备
        pred_rcs = pred_rcs.to(self.device)
        true_rcs = true_rcs.to(self.device)

        # MSE (均方误差)
        mse = F.mse_loss(pred_rcs, true_rcs).item()

        # MAE (平均绝对误差)
        mae = F.l1_loss(pred_rcs, true_rcs).item()

        # RMSE (均方根误差)
        rmse = torch.sqrt(F.mse_loss(pred_rcs, true_rcs)).item()

        # 相对误差
        true_rcs_safe = torch.where(torch.abs(true_rcs) < 1e-8,
                                   torch.sign(true_rcs) * 1e-8, true_rcs)
        relative_error = torch.mean(torch.abs((pred_rcs - true_rcs) / true_rcs_safe)).item()

        # 最大误差
        max_error = torch.max(torch.abs(pred_rcs - true_rcs)).item()

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'relative_error': relative_error,
            'max_error': max_error
        }

    def compute_ssim_metrics(self,
                           pred_rcs: torch.Tensor,
                           true_rcs: torch.Tensor) -> Dict[str, float]:
        """计算结构相似性指标"""

        pred_np = pred_rcs.detach().cpu().numpy()
        true_np = true_rcs.detach().cpu().numpy()

        batch_size = pred_np.shape[0]
        ssim_scores = []

        # 对每个样本和频率计算SSIM
        for b in range(batch_size):
            for freq in range(2):  # 两个频率
                pred_freq = pred_np[b, :, :, freq]
                true_freq = true_np[b, :, :, freq]

                # 标准化到[0,1]范围
                pred_norm = self._normalize_for_ssim(pred_freq)
                true_norm = self._normalize_for_ssim(true_freq)

                try:
                    ssim_score = ssim(true_norm, pred_norm, data_range=1.0)
                    ssim_scores.append(ssim_score)
                except Exception:
                    # 如果SSIM计算失败，使用默认值
                    ssim_scores.append(0.0)

        avg_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        min_ssim = np.min(ssim_scores)

        return {
            'ssim_mean': avg_ssim,
            'ssim_std': std_ssim,
            'ssim_min': min_ssim
        }

    def _normalize_for_ssim(self, data: np.ndarray) -> np.ndarray:
        """为SSIM计算标准化数据"""
        data_min = np.min(data)
        data_max = np.max(data)

        if data_max - data_min < 1e-8:
            return np.zeros_like(data)

        return (data - data_min) / (data_max - data_min)

    def compute_frequency_metrics(self,
                                pred_rcs: torch.Tensor,
                                true_rcs: torch.Tensor) -> Dict[str, float]:
        """计算频域一致性指标"""

        pred_rcs = pred_rcs.to(self.device)
        true_rcs = true_rcs.to(self.device)

        # FFT分析
        pred_fft = torch.fft.fft2(pred_rcs, dim=[1, 2])
        true_fft = torch.fft.fft2(true_rcs, dim=[1, 2])

        # 幅度谱误差
        pred_magnitude = torch.abs(pred_fft)
        true_magnitude = torch.abs(true_fft)
        magnitude_error = F.mse_loss(pred_magnitude, true_magnitude).item()

        # 相位谱误差
        pred_phase = torch.angle(pred_fft)
        true_phase = torch.angle(true_fft)
        phase_error = F.mse_loss(pred_phase, true_phase).item()

        # 功率谱密度误差
        pred_power = pred_magnitude ** 2
        true_power = true_magnitude ** 2
        power_error = F.mse_loss(pred_power, true_power).item()

        # 频率间一致性
        freq_consistency = self._compute_frequency_consistency(pred_rcs, true_rcs)

        return {
            'freq_magnitude_error': magnitude_error,
            'freq_phase_error': phase_error,
            'freq_power_error': power_error,
            'freq_consistency_error': freq_consistency
        }

    def _compute_frequency_consistency(self,
                                     pred_rcs: torch.Tensor,
                                     true_rcs: torch.Tensor) -> float:
        """计算频率间一致性"""

        # 1.5GHz和3GHz的一致性分析
        pred_1_5g = pred_rcs[:, :, :, 0]  # [B, 91, 91]
        pred_3g = pred_rcs[:, :, :, 1]
        true_1_5g = true_rcs[:, :, :, 0]
        true_3g = true_rcs[:, :, :, 1]

        # 频率差异的一致性
        pred_diff = pred_3g - pred_1_5g
        true_diff = true_3g - true_1_5g

        diff_error = F.mse_loss(pred_diff, true_diff).item()

        return diff_error

    def compute_physics_metrics(self,
                              pred_rcs: torch.Tensor,
                              true_rcs: torch.Tensor) -> Dict[str, float]:
        """计算物理约束指标"""

        pred_rcs = pred_rcs.to(self.device)
        true_rcs = true_rcs.to(self.device)

        # 对称性误差 (φ=0°平面对称性)
        symmetry_error = self._compute_symmetry_error(pred_rcs, true_rcs)

        # 连续性误差 (空间梯度连续性)
        continuity_error = self._compute_continuity_error(pred_rcs, true_rcs)

        # 非负性检查 (RCS通常非负，但这里可能有负值由于预处理)
        negative_ratio = self._compute_negative_ratio(pred_rcs)

        return {
            'symmetry_error': symmetry_error,
            'continuity_error': continuity_error,
            'negative_ratio': negative_ratio
        }

    def _compute_symmetry_error(self,
                              pred_rcs: torch.Tensor,
                              true_rcs: torch.Tensor) -> float:
        """计算对称性误差"""

        center_phi = 45  # φ=0°对应第45列
        symmetry_errors = []

        for i in range(1, min(center_phi + 1, 46)):  # 最多检查45度范围
            left_idx = center_phi - i
            right_idx = center_phi + i

            if right_idx < 91:
                # 预测的对称性
                pred_left = pred_rcs[:, :, left_idx, :]
                pred_right = pred_rcs[:, :, right_idx, :]
                pred_sym_diff = pred_left - pred_right

                # 真实的对称性
                true_left = true_rcs[:, :, left_idx, :]
                true_right = true_rcs[:, :, right_idx, :]
                true_sym_diff = true_left - true_right

                # 对称性误差
                sym_error = F.mse_loss(pred_sym_diff, true_sym_diff)
                symmetry_errors.append(sym_error.item())

        return np.mean(symmetry_errors) if symmetry_errors else 0.0

    def _compute_continuity_error(self,
                                pred_rcs: torch.Tensor,
                                true_rcs: torch.Tensor) -> float:
        """计算连续性误差"""

        # θ方向梯度
        pred_grad_theta = pred_rcs[:, 1:, :, :] - pred_rcs[:, :-1, :, :]
        true_grad_theta = true_rcs[:, 1:, :, :] - true_rcs[:, :-1, :, :]

        # φ方向梯度
        pred_grad_phi = pred_rcs[:, :, 1:, :] - pred_rcs[:, :, :-1, :]
        true_grad_phi = true_rcs[:, :, 1:, :] - true_rcs[:, :, :-1, :]

        # 梯度误差
        grad_error_theta = F.mse_loss(pred_grad_theta, true_grad_theta)
        grad_error_phi = F.mse_loss(pred_grad_phi, true_grad_phi)

        return (grad_error_theta + grad_error_phi).item() / 2

    def _compute_negative_ratio(self, pred_rcs: torch.Tensor) -> float:
        """计算负值比例"""
        total_elements = pred_rcs.numel()
        negative_elements = torch.sum(pred_rcs < 0).item()
        return negative_elements / total_elements

    def compute_statistical_metrics(self,
                                  pred_rcs: torch.Tensor,
                                  true_rcs: torch.Tensor) -> Dict[str, float]:
        """计算统计指标"""

        pred_rcs = pred_rcs.to(self.device)
        true_rcs = true_rcs.to(self.device)

        # 皮尔逊相关系数
        correlation = self._compute_correlation(pred_rcs, true_rcs)

        # R²决定系数
        r2_score = self._compute_r2_score(pred_rcs, true_rcs)

        # 分布差异 (KL散度近似)
        kl_divergence = self._compute_kl_divergence(pred_rcs, true_rcs)

        # 数值范围比较
        range_metrics = self._compute_range_metrics(pred_rcs, true_rcs)

        metrics = {
            'correlation': correlation,
            'r2_score': r2_score,
            'kl_divergence': kl_divergence
        }
        metrics.update(range_metrics)

        return metrics

    def _compute_correlation(self,
                           pred_rcs: torch.Tensor,
                           true_rcs: torch.Tensor) -> float:
        """计算皮尔逊相关系数"""

        pred_flat = pred_rcs.flatten()
        true_flat = true_rcs.flatten()

        # 计算相关系数
        pred_mean = torch.mean(pred_flat)
        true_mean = torch.mean(true_flat)

        numerator = torch.sum((pred_flat - pred_mean) * (true_flat - true_mean))
        pred_std = torch.sqrt(torch.sum((pred_flat - pred_mean) ** 2))
        true_std = torch.sqrt(torch.sum((true_flat - true_mean) ** 2))

        denominator = pred_std * true_std

        if denominator < 1e-8:
            return 0.0

        correlation = (numerator / denominator).item()
        return correlation

    def _compute_r2_score(self,
                        pred_rcs: torch.Tensor,
                        true_rcs: torch.Tensor) -> float:
        """计算R²决定系数"""

        ss_res = torch.sum((true_rcs - pred_rcs) ** 2)
        ss_tot = torch.sum((true_rcs - torch.mean(true_rcs)) ** 2)

        if ss_tot < 1e-8:
            return 0.0

        r2 = 1 - ss_res / ss_tot
        return r2.item()

    def _compute_kl_divergence(self,
                             pred_rcs: torch.Tensor,
                             true_rcs: torch.Tensor) -> float:
        """计算KL散度近似"""

        # 将数据转换为概率分布（简化处理）
        pred_hist = torch.histc(pred_rcs, bins=50, min=pred_rcs.min(), max=pred_rcs.max())
        true_hist = torch.histc(true_rcs, bins=50, min=true_rcs.min(), max=true_rcs.max())

        # 标准化为概率分布
        pred_prob = pred_hist / torch.sum(pred_hist)
        true_prob = true_hist / torch.sum(true_hist)

        # 避免log(0)
        pred_prob = torch.clamp(pred_prob, min=1e-8)
        true_prob = torch.clamp(true_prob, min=1e-8)

        # KL散度
        kl = torch.sum(true_prob * torch.log(true_prob / pred_prob))

        return kl.item()

    def _compute_range_metrics(self,
                             pred_rcs: torch.Tensor,
                             true_rcs: torch.Tensor) -> Dict[str, float]:
        """计算数值范围指标"""

        pred_min, pred_max = torch.min(pred_rcs).item(), torch.max(pred_rcs).item()
        true_min, true_max = torch.min(true_rcs).item(), torch.max(true_rcs).item()

        pred_range = pred_max - pred_min
        true_range = true_max - true_min

        range_error = abs(pred_range - true_range) / max(true_range, 1e-8)

        return {
            'pred_min': pred_min,
            'pred_max': pred_max,
            'true_min': true_min,
            'true_max': true_max,
            'range_error': range_error
        }

    def generate_report(self,
                       metrics: Dict[str, float],
                       detailed: bool = True) -> str:
        """生成评估报告"""

        report = "\\n=== RCS重建质量评估报告 ===\\n"

        # 基础误差指标
        report += "\\n📊 基础误差指标:\\n"
        report += f"  MSE (均方误差):      {metrics.get('mse', 0):.6f}\\n"
        report += f"  MAE (平均绝对误差):   {metrics.get('mae', 0):.6f}\\n"
        report += f"  RMSE (均方根误差):    {metrics.get('rmse', 0):.6f}\\n"
        report += f"  相对误差:            {metrics.get('relative_error', 0):.6f}\\n"

        # 结构相似性
        report += "\\n🔍 结构相似性指标:\\n"
        report += f"  SSIM (平均):         {metrics.get('ssim_mean', 0):.4f}\\n"
        report += f"  SSIM (标准差):       {metrics.get('ssim_std', 0):.4f}\\n"

        # 物理约束
        report += "\\n⚖️ 物理约束指标:\\n"
        report += f"  对称性误差:          {metrics.get('symmetry_error', 0):.6f}\\n"
        report += f"  连续性误差:          {metrics.get('continuity_error', 0):.6f}\\n"

        # 统计指标
        report += "\\n📈 统计指标:\\n"
        report += f"  相关系数:            {metrics.get('correlation', 0):.4f}\\n"
        report += f"  R²决定系数:          {metrics.get('r2_score', 0):.4f}\\n"

        if detailed:
            report += "\\n🔬 详细指标:\\n"
            report += f"  频域幅度误差:        {metrics.get('freq_magnitude_error', 0):.6f}\\n"
            report += f"  频域相位误差:        {metrics.get('freq_phase_error', 0):.6f}\\n"
            report += f"  负值比例:            {metrics.get('negative_ratio', 0):.4f}\\n"
            report += f"  数值范围误差:        {metrics.get('range_error', 0):.4f}\\n"

        return report


def test_reconstruction_metrics():
    """测试重建质量评估"""
    print("=== 重建质量评估测试 ===")

    # 创建测试数据
    batch_size = 5
    true_rcs = torch.randn(batch_size, 91, 91, 2) * 10

    # 创建不同质量的预测数据进行测试
    test_cases = [
        ("完美重建", true_rcs),
        ("添加噪声", true_rcs + torch.randn_like(true_rcs) * 0.5),
        ("系统偏移", true_rcs + 2.0),
        ("比例缩放", true_rcs * 0.8),
        ("随机噪声", torch.randn_like(true_rcs) * 5)
    ]

    # 创建评估器
    evaluator = ReconstructionMetrics()

    # 测试每种情况
    for case_name, pred_rcs in test_cases:
        print(f"\\n--- {case_name} ---")

        metrics = evaluator.compute_all_metrics(pred_rcs, true_rcs)

        # 显示关键指标
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"SSIM: {metrics['ssim_mean']:.4f}")
        print(f"相关系数: {metrics['correlation']:.4f}")
        print(f"R²: {metrics['r2_score']:.4f}")

    # 生成详细报告
    print("\\n" + "="*50)
    print("详细评估报告示例:")

    best_metrics = evaluator.compute_all_metrics(test_cases[1][1], true_rcs)
    report = evaluator.generate_report(best_metrics, detailed=True)
    print(report)

    return True


if __name__ == "__main__":
    test_reconstruction_metrics()