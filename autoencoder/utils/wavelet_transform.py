"""
小波变换工具模块
实现RCS数据的小波变换预处理和逆变换
方案A: 小波预处理 + 单AutoEncoder
"""

import torch
import torch.nn.functional as F
import numpy as np
import pywt
from typing import Tuple, Union, List
import warnings


class WaveletTransform:
    """
    小波变换处理器
    支持RCS数据的多频带小波分解和重建
    """

    def __init__(self, wavelet: str = 'db4', mode: str = 'symmetric'):
        """
        初始化小波变换器

        Args:
            wavelet: 小波基函数 ('db1', 'db4', 'db8', 'haar', 'bior2.2')
            mode: 边界处理模式 ('symmetric', 'periodization', 'zero')
        """
        self.wavelet = wavelet
        self.mode = mode

        # 验证小波类型
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Unsupported wavelet: {wavelet}")

        # 获取小波属性
        self.wavelet_obj = pywt.Wavelet(wavelet)

        print(f"初始化小波变换器: {wavelet}, 模式: {mode}")

    def forward_transform(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """
        RCS数据 → 小波系数

        Args:
            rcs_data: [B, 91, 91, 2] RCS数据 (批次, 高度, 宽度, 频率)

        Returns:
            wavelet_coeffs: [B, 91, 91, 8] 小波系数 (2频率 × 4频带)
        """
        batch_size = rcs_data.shape[0]
        height, width = rcs_data.shape[1], rcs_data.shape[2]

        # 存储所有小波系数
        all_coeffs = []

        # 对每个样本的每个频率进行小波变换
        for batch_idx in range(batch_size):
            batch_coeffs = []

            for freq_idx in range(2):  # 1.5GHz, 3GHz
                # 提取单频数据 [91, 91]
                freq_data = rcs_data[batch_idx, :, :, freq_idx].detach().cpu().numpy()

                # 2D离散小波变换
                coeffs = pywt.dwt2(freq_data, self.wavelet, mode=self.mode)
                cA, (cH, cV, cD) = coeffs  # LL, LH, HL, HH

                # 调整系数尺寸到统一大小
                target_size = (height, width)
                cA_resized = self._resize_coeffs(cA, target_size)
                cH_resized = self._resize_coeffs(cH, target_size)
                cV_resized = self._resize_coeffs(cV, target_size)
                cD_resized = self._resize_coeffs(cD, target_size)

                # 将4个频带转换为torch tensor
                freq_coeffs = [
                    torch.from_numpy(cA_resized).float(),  # LL - 低频分量
                    torch.from_numpy(cH_resized).float(),  # LH - 水平边缘
                    torch.from_numpy(cV_resized).float(),  # HL - 垂直边缘
                    torch.from_numpy(cD_resized).float()   # HH - 对角边缘
                ]

                batch_coeffs.extend(freq_coeffs)

            # 堆叠为 [8, 91, 91]，然后转置为 [91, 91, 8]
            sample_coeffs = torch.stack(batch_coeffs, dim=0).permute(1, 2, 0)
            all_coeffs.append(sample_coeffs)

        # 最终输出: [B, 91, 91, 8]
        wavelet_tensor = torch.stack(all_coeffs, dim=0)

        # 移到原始设备
        if rcs_data.is_cuda:
            wavelet_tensor = wavelet_tensor.cuda()

        return wavelet_tensor

    def inverse_transform(self, wavelet_coeffs: torch.Tensor) -> torch.Tensor:
        """
        小波系数 → RCS数据

        Args:
            wavelet_coeffs: [B, 91, 91, 8] 小波系数

        Returns:
            rcs_data: [B, 91, 91, 2] 重建的RCS数据
        """
        batch_size = wavelet_coeffs.shape[0]
        height, width = wavelet_coeffs.shape[1], wavelet_coeffs.shape[2]

        # 存储重建的RCS数据
        reconstructed_rcs = []

        # 对每个样本进行逆变换
        for batch_idx in range(batch_size):
            batch_rcs = []

            # 提取当前样本的系数 [91, 91, 8]
            sample_coeffs = wavelet_coeffs[batch_idx].detach().cpu().numpy()

            for freq_idx in range(2):  # 两个频率
                # 提取当前频率的4个频带 [91, 91, 4]
                start_idx = freq_idx * 4
                cA = sample_coeffs[:, :, start_idx]     # LL
                cH = sample_coeffs[:, :, start_idx + 1] # LH
                cV = sample_coeffs[:, :, start_idx + 2] # HL
                cD = sample_coeffs[:, :, start_idx + 3] # HH

                # 计算逆变换所需的原始尺寸
                # 由于我们调整过尺寸，需要估算原始系数尺寸
                orig_h = height // 2 if height % 2 == 0 else height // 2 + 1
                orig_w = width // 2 if width % 2 == 0 else width // 2 + 1

                # 调整系数尺寸到逆变换所需尺寸
                cA_orig = self._resize_coeffs(cA, (orig_h, orig_w))
                cH_orig = self._resize_coeffs(cH, (orig_h, orig_w))
                cV_orig = self._resize_coeffs(cV, (orig_h, orig_w))
                cD_orig = self._resize_coeffs(cD, (orig_h, orig_w))

                # 重建小波系数结构
                coeffs_tuple = (cA_orig, (cH_orig, cV_orig, cD_orig))

                # 2D逆离散小波变换
                try:
                    reconstructed_freq = pywt.idwt2(coeffs_tuple, self.wavelet, mode=self.mode)

                    # 调整到目标尺寸
                    if reconstructed_freq.shape != (height, width):
                        reconstructed_freq = self._resize_coeffs(reconstructed_freq, (height, width))

                    batch_rcs.append(torch.from_numpy(reconstructed_freq).float())

                except Exception as e:
                    warnings.warn(f"逆小波变换失败: {e}, 使用近似重建")
                    # 使用LL分量作为近似重建
                    approx_recon = self._resize_coeffs(cA, (height, width))
                    batch_rcs.append(torch.from_numpy(approx_recon).float())

            # 堆叠频率维度 [2, 91, 91] -> [91, 91, 2]
            sample_rcs = torch.stack(batch_rcs, dim=-1)
            reconstructed_rcs.append(sample_rcs)

        # 最终输出: [B, 91, 91, 2]
        rcs_tensor = torch.stack(reconstructed_rcs, dim=0)

        # 移到原始设备
        if wavelet_coeffs.is_cuda:
            rcs_tensor = rcs_tensor.cuda()

        return rcs_tensor

    def _resize_coeffs(self, coeffs: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整小波系数尺寸 - 优化版本，减少插值误差

        Args:
            coeffs: 小波系数
            target_size: 目标尺寸 (height, width)

        Returns:
            resized_coeffs: 调整后的系数
        """
        if coeffs.shape == target_size:
            return coeffs

        # 如果尺寸很接近，使用简单的pad/crop
        h_diff = abs(coeffs.shape[0] - target_size[0])
        w_diff = abs(coeffs.shape[1] - target_size[1])

        if h_diff <= 2 and w_diff <= 2:
            # 使用numpy的pad/crop，保持数值精度
            result = coeffs.copy()

            # 高度调整
            if coeffs.shape[0] < target_size[0]:
                pad_h = target_size[0] - coeffs.shape[0]
                result = np.pad(result, ((0, pad_h), (0, 0)), mode='edge')
            elif coeffs.shape[0] > target_size[0]:
                result = result[:target_size[0], :]

            # 宽度调整
            if result.shape[1] < target_size[1]:
                pad_w = target_size[1] - result.shape[1]
                result = np.pad(result, ((0, 0), (0, pad_w)), mode='edge')
            elif result.shape[1] > target_size[1]:
                result = result[:, :target_size[1]]

            return result

        # 尺寸差异较大时使用插值
        coeffs_tensor = torch.from_numpy(coeffs).float().unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(
            coeffs_tensor,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        return resized_tensor.squeeze(0).squeeze(0).numpy()

    def get_transform_info(self) -> dict:
        """获取变换器信息"""
        return {
            'wavelet': self.wavelet,
            'mode': self.mode,
            'family': self.wavelet_obj.family_name,
            'orthogonal': self.wavelet_obj.orthogonal,
            'biorthogonal': self.wavelet_obj.biorthogonal,
            'input_shape': '[B, 91, 91, 2]',
            'output_shape': '[B, 91, 91, 8]',
            'frequency_bands': ['1.5GHz_LL', '1.5GHz_LH', '1.5GHz_HL', '1.5GHz_HH',
                              '3GHz_LL', '3GHz_LH', '3GHz_HL', '3GHz_HH']
        }

    def analyze_frequency_content(self, rcs_data: torch.Tensor) -> dict:
        """
        分析RCS数据的频率成分

        Args:
            rcs_data: [B, 91, 91, 2] RCS数据

        Returns:
            analysis: 频率分析结果
        """
        wavelet_coeffs = self.forward_transform(rcs_data)

        # 计算各频带的能量
        batch_size = wavelet_coeffs.shape[0]
        band_names = ['1.5GHz_LL', '1.5GHz_LH', '1.5GHz_HL', '1.5GHz_HH',
                     '3GHz_LL', '3GHz_LH', '3GHz_HL', '3GHz_HH']

        energy_analysis = {}

        for i, band_name in enumerate(band_names):
            band_coeffs = wavelet_coeffs[:, :, :, i]
            energy = torch.mean(torch.square(band_coeffs), dim=[1, 2])  # [B]
            energy_analysis[band_name] = {
                'mean_energy': energy.mean().item(),
                'std_energy': energy.std().item(),
                'max_energy': energy.max().item(),
                'min_energy': energy.min().item()
            }

        return {
            'band_energy': energy_analysis,
            'total_samples': batch_size,
            'transform_info': self.get_transform_info()
        }


def test_wavelet_transform():
    """测试小波变换功能"""
    print("=== 小波变换测试 ===")

    # 创建测试数据
    batch_size = 2
    test_rcs = torch.randn(batch_size, 91, 91, 2) * 10  # 模拟RCS数据

    # 初始化变换器
    wt = WaveletTransform(wavelet='db4')

    print(f"输入RCS形状: {test_rcs.shape}")
    print(f"RCS数值范围: [{test_rcs.min():.3f}, {test_rcs.max():.3f}]")

    # 前向变换
    wavelet_coeffs = wt.forward_transform(test_rcs)
    print(f"小波系数形状: {wavelet_coeffs.shape}")
    print(f"系数数值范围: [{wavelet_coeffs.min():.3f}, {wavelet_coeffs.max():.3f}]")

    # 逆变换
    reconstructed_rcs = wt.inverse_transform(wavelet_coeffs)
    print(f"重建RCS形状: {reconstructed_rcs.shape}")
    print(f"重建数值范围: [{reconstructed_rcs.min():.3f}, {reconstructed_rcs.max():.3f}]")

    # 计算重建误差
    reconstruction_error = F.mse_loss(reconstructed_rcs, test_rcs)
    print(f"重建MSE误差: {reconstruction_error:.6f}")

    # 频率分析
    analysis = wt.analyze_frequency_content(test_rcs)
    print(f"频率分析完成，检测到 {len(analysis['band_energy'])} 个频带")

    # 变换器信息
    info = wt.get_transform_info()
    print(f"使用小波: {info['wavelet']} ({info['family']})")

    return reconstruction_error < 1e-3  # 重建误差应该很小


if __name__ == "__main__":
    test_wavelet_transform()