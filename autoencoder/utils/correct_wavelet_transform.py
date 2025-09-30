"""
正确的小波变换实现
保持小波系数的原始尺寸，不进行任何插值
"""

import torch
import torch.nn.functional as F
import numpy as np
import pywt
from typing import Tuple, Union, List
import warnings


class CorrectWaveletTransform:
    """
    正确的小波变换处理器
    保持小波系数的原始尺寸 - 这是关键！
    """

    def __init__(self,
                 wavelet: str = 'db4',
                 mode: str = 'symmetric',
                 num_frequencies: int = 2):
        """
        初始化小波变换器

        Args:
            wavelet: 小波基函数
            mode: 边界处理模式
            num_frequencies: 频率数量
        """
        self.wavelet = wavelet
        self.mode = mode
        self.num_frequencies = num_frequencies

        # 验证小波类型
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Unsupported wavelet: {wavelet}")

        self.wavelet_obj = pywt.Wavelet(wavelet)

        # 计算小波变换后的尺寸
        # 对于91x91输入，小波变换后实际是49x49
        test_size = (91, 91)
        test_data = np.random.randn(*test_size)
        coeffs = pywt.dwt2(test_data, wavelet, mode=mode)
        cA, (cH, cV, cD) = coeffs

        self.wavelet_size = cA.shape  # 记录实际的小波系数尺寸 (49, 49)
        print(f"小波变换: {test_size} → {self.wavelet_size}")
        print(f"初始化正确小波变换器: {wavelet}, 模式: {mode}, 小波尺寸: {self.wavelet_size}")

    def forward_transform(self, rcs_data: torch.Tensor) -> torch.Tensor:
        """
        RCS数据 → 小波系数 (保持原始小波尺寸)

        Args:
            rcs_data: [B, 91, 91, num_freq] RCS数据

        Returns:
            wavelet_coeffs: [B, wavelet_h, wavelet_w, num_freq*4] 小波系数
                           例如: [B, 49, 49, 8] for 2频率
        """
        batch_size = rcs_data.shape[0]

        # 存储所有小波系数
        all_coeffs = []

        # 对每个样本的每个频率进行小波变换
        for batch_idx in range(batch_size):
            batch_coeffs = []

            for freq_idx in range(self.num_frequencies):
                # 提取单频数据 [91, 91]
                freq_data = rcs_data[batch_idx, :, :, freq_idx].detach().cpu().numpy()

                # 2D离散小波变换
                coeffs = pywt.dwt2(freq_data, self.wavelet, mode=self.mode)
                cA, (cH, cV, cD) = coeffs  # LL, LH, HL, HH

                # ⚠️ 关键：保持小波系数的原始尺寸，不进行任何插值！
                freq_coeffs = [
                    torch.from_numpy(cA).float(),  # LL - 低频分量
                    torch.from_numpy(cH).float(),  # LH - 水平边缘
                    torch.from_numpy(cV).float(),  # HL - 垂直边缘
                    torch.from_numpy(cD).float()   # HH - 对角边缘
                ]

                batch_coeffs.extend(freq_coeffs)

            # 堆叠为 [num_freq*4, wavelet_h, wavelet_w]，然后转置
            sample_coeffs = torch.stack(batch_coeffs, dim=0).permute(1, 2, 0)
            all_coeffs.append(sample_coeffs)

        # 最终输出: [B, wavelet_h, wavelet_w, num_freq*4]
        result = torch.stack(all_coeffs, dim=0)

        print(f"小波变换结果形状: {rcs_data.shape} → {result.shape}")
        return result

    def inverse_transform(self, wavelet_coeffs: torch.Tensor) -> torch.Tensor:
        """
        小波系数 → RCS数据

        Args:
            wavelet_coeffs: [B, wavelet_h, wavelet_w, num_freq*4] 小波系数

        Returns:
            rcs_data: [B, 91, 91, num_freq] 重建的RCS数据
        """
        batch_size = wavelet_coeffs.shape[0]
        wavelet_h, wavelet_w = wavelet_coeffs.shape[1], wavelet_coeffs.shape[2]

        all_rcs = []

        for batch_idx in range(batch_size):
            batch_rcs = []

            for freq_idx in range(self.num_frequencies):
                # 提取该频率的4个小波分量
                base_idx = freq_idx * 4
                cA = wavelet_coeffs[batch_idx, :, :, base_idx + 0].detach().cpu().numpy()
                cH = wavelet_coeffs[batch_idx, :, :, base_idx + 1].detach().cpu().numpy()
                cV = wavelet_coeffs[batch_idx, :, :, base_idx + 2].detach().cpu().numpy()
                cD = wavelet_coeffs[batch_idx, :, :, base_idx + 3].detach().cpu().numpy()

                # 重组小波系数
                coeffs = (cA, (cH, cV, cD))

                # 逆小波变换
                reconstructed = pywt.idwt2(coeffs, self.wavelet, mode=self.mode)

                # 裁剪到原始尺寸 (91, 91) - 解决pywt尺寸扩张问题
                if reconstructed.shape != (91, 91):
                    reconstructed = reconstructed[:91, :91]

                # 转换为tensor并添加到列表
                batch_rcs.append(torch.from_numpy(reconstructed).float())

            # 堆叠频率维度
            sample_rcs = torch.stack(batch_rcs, dim=2)  # [91, 91, num_freq]
            all_rcs.append(sample_rcs)

        # 最终输出: [B, 91, 91, num_freq]
        result = torch.stack(all_rcs, dim=0)

        print(f"逆小波变换结果形状: {wavelet_coeffs.shape} → {result.shape}")
        return result

    def get_wavelet_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        获取小波变换后的输出尺寸

        Args:
            input_shape: 输入尺寸 (H, W)

        Returns:
            output_shape: 小波变换后的尺寸
        """
        return self.wavelet_size

    def get_transform_info(self) -> dict:
        """获取变换信息"""
        return {
            'wavelet': self.wavelet,
            'mode': self.mode,
            'num_frequencies': self.num_frequencies,
            'input_size': (91, 91),
            'wavelet_size': self.wavelet_size,
            'size_reduction': f"{91*91} → {self.wavelet_size[0]*self.wavelet_size[1]} (49x49)",
            'compression_ratio': f"{(91*91) / (self.wavelet_size[0]*self.wavelet_size[1]):.1f}x"
        }


def test_correct_wavelet_transform():
    """测试正确的小波变换"""
    print("=== 测试正确的小波变换 ===")

    # 创建变换器
    wt = CorrectWaveletTransform(wavelet='db4', num_frequencies=2)

    # 创建测试数据 [B, 91, 91, 2]
    test_rcs = torch.randn(4, 91, 91, 2)
    print(f"输入RCS数据: {test_rcs.shape}")

    # 前向变换
    wavelet_coeffs = wt.forward_transform(test_rcs)
    print(f"小波系数: {wavelet_coeffs.shape}")

    # 逆变换
    reconstructed_rcs = wt.inverse_transform(wavelet_coeffs)
    print(f"重建RCS数据: {reconstructed_rcs.shape}")

    # 计算重建误差
    mse = torch.mean((test_rcs - reconstructed_rcs)**2).item()
    print(f"重建MSE: {mse:.6f}")

    # 显示变换信息
    info = wt.get_transform_info()
    print(f"\n变换信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    return wt


if __name__ == "__main__":
    test_correct_wavelet_transform()