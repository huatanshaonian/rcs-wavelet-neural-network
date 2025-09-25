"""
小波多尺度神经网络模块
用于从9个飞行器参数预测三维RCS数据张量[91_phi, 91_theta, 2_freq]

网络架构:
1. 参数编码器: Linear(9→128→256)
2. 多尺度小波特征提取器 (4个尺度, φ-θ平面2D小波变换)
3. 双频分支网络
4. 频率交互模块
5. 输出: [batch_size, 91, 91, 2] (偏航×俯仰×频率)

物理约束:
- φ=0°平面对称性: σ(φ,θ,f) = σ(-φ,θ,f)
- 双频间的物理关系建模

作者: 基于小波多尺度理论设计
版本: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class Wavelet2DConv(nn.Module):
    """
    2D小波卷积层

    对φ-θ平面进行2D小波变换，支持多种小波基
    数学原理:
    W(a,b) = ∫∫ f(x,y) ψ*((x-b_x)/a, (y-b_y)/a) dx dy
    其中 ψ 为母小波，a为尺度参数，b为平移参数
    """

    def __init__(self, in_channels: int, out_channels: int,
                 wavelet: str = 'db4', scale: int = 2):
        """
        初始化2D小波卷积层

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            wavelet: 小波类型 ('db4', 'bior2.2', 'haar' 等)
            scale: 小波尺度
        """
        super(Wavelet2DConv, self).__init__()

        self.wavelet = wavelet
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 获取小波滤波器系数
        wavelet_obj = pywt.Wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet_obj.filter_bank

        # 转换为2D卷积核
        self.register_buffer('dec_lo', torch.tensor(dec_lo, dtype=torch.float32))
        self.register_buffer('dec_hi', torch.tensor(dec_hi, dtype=torch.float32))

        # 可学习的小波参数
        self.conv_ll = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)
        self.conv_lh = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)
        self.conv_hl = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)
        self.conv_hh = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def wavelet_decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        执行简化的2D小波分解（可微分版本）

        参数:
            x: 输入张量 [B, C, H, W]

        返回:
            (LL, LH, HL, HH): 小波子带系数
        """
        # 使用平均池化和卷积模拟小波分解
        # LL: 低通-低通 (平均池化)
        LL = F.avg_pool2d(x, kernel_size=2, stride=2)

        # LH: 低通-高通 (水平差分)
        h_diff = x[:, :, :, 1:] - x[:, :, :, :-1]  # 水平差分
        h_diff_padded = F.pad(h_diff, (0, 1, 0, 0))  # 填充
        LH = F.avg_pool2d(h_diff_padded, kernel_size=2, stride=2)

        # HL: 高通-低通 (垂直差分)
        v_diff = x[:, :, 1:, :] - x[:, :, :-1, :]  # 垂直差分
        v_diff_padded = F.pad(v_diff, (0, 0, 0, 1))  # 填充
        HL = F.avg_pool2d(v_diff_padded, kernel_size=2, stride=2)

        # HH: 高通-高通 (对角差分)
        diag_diff = x[:, :, 1:, 1:] + x[:, :, :-1, :-1] - x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        diag_diff_padded = F.pad(diag_diff, (0, 1, 0, 1))  # 填充
        HH = F.avg_pool2d(diag_diff_padded, kernel_size=2, stride=2)

        return LL, LH, HL, HH

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 [B, C, H, W]

        返回:
            输出张量 [B, out_channels, H', W']
        """
        # 小波分解
        LL, LH, HL, HH = self.wavelet_decompose(x)

        # 对各子带应用卷积
        ll_feat = self.conv_ll(LL)
        lh_feat = self.conv_lh(LH)
        hl_feat = self.conv_hl(HL)
        hh_feat = self.conv_hh(HH)

        # 拼接所有子带特征
        out = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)

        # 批标准化和激活
        out = self.bn(out)
        out = self.activation(out)

        return out


class MultiScaleWaveletExtractor(nn.Module):
    """
    多尺度小波特征提取器

    在φ-θ平面使用4个不同尺度的2D小波变换
    实现多分辨率特征提取，捕获不同尺度的散射特性
    """

    def __init__(self, input_dim: int = 256, wavelet_config: List[str] = None):
        """
        初始化多尺度特征提取器

        参数:
            input_dim: 输入特征维度
            wavelet_config: 小波配置列表，包含4个小波类型 [scale1, scale2, scale3, scale4]
        """
        super(MultiScaleWaveletExtractor, self).__init__()

        # 默认小波配置
        if wavelet_config is None:
            wavelet_config = ['db4', 'db4', 'bior2.2', 'bior2.2']

        # 将1D特征映射到2D特征图
        self.feature_to_2d = nn.Sequential(
            nn.Linear(input_dim, 32*32*4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 4个不同尺度的小波层，使用配置的小波类型
        self.wavelet_scale1 = Wavelet2DConv(4, 64, wavelet=wavelet_config[0], scale=1)   # 细节尺度
        self.wavelet_scale2 = Wavelet2DConv(64, 64, wavelet=wavelet_config[1], scale=2)  # 中等尺度
        self.wavelet_scale3 = Wavelet2DConv(64, 64, wavelet=wavelet_config[2], scale=3)  # 粗糙尺度
        self.wavelet_scale4 = Wavelet2DConv(64, 64, wavelet=wavelet_config[3], scale=4)  # 最粗尺度

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 自适应池化到目标尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((23, 23))  # 91/4 ≈ 23

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入特征 [B, input_dim]

        返回:
            多尺度特征 [B, 64, 23, 23]
        """
        batch_size = x.size(0)

        # 转换为2D特征图
        x_2d = self.feature_to_2d(x)
        x_2d = x_2d.view(batch_size, 4, 32, 32)

        # 多尺度小波特征提取
        feat1 = self.wavelet_scale1(x_2d)  # [B, 64, H/2, W/2]
        feat2 = self.wavelet_scale2(feat1)  # [B, 64, H/4, W/4]
        feat3 = self.wavelet_scale3(feat2)  # [B, 64, H/8, W/8]
        feat4 = self.wavelet_scale4(feat3)  # [B, 64, H/16, W/16]

        # 上采样到统一尺寸 (使用feat1的尺寸作为目标)
        target_size = feat1.shape[2:]
        feat2_up = F.interpolate(feat2, size=target_size, mode='bilinear', align_corners=False)
        feat3_up = F.interpolate(feat3, size=target_size, mode='bilinear', align_corners=False)
        feat4_up = F.interpolate(feat4, size=target_size, mode='bilinear', align_corners=False)

        # 特征融合
        multi_scale_feat = torch.cat([feat1, feat2_up, feat3_up, feat4_up], dim=1)
        fused_feat = self.feature_fusion(multi_scale_feat)

        # 自适应池化
        output = self.adaptive_pool(fused_feat)

        return output


class FrequencyInteractionModule(nn.Module):
    """
    频率交互模块

    建模两个频率间的物理关系:
    1. 频率比例关系 (f2/f1 = 2)
    2. 散射机制的频率依赖性
    3. 跨频率信息交换
    """

    def __init__(self, feature_dim: int = 64):
        """
        初始化频率交互模块

        参数:
            feature_dim: 特征维度
        """
        super(FrequencyInteractionModule, self).__init__()

        # 频率特异性编码
        self.freq1_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

        self.freq2_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

        # 跨频率注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=True
        )

        # 频率融合层
        self.frequency_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, shared_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            shared_features: 共享特征 [B, feature_dim, H, W]

        返回:
            (freq1_features, freq2_features): 双频特征
        """
        # 分别编码两个频率的特征
        freq1_feat = self.freq1_encoder(shared_features)
        freq2_feat = self.freq2_encoder(shared_features)

        batch_size, channels, height, width = freq1_feat.shape

        # 重塑为序列格式用于注意力机制
        freq1_seq = freq1_feat.view(batch_size, channels, -1).transpose(1, 2)
        freq2_seq = freq2_feat.view(batch_size, channels, -1).transpose(1, 2)

        # 跨频率注意力
        freq1_attended, _ = self.cross_attention(freq1_seq, freq2_seq, freq2_seq)
        freq2_attended, _ = self.cross_attention(freq2_seq, freq1_seq, freq1_seq)

        # 重塑回2D格式
        freq1_attended = freq1_attended.transpose(1, 2).view(batch_size, channels, height, width)
        freq2_attended = freq2_attended.transpose(1, 2).view(batch_size, channels, height, width)

        # 特征融合
        freq1_enhanced = self.frequency_fusion(torch.cat([freq1_feat, freq1_attended], dim=1))
        freq2_enhanced = self.frequency_fusion(torch.cat([freq2_feat, freq2_attended], dim=1))

        return freq1_enhanced, freq2_enhanced


class ProgressiveDecoder(nn.Module):
    """
    渐进式解码器

    采用渐进式上采样策略，避免直接生成91×91高分辨率输出
    23×23 → 46×46 → 91×91
    """

    def __init__(self, input_channels: int = 64, use_log_output: bool = False):
        """
        初始化渐进式解码器

        参数:
            input_channels: 输入通道数
            use_log_output: 是否输出对数域数据（无激活函数）
        """
        super(ProgressiveDecoder, self).__init__()

        # 第一阶段: 23×23 → 46×46
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 第二阶段: 46×46 → 91×91 (使用插值+卷积)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # 最终输出层
        self.final_conv = nn.Conv2d(8, 1, 1)
        # 根据输出类型选择激活函数
        if use_log_output:
            # 对数域输出，不需要激活函数
            self.output_activation = nn.Identity()
        else:
            # 线性域输出，使用Softplus确保正值
            self.output_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入特征 [B, input_channels, 23, 23]

        返回:
            输出RCS [B, 1, 91, 91]
        """
        # 第一阶段上采样
        x = self.stage1(x)  # [B, 32, 46, 46]

        # 第二阶段上采样到91×91
        x = F.interpolate(x, size=(91, 91), mode='bilinear', align_corners=False)
        x = self.stage2(x)  # [B, 8, 91, 91]

        # 最终输出
        output = self.final_conv(x)  # [B, 1, 91, 91]

        # 应用激活函数确保输出为正值
        output = self.output_activation(output)

        return output


class TriDimensionalRCSNet(nn.Module):
    """
    三维RCS预测网络

    网络架构:
    输入: 9维参数向量
    输出: [batch_size, 91, 91, 2] 双频RCS数据

    核心组件:
    1. 深层参数编码器
    2. 多尺度小波特征提取器
    3. 频率交互模块
    4. 双频分支解码器
    """

    def __init__(self, input_dim: int = 9,
                 hidden_dims: List[int] = [128, 256],
                 dropout_rate: float = 0.2,
                 wavelet_config: List[str] = None,
                 use_log_output: bool = False):
        """
        初始化三维RCS网络

        参数:
            input_dim: 输入参数维度 (9个飞行器参数)
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            wavelet_config: 小波配置列表，包含4个小波类型
            use_log_output: 是否输出对数域数据（无激活函数）
        """
        super(TriDimensionalRCSNet, self).__init__()

        self.input_dim = input_dim
        self.wavelet_config = wavelet_config or ['db4', 'db4', 'bior2.2', 'bior2.2']

        # 深层参数编码器
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.parameter_encoder = nn.Sequential(*encoder_layers)

        # 多尺度小波特征提取器
        self.wavelet_extractor = MultiScaleWaveletExtractor(prev_dim, self.wavelet_config)

        # 频率交互模块
        self.frequency_interaction = FrequencyInteractionModule(64)

        # 双频分支解码器
        self.freq1_decoder = ProgressiveDecoder(64, use_log_output)  # 1.5GHz解码器
        self.freq2_decoder = ProgressiveDecoder(64, use_log_output)  # 3GHz解码器

        # 物理约束层 (φ=0°平面对称性)
        self.symmetry_constraint = self._build_symmetry_constraint()

        # 输出配置
        self.use_log_output = use_log_output

    def _build_symmetry_constraint(self):
        """构建对称性约束层"""
        return lambda x: self._apply_phi_symmetry(x)

    def _apply_phi_symmetry(self, rcs_tensor: torch.Tensor) -> torch.Tensor:
        """
        应用φ=0°平面对称性约束: σ(φ,θ,f) = σ(-φ,θ,f)

        参数:
            rcs_tensor: RCS张量 [B, theta, phi, freq] = [B, 91, 91, 2]
                       theta维度: 索引0-90对应45-135°
                       phi维度: 索引0-90对应-45到45°, 索引45是phi=0°

        返回:
            对称约束后的RCS张量
        """
        # 克隆张量以避免就地修改导致的CUDA错误
        rcs_symmetric = rcs_tensor.clone()

        # 数据维度: [batch, theta, phi, freq]
        # phi=0°对应第2维的索引45 (phi从-45到45°)
        center_phi = 45

        # 应用phi对称性约束: 对每个theta, phi(-a) = phi(a)
        for i in range(1, center_phi + 1):
            left_idx = center_phi - i   # phi < 0°
            right_idx = center_phi + i  # phi > 0°

            # 边界检查
            if left_idx >= 0 and right_idx < 91:
                # 取两侧的平均值实现对称 (注意维度: [:, :, phi, :])
                avg_values = (rcs_symmetric[:, :, left_idx, :] + rcs_symmetric[:, :, right_idx, :]) / 2
                rcs_symmetric[:, :, left_idx, :] = avg_values
                rcs_symmetric[:, :, right_idx, :] = avg_values

        return rcs_symmetric

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            parameters: 输入参数 [B, 9]

        返回:
            RCS预测结果 [B, 91, 91, 2]
        """
        # 参数编码
        encoded_params = self.parameter_encoder(parameters)

        # 多尺度小波特征提取
        wavelet_features = self.wavelet_extractor(encoded_params)

        # 频率交互
        freq1_features, freq2_features = self.frequency_interaction(wavelet_features)

        # 双频分支解码
        freq1_rcs = self.freq1_decoder(freq1_features)  # [B, 1, 91, 91]
        freq2_rcs = self.freq2_decoder(freq2_features)  # [B, 1, 91, 91]

        # 组合双频输出
        rcs_output = torch.cat([freq1_rcs, freq2_rcs], dim=1)  # [B, 2, 91, 91]

        # 重排为目标格式 [B, 91, 91, 2]
        rcs_output = rcs_output.permute(0, 2, 3, 1)

        # 应用物理约束
        rcs_output = self.symmetry_constraint(rcs_output)

        return rcs_output

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'input_shape': (self.input_dim,),
            'output_shape': (91, 91, 2),
            'architecture': 'Wavelet Multi-scale RCS Prediction Network'
        }


class TriDimensionalRCSLoss(nn.Module):
    """
    三维RCS损失函数

    包含:
    1. 每个频率的2D多尺度损失
    2. φ=0°平面对称性约束损失
    3. 渐进式训练损失
    4. 频率一致性损失
    """

    def __init__(self, loss_weights: Dict[str, float] = None):
        """
        初始化损失函数

        参数:
            loss_weights: 各损失项权重
        """
        super(TriDimensionalRCSLoss, self).__init__()

        self.loss_weights = loss_weights or {
            'mse': 1.0,
            'symmetry': 0.02,  # 降低对称性权重，避免主导训练
            'multiscale': 0.1
        }

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def _symmetry_loss(self, pred_rcs: torch.Tensor, target_rcs: torch.Tensor) -> torch.Tensor:
        """
        计算φ=0°平面对称性损失

        修改：计算预测和目标的对称性差异，而非仅要求预测自身对称
        这避免了网络学习"对称但错误"的输出

        参数:
            pred_rcs: 预测RCS [B, theta, phi, freq] = [B, 91, 91, 2]
            target_rcs: 目标RCS [B, theta, phi, freq] = [B, 91, 91, 2]

        返回:
            对称性损失
        """
        # phi=0°在第2维的索引45
        center_phi = 45
        symmetry_loss = 0.0
        count = 0

        for i in range(1, center_phi + 1):
            left_idx = center_phi - i
            right_idx = center_phi + i

            if right_idx < 91:
                # 计算预测的对称性误差
                pred_left = pred_rcs[:, :, left_idx, :]
                pred_right = pred_rcs[:, :, right_idx, :]
                pred_sym_diff = pred_left - pred_right

                # 计算目标的对称性误差（理想情况下应该为0）
                target_left = target_rcs[:, :, left_idx, :]
                target_right = target_rcs[:, :, right_idx, :]
                target_sym_diff = target_left - target_right

                # 损失：预测的对称性差异应该匹配目标的对称性差异
                symmetry_loss += self.mse_loss(pred_sym_diff, target_sym_diff)
                count += 1

        return symmetry_loss / max(count, 1)


    def _multiscale_loss(self, pred_rcs: torch.Tensor, target_rcs: torch.Tensor) -> torch.Tensor:
        """
        计算多尺度损失

        在不同分辨率下计算损失

        参数:
            pred_rcs: 预测RCS [B, 91, 91, 2]
            target_rcs: 真实RCS [B, 91, 91, 2]

        返回:
            多尺度损失
        """
        scales = [1, 2, 4]  # 不同下采样尺度
        total_loss = 0.0

        for scale in scales:
            if scale == 1:
                pred_scale = pred_rcs
                target_scale = target_rcs
            else:
                # 下采样
                pred_scale = F.avg_pool2d(
                    pred_rcs.permute(0, 3, 1, 2),
                    kernel_size=scale, stride=scale
                ).permute(0, 2, 3, 1)

                target_scale = F.avg_pool2d(
                    target_rcs.permute(0, 3, 1, 2),
                    kernel_size=scale, stride=scale
                ).permute(0, 2, 3, 1)

            scale_loss = self.mse_loss(pred_scale, target_scale)
            total_loss += scale_loss / scale  # 高分辨率权重更大

        return total_loss / len(scales)

    def forward(self, pred_rcs: torch.Tensor, target_rcs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        参数:
            pred_rcs: 预测RCS [B, 91, 91, 2]
            target_rcs: 真实RCS [B, 91, 91, 2]

        返回:
            损失字典
        """
        losses = {}

        # 主要MSE损失
        losses['mse'] = self.mse_loss(pred_rcs, target_rcs)

        # 对称性损失（需要target）
        losses['symmetry'] = self._symmetry_loss(pred_rcs, target_rcs)

        # 多尺度损失
        losses['multiscale'] = self._multiscale_loss(pred_rcs, target_rcs)

        # 加权总损失
        total_loss = sum(
            self.loss_weights[key] * loss
            for key, loss in losses.items()
            if key in self.loss_weights
        )

        losses['total'] = total_loss

        return losses


# 辅助函数
def create_model(input_dim: int = 9, wavelet_config: List[str] = None, use_log_output: bool = False,
                 model_type: str = 'original', **kwargs) -> TriDimensionalRCSNet:
    """
    创建RCS预测模型

    参数:
        input_dim: 输入维度
        wavelet_config: 小波配置列表，包含4个小波类型
        use_log_output: 是否输出对数域数据
        model_type: 模型类型 ('original' 或 'enhanced')
        **kwargs: 其他模型参数

    返回:
        模型实例
    """
    if model_type == 'enhanced':
        # 使用增强版网络架构
        try:
            from enhanced_network import EnhancedTriDimensionalRCSNet
            print("使用增强版RCS网络架构")
            return EnhancedTriDimensionalRCSNet(input_dim=input_dim, use_log_output=use_log_output)
        except ImportError:
            print("警告: enhanced_network模块未找到，回退到原始架构")
            return TriDimensionalRCSNet(input_dim=input_dim, wavelet_config=wavelet_config,
                                      use_log_output=use_log_output, **kwargs)
    else:
        # 使用原始网络架构
        return TriDimensionalRCSNet(input_dim=input_dim, wavelet_config=wavelet_config,
                                  use_log_output=use_log_output, **kwargs)


def create_loss_function(loss_type: str = 'original', **kwargs):
    """
    创建损失函数

    参数:
        loss_type: 损失函数类型 ('original' 或 'improved')
        **kwargs: 损失函数参数

    返回:
        损失函数实例
    """
    if loss_type == 'improved':
        # 使用改进的损失函数
        try:
            from enhanced_network import ImprovedRCSLoss
            print("使用改进版损失函数")
            return ImprovedRCSLoss(**kwargs)
        except ImportError:
            print("警告: enhanced_network模块未找到，回退到原始损失函数")
            return TriDimensionalRCSLoss(**kwargs)
    else:
        # 使用原始损失函数
        return TriDimensionalRCSLoss(**kwargs)


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = create_model().to(device)
    print(f"模型信息: {model.get_model_info()}")

    # 创建测试数据
    batch_size = 4
    test_params = torch.randn(batch_size, 9).to(device)

    # 前向传播测试
    with torch.no_grad():
        output = model(test_params)
        print(f"输入形状: {test_params.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")

    # 损失函数测试
    loss_fn = create_loss_function()
    target = torch.randn_like(output)
    losses = loss_fn(output, target)

    print("\n损失项:")
    for key, value in losses.items():
        print(f"{key}: {value.item():.4f}")