"""
通道注意力权重可视化工具

展示如何查看和分析AutoEncoder的通道注意力权重
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_channel_attention_weights(model, sample_data, mode='wavelet', save_path=None):
    """
    可视化AutoEncoder的通道注意力权重

    Args:
        model: AutoEncoder模型（需要启用了use_channel_attention）
        sample_data: 样本数据 [B, H, W, C]
        mode: 'wavelet' 或 'direct'
        save_path: 如果提供，保存图片到指定路径

    Example:
        >>> from autoencoder.models import WaveletAutoEncoder
        >>> model = WaveletAutoEncoder(latent_dim=256, num_frequencies=2, use_channel_attention=True)
        >>> sample_data = torch.randn(4, 49, 49, 8)  # 模拟小波系数
        >>> visualize_channel_attention_weights(model, sample_data, mode='wavelet')
    """
    # 确保模型在eval模式
    model.eval()

    # 运行一次前向传播以获取注意力权重
    with torch.no_grad():
        _ = model.encode(sample_data)

    # 获取注意力权重
    weights_info = model.get_channel_attention_weights()

    if not weights_info['enabled']:
        print("❌ 该模型未启用通道注意力机制")
        print("   请在创建模型时设置 use_channel_attention=True")
        return

    if weights_info['weights'] is None:
        print("❌ 无法获取注意力权重")
        if 'message' in weights_info:
            print(f"   {weights_info['message']}")
        return

    weights = weights_info['weights']
    channel_names = weights_info['channel_names']

    # 创建可视化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # === 子图1: 柱状图 ===
    ax1 = axes[0]
    colors = []
    if mode == 'wavelet':
        # 小波模式：LL通道用蓝色，高频通道用红色
        for name in channel_names:
            if name.startswith('LL'):
                colors.append('#2E86DE')  # 蓝色 - LL
            else:
                colors.append('#EE5A6F')  # 红色 - 高频
    else:
        # Direct模式：使用渐变色
        colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))

    bars = ax1.bar(range(len(weights)), weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(weights)))
    ax1.set_xticklabels(channel_names, rotation=45, ha='right')
    ax1.set_ylabel('注意力权重', fontsize=12, fontweight='bold')
    ax1.set_title('通道注意力权重分布', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.0)

    # 在柱子上方标注数值
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # === 子图2: 热力图 ===
    ax2 = axes[1]

    if mode == 'wavelet':
        # 重塑为 [频率, 频带] 矩阵
        num_freqs = len(weights) // 4
        weights_matrix = weights.reshape(num_freqs, 4)

        im = ax2.imshow(weights_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['LL', 'LH', 'HL', 'HH'])
        ax2.set_yticks(range(num_freqs))
        ax2.set_yticklabels([f'频率{i+1}' for i in range(num_freqs)])
        ax2.set_xlabel('小波频带', fontsize=12, fontweight='bold')
        ax2.set_ylabel('频率', fontsize=12, fontweight='bold')

        # 在每个cell中标注数值
        for i in range(num_freqs):
            for j in range(4):
                text = ax2.text(j, i, f'{weights_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    else:
        # Direct模式：横向显示
        weights_matrix = weights.reshape(1, -1)
        im = ax2.imshow(weights_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(weights)))
        ax2.set_xticklabels(channel_names)
        ax2.set_yticks([])
        ax2.set_xlabel('频率通道', fontsize=12, fontweight='bold')

        for j in range(len(weights)):
            text = ax2.text(j, 0, f'{weights[j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax2.set_title('通道注意力热力图', fontsize=14, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('注意力权重', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图片已保存至: {save_path}")

    plt.show()

    # === 打印统计信息 ===
    print("\n" + "="*60)
    print("📊 通道注意力权重统计")
    print("="*60)

    for name, weight in zip(channel_names, weights):
        bar = '█' * int(weight * 30)  # 简单的文本条形图
        print(f"{name:12s}: {weight:.4f}  {bar}")

    print(f"\n最大权重: {weights.max():.4f} ({channel_names[weights.argmax()]})")
    print(f"最小权重: {weights.min():.4f} ({channel_names[weights.argmin()]})")
    print(f"平均权重: {weights.mean():.4f}")
    print(f"标准差:   {weights.std():.4f}")

    if mode == 'wavelet':
        # 分析LL vs 高频
        num_freqs = len(weights) // 4
        ll_weights = weights[::4]  # 每4个取1个（LL）
        hf_weights = np.delete(weights, np.arange(0, len(weights), 4))  # 去除LL

        print(f"\n🔷 LL通道平均权重: {ll_weights.mean():.4f}")
        print(f"🔶 高频通道平均权重: {hf_weights.mean():.4f}")
        print(f"📈 LL/高频比值: {ll_weights.mean() / hf_weights.mean():.2f}:1")

    print("="*60)


def print_attention_weights_simple(model, sample_data):
    """
    简单打印注意力权重（不需要matplotlib）

    Args:
        model: AutoEncoder模型
        sample_data: 样本数据

    Example:
        >>> print_attention_weights_simple(model, sample_data)
    """
    model.eval()
    with torch.no_grad():
        _ = model.encode(sample_data)

    weights_info = model.get_channel_attention_weights()

    if not weights_info['enabled']:
        print("通道注意力未启用")
        return

    if weights_info['weights'] is None:
        print("无法获取权重")
        return

    weights = weights_info['weights']
    channel_names = weights_info['channel_names']

    print("\n通道注意力权重:")
    print("-" * 40)
    for name, weight in zip(channel_names, weights):
        print(f"  {name:12s}: {weight:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    print("通道注意力权重可视化示例\n")

    # === 示例1: Wavelet模式 ===
    print("【示例1】Wavelet模式 (2频率)")
    from autoencoder.models import WaveletAutoEncoder

    model_wavelet = WaveletAutoEncoder(
        latent_dim=256,
        num_frequencies=2,
        wavelet_bands=4,
        use_channel_attention=True  # 启用通道注意力
    )

    # 模拟数据：[B, 49, 49, 8]
    sample_wavelet = torch.randn(8, 49, 49, 8)

    # 可视化
    visualize_channel_attention_weights(
        model_wavelet,
        sample_wavelet,
        mode='wavelet',
        save_path='attention_weights_wavelet.png'
    )

    # === 示例2: Direct模式 ===
    print("\n\n【示例2】Direct模式")
    from autoencoder.models import DirectAutoEncoder

    model_direct = DirectAutoEncoder(
        latent_dim=256,
        num_frequencies=2,
        use_channel_attention=True
    )

    sample_direct = torch.randn(8, 91, 91, 2)

    visualize_channel_attention_weights(
        model_direct,
        sample_direct,
        mode='direct',
        save_path='attention_weights_direct.png'
    )

    print("\n✅ 示例运行完成！")
    print("   - 图片已保存")
    print("   - 权重统计已打印")
