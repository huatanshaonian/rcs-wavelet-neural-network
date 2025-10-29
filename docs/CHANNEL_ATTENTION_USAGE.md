# 通道注意力权重查看与分析指南

## 📋 概述

通道注意力机制会为每个输入通道（如小波系数的LL/LH/HL/HH通道）学习一个权重（0-1之间），表示该通道的重要性。训练后，你可以查看这些权重来理解网络学到了什么。

---

## 🎯 为什么要查看注意力权重？

1. **可解释性**：了解网络认为哪些通道更重要
2. **验证假设**：确认LL通道是否真的比高频通道获得更高权重
3. **调试优化**：如果权重分布不合理，可能需要调整训练策略
4. **频率分析**：对比不同频率（1.5GHz vs 3GHz）的权重差异

---

## 💡 快速使用

### 方法1：使用可视化脚本（推荐）

```bash
python visualize_attention_weights.py
```

这个脚本会：
- ✅ 创建柱状图显示每个通道的权重
- ✅ 创建热力图显示频率×频带的权重矩阵
- ✅ 打印详细的统计信息
- ✅ 自动保存图片

### 方法2：在训练/评估代码中查看

```python
import torch
from autoencoder.models import WaveletAutoEncoder

# 1. 创建模型（确保启用通道注意力）
model = WaveletAutoEncoder(
    latent_dim=256,
    num_frequencies=2,
    use_channel_attention=True  # 必须启用！
)

# 2. 加载训练好的模型权重
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# 3. 运行一次前向传播（必须！）
sample_data = torch.randn(4, 49, 49, 8)  # 你的实际数据
with torch.no_grad():
    latent = model.encode(sample_data)

# 4. 获取注意力权重
weights_info = model.get_channel_attention_weights()

if weights_info['enabled']:
    weights = weights_info['weights']  # numpy数组 [8]
    names = weights_info['channel_names']  # ['LL_F1', 'LH_F1', ...]

    # 打印权重
    for name, weight in zip(names, weights):
        print(f"{name}: {weight:.4f}")
```

### 方法3：在GUI中查看（即将添加）

训练完成后，在AutoEncoder评估界面会自动显示注意力权重分布图。

---

## 📊 输出示例

### 文本输出
```
通道注意力权重:
────────────────────────────────────────
  LL_F1       : 0.7823  ████████████████████████
  LH_F1       : 0.2145  ██████
  HL_F1       : 0.2034  ██████
  HH_F1       : 0.1876  █████
  LL_F2       : 0.8012  ████████████████████████
  LH_F2       : 0.2267  ██████
  HL_F2       : 0.2156  ██████
  HH_F2       : 0.1923  █████
────────────────────────────────────────

🔷 LL通道平均权重: 0.7918
🔶 高频通道平均权重: 0.2067
📈 LL/高频比值: 3.83:1
```

### 可视化输出

运行 `visualize_attention_weights.py` 会生成：

1. **柱状图**：清晰显示每个通道的权重
2. **热力图**：矩阵视图，便于对比不同频率和频带

---

## 🔍 如何解读权重？

### Wavelet模式（小波系数）

**预期权重分布**：
- **LL通道（低频近似）**：权重应该较高（0.6-0.9）
  - 包含大部分能量和全局结构
  - 对重建质量至关重要
- **LH/HL/HH（高频细节）**：权重应该较低（0.1-0.4）
  - 包含边缘和纹理信息
  - 能量较小但包含重要细节

**示例分析**：
```
LL_F1: 0.78  ✓ 正常（LL权重高）
LH_F1: 0.21  ✓ 正常（高频权重低）
HL_F1: 0.20  ✓ 正常
HH_F1: 0.19  ✓ 正常
```

**异常情况**：
- ❌ **LL权重过低（<0.5）**：可能导致重建模糊
- ❌ **所有权重相似（±0.1内）**：注意力机制未生效
- ❌ **某个高频权重异常高**：可能过拟合噪声

### Direct模式（RCS数据）

**预期权重分布**：
- 不同频率的相对重要性
- 通常较均匀（因为只有2-3个通道）

---

## 🛠️ 训练中的最佳实践

### 1. 权重演变分析

在训练的不同阶段查看权重变化：

```python
# 训练开始
epoch 0:   LL=0.52, LH=0.48  # 未学习差异

# 训练中期
epoch 50:  LL=0.65, LH=0.35  # 开始区分

# 训练结束
epoch 100: LL=0.78, LH=0.22  # 明显区分
```

### 2. 对比不同配置

```python
# 配置1：标准CNN + 注意力
weights_1 = [0.78, 0.21, 0.20, 0.19]  # LL权重高

# 配置2：Deep CNN + 注意力
weights_2 = [0.82, 0.18, 0.17, 0.16]  # LL权重更高（更好）
```

### 3. 保存权重历史

```python
import json

weights_history = []
for epoch in range(num_epochs):
    train_one_epoch()

    # 每10个epoch保存一次
    if epoch % 10 == 0:
        weights_info = model.get_channel_attention_weights()
        weights_history.append({
            'epoch': epoch,
            'weights': weights_info['weights'].tolist()
        })

# 保存到文件
with open('attention_weights_history.json', 'w') as f:
    json.dump(weights_history, f, indent=2)
```

---

## 📈 高级用法：多样本统计

```python
import numpy as np

# 收集多个样本的权重
all_weights = []

for batch in dataloader:
    with torch.no_grad():
        _ = model.encode(batch)
    weights_info = model.get_channel_attention_weights()
    all_weights.append(weights_info['weights'])

# 统计分析
all_weights = np.array(all_weights)  # [N_samples, N_channels]

mean_weights = all_weights.mean(axis=0)
std_weights = all_weights.std(axis=0)

print("平均权重 ± 标准差:")
for name, mean, std in zip(names, mean_weights, std_weights):
    print(f"{name}: {mean:.4f} ± {std:.4f}")
```

---

## ⚠️ 常见问题

### Q1: 获取权重返回None？

**A**: 必须先运行一次前向传播！

```python
# ❌ 错误
weights_info = model.get_channel_attention_weights()  # weights=None

# ✓ 正确
with torch.no_grad():
    _ = model.encode(sample_data)  # 先运行前向传播
weights_info = model.get_channel_attention_weights()  # 现在有权重了
```

### Q2: enabled=False？

**A**: 模型未启用通道注意力

```python
# 创建模型时确保设置
model = WaveletAutoEncoder(
    latent_dim=256,
    use_channel_attention=True  # 必须启用
)
```

### Q3: 权重变化很小或全部相似？

**A**: 可能的原因：
1. 训练epoch数不够，注意力机制还未充分学习
2. 学习率过小，权重更新缓慢
3. 数据问题：如果所有通道确实同等重要，权重会相似

---

## 🔗 相关文件

- `autoencoder/models/channel_attention.py` - ChannelAttention类定义
- `visualize_attention_weights.py` - 可视化工具脚本
- `attention_integration_guide.py` - 集成指南

---

## 📝 总结

✅ **使用通道注意力权重可以**：
- 验证网络学习是否符合预期
- 理解不同通道的相对重要性
- 调试和优化模型

✅ **记得**：
- 启用 `use_channel_attention=True`
- 运行前向传播后再获取权重
- 多样本统计更可靠

✅ **工具**：
- `visualize_attention_weights.py` - 一键可视化
- `model.get_channel_attention_weights()` - API调用
- GUI集成（即将推出）

有问题？检查 `visualize_attention_weights.py` 中的示例代码！
