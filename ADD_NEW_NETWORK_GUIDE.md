# 添加新网络架构指南

本指南展示如何轻松向系统添加新的网络架构，而无需修改现有的前处理、后处理和训练代码。

## 📋 目录
- [快速开始](#快速开始)
- [完整示例](#完整示例)
- [高级功能](#高级功能)
- [集成到GUI](#集成到GUI)
- [最佳实践](#最佳实践)

## 🚀 快速开始

### 1. 创建新网络文件

创建 `networks/my_new_network.py`:

```python
import torch
import torch.nn as nn
from network_registry import BaseNetwork, NetworkConfig, register_network

@register_network
class MyAwesomeNetwork(BaseNetwork):
    """我的新网络架构"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        # 计算输出大小
        output_size = 1
        for dim in self.output_shape:
            output_size *= dim

        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        # 输出激活函数
        if self.use_log_output:
            self.activation = nn.Identity()
        else:
            self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        output = self.network(x)
        output = output.view(batch_size, *self.output_shape)
        return self.activation(output)

    @classmethod
    def get_name(cls) -> str:
        return "my_awesome_network"

    @classmethod
    def get_description(cls) -> str:
        return "My awesome network for RCS prediction"
```

### 2. 使用新网络

```python
from modern_wavelet_network import create_model

# 自动发现并使用新网络
model = create_model(model_type="my_awesome_network")

# 训练和预测代码完全不变！
# 所有前处理、后处理、可视化都自动适配
```

就这么简单！✨

---

## 💡 完整示例

### Transformer架构示例

```python
# networks/transformer_network.py
import torch
import torch.nn as nn
import math
from network_registry import BaseNetwork, NetworkConfig, register_network

@register_network
class TransformerRCS(BaseNetwork):
    """Transformer架构用于RCS预测"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        self.d_model = 512
        self.nhead = 8
        self.num_layers = 6

        # 参数编码
        self.param_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model)
        )

        # 位置编码
        self.pos_encoding = self._create_positional_encoding()

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # 输出层
        output_size = math.prod(self.output_shape)
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, output_size),
            nn.Softplus() if not self.use_log_output else nn.Identity()
        )

    def _create_positional_encoding(self):
        """创建位置编码"""
        max_len = 1000
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                           -(math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 参数嵌入 [B, 9] -> [B, 1, d_model]
        embedded = self.param_embedding(x).unsqueeze(1)

        # 添加位置编码
        embedded = embedded + self.pos_encoding[:1]

        # Transformer处理
        transformed = self.transformer(embedded)  # [B, 1, d_model]

        # 输出投影
        output = self.output_projection(transformed.squeeze(1))  # [B, output_size]
        output = output.view(batch_size, *self.output_shape)

        return output

    @classmethod
    def get_name(cls) -> str:
        return "transformer_rcs"

    @classmethod
    def get_description(cls) -> str:
        return "Transformer-based network for RCS prediction with attention mechanism"

    @classmethod
    def validate_config(cls, config: NetworkConfig) -> bool:
        # Transformer需要足够的计算资源，限制输出大小
        output_size = math.prod(config.output_shape)
        return output_size <= 50000  # 限制输出不超过50K元素

# 使用示例
if __name__ == "__main__":
    from modern_wavelet_network import create_model

    # 创建Transformer网络
    model = create_model(
        model_type="transformer_rcs",
        input_dim=9,
        output_shape=(91, 91, 2)
    )

    # 测试
    x = torch.randn(4, 9)
    output = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### CNN架构示例

```python
# networks/cnn_network.py
import torch
import torch.nn as nn
from network_registry import BaseNetwork, NetworkConfig, register_network

@register_network
class CNNReconstructionNetwork(BaseNetwork):
    """CNN重建网络"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        # 参数编码器
        self.param_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 8 * 8),  # 编码到8x8特征图
        )

        # CNN上采样网络
        self.upsampler = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 64x64 -> 91x91 (使用插值 + 卷积)
            nn.Upsample(size=(91, 91), mode='bilinear'),
            nn.Conv2d(32, self.output_shape[-1], 3, padding=1)
        )

        self.activation = nn.Softplus() if not self.use_log_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 编码参数
        encoded = self.param_encoder(x)  # [B, 64*8*8]

        # 重塑为特征图
        features = encoded.view(batch_size, 64, 8, 8)  # [B, 64, 8, 8]

        # CNN上采样
        output = self.upsampler(features)  # [B, 2, 91, 91]
        output = output.permute(0, 2, 3, 1)  # [B, 91, 91, 2]

        return self.activation(output)

    @classmethod
    def get_name(cls) -> str:
        return "cnn_reconstruction"

    @classmethod
    def get_description(cls) -> str:
        return "CNN-based reconstruction network with transposed convolutions"
```

---

## ⚙️ 高级功能

### 1. 自定义损失函数

```python
from network_registry import BaseLoss, LossConfig, register_loss

@register_loss
class PerceptualLoss(BaseLoss):
    """感知损失函数"""

    def __init__(self, config: LossConfig):
        super().__init__(config)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)

        # 感知损失：边缘检测
        pred_edges = self._detect_edges(pred)
        target_edges = self._detect_edges(target)
        edge_loss = self.mse(pred_edges, target_edges)

        total = (
            self.loss_weights.get('mse', 0.5) * mse_loss +
            self.loss_weights.get('l1', 0.3) * l1_loss +
            self.loss_weights.get('edge', 0.2) * edge_loss
        )

        return {
            'total': total,
            'mse': mse_loss,
            'l1': l1_loss,
            'edge': edge_loss
        }

    def _detect_edges(self, x):
        """简单边缘检测"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        # 对最后一个维度的每个通道分别处理
        edges = []
        for i in range(x.shape[-1]):
            channel = x[:, :, :, i:i+1].permute(0, 3, 1, 2)  # [B, 1, H, W]
            edge_x = F.conv2d(channel, sobel_x, padding=1)
            edge_y = F.conv2d(channel, sobel_y, padding=1)
            edge = torch.sqrt(edge_x**2 + edge_y**2)
            edges.append(edge)

        return torch.cat(edges, dim=1).permute(0, 2, 3, 1)  # [B, H, W, C]

    @classmethod
    def get_name(cls) -> str:
        return "perceptual_loss"
```

### 2. 动态网络配置

```python
@register_network
class AdaptiveNetwork(BaseNetwork):
    """自适应网络 - 根据数据集大小自动调整结构"""

    def __init__(self, config: NetworkConfig):
        super().__init__(config)

        # 根据输出大小自适应调整网络深度
        output_size = math.prod(self.output_shape)

        if output_size < 1000:
            hidden_sizes = [128, 256]
        elif output_size < 10000:
            hidden_sizes = [256, 512, 1024]
        else:
            hidden_sizes = [512, 1024, 2048, 1024]

        # 构建动态网络
        layers = []
        input_size = self.input_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size

        layers.append(nn.Linear(input_size, output_size))

        self.network = nn.Sequential(*layers)
        self.activation = nn.Softplus() if not self.use_log_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        output = self.network(x)
        output = output.view(batch_size, *self.output_shape)
        return self.activation(output)

    @classmethod
    def get_name(cls) -> str:
        return "adaptive_network"

    @classmethod
    def get_description(cls) -> str:
        return "Adaptive network that adjusts structure based on output size"
```

---

## 🖥️ 集成到GUI

修改GUI以支持新网络（只需要很小的改动）:

```python
# 在gui.py中添加:
from modern_wavelet_network import get_available_networks, get_network_info

def update_network_dropdown(self):
    """更新网络选择下拉框"""
    available_networks = get_available_networks()

    # 更新下拉框选项
    self.architecture_var.set('enhanced')  # 默认值
    self.architecture_combo['values'] = list(available_networks.keys())

def on_network_selection_changed(self, event):
    """网络选择改变时的回调"""
    selected_network = self.architecture_var.get()

    try:
        info = get_network_info(selected_network)
        param_count = info['parameters']['total']
        description = info['description']

        # 更新显示信息
        self.network_info_label.config(
            text=f"{description}\\n参数量: {param_count:,}"
        )
    except Exception as e:
        self.network_info_label.config(text=f"错误: {e}")
```

---

## 📝 最佳实践

### 1. 网络命名规范
- 使用小写和下划线：`my_network_v2`
- 包含架构信息：`resnet_18_rcs`, `transformer_small`
- 避免与现有网络冲突

### 2. 配置验证
```python
@classmethod
def validate_config(cls, config: NetworkConfig) -> bool:
    # 检查输入维度
    if config.input_dim < 3:
        return False

    # 检查输出形状
    output_size = math.prod(config.output_shape)
    if output_size > 1000000:  # 避免内存问题
        return False

    return True
```

### 3. 参数统计和文档
```python
def get_info(self) -> Dict[str, Any]:
    info = super().get_info()

    # 添加自定义信息
    info.update({
        'memory_usage_mb': self._estimate_memory(),
        'flops': self._estimate_flops(),
        'recommended_batch_size': self._get_recommended_batch_size()
    })

    return info
```

### 4. 单元测试
```python
def test_my_network():
    """测试新网络"""
    config = NetworkConfig(input_dim=9, output_shape=(91, 91, 2))
    network = MyNetwork(config)

    # 测试前向传播
    x = torch.randn(4, 9)
    output = network(x)

    assert output.shape == (4, 91, 91, 2)
    assert torch.all(output >= 0)  # 如果使用Softplus
```

---

## 🎯 总结

通过这个插件化架构，你可以：

✅ **快速添加新网络** - 只需创建一个文件并注册
✅ **零修改集成** - 前处理、后处理、训练代码完全不变
✅ **自动GUI支持** - 新网络自动出现在界面选项中
✅ **灵活配置** - 支持任意输入输出形状
✅ **智能推荐** - 根据数据集特点推荐合适网络
✅ **统一接口** - 所有网络使用相同的创建和训练流程

现在你可以专注于网络设计的创新，而不用担心工程集成的复杂性！🚀