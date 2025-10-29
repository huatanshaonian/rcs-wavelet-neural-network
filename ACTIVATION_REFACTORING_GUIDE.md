# 激活函数参数化重构指南

> **创建时间**: 2025-01-XX
> **目标**: 将所有AutoEncoder模型参数化，支持多种激活函数
> **状态**: 第1阶段完成（可微分小波系列），待Codex完成剩余模型

---

## 📋 重构概述

### 目标
将19个AutoEncoder类减少到约10个，通过activation参数支持多种激活函数，消除95%的代码重复。

### 已完成（第1阶段）
✅ 创建activation_factory.py工厂模块
✅ 重构DifferentiableWaveletAutoEncoder (CNN)
✅ 重构DifferentiableWaveletMLPAutoEncoder (MLP)
✅ 简化DifferentiableSineWaveletMLPAutoEncoder为别名

### 待完成（第2阶段 - Codex任务）
- [ ] 重构DualBranchDifferentiableWaveletAutoEncoder (CNN)
- [ ] 重构DualBranchDifferentiableWaveletMLPAutoEncoder (MLP)
- [ ] 重构WaveletAutoEncoder (CNN)
- [ ] 重构WaveletMLPAutoEncoder (MLP)
- [ ] 重构SinWaveletAutoEncoder → 别名化
- [ ] 重构SinWaveletMLPAutoEncoder → 别名化
- [ ] 重构DualBranchWaveletAutoEncoder (CNN)
- [ ] 重构DualBranchWaveletMLPAutoEncoder (MLP)
- [ ] 重构DirectAutoEncoder (CNN)
- [ ] 重构DirectMLPAutoEncoder (MLP)
- [ ] 重构SinDirectAutoEncoder → 别名化
- [ ] 重构SinDirectMLPAutoEncoder → 别名化
- [ ] 重构EnhancedWaveletAutoEncoder (CNN)
- [ ] 重构EnhancedDirectAutoEncoder (CNN)
- [ ] 重构DeepWaveletAutoEncoder (CNN)
- [ ] 重构DeepDirectAutoEncoder (CNN)

---

## 🔧 重构步骤（详细指南）

### 步骤1: 导入激活函数工厂

在文件顶部添加导入：

```python
from autoencoder.utils.activation_factory import get_activation, get_activation_name
```

**示例** (differentiable_wavelet_autoencoder.py line 26):
```python
from autoencoder.utils.activation_factory import get_activation, get_activation_name
```

---

### 步骤2: 添加activation参数到__init__

在`__init__`方法中添加`activation='relu'`参数，并保存标准化的激活函数名称。

**模板**:
```python
def __init__(self,
             latent_dim: int = 256,
             ...现有参数...,
             activation: str = 'relu'):  # ← 添加这个参数
    """
    Args:
        ...现有文档...
        activation: 激活函数类型 ('relu', 'sin', 'gelu', 'swish'等，默认: 'relu')
    """
    super().__init__()

    ...现有属性...
    self.activation_type = get_activation_name(activation)  # ← 添加这行
```

**示例** (differentiable_wavelet_autoencoder.py line 88-120):
```python
def __init__(self,
             latent_dim: int = 256,
             num_frequencies: int = 2,
             wavelet_bands: int = 4,
             dropout_rate: float = 0.2,
             wavelet_type: str = 'db4',
             input_size: int = 49,
             use_channel_attention: bool = False,
             activation: str = 'relu'):  # ✓ 添加
    super().__init__()

    self.latent_dim = latent_dim
    ...
    self.activation_type = get_activation_name(activation)  # ✓ 添加
```

---

### 步骤3: 替换所有硬编码的激活函数

在所有`nn.Sequential`中，将`nn.ReLU(inplace=True)`替换为`get_activation(activation)`。

#### 3.1 CNN架构

**替换前**:
```python
self.encoder = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, ...),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),  # ← 删除
    nn.Dropout2d(dropout_rate),
    ...
)
```

**替换后**:
```python
self.encoder = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, ...),
    nn.BatchNorm2d(out_channels),
    get_activation(activation),  # ← 替换为这个
    nn.Dropout2d(dropout_rate),
    ...
)
```

**示例** (differentiable_wavelet_autoencoder.py line 141-161):
```python
self.encoder = nn.Sequential(
    nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    get_activation(activation),  # ✓ 替换
    nn.Dropout2d(dropout_rate),

    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(128),
    get_activation(activation),  # ✓ 替换
    nn.Dropout2d(dropout_rate),
    ...
)
```

#### 3.2 MLP架构

**替换前**:
```python
self.encoder = nn.Sequential(
    nn.Linear(in_dim, out_dim),
    nn.BatchNorm1d(out_dim),
    nn.ReLU(inplace=True),  # ← 删除
    nn.Dropout(dropout_rate),
    ...
)
```

**替换后**:
```python
self.encoder = nn.Sequential(
    nn.Linear(in_dim, out_dim),
    nn.BatchNorm1d(out_dim),
    get_activation(activation),  # ← 替换为这个
    nn.Dropout(dropout_rate),
    ...
)
```

**示例** (differentiable_wavelet_autoencoder.py line 420-438):
```python
self.encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(self.input_dim, 4096),
    nn.BatchNorm1d(4096),
    get_activation(activation),  # ✓ 替换
    nn.Dropout(dropout_rate),

    nn.Linear(4096, 1024),
    nn.BatchNorm1d(1024),
    get_activation(activation),  # ✓ 替换
    nn.Dropout(dropout_rate),
    ...
)
```

#### 3.3 FC层（循环构建）

如果有动态构建的FC层（如小隐空间适配）：

**替换前**:
```python
for intermediate_dim in self.intermediate_dims:
    encoder_fc_layers.extend([
        nn.Linear(current_dim, intermediate_dim),
        nn.ReLU(inplace=True),  # ← 删除
        nn.Dropout(dropout_rate)
    ])
```

**替换后**:
```python
for intermediate_dim in self.intermediate_dims:
    encoder_fc_layers.extend([
        nn.Linear(current_dim, intermediate_dim),
        get_activation(activation),  # ← 替换为这个
        nn.Dropout(dropout_rate)
    ])
```

**示例** (differentiable_wavelet_autoencoder.py line 190-196):
```python
for intermediate_dim in self.intermediate_dims:
    encoder_fc_layers.extend([
        nn.Linear(current_dim, intermediate_dim),
        get_activation(activation),  # ✓ 替换
        nn.Dropout(dropout_rate)
    ])
```

#### 3.4 注意事项

⚠️ **不要替换最后一层的激活函数**（如果存在），因为：
- Decoder最后一层通常不加激活，允许输出负值（小波系数可以为负）
- Encoder最后一层（到latent）通常也不加激活

✅ **正确示例**:
```python
nn.ConvTranspose2d(64, self.input_channels, kernel_size=3, stride=2, padding=1),
# 最后一层不加激活函数，允许小波系数为负值
```

---

### 步骤4: 更新get_model_info方法

在`get_model_info()`返回的字典中添加`activation`字段。

**模板**:
```python
def get_model_info(self) -> Dict[str, Any]:
    """获取模型信息"""
    ...
    return {
        'type': '模型类名',
        'architecture': 'CNN' or 'MLP',
        'latent_dim': self.latent_dim,
        'activation': self.activation_type,  # ← 添加这行
        ...其他字段...
    }
```

**示例** (differentiable_wavelet_autoencoder.py line 350-367):
```python
def get_model_info(self) -> Dict[str, Any]:
    return {
        'type': 'DifferentiableWaveletAutoEncoder',
        'architecture': 'CNN',
        'latent_dim': self.latent_dim,
        'num_frequencies': self.num_frequencies,
        'wavelet_type': self.wavelet_type,
        'activation': self.activation_type,  # ✓ 添加
        'differentiable': True,
        ...
    }
```

---

### 步骤5: 处理Sine系列模型（别名化）

对于已存在的Sine系列模型（如SinWaveletAutoEncoder），将其简化为向后兼容的别名。

#### 5.1 简化__init__方法

**原始代码** (完整重复实现，80-100行):
```python
class SinWaveletAutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)

        # 大量重复的初始化代码...
        self.encoder = nn.Sequential(
            nn.Conv2d(...),
            nn.BatchNorm2d(...),
            SinActivation(),  # 硬编码
            ...
        )
```

**重构后** (3行):
```python
class SinWaveletAutoEncoder(WaveletAutoEncoder):
    """Sin激活CNN AutoEncoder (向后兼容别名)"""

    def __init__(self, *args, **kwargs):
        # 强制设置activation='sin'，其他参数传递给父类
        kwargs['activation'] = 'sin'
        super().__init__(*args, **kwargs)
```

#### 5.2 重写get_model_info（可选）

如果需要保持旧的类型名：

```python
def get_model_info(self) -> Dict[str, Any]:
    """获取模型信息（重写以返回正确类型名）"""
    info = super().get_model_info()
    info['type'] = 'SinWaveletAutoEncoder'  # 保持旧类型名
    return info
```

**示例** (differentiable_wavelet_autoencoder.py line 550-577):
```python
class DifferentiableSineWaveletMLPAutoEncoder(DifferentiableWaveletMLPAutoEncoder):
    """可微分Sine激活MLP AutoEncoder (向后兼容别名)"""

    def __init__(self, *args, **kwargs):
        kwargs['activation'] = 'sin'
        super().__init__(*args, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info['type'] = 'DifferentiableSineWaveletMLPAutoEncoder'
        return info
```

---

## 📝 完整示例：重构一个文件

### 示例1: cnn_autoencoder.py (Wavelet模式 CNN)

#### 重构前关键代码片段:
```python
# line 10-15: 缺少导入
import torch
import torch.nn as nn
# 没有导入activation_factory

# line 50-80: __init__没有activation参数
def __init__(self,
             latent_dim: int = 256,
             num_frequencies: int = 2,
             wavelet_bands: int = 4,
             dropout_rate: float = 0.2,
             input_size: int = 49,
             use_channel_attention: bool = False):
    super().__init__()

    self.latent_dim = latent_dim
    ...
    # 缺少 self.activation_type

# line 100-120: 硬编码ReLU
self.encoder = nn.Sequential(
    nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),  # ← 硬编码
    nn.Dropout2d(dropout_rate),
    ...
)
```

#### 重构后关键代码片段:
```python
# line 10-15: 添加导入
import torch
import torch.nn as nn
from autoencoder.utils.activation_factory import get_activation, get_activation_name

# line 50-80: 添加activation参数
def __init__(self,
             latent_dim: int = 256,
             num_frequencies: int = 2,
             wavelet_bands: int = 4,
             dropout_rate: float = 0.2,
             input_size: int = 49,
             use_channel_attention: bool = False,
             activation: str = 'relu'):  # ← 添加
    """
    Args:
        ...
        activation: 激活函数类型 ('relu', 'sin', 'gelu', 'swish'等，默认: 'relu')
    """
    super().__init__()

    self.latent_dim = latent_dim
    ...
    self.activation_type = get_activation_name(activation)  # ← 添加

# line 100-120: 替换ReLU
self.encoder = nn.Sequential(
    nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    get_activation(activation),  # ← 替换
    nn.Dropout2d(dropout_rate),
    ...
)
```

### 示例2: sine_cnn_autoencoder.py (Sine CNN) → 别名化

#### 重构前 (完整文件，200+行):
```python
"""Sin激活CNN AutoEncoder"""
import torch
import torch.nn as nn

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SinWaveletAutoEncoder(nn.Module):
    def __init__(self, ...):
        # 80行重复代码...
        self.encoder = nn.Sequential(
            nn.Conv2d(...),
            SinActivation(),  # 硬编码
            ...
        )
```

#### 重构后 (10行):
```python
"""Sin激活CNN AutoEncoder (向后兼容别名)"""
from .cnn_autoencoder import WaveletAutoEncoder

class SinWaveletAutoEncoder(WaveletAutoEncoder):
    """Sin激活CNN AutoEncoder (向后兼容别名)

    注意：现在是WaveletAutoEncoder(activation='sin')的别名。
    推荐直接使用父类并指定activation参数。
    """
    def __init__(self, *args, **kwargs):
        kwargs['activation'] = 'sin'
        super().__init__(*args, **kwargs)
```

---

## ✅ 检查清单

重构每个模型文件后，请检查：

### 必做检查
- [ ] 导入了`get_activation`和`get_activation_name`
- [ ] `__init__`添加了`activation='relu'`参数
- [ ] 保存了`self.activation_type = get_activation_name(activation)`
- [ ] 所有`nn.ReLU(inplace=True)`替换为`get_activation(activation)`
- [ ] 所有`nn.ReLU()`替换为`get_activation(activation)`（如果没有inplace）
- [ ] `get_model_info()`添加了`'activation': self.activation_type`字段
- [ ] Decoder最后一层没有误加激活函数

### 特殊情况
- [ ] MLP架构：`nn.ReLU(inplace=True)` → `get_activation(activation)`（MLP中ReLU通常也用inplace）
- [ ] 动态构建的FC层：循环中也替换了ReLU
- [ ] Sine系列模型：删除重复代码，改为别名
- [ ] 双分支模型：LL和HF分支都替换了激活函数

### 测试验证
- [ ] 模型可以正常导入
- [ ] 可以用`activation='relu'`创建模型
- [ ] 可以用`activation='sin'`创建模型
- [ ] 前向传播正常工作
- [ ] `get_model_info()`包含activation字段

---

## 🧪 测试模板

重构完成后，使用以下测试代码验证：

```python
import torch
from autoencoder.models.XXX import YourModel

print('=== Testing YourModel ===')

# Test 1: 创建模型with different activations
for act in ['relu', 'sin', 'gelu']:
    model = YourModel(latent_dim=32, activation=act, num_frequencies=2)
    info = model.get_model_info()
    print(f'{act}: activation={info["activation"]}, params={info["total_params"]:,}')

    # Test forward pass
    x = torch.randn(2, 91, 91, 2)  # or appropriate input shape
    recon, latent = model(x)
    print(f'  Input {x.shape} -> Latent {latent.shape} -> Recon {recon.shape}')

# Test 2: 向后兼容（如果有Sine别名）
if hasattr(module, 'SinYourModel'):
    model = module.SinYourModel(latent_dim=32, num_frequencies=2)
    info = model.get_model_info()
    assert info['activation'] == 'sin', "Sine alias should have activation='sin'"
    print('Backward compatibility: OK')

print('=== All Tests Passed! ===')
```

---

## 🚨 常见错误和修复

### 错误1: NameError: name 'get_activation' is not defined

**原因**: 忘记导入激活函数工厂

**修复**: 在文件顶部添加
```python
from autoencoder.utils.activation_factory import get_activation, get_activation_name
```

### 错误2: TypeError: __init__() got an unexpected keyword argument 'activation'

**原因**:
1. 忘记在`__init__`中添加`activation`参数
2. Sine别名类中，父类还未重构

**修复**:
1. 添加`activation: str = 'relu'`到`__init__`参数列表
2. 确保先重构父类，再重构Sine别名

### 错误3: 重构Sine别名后，旧代码还在

**原因**: 删除重复代码不彻底

**修复**: Sine别名类应该只有：
```python
def __init__(self, *args, **kwargs):
    kwargs['activation'] = 'sin'
    super().__init__(*args, **kwargs)

def get_model_info(self) -> Dict[str, Any]:  # 可选
    info = super().get_model_info()
    info['type'] = 'SinXXXAutoEncoder'
    return info
```

**删除以下所有内容**:
- 重复的`self.encoder`定义
- 重复的`self.decoder`定义
- 重复的`encode()`和`decode()`方法
- 重复的属性初始化

### 错误4: Decoder输出异常（全是NaN或很大的值）

**原因**: 误在decoder最后一层添加了激活函数

**修复**: Decoder最后一层**不应该**有激活函数
```python
nn.ConvTranspose2d(64, self.input_channels, kernel_size=3, stride=2, padding=1),
# ← 这里不要加get_activation()！
```

### 错误5: 模型信息中缺少activation字段

**原因**: 忘记在`get_model_info()`中添加

**修复**:
```python
def get_model_info(self) -> Dict[str, Any]:
    return {
        ...
        'activation': self.activation_type,  # ← 添加这行
        ...
    }
```

---

## 📂 文件清单和优先级

### 高优先级（用户主要使用）

1. **dual_branch_differentiable_autoencoder.py** ⭐⭐⭐
   - DualBranchDifferentiableWaveletAutoEncoder (CNN)
   - DualBranchDifferentiableWaveletMLPAutoEncoder (MLP)
   - **重要**: 用户优先使用可微分小波

2. **cnn_autoencoder.py** ⭐⭐
   - WaveletAutoEncoder (CNN)

3. **mlp_autoencoder.py** ⭐⭐
   - WaveletMLPAutoEncoder (MLP)

4. **dual_branch_autoencoder.py** ⭐⭐
   - DualBranchWaveletAutoEncoder (CNN)
   - DualBranchWaveletMLPAutoEncoder (MLP)

### 中优先级

5. **direct_autoencoder.py** ⭐
   - DirectAutoEncoder (CNN)

6. **mlp_autoencoder.py** ⭐
   - DirectMLPAutoEncoder (MLP)

7. **enhanced_cnn_autoencoder.py** ⭐
   - EnhancedWaveletAutoEncoder (CNN)
   - EnhancedDirectAutoEncoder (CNN)

8. **deep_autoencoder.py** ⭐
   - DeepWaveletAutoEncoder (CNN)
   - DeepDirectAutoEncoder (CNN)

### 低优先级（别名化）

9. **sine_cnn_autoencoder.py**
   - SinWaveletAutoEncoder → 别名
   - SinDirectAutoEncoder → 别名

10. **sine_mlp_autoencoder.py**
    - SinWaveletMLPAutoEncoder → 别名
    - SinDirectMLPAutoEncoder → 别名

---

## 📊 进度追踪

### 已完成 ✅ (3/19)
- [x] DifferentiableWaveletAutoEncoder (CNN)
- [x] DifferentiableWaveletMLPAutoEncoder (MLP)
- [x] DifferentiableSineWaveletMLPAutoEncoder (别名)

### 待重构 ⏳ (16/19)

#### 可微分小波系列 (2个)
- [ ] DualBranchDifferentiableWaveletAutoEncoder (CNN)
- [ ] DualBranchDifferentiableWaveletMLPAutoEncoder (MLP)

#### Wavelet模式 (6个)
- [ ] WaveletAutoEncoder (CNN)
- [ ] WaveletMLPAutoEncoder (MLP)
- [ ] SinWaveletAutoEncoder (别名)
- [ ] SinWaveletMLPAutoEncoder (别名)
- [ ] DualBranchWaveletAutoEncoder (CNN)
- [ ] DualBranchWaveletMLPAutoEncoder (MLP)

#### Direct模式 (4个)
- [ ] DirectAutoEncoder (CNN)
- [ ] DirectMLPAutoEncoder (MLP)
- [ ] SinDirectAutoEncoder (别名)
- [ ] SinDirectMLPAutoEncoder (别名)

#### Enhanced系列 (2个)
- [ ] EnhancedWaveletAutoEncoder (CNN)
- [ ] EnhancedDirectAutoEncoder (CNN)

#### Deep系列 (2个)
- [ ] DeepWaveletAutoEncoder (CNN)
- [ ] DeepDirectAutoEncoder (CNN)

---

## 🎯 Claude后续工作（第3-5阶段）

Codex完成所有模型重构后，Claude将负责：

1. **检查重构质量**
   - 验证所有模型测试通过
   - 检查代码一致性
   - 修复发现的问题

2. **更新frequency_config.py**
   - 添加activation参数支持
   - 更新所有create_autoencoder_system调用
   - 示例: `WaveletAutoEncoder(activation=activation, ...)`

3. **更新GUI (gui_autoencoder_extension.py)**
   - 添加激活函数选择下拉框
   - 选项: ["ReLU", "Sin", "GELU", "Swish", "Tanh"]
   - 传递activation参数到模型创建

4. **更新models/__init__.py**
   - 确保所有别名正确导出
   - 添加注释说明向后兼容性

5. **更新README.md**
   - 移除独立的Sine行
   - 在表格中添加"支持激活函数"列
   - 更新使用示例

6. **标记废弃Sine文件**
   - 在sine_cnn_autoencoder.py添加@deprecated注释
   - 在sine_mlp_autoencoder.py添加@deprecated注释
   - 提交最终版本

---

## 💡 最佳实践

1. **一次重构一个文件**，完成后立即测试
2. **先重构父类，再重构Sine别名**
3. **保持代码格式一致**（缩进、空行等）
4. **复制粘贴时注意变量名**（不要出现WaveletXXX in DirectXXX）
5. **git commit时写清楚重构了哪个文件**

---

## 📧 问题反馈

如果遇到问题或不确定的地方，在重构完成后告知Claude，我会检查并修复。

**常见需要Claude检查的情况**:
- 模型测试失败
- 不确定某段代码是否需要修改
- Decoder最后一层激活函数的处理
- 复杂的动态构建逻辑

---

**准备好了吗？开始重构吧！** 🚀

记住：**按照文件清单的顺序，一个一个来！**
