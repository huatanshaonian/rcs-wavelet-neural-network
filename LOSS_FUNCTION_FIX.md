# 损失函数关键修复说明

## 问题描述

用户报告训练后出现严重问题：
- 训练loss可以降到10^-3水平（看起来很好）
- 但预测结果全是-100dB（完全错误）
- R²值为负数且极小（10^-9量级，说明模型完全无预测能力）

## 根本原因分析

### 1. 对称性损失的设计缺陷

**原始设计**：
```python
def _symmetry_loss(self, pred_rcs: torch.Tensor) -> torch.Tensor:
    # 只要求预测值自身左右对称
    for i in range(1, center_phi + 1):
        left_values = pred_rcs[:, :, left_idx, :]
        right_values = pred_rcs[:, :, right_idx, :]
        symmetry_loss += mse_loss(left_values, right_values)
```

**问题**：
- 只要求预测值自身对称，**不管是否接近真实RCS**
- 如果网络输出全是同一个常数，对称性loss = 0（完美对称！）
- 这导致网络学会输出"对称但错误"的值

### 2. 损失权重配置问题

**原始权重**（渐进式）：
```python
# 早期epoch
'mse': 1.0
'symmetry': 0.3  # 太高！
'multiscale': 0.5
```

**问题**：
- 对称性损失权重0.3，在早期训练中**主导了优化方向**
- 网络学习策略：输出一个对称的常数，让对称性loss=0
- 即使MSE loss较高，总loss仍可以很低

### 3. 导致的训练病态

**网络学习到错误模式**：
1. 输出全是标准化后的某个负值（比如-10）
2. 对称性loss = 0（完美）
3. 多尺度loss很低（所有尺度都是常数）
4. MSE loss虽然存在，但被对称性loss主导
5. 总loss下降，但预测完全错误

**为什么是-100dB**：
- 网络输出：标准化后的某个负值
- 反标准化：value_db = value_std * std + mean
- 如果网络学会了输出极小值，反标准化后就是-100dB量级

## 解决方案

### 1. 改进对称性损失定义

**新设计**：
```python
def _symmetry_loss(self, pred_rcs: torch.Tensor, target_rcs: torch.Tensor) -> torch.Tensor:
    # 计算预测和目标的对称性差异
    pred_sym_diff = pred_left - pred_right
    target_sym_diff = target_left - target_right

    # 要求预测的对称性差异匹配目标的对称性差异
    symmetry_loss += mse_loss(pred_sym_diff, target_sym_diff)
```

**优势**：
- 现在对称性损失与target相关
- 网络无法通过输出常数来"欺骗"对称性损失
- 对称性损失真正成为MSE的补充，而非替代

### 2. 显著降低对称性权重

**新权重**（渐进式）：
```python
# 早期epoch
'mse': 1.0
'symmetry': 0.05  # 从0.3降到0.05
'multiscale': 0.3
```

**默认权重**：
```python
{
    'mse': 1.0,
    'symmetry': 0.02,  # 从0.1降到0.02
    'multiscale': 0.1
}
```

**理由**：
- MSE应该是主导损失
- 对称性只是辅助约束，不应主导训练
- 权重降低5-10倍，确保MSE优先

## 修复文件清单

1. **wavelet_network.py**:
   - 修改 `_symmetry_loss()` 函数，增加target参数
   - 计算pred和target的对称性差异匹配
   - 降低默认对称性权重：0.1 → 0.02

2. **training.py**:
   - 修改 `_create_progressive_loss_weights()` 函数
   - 降低渐进式对称性权重：0.3-0.2*progress → 0.05-0.03*progress
   - 降低默认对称性权重：0.1 → 0.02

## 预期效果

修复后的训练应该：
1. MSE损失主导训练，确保预测接近真实RCS
2. 对称性损失作为辅助约束，微调对称性
3. 网络输出合理的RCS值（不会是-100dB）
4. R²值应该为正数且较高
5. loss和实际预测质量相关

## 训练建议

使用修复后的代码重新训练：
```bash
python main.py --mode train
```

或使用GUI：
1. 启动GUI：`python main.py --mode gui`
2. 确认对数预处理已启用
3. 检查学习率设置（初始LR=0.003, min_lr=2e-5）
4. 开始训练并观察：
   - loss应该稳定下降
   - 预测RCS应该在合理范围（不是-100dB）
   - R²应该为正数

## 总结

这是一个**损失函数设计缺陷**导致的训练失败案例：
- 辅助损失（对称性）权重过高
- 辅助损失设计不当（不依赖target）
- 导致网络学习错误模式
- loss下降但预测失败

修复的核心：
1. 对称性损失必须基于target
2. MSE必须是主导损失
3. 辅助损失只能是补充，不能主导