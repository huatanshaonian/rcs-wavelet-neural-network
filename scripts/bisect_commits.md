# 从 aade9e7 到 HEAD 的提交列表（用于二分查找问题）

## 完整提交列表（时间顺序）

以下是从 aade9e7 之后到当前的所有 18 个提交：

```
基准 → aade9e7  fix(model): 修复联合训练模型加载时mapper配置丢失问题

1.  798e2c0  refactor: 清理根目录并优化项目结构
2.  5f46952  fix(ae): 修复联合训练模式评估失败问题
3.  a4c6803  feat(ae): 添加RCS非负物理约束选项                    ⚠️ 可疑
4.  0121be0  fix(ae): 使用Softplus替代ReLU，避免dying ReLU问题    ⚠️ 高度可疑
5.  bd34e38  feat(loss): 添加均值损失函数提升统计特性学习
6.  ea8f349  feat(ae): 训练时在RCS空间应用非负约束               ⚠️ 可疑
7.  753a245  feat(ae): 为Joint Training添加RCS非负约束支持
8.  2a82147  feat(ae): 实现叠加型双分支AutoEncoder架构
9.  a4f4e3a  Fix GUI layout bugs and integrate Additive Dual-Branch features
10. 6efb75c  fix(ae): 增强逆标准化健壮性和修复损失函数调用路径    ⚠️ 可疑
11. a25b40b  feat(ae): 联合训练支持绘制训练进度图
12. 6d3170e  fix(model-io): 修复enforce_nonnegative_rcs配置未保存的Bug  ⚠️ 可疑
13. e2ae8aa  fix(gui): 修复enforce_nonnegative_rcs默认值和参数传递      ⚠️ 高度可疑
14. 0f2f152  Revert "fix(gui): 修复enforce_nonnegative_rcs默认值和参数传递"
15. f6fae0a  Revert "fix(model-io): 修复enforce_nonnegative_rcs配置未保存的Bug"
16. 60e170a  Revert "feat(ae): 为Joint Training添加RCS非负约束支持"
17. 55d04cf  Revert "feat(ae): 训练时在RCS空间应用非负约束（消除train-test mismatch）"
18. 945560b  Revert "feat(ae): 添加RCS非负物理约束选项"

当前 → HEAD (945560b)
```

## 高度可疑提交（优先测试）

### 🔴 #13: e2ae8aa - 修复enforce_nonnegative_rcs默认值和参数传递
**最可能的罪魁祸首！**

- **问题**: 这个提交修改了旧模型加载时的默认行为
- **可能影响**: 旧模型没有 `enforce_nonnegative_rcs` 配置，加载时默认值可能设置错误
- **文件**: `gui.py`, `gui_autoencoder_extension.py`

### 🟠 #12: 6d3170e - 修复enforce_nonnegative_rcs配置未保存的Bug
**次要可疑**

- **问题**: 修改了模型保存逻辑
- **可能影响**: 如果加载逻辑依赖这个配置，可能导致行为不一致

### 🟠 #4: 0121be0 - 使用Softplus替代ReLU
**直接改变输出的提交**

- **问题**: 将 ReLU 改为 Softplus
- **影响**: 如果 `enforce_nonnegative_rcs=True`，输出会被 Softplus 压缩
- **文件**: `reconstruction_manager.py`

### 🟠 #10: 6efb75c - 增强逆标准化健壮性和修复损失函数调用路径
**可能影响数据处理**

- **问题**: 修改了逆标准化逻辑
- **可能影响**: 如果逆标准化有bug，重建结果会不同

## 测试方法

### 方法1: 快速定位（推荐）

直接测试最可疑的 3 个提交：

```bash
# 1. 测试 e2ae8aa (最可疑)
git checkout e2ae8aa
python -c "你的测试代码"  # 记录结果

# 2. 测试 6d3170e
git checkout 6d3170e
python -c "你的测试代码"  # 记录结果

# 3. 测试 0121be0
git checkout 0121be0
python -c "你的测试代码"  # 记录结果

# 4. 测试 6efb75c
git checkout 6efb75c
python -c "你的测试代码"  # 记录结果

# 恢复到当前
git checkout dev_ae
```

### 方法2: 二分查找（系统化）

如果快速测试无法定位，使用 git bisect：

```bash
# 1. 开始二分查找
git bisect start

# 2. 标记当前版本为坏的（重建结果不对）
git bisect bad HEAD

# 3. 标记 aade9e7 为好的（重建结果正确）
git bisect good aade9e7

# 4. Git 会自动 checkout 到中间版本，测试后标记
python your_test_script.py
# 如果结果正确: git bisect good
# 如果结果错误: git bisect bad

# 5. 重复步骤4，直到找到第一个坏提交

# 6. 完成后重置
git bisect reset
```

### 方法3: 逐个测试（最全面）

```bash
# 创建测试脚本
cat > test_commits.sh << 'EOF'
#!/bin/bash

commits=(
    "798e2c0"
    "5f46952"
    "a4c6803"
    "0121be0"
    "bd34e38"
    "ea8f349"
    "753a245"
    "2a82147"
    "a4f4e3a"
    "6efb75c"
    "a25b40b"
    "6d3170e"
    "e2ae8aa"
    "0f2f152"
    "f6fae0a"
    "60e170a"
    "55d04cf"
    "945560b"
)

for commit in "${commits[@]}"; do
    echo "Testing commit: $commit"
    git checkout $commit
    python your_test_script.py
    echo "Press Enter to continue..."
    read
done

git checkout dev_ae
EOF

chmod +x test_commits.sh
./test_commits.sh
```

## 测试脚本模板

创建 `scripts/test_model_reconstruction.py`：

```python
"""测试模型重建结果的脚本"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from autoencoder.utils.frequency_config import create_autoencoder_system

# 模型路径
MODEL_PATH = 'G:/feko_data/wavelet/models/direct_mlp_sin_raw_20251221_220849.pth'

def test_reconstruction():
    # 加载模型
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    config = checkpoint['config']

    # 创建系统
    ae_system = create_autoencoder_system(
        config_name=config.get('config_name', '2freq'),
        mode=config.get('mode', 'direct'),
        architecture=config.get('architecture', 'mlp'),
        latent_dim=config.get('latent_dim', 64),
        activation=config.get('activation', 'sin'),
        dropout_rate=config.get('dropout_rate', 0.2),
        normalize=config.get('normalize', False),
        db_transform=config.get('db_transform', False),
        normalization_method=config.get('normalization_method', 'none')
    )

    # 加载权重
    ae_system['autoencoder'].load_state_dict(checkpoint['autoencoder'])
    ae_system['autoencoder'].eval()

    # 创建固定的测试输入（确保每次相同）
    torch.manual_seed(42)
    test_input = torch.randn(1, 91*91*2)

    # 推理
    with torch.no_grad():
        reconstructed, latent = ae_system['autoencoder'](test_input)

    # 输出关键统计信息
    print(f"重建值范围: [{reconstructed.min():.8f}, {reconstructed.max():.8f}]")
    print(f"重建值均值: {reconstructed.mean():.8f}")
    print(f"重建值标准差: {reconstructed.std():.8f}")
    print(f"负值数量: {(reconstructed < 0).sum().item()}")
    print(f"前10个值: {reconstructed[0, :10].tolist()}")

    return reconstructed.numpy()

if __name__ == '__main__':
    result = test_reconstruction()
```

## 预期结果

- **aade9e7（好的）**: 重建结果正常，可能包含负值
- **某个提交（坏的）**: 重建结果异常，可能全为正值或值域被压缩

## 关键检查点

在每个提交测试时，记录：
1. ✅ 重建值范围（特别是最小值是否<0）
2. ✅ 前10个重建值（用于精确对比）
3. ✅ 是否有报错或警告信息
4. ✅ 是否看到 "enforce_nonnegative_rcs" 相关日志

## 预测

我强烈怀疑问题出在 **e2ae8aa** 或 **0121be0**：

- **e2ae8aa**: 修改了旧模型加载时的默认值处理
- **0121be0**: 将约束从 ReLU 改为 Softplus

如果在这两个提交中发现问题，我可以立即提供修复方案。
