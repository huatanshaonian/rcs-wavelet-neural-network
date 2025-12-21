# JSON参数空间批量训练使用指南

> 实现日期: 2025-01-18  
> 功能: 支持通过JSON配置文件定义任意超参数组合进行批量训练

---

## 功能概述

JSON批量训练功能允许你通过JSON文件定义实验参数空间，自动生成所有实验组合并批量执行训练。

**核心优势**：
- ✅ **任意超参数**: JSON中任何参数都会传递给训练器，无需修改代码
- ✅ **完全可扩展**: 新增超参数时只需修改JSON，无需改动程序
- ✅ **配置即代码**: JSON文件可版本控制、分享、复现
- ✅ **GUI集成**: 支持从GUI直接导入JSON并执行
- ✅ **独立运行**: 支持CLI脚本运行，无需GUI

---

## 快速开始

### 1. 从模板开始

系统已生成配置模板：`experiments/template.json`

```json
{
  "experiment_name": "example_experiment",
  "description": "示例实验：探索隐空间维度和学习率的影响",
  
  "base_config": {
    "mode": "wavelet",
    "architecture": "cnn",
    "latent_dim": 256,
    "batch_size": 8,
    "learning_rate": 0.001,
    "epochs": {"stage1": 100, "stage2": 50, "stage3": 30}
  },
  
  "parameter_grid": {
    "latent_dim": [128, 256, 512],
    "learning_rate": [0.001, 0.0001],
    "dropout_rate": [0.1, 0.2, 0.3]
  }
}
```

**这将生成 3 × 2 × 3 = 18 个实验！**

### 2. GUI使用方法

1. 打开主GUI → 【批量实验】标签页
2. 点击"📂 导入JSON配置"按钮
3. 选择你的JSON配置文件
4. 点击"▶ 开始批量训练"

系统会：
- 自动创建所有实验配置
- 逐个训练模型
- 保存所有模型和日志
- 显示训练进度和指标

### 3. CLI使用方法

```python
# simple_batch.py
from autoencoder.training import batch_train_from_json
import numpy as np

# 加载数据
rcs_data = np.load('data/rcs_data.npy')
param_data = np.load('data/param_data.npy')

# 批量训练
results = batch_train_from_json(
    json_path='experiments/my_experiment.json',
    rcs_data=rcs_data,
    param_data=param_data
)

# 查看结果
for i, result in enumerate(results):
    if result['success']:
        print(f"{i}: {result['final_metrics']}")
```

---

## JSON配置详解

### 结构说明

```json
{
  "experiment_name": "实验名称（用于目录命名）",
  "description": "实验描述（可选）",
  
  "base_config": {
    // 固定参数：所有实验共享
  },
  
  "parameter_grid": {
    // 搜索空间：列表值会生成笛卡尔积
  }
}
```

### base_config（必需参数）

```json
{
  "mode": "wavelet | direct | differentiable_wavelet",
  "architecture": "cnn | mlp | enhanced_cnn | deep_cnn",
  "latent_dim": 256,
  "batch_size": 8,
  "learning_rate": 0.001,
  "training_mode": "three_stage | stage1_only",
  "epochs": {
    "stage1": 100,
    "stage2": 50,
    "stage3": 30
  }
}
```

### base_config（可选参数）

```json
{
  // 激活函数
  "activation": "relu | sin | gelu | swish | tanh | mish | elu",
  
  // 数据预处理
  "normalization_method": "zscore | minmax",
  "db_transform": true | false,
  "wavelet_type": "db4 | db8 | haar | bior2.2",
  
  // 优化器
  "optimizer_type": "adam | adamw | sgd | lbfgs",
  "lr_scheduler": "plateau | cosine | multi_stage | adaptive_multi_stage",
  "momentum": 0.9,
  
  // 正则化
  "dropout_rate": 0.2,
  
  // 高级功能
  "gradient_monitoring": false,
  "channel_attention": false,
  "loss_normalization": false,
  "ssim_weight": 0.0,
  "num_lr_stages": 3,
  "lr_decay_factor": 0.1
}
```

### parameter_grid（搜索空间）

parameter_grid中的参数会覆盖base_config，并生成笛卡尔积：

```json
{
  "latent_dim": [64, 128, 256, 512],      // 4个选项
  "learning_rate": [0.001, 0.0001],       // 2个选项
  "dropout_rate": [0.1, 0.2, 0.3]         // 3个选项
}
// 总实验数: 4 × 2 × 3 = 24
```

**支持所有参数**：任何base_config中的参数都可以放入parameter_grid！

---

## 实战示例

### 示例 1: 隐空间维度搜索

```json
{
  "experiment_name": "latent_dim_search",
  "base_config": {
    "mode": "wavelet",
    "architecture": "cnn",
    "batch_size": 8,
    "learning_rate": 0.001,
    "epochs": {"stage1": 100, "stage2": 50, "stage3": 30}
  },
  "parameter_grid": {
    "latent_dim": [32, 64, 128, 256, 512, 1024]
  }
}
```

### 示例 2: 激活函数对比

```json
{
  "experiment_name": "activation_comparison",
  "base_config": {
    "mode": "wavelet",
    "architecture": "cnn",
    "latent_dim": 256
  },
  "parameter_grid": {
    "activation": ["relu", "sin", "gelu", "swish", "tanh", "mish"]
  }
}
```

### 示例 3: 多维度网格搜索

```json
{
  "experiment_name": "grid_search",
  "parameter_grid": {
    "latent_dim": [128, 256],
    "learning_rate": [0.001, 0.0001],
    "batch_size": [8, 16],
    "dropout_rate": [0.1, 0.2, 0.3],
    "lr_scheduler": ["cosine", "plateau"]
  }
}
// 总实验数: 2 × 2 × 2 × 3 × 2 = 48
```

### 示例 4: 优化器对比

```json
{
  "experiment_name": "optimizer_comparison",
  "base_config": {
    "latent_dim": 256,
    "batch_size": 16,
    "epochs": {"stage1": 50, "stage2": 30, "stage3": 20}
  },
  "parameter_grid": {
    "optimizer_type": ["adam", "adamw", "sgd", "lbfgs"],
    "learning_rate": [0.001, 0.0001]
  }
}
```

---

## 结果目录结构

```
batch_experiments/
└── experiment_name_20250118_143022/
    ├── experiment_config.json          # 实验配置备份
    ├── batch_training_20250118_143022.log  # 训练日志
    │
    ├── models/                         # 模型文件
    │   ├── exp000_ld128_lr1e-03.pth
    │   ├── exp001_ld128_lr1e-04.pth
    │   └── ...
    │
    ├── training_plots/                 # ✨ 训练进度图（JSON模式自动生成）
    │   ├── exp000_ld128_lr1e-03_training_progress.png
    │   ├── exp001_ld128_lr1e-04_training_progress.png
    │   └── ... (每个实验1张，显示3阶段训练曲线)
    │
    ├── rcs_comparison/                 # ✨ RCS重建对比图（JSON模式自动生成）
    │   ├── exp000_ld128_lr1e-03_train_sample000_freq0.png
    │   ├── exp000_ld128_lr1e-03_train_sample001_freq0.png
    │   ├── exp000_ld128_lr1e-03_train_sample002_freq0.png
    │   ├── exp000_ld128_lr1e-03_test_sample400_freq0.png
    │   ├── exp000_ld128_lr1e-03_test_sample401_freq0.png
    │   ├── exp000_ld128_lr1e-03_test_sample402_freq0.png
    │   └── ... (每实验: 3训练样本+3测试样本 × 2频率 = 12张图)
    │
    └── comparison_plots/               # GUI模式对比图表（仅GUI模式）
        ├── loss_curves.png
        ├── metrics_bar.png
        └── ...
```

### 可视化图表说明

**训练进度图** (`training_plots/`)：
- 每个实验1张图
- 显示3个训练阶段的Train/Val Loss曲线
- 对数刻度，标注最佳epoch
- 快速了解训练收敛情况

**RCS重建对比图** (`rcs_comparison/`)：
- 每个实验12张图（3训练+3测试 × 2频率）
- 每张图包含：真实RCS、预测RCS、残差图
- 直观评估模型重建性能
- 训练集样本：验证过拟合
- 测试集样本：验证泛化能力


---

## 高级用法

### 1. 单个实验训练（CLI）

```python
from autoencoder.training import train_from_json

# 只训练第一个实验
result = train_from_json(
    json_path='experiments/my_experiment.json',
    rcs_data=rcs_data,
    param_data=param_data,
    experiment_index=0  # 使用第0个实验配置
)
```

### 2. 自定义日志

```python
from autoencoder.training import train_autoencoder_standalone

# 自定义日志函数
def my_log(message):
    print(f"[{datetime.now()}] {message}")
    # 或写入自定义日志文件

result = train_autoencoder_standalone(
    rcs_data=rcs_data,
    param_data=param_data,
    training_config=config,
    log_callback=my_log
)
```

### 3. 进度回调

```python
def progress_callback(stage, epoch, total):
    print(f"{stage}: Epoch {epoch}/{total}")

result = train_autoencoder_standalone(
    rcs_data=rcs_data,
    param_data=param_data,
    training_config=config,
    progress_callback=progress_callback
)
```

---

## 可扩展性示例

**新增超参数无需修改代码！**

假设未来添加了新参数`new_feature_enabled`：

```json
{
  "base_config": {
    "mode": "wavelet",
    "new_feature_enabled": true  // ← 新参数
  },
  "parameter_grid": {
    "new_feature_enabled": [true, false]  // ← 直接搜索
  }
}
```

系统会自动：
1. 解析JSON中的所有参数
2. 传递给训练器
3. 训练器传递给create_autoencoder_system
4. 最终传递给模型构造函数

**完全无缝！**

---

## 常见问题

### Q: JSON中可以包含哪些参数？

A: 理论上任何参数！只要训练流程支持，就可以放入JSON。系统不做硬编码限制。

### Q: 如何查看实验进度？

A: 
- GUI模式：实时显示在右侧进度面板
- CLI模式：终端输出 + `batch_training_*.log`文件

### Q: 实验失败会怎样？

A: 单个实验失败不影响其他实验。结果中会标记失败并记录错误信息。

### Q: 如何选择最佳模型？

A: 查看`batch_training_*.log`或结果JSON中的final_metrics，比较各实验的MSE/RMSE等指标。

### Q: JSON模式和GUI模式有什么区别？

A:
- **JSON模式**:
  - 使用standalone_trainer，完全独立，支持CLI
  - 自动生成训练进度图 + RCS重建对比图（每个实验）
  - 适合：超参数搜索、大量实验、自动化脚本

- **GUI模式**:
  - 使用BatchExperimentManager，GUI操作
  - 除了单实验可视化，还生成6种跨实验对比图表
  - 适合：架构对比、深度分析、交互式探索

### Q: 如何快速查看实验结果？

A:
1. **查看训练日志**：`batch_training_*.log`（所有实验指标汇总）
2. **查看训练进度图**：`training_plots/` 目录（每个实验的收敛曲线）
3. **查看RCS重建效果**：`rcs_comparison/` 目录（真实vs预测对比）
4. **选择最佳模型**：根据测试集样本的重建质量和final_metrics

---

## 参考文件

- **JSON解析器**: `autoencoder/utils/json_experiment.py`
- **独立训练函数**: `autoencoder/training/standalone_trainer.py`
- **GUI集成**: `gui_batch_experiment_extension.py`
- **配置模板**: `experiments/template.json`

---

**享受高效的超参数搜索！** 🚀
