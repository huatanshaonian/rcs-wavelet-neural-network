# RCS小波神经网络预测系统

基于小波多尺度理论的飞行器RCS（雷达散射截面）预测系统，使用深度学习技术从9个飞行器参数预测多频RCS分布数据。

## 🚀 项目特色

- **小波多尺度架构**: 在φ-θ平面使用2D小波变换，支持4个不同尺度的特征提取
- **多频率数据支持**: 完整支持2频(1.5GHz+3GHz)和3频(1.5GHz+3GHz+6GHz)配置 [91×91×2/3]
- **双模式AutoEncoder系统**:
  - **小波增强模式**: 结合小波变换的深度压缩，256维隐空间
  - **直接CNN模式**: 纯卷积架构，更快推理速度
  - **三阶段训练**: AE预训练 → 参数映射 → 端到端微调
- **物理约束**: 集成φ=0°平面对称性约束和频率一致性损失
- **插件化架构**: 零修改添加新网络，支持original/enhanced/自定义架构
- **小数据集优化**: 针对~100样本设计的数据增强和交叉验证策略
- **完整工作流**: 数据加载→训练→评估→预测→可视化的完整pipeline
- **现代化GUI**: 基于tkinter的直观操作界面，支持多频配置和小波分析

## 📁 项目结构与文件说明

### 核心程序文件

#### 主程序入口
- **`gui.py`** - 图形界面主程序
  - 基于tkinter的完整GUI系统
  - 集成数据管理、模型训练、评估、可视化功能
  - 管理AutoEncoder和传统网络两套系统
  - 提供频率配置验证和会话时间戳管理

- **`main.py`** - 命令行工具入口
  - 支持训练、评估、预测三种模式
  - 命令行参数配置
  - 适合脚本化和批处理任务

#### 数据处理模块
- **`rcs_data_reader.py`** - RCS数据读取器
  - 从CSV文件加载RCS矩阵数据（91×91）
  - 支持多频率数据读取（1.5G/3G/6G）
  - 提供线性域和dB域转换

- **`data_cache.py`** - 数据缓存管理器
  - 智能缓存机制，加速数据加载
  - 支持2频/3频配置的缓存管理
  - 自动验证缓存有效性

- **`rcs_visual.py`** - RCS数据可视化
  - 2D热图、3D表面图、球坐标图
  - 多模型对比可视化
  - 集成到GUI和命令行工具

#### 网络架构系统

##### 传统小波网络
- **`wavelet_network.py`** - 小波神经网络核心
  - TriDimensionalRCSNet：基础小波多尺度网络
  - 支持2D小波变换特征提取
  - 包含对称性和频率一致性约束

- **`enhanced_network.py`** - 增强版小波网络
  - EnhancedTriDimensionalRCSNet：改进架构
  - 增强的特征提取和跨频率交互
  - 优化的损失函数（ImprovedRCSLoss）

##### 插件化架构系统
- **`network_registry.py`** - 网络注册中心
  - BaseNetwork/BaseLoss基类定义
  - 网络和损失函数的注册机制
  - 自动发现和加载插件
  - 避免重复注册，优化启动速度

- **`modern_wavelet_network.py`** - 现代化网络接口
  - 统一的网络创建接口
  - 兼容传统和插件化两种模式
  - 集成网络注册系统

- **`networks/example_networks.py`** - 示例网络实现
  - WaveletRCSNetwork：小波多尺度网络
  - SimpleFCNetwork：全连接基线网络
  - ResNetRCSNetwork：残差连接网络
  - FlexibleOutputNetwork：灵活输出网络

#### 损失函数系统
- **`configurable_loss.py`** - 可配置损失函数
  - ConfigurableLoss：支持多组件损失
  - 物理约束损失（对称性、频率一致性）
  - 动态权重调整

### AutoEncoder系统 (`autoencoder/`)

#### 模型定义 (`models/`)
- **`cnn_autoencoder.py`** - CNN AutoEncoder（标准版）
  - WaveletAutoEncoder：小波增强CNN架构
  - 支持2频/3频配置
  - 编码器-解码器对称结构

- **`direct_autoencoder.py`** - 直接CNN AutoEncoder
  - DirectAutoEncoder：无小波变换的纯CNN
  - 直接处理91×91×N数据
  - 更快的推理速度

- **`mlp_autoencoder.py`** - MLP AutoEncoder
  - WaveletMLPAutoEncoder：小波+MLP架构
  - DirectMLPAutoEncoder：纯MLP架构
  - 适用于参数敏感性分析

- **`enhanced_cnn_autoencoder.py`** - 增强感受野CNN
  - EnhancedWaveletAutoEncoder：多尺度卷积
  - EnhancedDirectAutoEncoder：增强感受野
  - 更好的全局特征捕捉

- **`parameter_mapper.py`** - 参数映射器
  - ParameterMapper：9维参数→隐空间映射
  - 多层全连接网络
  - 支持Stage2和Stage3训练

- **`correct_cnn_autoencoder.py`** - 修正版CNN（实验性）
- **`deep_cnn_autoencoder.py`** - 深层CNN（实验性）
- **`efficient_cnn_autoencoder.py`** - 高效CNN（实验性）
- **`micro_latent_autoencoder.py`** - 微隐空间AE（实验性）

#### 训练模块 (`training/`)
- **`ae_trainer.py`** - AutoEncoder训练器
  - 三阶段训练流程实现
  - Stage1：AutoEncoder预训练
  - Stage2：ParameterMapper训练（固定AE）
  - Stage3：端到端微调
  - 训练历史记录和Early Stopping

#### 评估模块 (`evaluation/`)
- **`ae_evaluator.py`** - AutoEncoder评估器
  - 重建质量评估（RMSE、R²、相关系数）
  - 频率分离评估
  - 对比图生成

- **`reconstruction_metrics.py`** - 重建指标计算
  - SSIM（结构相似性）
  - PSNR（峰值信噪比）
  - 多频率统计分析

#### 工具模块 (`utils/`)
- **`frequency_config.py`** - 频率配置系统
  - FrequencyConfig：2freq/3freq配置管理
  - create_autoencoder_system()：一键创建完整系统
  - 自动检测和验证频率配置

- **`correct_wavelet_transform.py`** - 小波变换工具
  - CorrectWaveletTransform：正确的2D小波变换
  - 支持db4/db6/db8等小波基
  - 前向变换和逆变换

- **`data_adapters.py`** - 数据适配器
  - RCS_DataAdapter：数据标准化和预处理
  - 支持对数变换和归一化
  - 频率数量验证

- **`comparison_system.py`** - 对比分析系统
  - 双系统性能对比
  - 统计分析和可视化
  - 保存对比报告

### GUI扩展模块
- **`gui_autoencoder_extension.py`** - AutoEncoder GUI扩展
  - AE页面的完整实现
  - 系统创建、训练、评估界面
  - 小波分析和可视化面板

- **`gui_training_config.py`** - 训练配置GUI
  - 训练参数配置界面
  - 三阶段训练参数设置
  - 配置保存和加载

- **`wavelet_gui_helper.py`** - 小波分析GUI辅助
  - 小波变换可视化
  - 系数分析和展示
  - 集成到主GUI

### 训练和评估框架
- **`training.py`** - 传统网络训练框架
  - RCSDataset：数据集封装
  - CrossValidationTrainer：交叉验证训练器
  - 物理约束损失集成
  - 早停和学习率调度

- **`evaluation.py`** - 传统网络评估模块
  - RCSEvaluator：全面评估工具
  - 线性域和dB域指标
  - 对称性和一致性分析
  - 生成评估报告

### 工具和测试脚本
- **`unicode_fix.py`** - Unicode编码修复
  - Windows平台Unicode输出修复
  - 支持表情符号和中文字符

- **`fix_ae112_config.py`** - 模型配置修复工具
  - 修复旧模型的config信息
  - 推断缺失的配置参数
  - 生成修复后的模型文件

- **`debug_ae_training.py`** - AE训练调试脚本
  - 独立测试AE训练流程
  - 验证数据流和梯度

- **`test_enhanced_cnn.py`** - 增强CNN测试
  - 测试增强感受野CNN
  - 性能基准测试

- **`verify_loss.py`** - 损失函数验证
  - 验证损失函数计算正确性
  - 测试物理约束

- **`cnn_convolution_explanation.py`** - CNN卷积说明
  - CNN架构可视化说明
  - 卷积层参数解释

- **`wavelet_size_explanation.py`** - 小波尺寸说明
  - 小波变换尺寸计算说明
  - 帮助理解特征维度

- **`two_level_wavelet_analysis.py`** - 两级小波分析
  - 实验性多级小波分析
  - 深入分析频率特性

### 配置和文档
- **`README.md`** - 项目说明文档（本文件）
- **`requirements.txt`** - Python依赖包列表
- **`config.json`** - 运行时配置文件（自动生成）
- **`.claude/settings.local.json`** - Claude Code配置

### 目录结构
```
wavelet/
├── 核心程序（如上所述）
├── autoencoder/              # AutoEncoder完整系统
│   ├── models/              # 多种AE架构实现
│   ├── training/            # 三阶段训练器
│   ├── evaluation/          # 评估和指标
│   └── utils/               # 工具（频率配置、小波变换等）
├── networks/                 # 插件化网络
│   ├── __init__.py
│   └── example_networks.py  # 示例网络实现
├── models/                   # 保存的模型文件
├── logs/                     # 训练日志
├── figures/                  # 生成的图表
└── checkpoints/              # 训练检查点
```

## 🛠️ 安装和配置

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 数据准备

确保数据文件位于正确位置：
- 飞行器参数: `../parameter/parameters_sorted.csv`
- RCS数据: `../parameter/csv_output/`
  - 2频配置: `001_1.5G.csv`, `001_3G.csv`, ... , `100_3G.csv`
  - 3频配置: 额外需要 `001_6G.csv`, ... , `100_6G.csv`

## 🎯 快速开始

### 方式1: GUI界面（推荐）

```bash
python gui.py
```

启动图形界面，提供完整的可视化操作流程：
- **数据管理**: 配置数据路径、选择频率配置(2freq/3freq)、加载数据
- **AutoEncoder**: 创建双模式AE系统、三阶段训练、性能对比分析
- **小波分析**: 选择模型和频率进行小波变换分析和可视化

### 方式2: 命令行训练

```bash
# 使用默认配置训练
python main.py --mode train

# 自定义训练参数
python main.py --mode train --epochs 150 --batch-size 16 --learning-rate 0.001
```

### 方式3: 模型评估

```bash
python main.py --mode evaluate --model-path models/best_model.pth
```

### 方式4: RCS预测

```bash
# 使用参数字符串预测
python main.py --mode predict --model-path models/best_model.pth --params "1.2,0.8,2.1,1.5,0.9,1.8,2.3,1.1,0.7" --visualize

# 从文件读取参数
python main.py --mode predict --model-path models/best_model.pth --params input_params.txt --output prediction_result.npz
```

## 🧠 网络架构详解

### 核心组件

1. **参数编码器**
   - 输入: 9维飞行器参数
   - 结构: Linear(9→128→256) + BatchNorm + ReLU + Dropout

2. **多尺度小波特征提取器**
   - 4个不同尺度的2D小波层
   - 支持Daubechies、双正交等小波基
   - 在φ-θ平面进行多分辨率分析

3. **频率交互模块**
   - 跨频率注意力机制
   - 建模1.5GHz和3GHz间的物理关系

4. **渐进式解码器**
   - 23×23 → 46×46 → 91×91
   - 避免直接生成高分辨率输出的挑战

### 损失函数

- **MSE损失**: 主要回归损失
- **对称性损失**: φ=0°平面对称性约束
- **频率一致性损失**: 双频间物理关系
- **多尺度损失**: 不同分辨率下的一致性

## 📊 评估指标

### 基础指标
- RMSE (Root Mean Square Error) - 在线性RCS域计算
- R² (决定系数) - 在线性RCS域计算
- 相关系数 (Pearson Correlation) - 在线性RCS域计算

**重要说明**: 训练loss在对数域(标准化dB)计算，但评估指标(RMSE/R²)会自动转换到线性域以保证物理意义正确。

### 物理一致性
- φ=0°平面对称性得分 (已修复：在正确的维度上应用)
- 频率比例一致性
- 角度域高频保持度

### 可视化对比
- 真实vs预测的双频RCS热图 (dB显示)
- 误差分布图 (dB显示)
- 参数敏感性分析

## 🔧 配置说明

系统会自动生成`config.json`配置文件，主要包含：

```json
{
  "data": {
    "params_file": "../parameter/parameters_sorted.csv",
    "rcs_data_dir": "../parameter/csv_output",
    "model_ids": ["001", "002", ..., "100"],
    "frequencies": ["1.5G", "3G"],
    "use_log_preprocessing": true
  },
  "model": {
    "input_dim": 9,
    "hidden_dims": [128, 256],
    "dropout_rate": 0.2,
    "wavelet_type": "db4"
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 0.003,
    "min_lr": 2e-5,
    "epochs": 200,
    "use_cross_validation": true,
    "n_folds": 5
  }
}
```

## 📈 训练策略

### 小数据集优化
- **交叉验证**: 5折交叉验证提高模型泛化性
- **数据增强**: 参数空间插值、角度域增强
- **渐进式训练**: 先低分辨率后高分辨率
- **早停机制**: 防止过拟合

### 物理约束
- **对称性**: 强制φ=0°平面对称性
- **频率关系**: 建模双频间的散射物理
- **正则化**: L2权重衰减 + Dropout

## 🎨 可视化功能

### 内置可视化类型
- **2D热图**: 显示RCS在φ-θ平面的分布
- **3D表面图**: 立体显示散射特性
- **球坐标图**: 全向散射模式分析
- **对比图**: 真实vs预测结果对比

### 调用现有可视化
```python
import rcs_visual as rv

# 使用现有的可视化函数
rv.plot_2d_heatmap("001", "1.5G")
rv.plot_3d_surface("001", "1.5G")
rv.compare_models(["001", "002", "003"], "1.5G")
```

## 🔍 使用示例

### Python API示例

```python
from wavelet_network import create_model
from training import RCSDataset, CrossValidationTrainer
from evaluation import RCSEvaluator
import numpy as np

# 1. 创建和训练模型
model = create_model(input_dim=9, hidden_dims=[128, 256])
trainer = CrossValidationTrainer(model_params={'input_dim': 9})

# 2. 加载数据（使用您的参数和RCS数据）
params = np.random.randn(100, 9)  # 替换为真实参数
rcs_data = np.random.randn(100, 91, 91, 2)  # 替换为真实RCS数据
dataset = RCSDataset(params, rcs_data)

# 3. 训练
results = trainer.cross_validate(dataset, training_config)

# 4. 评估
evaluator = RCSEvaluator(model)
eval_results = evaluator.evaluate_dataset(test_dataset)

# 5. 预测
prediction = model(torch.tensor(new_params))
```

## ⚠️ 注意事项

1. **GPU内存**: 训练时需要足够的GPU内存，推荐8GB+
2. **数据路径**: 确保数据文件路径正确设置
3. **CUDA版本**: 如使用GPU，确保CUDA版本兼容
4. **依赖冲突**: 如遇到包冲突，建议使用虚拟环境

## 🐛 故障排除

### 常见问题

1. **导入错误**: 检查所有模块文件是否在同一目录
2. **CUDA错误**: 确认CUDA安装和PyTorch版本匹配
3. **内存不足**: 减少batch_size或使用CPU训练
4. **数据加载失败**: 检查数据文件路径和格式

### 调试模式

```bash
python main.py --mode train --verbose
```

## 📝 更新日志

### v2.1 (最新版本 - 2025-01-13)
- ✅ **频率配置验证系统**: 防止模型与数据不匹配
  - 模型加载时自动验证频率配置
  - 数据加载时检查与模型的兼容性
  - 对比图生成前强制验证，避免系统性错误
- ✅ **网络注册优化**: 消除重复注册警告，加快GUI启动速度
- ✅ **删除clip操作**: 让错误数据直接暴露，便于问题诊断
- ✅ **会话时间戳机制**: 确保同一会话的所有输出文件使用相同时间戳
- ✅ **完整训练历史**: Stage2/Stage3训练历史保存和可视化

### v2.0 (AutoEncoder专注)
- ✅ **多频率数据支持**: 完整支持2频/3频配置，数据管理页可选择
- ✅ **双模式AutoEncoder系统**:
  - 小波增强模式 (WaveletAutoEncoder)
  - 直接CNN模式 (DirectAutoEncoder)
  - MLP架构支持 (WaveletMLPAutoEncoder/DirectMLPAutoEncoder)
  - 增强感受野CNN (EnhancedWaveletAutoEncoder)
- ✅ **三阶段训练流程**: AE预训练 → 参数映射 → 端到端微调
- ✅ **小波分析增强**: 支持选择模型和频率进行独立分析
- ✅ **性能对比工具**: 双系统对比分析功能
- ✅ **数据缓存优化**: 支持多频配置的智能缓存

### v1.1
- ✅ 修复φ=0°对称性约束维度错误 (CRITICAL)
- ✅ 修复评估指标域不匹配问题 (对数域→线性域转换)
- ✅ 所有可视化图表改用分贝(dB)显示
- ✅ 学习率调度优化 (初始LR: 0.003, eta_min: 2e-5)
- ✅ GUI增加初始学习率和最低学习率调节功能
- ✅ 多小波基选择功能 (db4, db6, db8, bior3.3等)
- ✅ 对数域预处理选项

### v1.0
- ✅ 完整的小波神经网络架构
- ✅ 双频RCS预测功能
- ✅ 物理约束损失函数
- ✅ 交叉验证训练策略
- ✅ 全面的评估指标
- ✅ tkinter图形界面
- ✅ 命令行工具

### 🔮 后续开发方向 (AutoEncoder优化)
- 🔄 **变分AutoEncoder (VAE)**: 概率生成模型
- 🔄 **条件AutoEncoder**: 条件化参数生成
- 🔄 **对抗训练 (GAN)**: 提升重建质量
- 🔄 **注意力机制**: 关键特征聚焦
- 🔄 **迁移学习**: 跨频率知识迁移
- 🔄 **模型压缩**: 量化和剪枝优化

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 项目: RCS Wavelet Network Project
- 版本: v1.0

---

**注意**: 本项目基于现有的RCS数据读取和可视化模块构建，充分复用了已有的功能模块，确保了系统的稳定性和兼容性。