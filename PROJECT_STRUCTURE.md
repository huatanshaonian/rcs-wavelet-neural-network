# 项目结构文档

> **最后更新**: 2025-12-25
> **用途**: 快速定位功能代码位置，理解模块职责和依赖关系
> **重要**: 搜索代码或功能时，**优先参考本文档**

---

## 📁 顶层目录结构

```
wavelet/
├── gui_managers/           # GUI管理器模块（10K+ 行，核心GUI逻辑）
├── autoencoder/            # AutoEncoder核心模块（15K+ 行，神经网络）
├── data_processing/        # 数据加载与预处理
├── networks/               # 传统网络定义与注册
├── scripts/                # 诊断、修复与运行脚本（20+ 个）
├── tools/                  # 可视化与教学工具
├── docs/                   # 技术文档目录
├── tests/                  # 单元测试
├── legacy/                 # 废弃代码（不使用）
├── refactoring_scripts/    # 重构辅助脚本（已完成）
├── refactoring_tools/      # 重构工具（已完成）
├── ae_checkpoints/         # AutoEncoder模型保存目录
├── batch_experiments/      # 批量实验结果目录
├── cache/                  # 数据缓存目录
├── checkpoints/            # 传统模型检查点
├── results/                # 训练结果
├── visualizations/         # 可视化输出
├── logs/                   # 日志文件
├── gui.py                  # 主GUI（旧版，7000+ 行，逐步迁移到gui_managers）
├── main.py                 # 命令行入口
├── CLAUDE.md               # Claude工作上下文（核心技术文档）
├── PARAMETERS_REFERENCE.md # 参数命名规范（必读）
└── PROJECT_STRUCTURE.md    # 本文档
```

---

## 🗂️ 核心模块详解

### 1. gui_managers/ - GUI管理器模块（模块化GUI架构）

**总览**: 10K+ 行代码，将 `gui.py` 的巨型类拆分为职责明确的模块

```
gui_managers/
├── extensions/             # GUI扩展插件（2.7K 行）
│   ├── gui_autoencoder_extension.py      # 1759 行 - AutoEncoder训练/评估界面
│   └── gui_batch_experiment_extension.py # 1023 行 - 批量实验界面
│
├── managers/               # 业务逻辑管理器（5K 行）
│   ├── training_manager.py        # 450 行 - 训练流程管理（协调）
│   ├── evaluation_manager.py      # 381 行 - 模型评估管理
│   ├── visualization_manager.py   # 3026 行 - 可视化管理（最大模块）
│   ├── statistics_manager.py      # 961 行 - 统计分析管理
│   └── reconstruction_manager.py  # 362 行 - 重建管理
│
├── tabs/                   # GUI标签页组件（仅UI逻辑）
│   ├── training_tab.py            # 训练界面标签页
│   ├── evaluation_tab.py          # 评估界面标签页
│   ├── prediction_tab.py          # 预测界面标签页
│   ├── visualization_tab.py       # 可视化界面标签页
│   ├── data_management_tab.py     # 数据管理标签页
│   └── loss_config_tab.py         # 损失函数配置标签页
│
└── trainers/               # 训练器封装（2.1K 行）
    ├── ae_trainer.py              # 1696 行 - AutoEncoder训练核心
    └── legacy_trainer.py          # 460 行 - 传统网络训练（旧代码）
```

#### 模块职责说明

##### extensions/ - 扩展插件
- **gui_autoencoder_extension.py**:
  - AutoEncoder训练/评估的完整界面
  - 集成到主GUI的扩展标签页
  - 支持三阶段训练、Stage 1 Only、端到端训练
  - 模型保存/加载、配置管理

- **gui_batch_experiment_extension.py**:
  - 批量实验管理界面
  - 自动化超参数搜索
  - 架构对比、激活函数对比、数据预处理对比

##### managers/ - 业务逻辑管理器
- **training_manager.py**:
  - **职责**: 协调训练流程（Controller）
  - **委托**: 具体训练任务委托给 `ae_trainer.py` 或 `legacy_trainer.py`
  - **功能**: 准备训练配置、创建优化器、管理训练历史

- **evaluation_manager.py**:
  - **职责**: 模型评估管理
  - **功能**: 计算评估指标、生成评估报告

- **visualization_manager.py** (最大模块):
  - **职责**: 所有可视化功能
  - **功能**: 训练曲线、RCS热图、频谱分析、残差图等

- **statistics_manager.py**:
  - **职责**: 统计分析管理
  - **功能**: 全局统计对比、分布分析

- **reconstruction_manager.py**:
  - **职责**: 重建管理
  - **功能**: RCS重建、误差计算

##### trainers/ - 训练器
- **ae_trainer.py** (核心训练逻辑):
  - **职责**: AutoEncoder三阶段训练的实际执行
  - **功能**:
    - Stage 1: AutoEncoder预训练
    - Stage 2: 参数映射器训练
    - Stage 3: 端到端微调
    - Stage 1 Only: 仅重建性能研究
    - 端到端训练: 联合训练模式
  - **支持**: L-BFGS/Adam/AdamW/SGD优化器，多阶段学习率调度

- **legacy_trainer.py**:
  - **职责**: 传统网络训练（向后兼容）
  - **状态**: 维护模式，新功能在 `ae_trainer.py`

##### tabs/ - 界面标签页
- **职责**: 纯UI组件，不包含业务逻辑
- **功能**: Tkinter界面布局、控件创建、事件绑定

---

### 2. autoencoder/ - AutoEncoder核心模块

**总览**: 15K+ 行代码，神经网络定义与训练评估工具

```
autoencoder/
├── models/                 # 网络架构定义（8K 行，15+ 模型）
│   ├── cnn_autoencoder.py                        # 561 行 - Wavelet标准CNN
│   ├── direct_autoencoder.py                     # 431 行 - Direct标准CNN
│   ├── mlp_autoencoder.py                        # 548 行 - MLP系列
│   ├── enhanced_cnn_autoencoder.py               # 850 行 - Enhanced CNN系列
│   ├── deep_autoencoder.py                       # 657 行 - Deep CNN系列
│   ├── differentiable_wavelet_autoencoder.py     # 596 行 - 可微分小波模式
│   ├── dual_branch_differentiable_autoencoder.py # 800 行 - 双分支V1（遗留）
│   ├── dual_branch_differentiable_autoencoder_v2.py # 961 行 - 双分支V2（推荐）
│   ├── additive_dual_branch_autoencoder.py       # 746 行 - 叠加型双分支CNN
│   ├── additive_dual_branch_mlp.py               # 457 行 - 叠加型双分支MLP
│   ├── parameter_mapper.py                       # 464 行 - 参数映射器
│   ├── channel_attention.py                      # 232 行 - 通道注意力模块
│   ├── base_autoencoder.py                       # 86 行 - 基类
│   ├── dual_branch_autoencoder.py                # 643 行 - 分离型双分支（旧）
│   ├── __init__.py                               # 63 行 - 导出核心模型
│   └── experimental/                             # 实验性模型（未使用）
│       ├── correct_cnn_autoencoder.py
│       ├── deep_cnn_autoencoder.py
│       ├── efficient_cnn_autoencoder.py
│       └── micro_latent_autoencoder.py
│
├── utils/                  # 工具模块（8.5K 行，20+ 工具）
│   ├── frequency_config.py                # 765 行 - 创建AutoEncoder系统（核心工厂）
│   ├── plotting.py                        # 1003 行 - 绘图工具
│   ├── configurable_loss.py               # 595 行 - 可配置损失函数
│   ├── model_io.py                        # 583 行 - 模型保存/加载
│   ├── json_experiment.py                 # 559 行 - JSON实验管理
│   ├── statistics_comparison.py           # 510 行 - 统计对比
│   ├── data_adapters.py                   # 366 行 - 数据预处理（标准化/dB）
│   ├── model_summary.py                   # 354 行 - 模型摘要
│   ├── gradient_monitor.py                # 350 行 - 梯度监控
│   ├── reconstruction.py                  # 305 行 - 重建工具
│   ├── data_cache.py                      # 285 行 - 数据缓存
│   ├── differentiable_wavelet_transform.py # 241 行 - 可微分小波变换
│   ├── correct_wavelet_transform.py       # 210 行 - 小波变换（统一接口）
│   ├── frequency_analysis.py              # 207 行 - 频率分析
│   ├── activation_factory.py              # 激活函数工厂
│   ├── adaptive_layers.py                 # 自适应层
│   ├── ae_evaluation.py                   # AE评估工具
│   ├── batch_experiment.py                # 批量实验管理器
│   └── comparison_system.py               # 网络对比框架
│
├── training/               # 训练逻辑
│   ├── ae_trainer.py              # 独立训练器（命令行）
│   ├── standalone_trainer.py      # 独立训练脚本
│   └── multi_stage_scheduler.py   # 多阶段学习率调度器
│
└── evaluation/             # 评估逻辑
    ├── ae_evaluator.py            # AE评估器
    └── reconstruction_metrics.py  # 重建评估指标
```

#### 核心文件说明

##### models/ - 网络架构
- **命名规范**: `<Mode><Architecture>AutoEncoder`
  - Mode: Wavelet/Direct/DifferentiableWavelet
  - Architecture: CNN/MLP/EnhancedCNN/DeepCNN/DualBranch

- **6个核心模型** (生产使用):
  1. `WaveletAutoEncoder` - Wavelet + 标准CNN
  2. `DirectAutoEncoder` - Direct + 标准CNN
  3. `WaveletMLPAutoEncoder` - Wavelet + MLP
  4. `DirectMLPAutoEncoder` - Direct + MLP
  5. `EnhancedWaveletAutoEncoder` - Wavelet + Enhanced CNN
  6. `EnhancedDirectAutoEncoder` - Direct + Enhanced CNN

- **高级模型**:
  - `DeepWaveletAutoEncoder` / `DeepDirectAutoEncoder` - 最强表达力
  - `DualBranchDifferentiableWaveletAutoEncoderV2` - 双分支架构（推荐V2）
  - `AdditiveDualBranch*AutoEncoder` - 叠加型双分支（新架构）

##### utils/ - 工具模块
- **frequency_config.py** (核心工厂):
  - `create_autoencoder_system()` - 创建完整的AE系统
  - 统一接口创建所有类型的AutoEncoder

- **data_adapters.py** (数据预处理):
  - `RCS_DataAdapter` - 标准化、dB变换
  - **关键**: 小波变换必须在标准化**之前**！

- **plotting.py** (可视化):
  - 所有绘图函数（训练曲线、热图、频谱、残差等）

- **configurable_loss.py** (损失函数):
  - 可配置的复合损失函数
  - 支持MSE/Huber/物理约束等

---

### 3. data_processing/ - 数据加载与预处理

```
data_processing/
├── data_loader.py          # 数据加载器
└── data_preprocessor.py    # 数据预处理
```

**职责**:
- 加载RCS数据文件
- 数据清洗与格式转换

---

### 4. networks/ - 传统网络模块

```
networks/
├── network_registry.py     # 网络注册系统
└── example_networks.py     # 示例网络定义
```

**职责**:
- 传统网络的注册与管理
- 向后兼容旧版本代码

---

### 5. scripts/ - 诊断与运行脚本

**常用脚本** (20+ 个):

- **训练相关**:
  - `debug_ae_training.py` - 独立训练脚本（调试用）

- **诊断相关**:
  - `diagnose_model_loading.py` - 诊断模型加载问题
  - `diagnose_stage1_reconstruction.py` - 诊断Stage 1重建问题
  - `diagnose_wavelet_mismatch.py` - 诊断小波不匹配
  - `diagnose_attention_standardization.py` - 诊断注意力标准化

- **修复相关**:
  - `fix_ae112_config.py` - 修复AE112配置文件
  - `convert_sine_mlp_model.py` - 转换Sin MLP模型

- **验证相关**:
  - `verify_activations.py` - 验证激活函数
  - `verify_activations_mlp.py` - 验证MLP激活
  - `verify_branch_order.py` - 验证分支顺序

- **工具相关**:
  - `list_cache.py` - 列出缓存文件
  - `apply_dynamic_adaptation.py` - 应用动态适配

---

### 6. tools/ - 可视化与教学工具

```
tools/
├── interactive_*.py        # 交互式可视化工具
└── teaching_*.py           # 教学演示工具
```

**用途**:
- 交互式参数调整
- 可视化教学演示

---

### 7. docs/ - 技术文档

**核心文档**:
- `README.md` - 技术文档索引
- `DATA_PIPELINE.md` - 数据流程完整说明（重要⚠️）
- `DUAL_BRANCH_V1_VS_V2_COMPARISON.md` - V1/V2架构对比
- `architecture/` - 架构详解（CNN/MLP）

---

## 🔍 功能快速索引

### 🎯 训练相关

| 功能 | 文件位置 | 行号/函数 |
|------|---------|---------|
| **启动三阶段训练** | `gui_managers/extensions/gui_autoencoder_extension.py` | `_run_three_stage_training()` |
| **启动Stage 1 Only** | 同上 | 同上（根据training_mode判断） |
| **训练配置管理** | `gui_managers/managers/training_manager.py` | `prepare_training_config()` |
| **Stage 1 训练循环** | `gui_managers/trainers/ae_trainer.py` | `_train_autoencoder_stage1_v2()` |
| **Stage 2 训练循环** | 同上 | `_train_parameter_mapping_stage2_v2()` |
| **Stage 3 训练循环** | 同上 | `_train_end_to_end_stage3_v2()` |
| **优化器创建** | `gui_managers/trainers/ae_trainer.py` | `_create_ae_optimizer_and_scheduler()` |
| **L-BFGS训练** | `gui_managers/trainers/ae_trainer.py` | `_train_batch_with_lbfgs()` |
| **多阶段学习率** | `autoencoder/training/multi_stage_scheduler.py` | `MultiStageLRScheduler` |

### 📊 评估相关

| 功能 | 文件位置 | 行号/函数 |
|------|---------|---------|
| **模型评估入口** | `gui_managers/managers/evaluation_manager.py` | `_evaluate_autoencoder_model()` |
| **重建指标计算** | `autoencoder/evaluation/reconstruction_metrics.py` | `ReconstructionMetrics` |
| **评估器** | `autoencoder/evaluation/ae_evaluator.py` | `AEEvaluator` |
| **评估界面** | `gui_managers/tabs/evaluation_tab.py` | - |

### 📈 可视化相关

| 功能 | 文件位置 | 行号/函数 |
|------|---------|---------|
| **可视化管理器** | `gui_managers/managers/visualization_manager.py` | 3026 行（所有可视化功能） |
| **核心绘图函数** | `autoencoder/utils/plotting.py` | 1003 行 |
| **训练曲线** | `visualization_manager.py` | `_plot_ae_training_curves()` |
| **RCS热图对比** | `visualization_manager.py` | `_plot_ae_2d_heatmap()` |
| **残差分析** | `visualization_manager.py` | `_plot_ae_comparison()` |
| **频谱分析** | `visualization_manager.py` | `plot_branch_comparison()` |
| **全局统计对比** | `visualization_manager.py` | `_plot_global_statistics_comparison()` |

### 📦 批量实验

| 功能 | 文件位置 | 行号/函数 |
|------|---------|---------|
| **批量实验界面** | `gui_managers/extensions/gui_batch_experiment_extension.py` | 1023 行 |
| **批量实验管理器** | `autoencoder/utils/batch_experiment.py` | `BatchExperimentManager` |
| **JSON实验管理** | `autoencoder/utils/json_experiment.py` | `JSONExperimentManager` |

### 🔧 模型管理

| 功能 | 文件位置 | 行号/函数 |
|------|---------|---------|
| **创建AE系统** | `autoencoder/utils/frequency_config.py` | `create_autoencoder_system()` |
| **模型保存** | `gui.py` (旧GUI) | `save_ae_model()` |
| **模型加载** | `gui.py` (旧GUI) | `load_ae_model()` |
| **模型I/O工具** | `autoencoder/utils/model_io.py` | 583 行 |
| **模型摘要** | `autoencoder/utils/model_summary.py` | 354 行 |

### 🗂️ 数据处理

| 功能 | 文件位置 | 行号/函数 |
|------|---------|---------|
| **数据预处理** | `autoencoder/utils/data_adapters.py` | `RCS_DataAdapter` |
| **小波变换** | `autoencoder/utils/correct_wavelet_transform.py` | `CorrectWaveletTransform` |
| **可微分小波** | `autoencoder/utils/differentiable_wavelet_transform.py` | `DifferentiableWaveletTransform` |
| **数据缓存** | `autoencoder/utils/data_cache.py` | `DataCacheManager` |
| **数据加载** | `data_processing/data_loader.py` | - |

---

## 🔄 核心数据流

### 训练流程 (Three Stage)

```
用户点击"开始训练"
  ↓
gui_autoencoder_extension.py:_run_three_stage_training()
  ↓
training_manager.py:prepare_training_config()  # 准备配置
  ↓
ae_trainer.py:_train_autoencoder_stage1_v2()   # Stage 1
  ├─ 数据: RCS → 小波变换 → 标准化 → Tensor
  ├─ 训练: AutoEncoder重建训练
  └─ 保存: 最佳模型到ae_system
  ↓
ae_trainer.py:_train_parameter_mapping_stage2_v2()  # Stage 2
  ├─ 数据: 参数 → ParameterMapper → latent
  ├─ 训练: 映射损失（冻结AutoEncoder）
  └─ 保存: 最佳映射器到ae_system
  ↓
ae_trainer.py:_train_end_to_end_stage3_v2()    # Stage 3
  ├─ 数据: 参数 → ParameterMapper → AutoEncoder → RCS重建
  ├─ 训练: 端到端微调（全部解冻）
  └─ 保存: 最终模型到ae_system
  ↓
gui.py:save_ae_model()  # 用户保存模型
  ├─ 模型权重: .pth文件
  └─ 配置文件: _config.json
```

### 评估流程

```
用户加载模型
  ↓
gui.py:load_ae_model()  # 加载权重 + 配置
  ↓
evaluation_manager.py:_evaluate_autoencoder_model()
  ├─ 获取测试数据
  ├─ Three Stage模式: 参数 → ParameterMapper → Decoder → RCS
  ├─ Stage 1 Only模式: RCS → Encoder → Decoder → RCS重建
  ├─ 逆变换: 标准化空间 → 原始RCS空间（重要⚠️）
  └─ 计算指标: MSE/RMSE/MAE
  ↓
reconstruction_metrics.py:ReconstructionMetrics
  ↓
visualization_manager.py:绘制各种图表
  ├─ RCS热图对比
  ├─ 残差分布
  ├─ 全局统计对比
  └─ 频谱分析（双分支模型）
```

### 数据预处理流程（关键⚠️）

详见 `docs/DATA_PIPELINE.md`

```
原始RCS数据 [N, 91, 91, 2]
  ↓
【Wavelet模式】
  Step 1: 小波变换（在原始线性数据上！）
    → wavelet_transform.forward_transform(rcs_data)
    → 小波系数 [N, 49, 49, 8]

  Step 2: 数据预处理（标准化/dB）
    → data_adapter.adapt_rcs_data(wavelet_coeffs)
    → 标准化小波系数 [N, 49, 49, 8]

  Step 3: Tensor转换
    → torch.FloatTensor(...)
    → 输入AutoEncoder

【Direct模式】
  Step 1: 数据预处理（标准化/dB）
    → data_adapter.adapt_rcs_data(rcs_data)
    → 标准化RCS [N, 91, 91, 2]

  Step 2: Tensor转换
    → torch.FloatTensor(...)
    → 输入AutoEncoder

⚠️ 重要规则：
1. 小波变换必须在标准化**之前**（破坏正交性！）
2. 所有训练/评估都必须使用data_adapter
3. decoder输出必须逆变换回原始RCS空间
```

---

## 🧩 模块依赖关系

```
gui.py (旧主GUI - 7000+ 行)
  ↓
gui_managers/  ← 逐步迁移到这里
  │
  ├── extensions/
  │     ├─ gui_autoencoder_extension.py
  │     │    ↓ 调用
  │     │    ├─ managers/training_manager.py
  │     │    ├─ managers/evaluation_manager.py
  │     │    ├─ managers/visualization_manager.py
  │     │    └─ managers/statistics_manager.py
  │     │
  │     └─ gui_batch_experiment_extension.py
  │          ↓ 调用
  │          └─ autoencoder/utils/batch_experiment.py
  │
  ├── managers/
  │     ├─ training_manager.py
  │     │    ↓ 委托
  │     │    └─ trainers/ae_trainer.py
  │     │
  │     ├─ evaluation_manager.py
  │     │    ↓ 调用
  │     │    └─ autoencoder/evaluation/reconstruction_metrics.py
  │     │
  │     └─ visualization_manager.py
  │          ↓ 调用
  │          └─ autoencoder/utils/plotting.py
  │
  └── trainers/
        ├─ ae_trainer.py
        │    ↓ 调用
        │    ├─ autoencoder/models/*
        │    ├─ autoencoder/utils/frequency_config.py
        │    ├─ autoencoder/utils/data_adapters.py
        │    ├─ autoencoder/utils/configurable_loss.py
        │    └─ autoencoder/training/multi_stage_scheduler.py
        │
        └─ legacy_trainer.py
             ↓ 调用
             └─ networks/* (传统网络)

autoencoder/  ← 核心神经网络模块（独立）
  │
  ├── models/  (15+ 模型定义)
  │     └─ 所有模型继承自 base_autoencoder.py
  │
  ├── utils/  (20+ 工具模块)
  │     ├─ frequency_config.py  ← 工厂函数（创建AE系统）
  │     ├─ data_adapters.py      ← 数据预处理
  │     ├─ plotting.py           ← 绘图函数
  │     └─ ...
  │
  ├── training/
  │     ├─ ae_trainer.py         ← 独立训练器（命令行）
  │     └─ multi_stage_scheduler.py
  │
  └── evaluation/
        ├─ ae_evaluator.py
        └─ reconstruction_metrics.py

data_processing/  ← 数据加载
networks/         ← 传统网络
scripts/          ← 独立脚本（不依赖GUI）
tools/            ← 可视化工具
```

### 关键依赖说明

1. **GUI层 → 业务逻辑层 → 核心模块层**:
   - `gui_managers/extensions/` → `gui_managers/managers/` → `autoencoder/`
   - 单向依赖，核心模块独立可用

2. **trainers/ 是桥梁**:
   - 连接GUI层和autoencoder核心模块
   - 处理训练流程的实际执行

3. **autoencoder/ 完全独立**:
   - 可以单独使用（命令行、脚本）
   - 不依赖GUI

---

## 📋 文件清单（按功能分类）

### GUI相关（用户界面）

| 文件 | 行数 | 职责 |
|------|------|------|
| `gui.py` | 7000+ | 旧主GUI（逐步迁移） |
| `gui_managers/extensions/gui_autoencoder_extension.py` | 1759 | AutoEncoder界面扩展 |
| `gui_managers/extensions/gui_batch_experiment_extension.py` | 1023 | 批量实验界面扩展 |
| `gui_managers/tabs/*.py` | - | 各个标签页UI组件 |

### 业务逻辑（管理器 + 训练器）

| 文件 | 行数 | 职责 |
|------|------|------|
| `gui_managers/managers/training_manager.py` | 450 | 训练流程管理 |
| `gui_managers/managers/evaluation_manager.py` | 381 | 评估管理 |
| `gui_managers/managers/visualization_manager.py` | 3026 | 可视化管理 |
| `gui_managers/managers/statistics_manager.py` | 961 | 统计分析管理 |
| `gui_managers/managers/reconstruction_manager.py` | 362 | 重建管理 |
| `gui_managers/trainers/ae_trainer.py` | 1696 | AutoEncoder训练核心 |
| `gui_managers/trainers/legacy_trainer.py` | 460 | 传统网络训练 |

### 神经网络模型

| 文件 | 行数 | 模型名称 |
|------|------|---------|
| `autoencoder/models/cnn_autoencoder.py` | 561 | WaveletAutoEncoder/DirectAutoEncoder |
| `autoencoder/models/mlp_autoencoder.py` | 548 | Wavelet/DirectMLPAutoEncoder |
| `autoencoder/models/enhanced_cnn_autoencoder.py` | 850 | Enhanced系列 |
| `autoencoder/models/deep_autoencoder.py` | 657 | Deep系列 |
| `autoencoder/models/differentiable_wavelet_autoencoder.py` | 596 | 可微分小波系列 |
| `autoencoder/models/dual_branch_differentiable_autoencoder_v2.py` | 961 | 双分支V2（推荐） |
| `autoencoder/models/additive_dual_branch_autoencoder.py` | 746 | 叠加型双分支CNN |
| `autoencoder/models/additive_dual_branch_mlp.py` | 457 | 叠加型双分支MLP |
| `autoencoder/models/parameter_mapper.py` | 464 | 参数映射器 |

### 工具模块

| 文件 | 行数 | 职责 |
|------|------|------|
| `autoencoder/utils/frequency_config.py` | 765 | 创建AE系统（工厂） |
| `autoencoder/utils/plotting.py` | 1003 | 绘图工具 |
| `autoencoder/utils/configurable_loss.py` | 595 | 可配置损失函数 |
| `autoencoder/utils/model_io.py` | 583 | 模型I/O |
| `autoencoder/utils/data_adapters.py` | 366 | 数据预处理 |
| `autoencoder/utils/correct_wavelet_transform.py` | 210 | 小波变换 |
| `autoencoder/utils/gradient_monitor.py` | 350 | 梯度监控 |

### 训练与评估

| 文件 | 行数 | 职责 |
|------|------|------|
| `autoencoder/training/ae_trainer.py` | - | 独立训练器（命令行） |
| `autoencoder/training/multi_stage_scheduler.py` | - | 多阶段学习率调度 |
| `autoencoder/evaluation/reconstruction_metrics.py` | - | 重建评估指标 |
| `autoencoder/evaluation/ae_evaluator.py` | - | AE评估器 |

### 数据处理

| 文件 | 行数 | 职责 |
|------|------|------|
| `data_processing/data_loader.py` | - | 数据加载 |
| `data_processing/data_preprocessor.py` | - | 数据预处理 |
| `autoencoder/utils/data_cache.py` | 285 | 数据缓存 |

### 文档

| 文件 | 职责 |
|------|------|
| `CLAUDE.md` | Claude工作上下文（核心技术文档） |
| `PROJECT_STRUCTURE.md` | 本文档（项目结构） |
| `PARAMETERS_REFERENCE.md` | 参数命名规范 |
| `README.md` | 项目说明文档 |
| `docs/DATA_PIPELINE.md` | 数据流程详解（重要⚠️） |
| `docs/DUAL_BRANCH_V1_VS_V2_COMPARISON.md` | 双分支架构对比 |
| `autoencoder/models/MODEL_INVENTORY.md` | 模型清单 |

---

## 🎯 常见任务快速指南

### 1. 我想修改训练流程

**路径**: `gui_managers/trainers/ae_trainer.py`

**函数**:
- `_train_autoencoder_stage1_v2()` - Stage 1训练
- `_train_parameter_mapping_stage2_v2()` - Stage 2训练
- `_train_end_to_end_stage3_v2()` - Stage 3训练

### 2. 我想添加新的可视化功能

**路径**: `gui_managers/managers/visualization_manager.py`

**步骤**:
1. 在 `visualization_manager.py` 添加管理函数
2. 调用 `autoencoder/utils/plotting.py` 的绘图函数
3. 在 `gui_managers/tabs/visualization_tab.py` 添加UI按钮

### 3. 我想添加新的AutoEncoder架构

**路径**: `autoencoder/models/`

**步骤**:
1. 创建新模型文件 `my_new_autoencoder.py`
2. 继承 `base_autoencoder.py` 或直接继承 `nn.Module`
3. 实现 `encode()`, `decode()`, `forward()` 方法
4. 提供 `self.encoder` 和 `self.decoder` 属性（必须⚠️）
5. 在 `__init__.py` 导出模型
6. 在 `frequency_config.py` 添加创建逻辑

### 4. 我想修改数据预处理逻辑

**路径**: `autoencoder/utils/data_adapters.py`

**类**: `RCS_DataAdapter`

**注意**: 小波变换必须在标准化**之前**（见 `docs/DATA_PIPELINE.md`）

### 5. 我想添加新的损失函数

**路径**: `autoencoder/utils/configurable_loss.py`

**函数**: `create_loss_function()`

### 6. 我想运行批量实验

**路径**:
- GUI界面: `gui_managers/extensions/gui_batch_experiment_extension.py`
- 核心逻辑: `autoencoder/utils/batch_experiment.py`

**步骤**:
1. 在【AutoEncoder】页面配置基准参数
2. 在【批量实验】页面选择对比维度
3. 执行批量训练
4. 查看结果: `batch_experiments/experiment_name_timestamp/`

### 7. 我想诊断训练问题

**路径**: `scripts/diagnose_*.py`

**常用脚本**:
- `diagnose_model_loading.py` - 模型加载问题
- `diagnose_stage1_reconstruction.py` - Stage 1重建问题
- `diagnose_wavelet_mismatch.py` - 小波不匹配

---

## ⚠️ 重要注意事项

### 1. 数据预处理顺序（关键⚠️⚠️⚠️）

**必须遵守**:
- ✅ Wavelet模式: `原始RCS → 小波变换 → 标准化`
- ❌ 错误: `原始RCS → 标准化 → 小波变换`（破坏正交性！）

详见: `docs/DATA_PIPELINE.md`

### 2. decoder输出必须逆变换（容易遗忘⚠️）

**规则**:
- 训练时: 损失在标准化空间计算（正确✅）
- 推理/评估/可视化: decoder输出必须逆变换到原始RCS空间

**检查清单**:
- [ ] 调用了 `data_adapter.inverse_adapt()`？
- [ ] Wavelet模式: 逆标准化 → 逆小波变换？
- [ ] Direct模式: 逆标准化（逆dB + 逆Z-score）？

### 3. 模型必须提供统一接口（模型定义规范⚠️）

**必须提供**:
```python
class MyAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 方式1: nn.Sequential
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)

        # 方式2: nn.ModuleList
        self.encoder = nn.ModuleList([...])
        self.decoder = nn.ModuleList([...])

    def encode(self, x): ...
    def decode(self, latent): ...
    def forward(self, x): ...
```

**原因**: Stage 2需要冻结/解冻encoder

### 4. 损失计算必须sample-weighted

**错误**:
```python
train_loss += loss.item()
avg_loss = train_loss / num_batches
```

**正确**:
```python
batch_size = data.size(0)
train_loss += loss.item() * batch_size
total_samples += batch_size
avg_loss = train_loss / total_samples
```

**原因**: 最后一个batch可能更小

---

## 🔧 代码搜索技巧

### 1. 按功能搜索

**使用本文档的"功能快速索引"**，直接定位到文件和函数。

### 2. 按模块搜索

**使用本文档的"核心模块详解"**，了解各目录职责。

### 3. 按关键词搜索

**常用关键词映射**:

| 关键词 | 可能位置 |
|--------|---------|
| `train` | `gui_managers/trainers/ae_trainer.py` |
| `evaluate` | `gui_managers/managers/evaluation_manager.py` |
| `plot` | `gui_managers/managers/visualization_manager.py` |
| `save_model` | `gui.py` |
| `load_model` | `gui.py` |
| `create_autoencoder` | `autoencoder/utils/frequency_config.py` |
| `data_adapter` | `autoencoder/utils/data_adapters.py` |
| `wavelet_transform` | `autoencoder/utils/correct_wavelet_transform.py` |
| `loss` | `autoencoder/utils/configurable_loss.py` |
| `optimizer` | `gui_managers/trainers/ae_trainer.py` |

### 4. 按文件大小搜索

**大文件通常是核心模块**:
- 3000+ 行: `visualization_manager.py` (可视化)
- 1700+ 行: `gui_autoencoder_extension.py` (AutoEncoder界面)
- 1700+ 行: `ae_trainer.py` (训练核心)
- 1000+ 行: `plotting.py` (绘图工具)

---

## 📚 相关文档链接

- **核心技术文档**: `CLAUDE.md`
- **参数命名规范**: `PARAMETERS_REFERENCE.md`
- **数据流程详解**: `docs/DATA_PIPELINE.md`
- **模型清单**: `autoencoder/models/MODEL_INVENTORY.md`
- **技术文档索引**: `docs/README.md`

---

**维护者**: Claude Code
**最后更新**: 2025-12-25
**版本**: v1.0

> 💡 **提示**: 搜索代码时，优先参考本文档的"功能快速索引"和"常见任务快速指南"！
