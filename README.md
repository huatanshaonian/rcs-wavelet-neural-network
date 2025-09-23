# RCS小波神经网络预测系统

基于小波多尺度理论的飞行器RCS（雷达散射截面）预测系统，使用深度学习技术从9个飞行器参数预测双频RCS分布数据。

## 🚀 项目特色

- **小波多尺度架构**: 在φ-θ平面使用2D小波变换，支持4个不同尺度的特征提取
- **双频预测**: 同时预测1.5GHz和3GHz频率的RCS分布 [91×91×2]
- **物理约束**: 集成φ=0°平面对称性约束和频率一致性损失
- **小数据集优化**: 针对~100样本设计的数据增强和交叉验证策略
- **完整工作流**: 数据加载→训练→评估→预测→可视化的完整pipeline
- **图形界面**: 基于tkinter的直观GUI操作界面

## 📁 项目结构

```
wavelet/
├── rcs_data_reader.py          # 数据读取模块（已有）
├── rcs_visual.py               # 可视化模块（已有）
├── wavelet_network.py          # 小波神经网络核心
├── training.py                 # 训练模块
├── evaluation.py               # 评估模块
├── gui.py                      # 图形界面
├── main.py                     # 主程序入口
├── config.json                 # 配置文件（自动生成）
├── requirements.txt            # 依赖列表
├── README.md                   # 本文件
└── checkpoints/                # 模型检查点目录
    models/                     # 训练好的模型
    results/                    # 评估结果
    logs/                       # 训练日志
    visualizations/             # 可视化图片
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
- RCS数据: `../parameter/csv_output/` (包含001_1.5G.csv到100_3G.csv)

## 🎯 快速开始

### 方式1: GUI界面（推荐）

```bash
python main.py --mode gui
```

启动图形界面，提供完整的可视化操作流程。

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

### v1.1 (最新版本)
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

### 计划功能
- 🔄 模型压缩和量化
- 🔄 分布式训练支持
- 🔄 Web界面版本

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 项目: RCS Wavelet Network Project
- 版本: v1.0

---

**注意**: 本项目基于现有的RCS数据读取和可视化模块构建，充分复用了已有的功能模块，确保了系统的稳定性和兼容性。