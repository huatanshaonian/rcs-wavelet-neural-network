# GUI重构辅助脚本

这个目录包含了GUI重构过程中使用的一次性辅助脚本。

## 重构概述

将原始的 `gui.py` (8,213行) 重构为模块化架构，拆分为5个管理器类：

1. **StatisticsManager** - 统计分析功能
2. **VisualizationManager** - 可视化功能
3. **TrainingManager** - 训练功能
4. **EvaluationManager** - 评估功能
5. **ReconstructionManager** - RCS重建功能

**最终结果**: gui.py 减少到 3,448行 (-58.0%)

## 脚本说明

### 提取脚本 (创建Manager类)
- `extract_training_methods.py` - 从gui.py提取训练相关方法
- `create_evaluation_manager.py` - 创建EvaluationManager
- `create_reconstruction_manager.py` - 创建ReconstructionManager

### 替换脚本 (修改gui.py)
- `replace_training_methods.py` - 将training方法替换为委托调用
- `replace_evaluation_methods.py` - 将evaluation方法替换为委托调用
- `replace_reconstruction_method.py` - 将reconstruction方法替换为委托调用

### 集成脚本
- `integrate_training_manager.py` - 集成TrainingManager到gui.py

## 工作流程

每个步骤的典型流程：

1. **提取**: 运行 `create_*_manager.py` 创建新的Manager类
2. **替换**: 运行 `replace_*_methods.py` 修改gui.py中的方法为委托调用
3. **测试**: 运行 `python -c "from gui import RCSWaveletGUI"`
4. **提交**: 提交到git

## 注意事项

这些脚本是**一次性使用**的重构工具，重构完成后仅作为参考保留。

如果需要类似的重构任务，可以参考这些脚本的实现思路：
- 基于行号范围提取方法
- 自动替换 `self.` 为 `self.gui.`
- 处理多行方法签名
- 保留文档字符串

## 重构时间线

- **2025-01-XX**: Steps 1-2 (StatisticsManager, VisualizationManager)
- **2025-01-XX**: Step 3 (TrainingManager) - 2259行，25个方法
- **2025-01-XX**: Step 4 (EvaluationManager) - 268行，5个方法
- **2025-01-XX**: Step 5 (ReconstructionManager) - 241行，1个方法
- **重构完成**: gui.py从8213行降至3448行 ✅
