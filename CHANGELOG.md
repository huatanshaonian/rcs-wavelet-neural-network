# RCS小波神经网络 - 版本修改日志

## 版本管理说明
- **大版本**: 核心架构、重要功能的重大改动
- **小版本**: 功能增强、Bug修复、界面优化等
- **Git分支**: master (稳定版) / dev (开发版)

---

## v2.0.0 - 增强版网络架构 (2025-XX-XX)

### 🚀 核心架构重构
**Git位置**: `dev` 分支 - commit `0fb09ee`

#### 主要改进
1. **增强编码器** (`enhanced_network.py::EnhancedParameterEncoder`)
   - 特征容量: 256维 → 1024维 (4倍提升)
   - 双分支设计: 主分支 + 几何特征分支
   - 渐进式扩展: 9→64→256→512→1024
   - LayerNorm + GELU激活函数替代BatchNorm + ReLU

2. **多尺度特征提取器** (`enhanced_network.py::MultiScaleFeatureExtractor`)
   - 三尺度并行: {64×64, 32×32, 16×16} 特征图
   - 替代原有单一32×32小波特征
   - 融合网络避免信息丢失
   - 自适应特征权重分配

3. **渐进式解码器** (`enhanced_network.py::ProgressiveDecoder`)
   - 平滑上采样: 64→72→84→91 (逐步细化)
   - 解决原有23×23→91×91跨度过大问题
   - 每阶段特征重建和细化
   - 双线性插值 + 卷积组合

4. **改进损失函数** (`enhanced_network.py::ImprovedRCSLoss`)
   - HuberLoss主损失 (对异常值鲁棒)
   - 频率一致性约束 (避免频率间不一致)
   - 空间连续性约束 (梯度平滑)
   - 对称性约束 (物理约束)
   - 自适应权重配比: main(0.7) + aux(0.25) + others(0.05)

#### 性能提升预期
- **收敛性**: loss卡在0.7 → 预期0.1-0.3
- **拟合能力**: R² 0.2 → 预期0.6-0.8
- **参数量**: 1.7M → 60.6M (34倍增长)
- **计算开销**: 前向传播时间增加33%

#### 技术创新
- **注意力机制**: 频率交互模块中的channel attention
- **数值稳定性**: 频率一致性损失使用对数比值避免除零
- **物理约束**: 多层次物理约束确保输出合理性

### 🔧 系统架构改进
**Git位置**: `dev` 分支 - commit `0fb09ee`

1. **架构选择系统** (`wavelet_network.py::create_model`)
   - 支持 `original` / `enhanced` 架构切换
   - 支持 `original` / `improved` 损失函数切换
   - 向后兼容原有架构

2. **GUI界面增强** (`gui.py` 新增选项)
   - 网络架构选择下拉框
   - 损失函数类型选择
   - 实时架构信息显示

3. **测试框架** (`test_architectures.py`)
   - 自动化架构性能对比
   - 参数量、速度、收敛性统计
   - 标准化测试报告生成

### 📋 架构验证完成 (2025-09-26)
**Git位置**: `dev` 分支 - 最新提交

#### 测试结果确认
1. **架构对比测试** (`test_architectures.py`)
   ```
   原始架构: 1,765,794 参数, 43.05ms前向传播, 12.01%收敛改善
   增强架构: 60,597,682 参数, 58.42ms前向传播, 43.22%收敛改善
   性能提升: 最终损失降低72.69% (0.3901→0.1065)
   ```

2. **功能验证测试** (`test_enhanced_loading.py`)
   - ✅ 增强架构模型创建: `EnhancedTriDimensionalRCSNet`
   - ✅ 参数统计: 60,597,682 参数完全可训练
   - ✅ 前向传播: 输出形状 (4,91,91,2) 正确
   - ✅ 损失函数集成: 6个损失组件 (total,main,aux,freq_consistency,continuity,symmetry)
   - ✅ 架构切换: original/enhanced 模式无缝切换

3. **兼容性修复验证**
   - ✅ 自动架构检测和回退机制
   - ✅ 模型加载兼容性处理
   - ✅ GUI界面架构选择功能

#### 生产就绪状态
- **状态**: ✅ 已完成验证，可投入生产使用
- **预期效果**: 训练损失从0.7降至0.1-0.3，R²从0.2提升至0.6-0.8
- **建议**: 优先使用增强架构进行新训练任务

### 🏗️ 插件化架构系统 (2025-09-26)
**Git位置**: `dev` 分支 - 当前开发

#### 系统概述
实现完全插件化的网络架构系统，支持零修改添加新网络，无需重写前处理、后处理、训练和GUI代码。

#### 核心组件
1. **网络注册系统** (`network_registry.py`)
   - `BaseNetwork`: 所有网络的抽象基类
   - `NetworkRegistry`: 全局网络注册中心
   - 装饰器注册: `@register_network` 自动发现
   ```python
   @register_network
   class MyNetwork(BaseNetwork):
       @classmethod
       def get_name(cls) -> str:
           return "my_network"
   ```

2. **统一创建接口** (`modern_wavelet_network.py`)
   - 兼容传统和插件化网络: `create_model(model_type)`
   - 智能回退机制: 插件化 → 传统网络 → 错误
   - 配置验证和参数统计
   ```python
   model = create_model(model_type="any_registered_network")
   ```

3. **示例网络集合** (`networks/example_networks.py`)
   - `wavelet_rcs`: 多尺度小波网络 (8.8M参数)
   - `simple_fc`: 全连接基线网络 (17.6M参数)
   - `resnet_rcs`: ResNet风格网络 (36M参数)
   - `flexible_output`: 任意输出形状网络 (17.6M参数)

4. **插件化损失函数**
   - `BaseLoss`: 损失函数基类
   - `simple_mse`, `robust_loss`: 示例损失函数
   - 动态损失组件适配

#### 技术创新
1. **零修改集成**
   - 新网络自动适配现有训练流程
   - GUI界面自动更新网络选择
   - 前后处理自动兼容

2. **智能配置系统**
   ```python
   class NetworkConfig:
       input_dim: int = 9
       output_shape: Tuple[int, ...] = (91, 91, 2)
       use_log_output: bool = False
   ```

3. **自动发现机制**
   ```python
   NetworkRegistry.auto_discover_networks(['networks.example_networks'])
   ```

4. **向后兼容性**
   - 完全保持`original`、`enhanced`网络支持
   - 渐进式迁移策略

#### 使用示例
```python
# 1. 创建新网络文件 networks/my_network.py
@register_network
class MyNetwork(BaseNetwork):
    # 网络实现

# 2. 自动可用，无需其他修改
model = create_model(model_type="my_network")

# 3. 训练、GUI、可视化完全不变
trainer.train_epoch(...)  # 自动适配
```

#### 性能对比
| 网络类型 | 参数量 | 前向传播时间 | 特点 |
|---------|--------|-------------|------|
| original | 1.77M | 43ms | 传统小波架构 |
| enhanced | 60.6M | 58ms | 增强版架构 |
| wavelet_rcs | 8.8M | 45ms | 插件化小波 |
| simple_fc | 17.6M | 52ms | 全连接基线 |
| resnet_rcs | 36M | 67ms | 残差网络 |

#### 文档和指南
- `ADD_NEW_NETWORK_GUIDE.md`: 详细的新网络添加指南
- `network_registry.py`: 完整的API文档
- `networks/example_networks.py`: 4个完整示例
- 智能推荐系统: 根据数据集大小推荐合适网络

#### 测试验证
- ✅ 所有示例网络前向传播正确
- ✅ 损失函数动态适配成功
- ✅ 传统网络向后兼容
- ✅ 网络参数统计和信息展示
- ✅ 异常处理和错误恢复

### 🔧 关键Bug修复 (2025-09-26)
**Git位置**: `dev` 分支 - 当前修复

#### 问题描述
- **症状**: 简单训练模式报错 `训练失败: 'main'`
- **根因**: 损失权重键不匹配 - 原始(`mse`,`symmetry`,`multiscale`) vs 增强版(`main`,`aux`,`freq_consistency`,`continuity`,`symmetry`)

#### 修复内容
1. **动态损失累积** (`training.py::train_epoch`, `validate_epoch`)
   - 替换固定键初始化为动态键适应: `epoch_losses = {'total': 0}`
   - 改进累积逻辑: 遍历损失函数返回的所有键而非预设键
   ```python
   for key, value in losses.items():
       if key not in epoch_losses:
           epoch_losses[key] = 0
       epoch_losses[key] += value.item()
   ```

2. **权重键自适应映射** (`training.py::_create_progressive_loss_weights`)
   - 检测损失函数类型: `ImprovedRCSLoss` vs `TriDimensionalRCSLoss`
   - 智能权重键转换: `mse→main`, `multiscale→aux`, 新增 `freq_consistency`, `continuity`
   ```python
   enhanced_weights = {
       'main': progressive_weights.get('mse', 1.0),
       'aux': progressive_weights.get('multiscale', 0.3),
       'freq_consistency': 0.1,
       'continuity': 0.05,
       'symmetry': progressive_weights.get('symmetry', 0.05)
   }
   ```

3. **历史记录兼容** (`training.py`, `gui.py`)
   - 训练历史键映射: `train_losses.get('mse', train_losses.get('main', 0))`
   - 保持向后兼容性，支持原始和增强架构的混合使用

#### 验证结果 (`test_training_fix.py`)
- ✅ 增强架构训练: 损失键 `['total', 'main', 'aux', 'freq_consistency', 'continuity', 'symmetry']`
- ✅ 原始架构训练: 损失键 `['total', 'mse', 'symmetry', 'multiscale']`
- ✅ 权重自适应: 自动检测并映射正确的权重键
- ✅ 完整兼容性: 两种架构可无缝切换使用

---

## v1.5.2 - 统计图表优化 (2025-XX-XX)

### 📊 可视化改进
**Git位置**: `master` 分支 - commit `dd61b42`

1. **统计对比图重构** (`gui.py::_plot_global_statistics_comparison`)
   - 前两子图: 散点图 → 模型均值对比图
   - 单位统一: 全部使用dBsm单位显示
   - 散点图独立保存: `results/statistics_comparison_*/scatter_plots.png`
   - 移除对数坐标使用线性坐标

2. **图表布局优化**
   - 6子图布局: 均值对比(2) + 统计指标(1) + 性能指标(2) + 汇总(1)
   - GUI实时显示 + 文件保存并行
   - 标题渲染修复: `\nR` → 正常换行显示

### 🔄 数据处理改进
**Git位置**: `master` 分支 - commit `dd61b42`

1. **模型保存格式标准化**
   - 完整checkpoint格式: `model_state_dict` + `preprocessing_stats` + `use_log_output`
   - 兼容新旧格式加载
   - tkinter变量序列化错误修复

2. **训练可重现性**
   - 全局种子控制: `torch.manual_seed(42)`
   - 数据分割种子固定
   - DataLoader种子固定
   - CUDA确定性设置

3. **预处理统计信息**
   - 训练时自动保存preprocessing_stats
   - 简单训练模式补充统计信息设置
   - 统计分析域转换修复

---

## v1.5.1 - 训练稳定性增强 (2025-XX-XX)

### 🛠️ CUDA稳定性修复
**Git位置**: `master` 分支 - commit `bfbe923`

1. **内存管理优化** (`training.py`)
   - 训练epoch > 200时定期清理GPU缓存
   - BatchNorm momentum自适应调整
   - 数值稳定性检查和恢复

2. **错误处理机制**
   - 多层级CUDA错误恢复
   - 批次大小自适应调整
   - 训练状态保存和恢复

### 🎯 损失函数优化
**Git位置**: `master` 分支 - commit `bfbe923`

1. **物理约束改进**
   - 对称性损失重构: 避免"对称但错误"的学习
   - 多尺度损失权重平衡
   - 数值范围限制和异常值处理

---

## v1.4.0 - 基础架构完成 (2025-XX-XX)

### 🏗️ 核心系统建立
**Git位置**: `master` 分支 - commit `44d465f`

1. **小波神经网络架构**
   - 三维RCS网络: 参数编码器 + 小波特征提取 + 双频解码器
   - 物理约束: 对称性 + 多尺度一致性
   - 损失函数: MSE + 对称性 + 多尺度

2. **GUI训练系统**
   - 可视化界面: 参数配置 + 训练监控 + 结果展示
   - 交叉验证 + 简单训练双模式
   - 实时日志和进度显示

3. **数据处理管道**
   - RCS数据加载和预处理
   - 缓存机制和内存优化
   - 批量预测和统计分析

---

## 待整合的小版本修改

### 🐛 Bug修复记录
- [ ] tkinter序列化错误修复 (已修复 v1.5.2)
- [ ] numpy作用域错误修复 (已修复 v1.5.2)
- [ ] 数据域转换统计错误 (已修复 v1.5.2)
- [ ] GUI统计图显示问题 (已修复 v1.5.2)

### ✨ 功能增强记录
- [ ] 网络架构文档生成 (`improvement_analysis.md`)
- [ ] 性能基准测试框架
- [ ] 自动化测试脚本
- [ ] 配置文件管理优化

### 🔮 计划中的改进
- [ ] Transformer架构实验
- [ ] 神经架构搜索 (NAS)
- [ ] 分布式训练支持
- [ ] 模型量化和推理优化
- [ ] Web界面部署

---

## 版本规范说明

### 版本号格式: `Major.Minor.Patch`
- **Major**: 架构重构、核心算法变更
- **Minor**: 新功能、重要优化
- **Patch**: Bug修复、小改进

### Git工作流
- `master`: 生产稳定版本
- `dev`: 开发测试版本
- `feature/*`: 功能开发分支
- `hotfix/*`: 紧急修复分支

### 文档更新规则
1. **立即记录**: 每次重要修改后更新此文档
2. **详细描述**: 包含修改原因、实现方式、预期效果
3. **Git关联**: 记录对应的commit hash和分支
4. **定期整理**: 小版本积累后整合成大版本发布