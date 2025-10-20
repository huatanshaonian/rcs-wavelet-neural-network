# GUI重构计划

## 📊 当前状态
- **gui.py**: 8,213行, 135个方法
- **gui_autoencoder_extension.py**: 1,107行
- **总计**: 9,320行

**问题**：GUI构建代码和功能性代码混合，难以维护

---

## 🎯 重构目标

将功能性代码分离到独立模块，保持GUI代码简洁清晰。

### 新文件结构
```
wavelet/
├── gui/
│   ├── __init__.py                    # 导出主窗口类
│   ├── main_window.py                 # GUI构建（从gui.py重构而来）
│   ├── managers/
│   │   ├── __init__.py
│   │   ├── statistics_manager.py      # 统计分析逻辑
│   │   ├── visualization_manager.py   # 可视化逻辑
│   │   ├── training_manager.py        # 训练逻辑
│   │   ├── evaluation_manager.py      # 评估逻辑
│   │   └── reconstruction_manager.py  # RCS重建逻辑
```

---

## 📋 分步重构计划

### 步骤0: 准备工作 ✓
- [x] 创建重构计划文档
- [x] 分析gui.py结构，分类所有方法
- [x] 创建git分支 (已在dev_ae)

---

### 步骤1: 提取统计分析模块 (最独立)

**目标文件**: `gui/managers/statistics_manager.py`

**要提取的方法** (2个方法, ~750行):
- `_plot_global_statistics_comparison()` (4239-4959, 720行)
- `_save_scatter_plots()` (4959-5022, 63行)

**依赖分析**:
- 需要访问: `self.ae_system`, `self.current_model`, `self.data_config`, `self.vis_fig`
- 需要导入: `torch`, `numpy`, `matplotlib`, `rcs_visual`, `rcs_data_reader`

**重构策略**:
```python
class StatisticsManager:
    def __init__(self, parent_gui):
        self.gui = parent_gui  # 保存GUI引用

    def plot_global_statistics_comparison(self):
        # 原 _plot_global_statistics_comparison 逻辑
        # 通过 self.gui.ae_system 访问数据
```

**测试点**:
- [ ] 加载数据 + 加载模型 → 统计对比正常显示
- [ ] Stage1-Only模式正常工作
- [ ] Three-Stage模式正常工作
- [ ] 散点图正常保存

**Commit消息**: `refactor: 提取统计分析模块到statistics_manager.py`

---

### 步骤2: 提取可视化模块

**目标文件**: `gui/managers/visualization_manager.py`

**要提取的方法** (17个方法, ~1500行):

**传统模型可视化**:
- `_plot_2d_heatmap()` (3467-3495)
- `_plot_3d_surface()` (3495-3542)
- `_plot_spherical()` (3542-3614)
- `_plot_comparison()` (3614-3748)
- `_plot_difference_analysis()` (3748-3852)
- `_plot_correlation_analysis()` (3852-3944)
- `_plot_training_history()` (3944-3992)
- `_save_fold_plot()` (3992-4076)
- `_display_fold_in_gui()` (4076-4155)
- `_display_simple_training_history()` (4155-4239)

**AutoEncoder可视化**:
- `_plot_autoencoder_visualization()` (7425-7436)
- `_plot_ae_latent_space()` (7436-7513)
- `_plot_ae_reconstruction_quality()` (7513-7575)
- `_plot_ae_parameter_mapping()` (7575-7650)
- `_plot_ae_training_progress_vis()` (7650-7754)
- `_plot_ae_2d_heatmap()` (7781-7872)
- `_plot_wavelet_coefficients_comparison()` (8043-8154)

**辅助方法**:
- `save_current_visualization()` (8154-end)

**测试点**:
- [ ] 传统模型：2D热图、3D表面图、球坐标图正常
- [ ] 传统模型：对比图、差值分析、相关性分析正常
- [ ] AutoEncoder：latent空间、重建质量可视化正常
- [ ] 小波系数对比正常显示
- [ ] 保存图片功能正常

**Commit消息**: `refactor: 提取可视化模块到visualization_manager.py`

---

### 步骤3: 提取训练逻辑模块

**目标文件**: `gui/managers/training_manager.py`

**要提取的方法** (20个方法, ~2500行):

**传统模型训练**:
- `_train_model()` (1833-2374)
- `_training_finished()` (2374-2380)
- `_set_random_seeds()` (2380-2440)
- `_initialize_cuda_safely()` (2440-2514)

**AutoEncoder三阶段训练（旧版）**:
- `_run_three_stage_training()` (5956-5985)
- `_train_autoencoder_stage1()` (6002-6123)
- `_train_parameter_mapping_stage2()` (6123-6260)
- `_train_end_to_end_stage3()` (6260-6388)
- `_train_full_end_to_end()` (6388-6520)

**AutoEncoder三阶段训练（v2版）**:
- `_run_three_stage_training_v2()` (6520-6578)
- `_run_end_to_end_training_v2()` (6578-6596)
- `_train_autoencoder_stage1_v2()` (6596-6776)
- `_train_parameter_mapping_stage2_v2()` (6776-6960)
- `_train_end_to_end_stage3_v2()` (6960-7143)
- `_train_full_end_to_end_v2()` (7143-7280)

**训练辅助函数**:
- `_create_ae_training_config()` (7286-7331)
- `_create_ae_optimizer_and_scheduler()` (7331-7373)
- `_create_ae_loss_function()` (7373-7386)
- `_create_end_to_end_loss_function()` (7386-7407)
- `_ae_step_scheduler()` (7407-7417)
- `_ae_log_training_progress()` (7417-7425)

**测试点**:
- [ ] 传统模型训练正常（单折、K折）
- [ ] AutoEncoder Stage1训练正常
- [ ] AutoEncoder Three-Stage训练正常
- [ ] Stage1-Only训练正常
- [ ] 训练日志正常显示
- [ ] 训练曲线正常保存

**Commit消息**: `refactor: 提取训练逻辑模块到training_manager.py`

---

### 步骤4: 提取评估逻辑模块

**目标文件**: `gui/managers/evaluation_manager.py`

**要提取的方法** (5个方法, ~500行):
- `_evaluate_traditional_model()` (2788-2849)
- `_evaluate_autoencoder_model()` (3070-3192)
- `_update_evaluation_display()` (3192-3206)
- `_display_autoencoder_results()` (3206-3234)
- `_display_traditional_results()` (3234-3258)

**测试点**:
- [ ] 传统模型评估正常显示
- [ ] AutoEncoder评估正常显示
- [ ] 评估结果展示正确

**Commit消息**: `refactor: 提取评估逻辑模块到evaluation_manager.py`

---

### 步骤5: 提取重建逻辑模块

**目标文件**: `gui/managers/reconstruction_manager.py`

**要提取的方法** (1个方法, ~220行):
- `_reconstruct_rcs()` (2849-3070, 核心重建逻辑)

**测试点**:
- [ ] RCS重建功能正常
- [ ] 小波系数返回正常
- [ ] latent返回正常
- [ ] 可视化调用正常

**Commit消息**: `refactor: 提取重建逻辑模块到reconstruction_manager.py`

---

### 步骤6: 整理主窗口

**目标**: 清理gui.py，保留纯GUI代码

**保留在main_window.py的内容**:
- `__init__()` - 初始化
- `create_widgets()` - 创建控件
- `create_*_tab()` - 创建各个标签页
- `setup_layout()` - 布局
- 各种按钮回调的**入口函数**（委托给manager）
- GUI状态管理

**测试点**:
- [ ] 完整测试所有功能
- [ ] 确保没有遗漏的功能
- [ ] 确保所有按钮都能正常工作

**Commit消息**: `refactor: 完成GUI重构，整理主窗口`

---

## 🔍 重构原则

1. **每次只提取一个模块**
2. **每个步骤都要commit**
3. **每个步骤都要测试**
4. **保持向后兼容**
5. **出问题可以快速回滚**

---

## 📊 预期效果

**重构前**:
- gui.py: 8,213行

**重构后**:
- gui/main_window.py: ~2,000行 (GUI构建)
- gui/managers/statistics_manager.py: ~800行
- gui/managers/visualization_manager.py: ~1,600行
- gui/managers/training_manager.py: ~2,600行
- gui/managers/evaluation_manager.py: ~500行
- gui/managers/reconstruction_manager.py: ~300行
- **总计**: ~7,800行 (功能代码分离，更易维护)

---

## ⚠️ 风险控制

1. **每步都commit** - 出问题可以回滚
2. **用户测试** - 关键步骤请用户测试
3. **保留原文件** - 重构完成前不删除gui.py
4. **渐进式** - 不强制一次性完成

---

## 📅 执行计划

- **步骤1**: 统计分析模块 (~30分钟) → 用户测试
- **步骤2**: 可视化模块 (~1小时) → 用户测试
- **步骤3**: 训练逻辑模块 (~1.5小时) → 用户测试
- **步骤4**: 评估逻辑模块 (~30分钟) → 用户测试
- **步骤5**: 重建逻辑模块 (~30分钟) → 用户测试
- **步骤6**: 整理主窗口 (~30分钟) → 完整测试

**总计**: ~4-5小时工作量

---

**创建时间**: 2025-01-20
**状态**: 待执行
**当前步骤**: 准备开始步骤1
