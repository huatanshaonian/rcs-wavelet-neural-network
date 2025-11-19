# RCS重建性能评估指标完整指南

> **创建日期**: 2025-01-19
> **用途**: RCS AutoEncoder重建质量的全面评估体系
> **项目**: wavelet小波RCS预测系统

---

## 📊 一、指标分类体系

### 1️⃣ 基础误差指标（Point-wise误差）
**用途**: 衡量预测值与真实值的逐点差异

| 指标 | 公式 | 优点 | 缺点 | 典型好值 |
|-----|------|------|------|---------|
| MSE | mean((pred - true)²) | 可微、凸函数 | 对大误差过敏感 | < 0.001 |
| RMSE | sqrt(MSE) | 与原始数据同单位 | 仍对离群值敏感 | < 0.03 |
| MAE | mean(\|pred - true\|) | 对异常值鲁棒 | 不可微 | < 0.02 |
| 相对误差 | mean(\|pred - true\| / \|true\|) | 考虑数值范围 | 分母接近0时不稳定 | < 5% |
| 最大误差 | max(\|pred - true\|) | 识别最坏情况 | 对单个异常值过敏 | - |

**适用场景**:
- MSE/RMSE: 训练优化、快速评估
- MAE: 存在离群值时的鲁棒评估
- 相对误差: 不同量级数据的比较

---

### 2️⃣ 结构相似性指标（感知质量）
**用途**: 衡量RCS模式的空间结构保持程度

| 指标 | 含义 | 取值范围 | 典型好值 |
|-----|------|---------|---------|
| SSIM | 结构相似性指数（亮度+对比度+结构） | 0-1 | > 0.8 |
| 对称性误差 | φ=0°平面对称性保持度 | 越小越好 | < 0.0001 |
| 连续性误差 | θ/φ方向梯度连续性 | 越小越好 | < 0.0005 |

**适用场景**:
- SSIM: 评估整体模式相似度（符合人类感知）
- 对称性误差: 检查物理约束（RCS应关于φ=0°对称）
- 连续性误差: 检测非物理突变

---

### 3️⃣ 统计一致性指标
**用途**: 评估重建分布与真实分布的匹配度

| 指标 | 含义 | 取值范围 | 典型好值 |
|-----|------|---------|---------|
| 皮尔逊相关系数 | 线性相关性 | -1到1 | > 0.95 |
| R²决定系数 | 解释方差比例 | 0-1 | > 0.85 |
| KL散度 | 分布差异 | ≥0 | < 0.1 |

**适用场景**:
- 相关系数: 评估线性关系（不受尺度影响）
- R²: 整体拟合优度（归一化指标）
- KL散度: 分布级别的差异度量

---

### 4️⃣ 频域指标
**用途**: 评估频率成分的保真度

| 指标 | 含义 | 作用 |
|-----|------|------|
| 频域幅度误差 | FFT幅度谱MSE | 检测能量损失 |
| 频域相位误差 | FFT相位谱MSE | 检测相位失真 |
| 功率谱密度误差 | PSD差异 | 检测能量分布变化 |
| 频率间一致性 | 1.5GHz与3GHz相对关系 | 检测频率间耦合 |

**适用场景**:
- 当空域指标良好但视觉效果差时，检查频域
- 高频细节损失检测
- 模糊/平滑问题诊断

---

### 5️⃣ RCS特定指标
**用途**: 针对RCS预测的特定需求

| 指标 | 定义 | 意义 |
|-----|------|------|
| 峰值保持度 | \|pred_peak - true_peak\| / true_peak | 主瓣/旁瓣峰值精度 |
| 零散射区准确度 | MSE(RCS < -30dB区域) | 低RCS区域预测能力 |
| 极化一致性 | HH/VV极化相对关系 | 极化间耦合关系 |

**适用场景**:
- 需要准确预测主瓣位置和强度
- 关注隐身特性（低RCS区域）
- 多极化应用

---

### 6️⃣ 系统性能指标
**用途**: 评估实用性

| 指标 | 含义 | 典型值 |
|-----|------|--------|
| 推理速度 | 样本/秒 | > 100 samples/s |
| 平均推理时间 | 单样本耗时（ms） | < 10 ms |
| 隐空间利用率 | 有效维度/总维度 | > 0.6 |
| 隐空间相关性 | 维度间平均相关系数 | < 0.3 (独立性) |

**适用场景**:
- 生产环境部署前的性能评估
- 模型压缩与加速决策
- 隐空间健康度诊断

---

## ✅ 二、当前项目已实现指标

### 已在GUI中使用
```python
✅ MSE (均方误差)
✅ RMSE (均方根误差)
✅ MAE (平均绝对误差)
✅ 按频率分解的 MSE/RMSE/MAE
```

### 已实现但未在GUI显示
```python
⚠️ SSIM (结构相似性)
⚠️ 相关系数
⚠️ R²决定系数
⚠️ 对称性误差
⚠️ 连续性误差
⚠️ 频域幅度/相位/功率误差
⚠️ 频率间一致性
⚠️ 负值比例
⚠️ KL散度
⚠️ 数值范围指标
```

**关键发现**: `autoencoder/evaluation/reconstruction_metrics.py` 已实现完整评估系统，但GUI只使用了3个基础指标！

---

## 🎯 三、推荐的最小指标集

如果只选择**5-6个核心指标**用于日常评估：

| 优先级 | 指标 | 理由 |
|--------|------|------|
| ⭐⭐⭐ | **MSE** | 训练优化核心指标 |
| ⭐⭐⭐ | **SSIM** | 结构质量黄金标准 |
| ⭐⭐⭐ | **R²决定系数** | 整体拟合优度 |
| ⭐⭐ | **推理速度** | 实用性指标 |
| ⭐⭐ | **相关系数** | 线性关系质量 |
| ⭐ | **隐空间利用率** | 模型健康度 |

---

## 📈 四、性能分析方法论

### 方法1: 快速诊断流程
```
1. 检查MSE/RMSE/MAE → 整体误差水平
   ↓ 误差大
2. 检查SSIM → 结构是否保持？
   ↓ SSIM低但MSE不太高
3. 检查频域指标 → 高频细节丢失？
   ↓ 频域误差大
4. 检查连续性误差 → 是否过度平滑？
```

### 方法2: 对比分析
```python
# 与基准对比
baseline_mse = nearest_neighbor_predictor.evaluate(test_data)
ae_mse = autoencoder_system.evaluate(test_data)
improvement = (baseline_mse - ae_mse) / baseline_mse * 100
print(f"相比最近邻基准改进: {improvement:.1f}%")
```

### 方法3: 残差分析
```python
# 空间残差分布
residual = predicted_rcs - true_rcs
plt.imshow(residual[..., 0])  # 可视化误差分布

# 残差vs真实值
plt.scatter(true_rcs.flatten(), residual.flatten())
# 理想情况: 随机分布在y=0附近
# 问题情况: 有系统性偏移/趋势
```

---

## 🚀 五、实施路线图

### ✅ 阶段1: 集成现有指标到GUI（1-2天）
**目标**: 充分利用已实现的`ReconstructionMetrics`类

**任务**:
- [x] 修改`evaluation_manager.py`调用`ReconstructionMetrics`
- [ ] 在GUI评估树中添加新指标节点
- [ ] 添加详细报告生成按钮

**预期效果**: 从3个指标扩展到20+指标

---

### ⏳ 阶段2: 添加可视化对比（1周）
**目标**: 图形化展示重建质量

**任务**:
- [ ] 残差分析图（热图、直方图、散点图）
- [ ] 频域对比图（FFT幅度谱）
- [ ] 切面对比图（θ=0°、φ=0°切面）

---

### ⏳ 阶段3: RCS特定指标（1周）
**目标**: 添加领域专用指标

**任务**:
- [ ] 峰值保持度指标
- [ ] 零散射区准确度
- [ ] 频率一致性评分
- [ ] 极化一致性（如有多极化数据）

---

### ⏳ 阶段4: 隐空间质量分析（1周）
**目标**: 诊断模型健康度

**任务**:
- [ ] 维度利用率分析
- [ ] 参数-隐空间映射质量
- [ ] 隐空间可视化（t-SNE/UMAP）

---

### ⏳ 阶段5: 对比基准系统（1-2周）
**目标**: 证明方法有效性

**任务**:
- [ ] 最近邻基准
- [ ] 线性插值基准
- [ ] 理论噪声下界估计
- [ ] 性能提升比计算

---

## 📋 六、常见问题与解答

### Q1: 为什么Train Loss > Val Loss?
**A**: Dropout=0.2导致，训练时20%神经元被关闭，验证时全部激活。这是正常现象。

### Q2: SSIM应该在什么数值范围？
**A**:
- SSIM > 0.9: 优秀（几乎完美重建）
- 0.8 < SSIM < 0.9: 良好（可接受）
- 0.6 < SSIM < 0.8: 一般（需要改进）
- SSIM < 0.6: 差（结构严重失真）

### Q3: 哪些指标应该关注变化趋势而非绝对值？
**A**:
- 训练曲线（应持续下降）
- 隐空间利用率（应随训练增加）
- 频域一致性（应随训练改善）

### Q4: 如何判断模型是否过拟合？
**A**:
- Val MSE持续上升而Train MSE下降
- 测试集SSIM显著低于验证集
- 隐空间相关性过高（>0.5）

### Q5: MSE很小但视觉效果差？
**A**:
- 检查SSIM（可能结构失真）
- 检查频域指标（可能高频丢失）
- 检查峰值保持度（主瓣可能偏移）

---

## 🔗 七、相关文件索引

### 核心实现
- `autoencoder/evaluation/reconstruction_metrics.py` - 完整指标计算
- `autoencoder/evaluation/ae_evaluator.py` - AE系统评估器
- `gui_managers/managers/evaluation_manager.py` - GUI评估管理

### 文档
- `CLAUDE.md` - 项目上下文
- `docs/DATA_PIPELINE.md` - 数据流程说明
- 本文档 - 评估指标指南

---

## 📞 使用示例

### 示例1: 完整评估
```python
from autoencoder.evaluation.reconstruction_metrics import ReconstructionMetrics

# 创建评估器
metrics_calc = ReconstructionMetrics(device='cuda')

# 计算所有指标
all_metrics = metrics_calc.compute_all_metrics(pred_rcs, true_rcs)

# 生成报告
report = metrics_calc.generate_report(all_metrics, detailed=True)
print(report)
```

### 示例2: 快速评估（仅基础指标）
```python
basic_metrics = metrics_calc.compute_basic_errors(pred_rcs, true_rcs)
print(f"MSE: {basic_metrics['mse']:.6f}")
print(f"RMSE: {basic_metrics['rmse']:.6f}")
print(f"MAE: {basic_metrics['mae']:.6f}")
```

### 示例3: SSIM专项评估
```python
ssim_metrics = metrics_calc.compute_ssim_metrics(pred_rcs, true_rcs)
print(f"平均SSIM: {ssim_metrics['ssim_mean']:.4f}")
print(f"SSIM标准差: {ssim_metrics['ssim_std']:.4f}")
print(f"最低SSIM: {ssim_metrics['ssim_min']:.4f}")
```

---

**维护者**: Claude Code
**最后更新**: 2025-01-19
