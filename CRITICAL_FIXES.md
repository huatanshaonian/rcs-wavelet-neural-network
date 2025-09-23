# 关键问题修复总结

## 发现的严重问题

### 问题1: 对称性约束的维度错误 ✅ 已修复

**问题描述:**
- RCS数据实际维度: `[batch, theta, phi, freq]`
  - theta维度 (第1维): 索引0-90 → 45°-135°
  - phi维度 (第2维): 索引0-90 → -45°-45°, phi=0°在索引45
- 原代码错误地假设: `[batch, phi, theta, freq]`
- 对称性约束应用在错误的维度上

**物理约束要求:**
- φ=0°平面对称: σ(φ, θ) = σ(-φ, θ)
- 即关于phi=0°对称,theta保持不变

**修复内容:**
1. `wavelet_network.py::_apply_phi_symmetry()` - 第459行
   - 修改前: `rcs_symmetric[:, left_idx, :, :]` (第1维)
   - 修改后: `rcs_symmetric[:, :, left_idx, :]` (第2维)

2. `wavelet_network.py::_symmetry_loss()` - 第554行
   - 修改前: `pred_rcs[:, left_idx, :, :]` (第1维)
   - 修改后: `pred_rcs[:, :, left_idx, :]` (第2维)

**影响:**
- 之前的对称性约束完全错误
- 导致重建图像的对称轴错误 (看到theta=90°对称而非phi=0°)

---

### 问题2: 损失函数与评估指标的尺度不匹配 ✅ 已修复

**问题描述:**
- 训练loss在对数域(dB)计算:
  - RCS数据 → 10*log10 → dB域 → 标准化 → 神经网络
  - loss = MSE(标准化后的dB值)

- 评估指标在对数域直接计算:
  - RMSE, R², 相关系数直接在标准化后的dB值上计算
  - **这是完全错误的!**

**为什么错误:**
1. RMSE应该在线性域计算,在对数域无物理意义
2. R²在对数域计算无法反映真实拟合优度
3. 导致: loss=0.01(很小) 但 RMSE=巨大, R²=负值

**正确流程:**
```
评估时应该:
1. 模型输出标准化dB值
2. 反标准化: dB = output * std + mean
3. dB转线性: RCS = 10^(dB/10)
4. 在线性域计算RMSE和R²
```

**修复内容:**
1. `evaluation.py::RCSEvaluator.__init__()` - 第55行
   - 添加参数: `use_log_output`, `preprocessing_stats`

2. `evaluation.py::_calculate_regression_metrics()` - 第149行
   - 添加对数域到线性域的转换逻辑
   - 转换公式:
     ```python
     pred_db = predictions * std + mean
     pred_linear = np.power(10, pred_db / 10)
     ```

**影响:**
- 之前的RMSE和R²完全不准确
- 修复后能正确反映模型性能

---

## 修复后的正确数据流

### 训练流程:
```
CSV数据 [theta, phi]
  ↓ 10*log10
dB域 [theta, phi]
  ↓ 标准化 (mean, std)
标准化dB [theta, phi]
  ↓ 堆叠 [models, theta, phi, freq]
训练数据
  ↓ 神经网络
预测输出 [batch, theta, phi, freq] (标准化dB域)
  ↓ 对称性约束 (正确的phi维度)
最终输出
  ↓ MSE loss (对数域)
训练loss
```

### 评估流程:
```
模型输出 (标准化dB域)
  ↓ 反标准化
dB域预测
  ↓ 10^(dB/10)
线性域预测
  ↓ RMSE, R², 相关系数
正确的评估指标
```

---

## 验证修复

### 如何验证对称性修复:
1. 可视化预测的RCS图
2. 检查是否关于phi=0°对称
3. theta方向不应有强制对称

### 如何验证评估指标修复:
1. 检查RMSE数值是否合理 (应该在1e-6到1e-1范围)
2. 检查R²是否在0-1之间 (负值说明有问题)
3. loss和RMSE应该有相关性

---

## 待办事项

### 1. GUI中传递preprocessing_stats ⚠️ 待完成
当前问题: GUI在评估时没有传递preprocessing_stats
- 需要在数据加载后保存: `self.preprocessing_stats = data_loader.preprocessing_stats`
- 需要在创建评估器时传递:
  ```python
  evaluator = RCSEvaluator(
      self.current_model,
      device,
      use_log_output=self.use_log_preprocessing.get(),
      preprocessing_stats=self.preprocessing_stats
  )
  ```

### 2. 测试完整训练流程
- 启用对数预处理训练
- 检查对称性是否正确
- 检查评估指标是否合理

---

## 影响范围

### 已修改文件:
1. ✅ `wavelet_network.py` - 对称性约束维度修复
2. ✅ `evaluation.py` - 评估指标域转换修复
3. ⚠️ `gui.py` - 需要传递preprocessing_stats (待完成)

### 需要重新训练:
- 所有使用对称性约束的模型需要重新训练
- 之前的评估结果不可信,需要重新评估

---

## 预期改进

修复后应该看到:
1. ✅ 对称性正确 (phi=0°平面对称)
2. ✅ RMSE数值合理 (不再是巨大的值)
3. ✅ R²在0-1之间 (不再是负值)
4. ✅ loss和RMSE有正相关
5. ✅ 可视化结果不再有闪烁斑点