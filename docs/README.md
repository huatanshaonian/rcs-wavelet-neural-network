# 技术文档索引

本目录包含RCS AutoEncoder系统的详细技术文档。

---

## 📚 核心技术文档

### 架构设计

| 文档 | 描述 | 用途 |
|------|------|------|
| [autoencoder_design.md](autoencoder_design.md) | AutoEncoder系统设计文档 | 了解系统整体架构和设计思想 |
| [architecture/CNN_Receptive_Field_Analysis.md](architecture/CNN_Receptive_Field_Analysis.md) | CNN感受野分析与改进方案 | CNN架构优化参考 |
| [architecture/MLP_Architecture.md](architecture/MLP_Architecture.md) | MLP架构详解 | MLP网络设计参考 |

### 数据处理

| 文档 | 描述 | 用途 |
|------|------|------|
| [DATA_PIPELINE.md](DATA_PIPELINE.md) | 数据流程完整说明 | 理解数据预处理流程 |
| [DATA_PREPROCESSING_ANALYSIS.md](DATA_PREPROCESSING_ANALYSIS.md) | 数据预处理分析报告 | 预处理策略选择参考 |
| [WAVELET_CHANNEL_SEPARATION_ANALYSIS.md](WAVELET_CHANNEL_SEPARATION_ANALYSIS.md) | 小波通道分离分析 | 理解小波频带处理 |

### 模型实现

| 文档 | 描述 | 用途 |
|------|------|------|
| [DUAL_BRANCH_IMPLEMENTATION.md](DUAL_BRANCH_IMPLEMENTATION.md) | 双分支模型实现指南 | 双分支架构开发参考 |
| [DIFFERENTIABLE_WAVELET_SMALL_LATENT_REPORT.md](DIFFERENTIABLE_WAVELET_SMALL_LATENT_REPORT.md) | 可微分小波+小隐空间报告 | 小隐空间实现参考 |
| [ADAPTIVE_LAYERS_GUIDE.md](ADAPTIVE_LAYERS_GUIDE.md) | 自适应层实现指南 | 支持小隐空间的层设计 |

### 注意力机制

| 文档 | 描述 | 用途 |
|------|------|------|
| [CHANNEL_ATTENTION_USAGE.md](CHANNEL_ATTENTION_USAGE.md) | 通道注意力使用指南 | 添加通道注意力模块 |
| [ATTENTION_MECHANISM_EXPLAINED.md](ATTENTION_MECHANISM_EXPLAINED.md) | 注意力机制详细说明 | 理解注意力机制原理 |
| [ATTENTION_STANDARDIZATION_SOLUTIONS.md](ATTENTION_STANDARDIZATION_SOLUTIONS.md) | 注意力标准化解决方案 | 注意力模块设计参考 |

### 训练监控

| 文档 | 描述 | 用途 |
|------|------|------|
| [GRADIENT_MONITORING_GUIDE.md](GRADIENT_MONITORING_GUIDE.md) | 梯度监控完整指南 | 添加梯度监控功能 |

### 可视化配置

| 文档 | 描述 | 用途 |
|------|------|------|
| [FONT_CONFIGURATION.md](FONT_CONFIGURATION.md) | 科研图片字体配置指南 | 配置论文级别图表字体 |

---

## 🗂️ 文档分类

### 入门必读
1. [autoencoder_design.md](autoencoder_design.md) - 系统整体设计
2. [DATA_PIPELINE.md](DATA_PIPELINE.md) - 数据流程

### 架构开发
- CNN架构：[CNN_Receptive_Field_Analysis.md](architecture/CNN_Receptive_Field_Analysis.md)
- MLP架构：[MLP_Architecture.md](architecture/MLP_Architecture.md)
- 双分支架构：[DUAL_BRANCH_IMPLEMENTATION.md](DUAL_BRANCH_IMPLEMENTATION.md)

### 高级特性
- 小隐空间支持：[ADAPTIVE_LAYERS_GUIDE.md](ADAPTIVE_LAYERS_GUIDE.md)
- 通道注意力：[CHANNEL_ATTENTION_USAGE.md](CHANNEL_ATTENTION_USAGE.md)
- 梯度监控：[GRADIENT_MONITORING_GUIDE.md](GRADIENT_MONITORING_GUIDE.md)

### 分析报告
- 数据预处理：[DATA_PREPROCESSING_ANALYSIS.md](DATA_PREPROCESSING_ANALYSIS.md)
- 小波通道分离：[WAVELET_CHANNEL_SEPARATION_ANALYSIS.md](WAVELET_CHANNEL_SEPARATION_ANALYSIS.md)
- 可微分小波：[DIFFERENTIABLE_WAVELET_SMALL_LATENT_REPORT.md](DIFFERENTIABLE_WAVELET_SMALL_LATENT_REPORT.md)

---

## 📌 快速查找

**想要实现新功能？**
- 添加新模型架构 → `architecture/` 目录
- 添加注意力机制 → `CHANNEL_ATTENTION_USAGE.md`
- 支持小隐空间 → `ADAPTIVE_LAYERS_GUIDE.md`
- 监控训练梯度 → `GRADIENT_MONITORING_GUIDE.md`

**遇到问题？**
- 数据预处理问题 → `DATA_PIPELINE.md`
- 小波变换问题 → `WAVELET_CHANNEL_SEPARATION_ANALYSIS.md`
- 架构设计问题 → `autoencoder_design.md`

**优化模型？**
- CNN感受野优化 → `architecture/CNN_Receptive_Field_Analysis.md`
- MLP层数优化 → `architecture/MLP_Architecture.md`
- 数据预处理优化 → `DATA_PREPROCESSING_ANALYSIS.md`

---

**最后更新**: 2025-01-20
