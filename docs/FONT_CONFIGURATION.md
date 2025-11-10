# 科研图片字体配置指南

> 为RCS AutoEncoder系统的所有可视化图片配置国际顶刊标准字体
>
> **配置文件**: `autoencoder/utils/plotting.py`
> **生效范围**: 所有通过plotting.py生成的图片（GUI + 批量实验）

---

## 📊 国际顶刊常用字体

### 推荐字体排序

| 优先级 | 字体 | 类型 | 使用期刊 | 系统状态 |
|-------|------|------|---------|---------|
| 1 | **Arial** | 无衬线 | Nature, Science, Cell, PNAS | ✅ 已安装 |
| 2 | Helvetica | 无衬线 | 专业期刊标准 | ❌ 未安装 |
| 3 | DejaVu Sans | 无衬线 | 开源，完整Unicode | ✅ 已安装 |
| 4 | Liberation Sans | 无衬线 | Linux标准 | ❌ 未安装 |

**当前配置**：matplotlib会按顺序查找，优先使用 **Arial**（已安装）

### 其他常用字体

- **Times New Roman** (已安装✅): 有衬线字体，传统学术风格，部分数学/物理期刊要求
- **Calibri**: Microsoft Office默认，商业报告常用
- **Computer Modern**: LaTeX默认，数学公式最佳

---

## ⚙️ 当前字体配置

### 字体族设置
```python
plt.rcParams['font.family'] = 'sans-serif'  # 无衬线字体族
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
```

### Unicode减号设置 ⭐ 重要
```python
plt.rcParams['axes.unicode_minus'] = False  # 禁用Unicode减号
```

**问题背景**：
- matplotlib默认使用Unicode减号 `U+2212` (−)，比普通连字符更美观
- 如果字体不支持该字符，会报错：`Font 'default' does not have a glyph for '\u2212'`

**解决方案**：
- 方案1: 使用支持Unicode的字体（Arial, DejaVu Sans）✅ 已配置
- 方案2: 禁用Unicode减号，使用普通连字符 `-` ✅ 已启用

**效果**：负数正常显示，无警告，图片质量不受影响

---

## 🎨 字号配置（科研标准）

| 元素 | 字号 | matplotlib参数 |
|------|------|---------------|
| 基础字号 | 10pt | `font.size` |
| 坐标轴标签 | 11pt | `axes.labelsize` |
| 子图标题 | 12pt | `axes.titlesize` |
| 整图标题 | 13pt | `figure.titlesize` |
| 刻度标签 | 10pt | `xtick/ytick.labelsize` |
| 图例 | 10pt | `legend.fontsize` |

**设计原则**：
- 层次清晰：标题 > 标签 > 正文
- 符合Nature, Science等顶刊投稿要求
- 屏幕显示和打印都清晰易读

---

## 🖼️ 高质量输出配置

### 分辨率设置
```python
plt.rcParams['figure.dpi'] = 100        # 屏幕显示：100 DPI
plt.rcParams['savefig.dpi'] = 300       # 文件保存：300 DPI（出版质量）
```

**说明**：
- **100 DPI**: GUI显示，平衡性能和质量
- **300 DPI**: 保存到文件，符合期刊要求（通常要求≥300 DPI）

### 布局设置
```python
plt.rcParams['savefig.bbox'] = 'tight'  # 紧凑布局，无多余空白
plt.rcParams['savefig.pad_inches'] = 0.1  # 适当边距
```

**效果**：图片边缘无大片空白，适合直接插入论文

### 线条和网格
```python
plt.rcParams['lines.linewidth'] = 1.5   # 数据线粗细
plt.rcParams['axes.linewidth'] = 1.0    # 坐标轴粗细
plt.rcParams['grid.linewidth'] = 0.5    # 网格线粗细
plt.rcParams['grid.alpha'] = 0.3        # 网格透明度
```

**效果**：专业的图表外观，清晰但不喧宾夺主

---

## 🔍 字体检查方法

### 快速检查
运行以下Python代码：
```python
import matplotlib.font_manager as fm

# 检查Arial是否可用
available = [f.name for f in fm.fontManager.ttflist]
print("Arial installed:", "Arial" in available)
```

### 完整检查脚本
```python
"""检查系统字体"""
import matplotlib.font_manager as fm

target_fonts = ['Arial', 'Helvetica', 'DejaVu Sans', 'Times New Roman']
available = [f.name for f in fm.fontManager.ttflist]

for font in target_fonts:
    status = "[OK]" if font in available else "[--]"
    print(f"{status} {font}")
```

### 系统字体状态（Windows）
```
[OK] Arial                - Installed ✓
[--] Helvetica            - Not found
[OK] DejaVu Sans          - Installed ✓
[--] Liberation Sans      - Not found
[OK] Times New Roman      - Installed ✓

Total: 404 fonts available
```

---

## 📝 常见问题

### Q1: 如何更改字体？

**A**: 修改 `autoencoder/utils/plotting.py` 中的配置：
```python
# 例如改为Times New Roman（有衬线字体）
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.family'] = 'serif'
```

### Q2: 为什么优先用Arial而不是Helvetica？

**A**:
- Helvetica是macOS的标准字体，Windows通常没有
- Arial是Helvetica的近似替代，Windows预装
- 两者视觉效果几乎相同，Arial可用性更好

### Q3: 如何安装缺失的字体？

**A**:

**Helvetica** (Windows):
- Helvetica是商业字体，需要购买
- 推荐使用Arial代替（视觉效果相似，免费）

**Liberation Sans** (Linux标准字体):
- 下载: [LibreOffice字体包](https://github.com/liberationfonts/liberation-fonts)
- 安装: 解压后双击.ttf文件安装
- 不是必需的（已有Arial和DejaVu Sans）

### Q4: 如何验证字体配置生效？

**A**: 生成一张图片，检查：
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [-1, -2, -3])  # 测试负号
ax.set_title('Font Test')
plt.savefig('font_test.png', dpi=300)
```

检查点：
- ✅ 文字清晰，使用Arial字体
- ✅ 负号正常显示，无警告
- ✅ 文件大小适中（300 DPI）

### Q5: 能否使用中文字体？

**A**: 当前配置不支持中文。如需中文：
```python
# 添加中文字体支持（例如微软雅黑）
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 保持禁用
```

**注意**：国际期刊通常要求全英文图片

---

## 🎯 最佳实践

### 论文投稿建议

1. **字体选择**：
   - ✅ 优先使用Arial（Nature, Science标准）
   - ✅ 或Times New Roman（传统期刊）
   - ❌ 避免使用非标准字体

2. **分辨率要求**：
   - ✅ 保存为300 DPI（已配置）
   - ✅ 文件格式：PNG, PDF, EPS
   - ❌ 不要使用JPEG（有损压缩）

3. **图片尺寸**：
   - 单栏图：宽度≤8.3 cm
   - 双栏图：宽度≤17.8 cm
   - 全页图：高度≤24.7 cm

4. **颜色选择**：
   - ✅ 使用色盲友好配色
   - ✅ 确保黑白打印时可区分
   - ❌ 避免红绿色对比

### 批量实验可视化

当前配置已应用于：
- ✅ 训练进度图 (`training_logs/`)
- ✅ RCS对比图 (`visualizations/`)
- ✅ 小波系数对比图 (`visualizations/`)
- ✅ 性能对比图表 (`comparison_plots/`)

所有图片统一使用Arial字体，300 DPI输出质量

---

## 📚 参考资料

### 期刊字体要求

- **Nature**: Arial or Helvetica, 5-7 pt最小字号
- **Science**: Arial or Helvetica, 6-8 pt推荐
- **Cell**: Arial, 8-12 pt推荐
- **IEEE**: Times New Roman或Arial
- **Springer**: Times Roman, Helvetica, Courier

### matplotlib文档

- [Text properties](https://matplotlib.org/stable/tutorials/text/text_props.html)
- [Customizing matplotlib](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
- [Font family](https://matplotlib.org/stable/gallery/text_labels_and_annotations/font_family_rc.html)

---

**配置版本**: v1.0
**最后更新**: 2025-01-11
**维护者**: Claude Code
**适用系统**: RCS AutoEncoder 可视化系统
