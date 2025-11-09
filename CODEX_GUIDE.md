修改步骤总结

  步骤1：在绘图函数开始处添加字号缩放因子获取代码

  在任何设置matplotlib文字元素之前，添加这段代码：

  # 获取字号缩放因子
  try:
      fontsize_scale = self.gui.fontsize_scale_var.get()
      # 限制范围在0.5-3.0之间
      fontsize_scale = max(0.5, min(3.0, fontsize_scale))
  except:
      fontsize_scale = 1.0

  步骤2：将所有固定字号改为动态计算

  修改前：
  ax.set_title('标题', fontsize=12)
  ax.set_xlabel('X轴', fontsize=10)
  ax.set_ylabel('Y轴', fontsize=10)
  ax.tick_params(axis='both', labelsize=9)

  修改后：
  ax.set_title('标题', fontsize=int(24*fontsize_scale), fontweight='bold')
  ax.set_xlabel('X轴', fontsize=int(20*fontsize_scale), fontweight='bold')
  ax.set_ylabel('Y轴', fontsize=int(20*fontsize_scale), fontweight='bold')
  ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))

  步骤3：处理colorbar

  修改前：
  self.gui.vis_fig.colorbar(im, ax=ax, label='RCS (dB)')

  修改后：
  cbar = self.gui.vis_fig.colorbar(im, ax=ax, label='RCS (dB)')
  cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
  cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

  步骤4：处理legend（如果有）

  修改前：
  ax.legend()

  修改后：
  ax.legend(fontsize=int(14*fontsize_scale))

  步骤5：处理文本注释（如果有）

  修改前：
  ax.text(0.5, 0.5, '文本', fontsize=10)

  修改后：
  ax.text(0.5, 0.5, '文本', fontsize=int(20*fontsize_scale))

  ---
  推荐的基础字号标准

  为了统一，建议使用以下基础字号：

  | 元素                    | 基础字号 | 说明      |
  |-----------------------|------|---------|
  | 标题 (title)            | 24   | 最大、最醒目  |
  | 坐标轴标签 (xlabel/ylabel) | 20   | 次要标题    |
  | Colorbar标签            | 20   | 与坐标轴同级  |
  | 刻度标签 (tick labels)    | 16   | 数值标注    |
  | 图例 (legend)           | 14   | 说明文字    |
  | 文本注释 (text)           | 20   | 根据重要性调整 |

  ---
  完整示例：修改3D表面图

  原始代码（gui_managers/managers/visualization_manager.py:87-94）：
  ax.set_xlabel('θ (俯仰角, °)')
  ax.set_ylabel('φ (偏航角, °)')
  ax.set_zlabel('RCS (dB)')
  ax.set_title(f'模型 {model_id} - {freq} RCS 3D表面图')
  self.gui.vis_fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='RCS (dB)')

  修改后：
  # 获取字号缩放因子
  try:
      fontsize_scale = self.gui.fontsize_scale_var.get()
      fontsize_scale = max(0.5, min(3.0, fontsize_scale))
  except:
      fontsize_scale = 1.0

  ax.set_xlabel('θ (俯仰角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
  ax.set_ylabel('φ (偏航角, °)', fontsize=int(20*fontsize_scale), fontweight='bold')
  ax.set_zlabel('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
  ax.set_title(f'模型 {model_id} - {freq} RCS 3D表面图', fontsize=int(24*fontsize_scale), fontweight='bold')
  ax.tick_params(axis='both', labelsize=int(16*fontsize_scale))

  cbar = self.gui.vis_fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='RCS (dB)')
  cbar.set_label('RCS (dB)', fontsize=int(20*fontsize_scale), fontweight='bold')
  cbar.ax.tick_params(labelsize=int(16*fontsize_scale))

  ---
  需要修改的文件位置

  所有绘图函数都在：
  - 文件路径：G:\feko_data\wavelet\gui_managers\managers\visualization_manager.py

  需要修改的函数列表：
  1. _plot_2d_heatmap ✅ 已完成
  2. _plot_3d_surface (line 60)
  3. _plot_spherical (line 107)
  4. _plot_comparison (line 150+)
  5. _plot_difference_analysis
  6. _plot_correlation_analysis
  7. _plot_training_history
  8. _plot_wavelet_coefficients_comparison
  9. _plot_ae_comparison
  10. _plot_ae_2d_heatmap ✅ 已完成（虽然不常用）
  11. _plot_original_rcs_fallback ✅ 已完成
  12. 以及其他所有 _plot_* 函数

  ---
  注意事项

  1. 所有绘图函数都通过 self.gui.fontsize_scale_var.get() 获取缩放因子
  2. 务必使用 int() 转换，因为fontsize需要整数
  3. 添加 try-except，防止变量不存在时报错
  4. 限制范围在0.5-3.0，防止极端值
  5. 为重要文字添加 fontweight='bold' 增强可读性

  ---
  这样codex就可以按照这个模板批量修改所有绘图函数了！