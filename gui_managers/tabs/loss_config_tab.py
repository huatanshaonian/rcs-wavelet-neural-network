import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class LossConfigTab(ttk.Frame):
    """
    损失函数配置标签页
    负责配置模型训练时的损失函数权重和类型。
    """

    def __init__(self, notebook, app):
        """
        初始化损失函数配置标签页。

        参数:
            notebook: 父容器 (ttk.Notebook)
            app: 主应用程序实例 (RCSWaveletGUI)，用于访问共享状态和配置。
        """
        super().__init__(notebook)
        self.app = app

        # 初始化配置变量
        self.init_loss_config_vars()
        
        # 构建界面
        self.create_widgets()

    def init_loss_config_vars(self):
        """初始化损失函数配置变量"""
        # 基础损失函数配置
        self.use_mse_loss = tk.BooleanVar(value=True)
        self.mse_weight = tk.StringVar(value="0.8")

        self.use_huber_loss = tk.BooleanVar(value=False)
        self.huber_weight = tk.StringVar(value="0.7")
        self.huber_delta = tk.StringVar(value="0.1")

        self.use_l1_loss = tk.BooleanVar(value=False)
        self.l1_weight = tk.StringVar(value="0.5")

        # SSIM Loss (感知/结构损失)
        self.use_ssim_loss = tk.BooleanVar(value=False)
        self.ssim_weight = tk.StringVar(value="0.8")

        # 物理约束损失配置
        self.use_symmetry_loss = tk.BooleanVar(value=True)
        self.symmetry_weight = tk.StringVar(value="0.01")

        self.use_freq_consistency = tk.BooleanVar(value=False)
        self.freq_consistency_weight = tk.StringVar(value="0.02")
        self.freq_consistency_type = tk.StringVar(value="diff")

        self.use_continuity_loss = tk.BooleanVar(value=False)
        self.continuity_weight = tk.StringVar(value="0.02")
        self.continuity_type = tk.StringVar(value="standard")

        self.use_multiscale_loss = tk.BooleanVar(value=False)
        self.multiscale_weight = tk.StringVar(value="0.1")

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧面板：损失函数组件配置
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 预设配置组
        preset_group = ttk.LabelFrame(left_panel, text="预设配置")
        preset_group.pack(fill=tk.X, pady=(0, 10))

        preset_frame = ttk.Frame(preset_group)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(preset_frame, text="Original", command=self.load_original_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Enhanced", command=self.load_enhanced_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Robust", command=self.load_robust_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="High-Freq", command=self.load_highfreq_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Smooth", command=self.load_smooth_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Perceptual", command=self.load_perceptual_preset).pack(side=tk.LEFT, padx=2)

        # 基础损失函数组
        basic_group = ttk.LabelFrame(left_panel, text="基础损失函数")
        basic_group.pack(fill=tk.X, pady=(0, 10))

        # MSE Loss
        mse_frame = ttk.Frame(basic_group)
        mse_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(mse_frame, text="MSE Loss", variable=self.use_mse_loss).pack(side=tk.LEFT)
        ttk.Label(mse_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(mse_frame, textvariable=self.mse_weight, width=8).pack(side=tk.LEFT)

        # Huber Loss
        huber_frame = ttk.Frame(basic_group)
        huber_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(huber_frame, text="Huber Loss", variable=self.use_huber_loss).pack(side=tk.LEFT)
        ttk.Label(huber_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(huber_frame, textvariable=self.huber_weight, width=8).pack(side=tk.LEFT)
        ttk.Label(huber_frame, text="Delta:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(huber_frame, textvariable=self.huber_delta, width=8).pack(side=tk.LEFT)

        # L1 Loss
        l1_frame = ttk.Frame(basic_group)
        l1_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(l1_frame, text="L1 Loss", variable=self.use_l1_loss).pack(side=tk.LEFT)
        ttk.Label(l1_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(l1_frame, textvariable=self.l1_weight, width=8).pack(side=tk.LEFT)

        # SSIM Loss
        ssim_frame = ttk.Frame(basic_group)
        ssim_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(ssim_frame, text="SSIM Loss", variable=self.use_ssim_loss).pack(side=tk.LEFT)
        ttk.Label(ssim_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(ssim_frame, textvariable=self.ssim_weight, width=8).pack(side=tk.LEFT)

        # 物理约束损失组
        physics_group = ttk.LabelFrame(left_panel, text="物理约束损失")
        physics_group.pack(fill=tk.X, pady=(0, 10))

        # 对称性损失
        symmetry_frame = ttk.Frame(physics_group)
        symmetry_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(symmetry_frame, text="对称性约束", variable=self.use_symmetry_loss).pack(side=tk.LEFT)
        ttk.Label(symmetry_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(symmetry_frame, textvariable=self.symmetry_weight, width=8).pack(side=tk.LEFT)

        # 频率一致性损失
        freq_frame = ttk.Frame(physics_group)
        freq_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(freq_frame, text="频率一致性", variable=self.use_freq_consistency).pack(side=tk.LEFT)
        ttk.Label(freq_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(freq_frame, textvariable=self.freq_consistency_weight, width=8).pack(side=tk.LEFT)

        freq_type_frame = ttk.Frame(physics_group)
        freq_type_frame.pack(fill=tk.X, padx=20, pady=2)
        ttk.Label(freq_type_frame, text="类型:").pack(side=tk.LEFT)
        ttk.Radiobutton(freq_type_frame, text="差值", variable=self.freq_consistency_type, value="diff").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(freq_type_frame, text="相关性", variable=self.freq_consistency_type, value="correlation").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(freq_type_frame, text="局部窗口", variable=self.freq_consistency_type, value="local").pack(side=tk.LEFT, padx=5)

        # 连续性损失
        continuity_frame = ttk.Frame(physics_group)
        continuity_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(continuity_frame, text="空间连续性", variable=self.use_continuity_loss).pack(side=tk.LEFT)
        ttk.Label(continuity_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(continuity_frame, textvariable=self.continuity_weight, width=8).pack(side=tk.LEFT)

        continuity_type_frame = ttk.Frame(physics_group)
        continuity_type_frame.pack(fill=tk.X, padx=20, pady=2)
        ttk.Label(continuity_type_frame, text="类型:").pack(side=tk.LEFT)
        ttk.Radiobutton(continuity_type_frame, text="标准", variable=self.continuity_type, value="standard").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(continuity_type_frame, text="自适应", variable=self.continuity_type, value="adaptive").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(continuity_type_frame, text="分区域", variable=self.continuity_type, value="regional").pack(side=tk.LEFT, padx=5)

        # 多尺度损失
        multiscale_frame = ttk.Frame(physics_group)
        multiscale_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(multiscale_frame, text="多尺度损失", variable=self.use_multiscale_loss).pack(side=tk.LEFT)
        ttk.Label(multiscale_frame, text="权重:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(multiscale_frame, textvariable=self.multiscale_weight, width=8).pack(side=tk.LEFT)

        # 右侧面板：配置预览和控制
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))

        # 当前配置预览
        preview_group = ttk.LabelFrame(right_panel, text="当前配置")
        preview_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.loss_config_text = scrolledtext.ScrolledText(preview_group, width=40, height=20)
        self.loss_config_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 控制按钮
        control_group = ttk.LabelFrame(right_panel, text="配置管理")
        control_group.pack(fill=tk.X)

        control_frame = ttk.Frame(control_group)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="更新预览", command=self.update_loss_config_preview).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="应用配置", command=self.apply_loss_config).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="重置为默认", command=self.reset_loss_config).pack(fill=tk.X, pady=2)

        # 初始化预览
        self.update_loss_config_preview()

    def update_loss_config_preview(self):
        """更新损失函数配置预览"""
        try:
            config_text = "=== 当前损失函数配置 ===\n\n"

            # 基础损失函数
            config_text += "📊 基础损失函数:\n"
            if self.use_mse_loss.get():
                config_text += f"  ✅ MSE Loss (权重: {self.mse_weight.get()})\n"
            if self.use_huber_loss.get():
                config_text += f"  ✅ Huber Loss (权重: {self.huber_weight.get()}, Delta: {self.huber_delta.get()})\n"
            if self.use_l1_loss.get():
                config_text += f"  ✅ L1 Loss (权重: {self.l1_weight.get()})\n"
            if self.use_ssim_loss.get():
                config_text += f"  ✅ SSIM Loss (权重: {self.ssim_weight.get()})\n"

            # 物理约束损失
            config_text += "\n🔬 物理约束损失:\n"
            if self.use_symmetry_loss.get():
                config_text += f"  ✅ 对称性约束 (权重: {self.symmetry_weight.get()})\n"
            if self.use_freq_consistency.get():
                config_text += f"  ✅ 频率一致性 (权重: {self.freq_consistency_weight.get()}, 类型: {self.freq_consistency_type.get()})\n"
            if self.use_continuity_loss.get():
                config_text += f"  ✅ 空间连续性 (权重: {self.continuity_weight.get()}, 类型: {self.continuity_type.get()})\n"
            if self.use_multiscale_loss.get():
                config_text += f"  ✅ 多尺度损失 (权重: {self.multiscale_weight.get()})\n"

            # 计算总权重
            total_weight = 0
            if self.use_mse_loss.get():
                total_weight += float(self.mse_weight.get())
            if self.use_huber_loss.get():
                total_weight += float(self.huber_weight.get())
            if self.use_l1_loss.get():
                total_weight += float(self.l1_weight.get())
            if self.use_ssim_loss.get():
                total_weight += float(self.ssim_weight.get())
            if self.use_symmetry_loss.get():
                total_weight += float(self.symmetry_weight.get())
            if self.use_freq_consistency.get():
                total_weight += float(self.freq_consistency_weight.get())
            if self.use_continuity_loss.get():
                total_weight += float(self.continuity_weight.get())
            if self.use_multiscale_loss.get():
                total_weight += float(self.multiscale_weight.get())

            config_text += f"\n📈 总权重: {total_weight:.3f}\n"

            # 显示配置建议
            config_text += "\n💡 配置建议:\n"
            if total_weight > 2.0:
                config_text += "  ⚠️ 总权重较高，可能导致过度约束\n"
            elif total_weight < 0.5:
                config_text += "  ⚠️ 总权重较低，约束可能不足\n"
            else:
                config_text += "  ✅ 权重配置合理\n"

            if self.use_freq_consistency.get() and self.use_continuity_loss.get():
                config_text += "  ⚠️ 同时启用频率和连续性约束可能过度平滑\n"

            self.loss_config_text.delete(1.0, tk.END)
            self.loss_config_text.insert(1.0, config_text)

        except Exception as e:
            self.loss_config_text.delete(1.0, tk.END)
            self.loss_config_text.insert(1.0, f"配置预览错误: {e}")

    def apply_loss_config(self):
        """应用损失函数配置"""
        try:
            # 构建损失函数配置字典
            loss_config = {
                'use_mse': self.use_mse_loss.get(),
                'mse_weight': float(self.mse_weight.get()) if self.use_mse_loss.get() else 0,

                'use_huber': self.use_huber_loss.get(),
                'huber_weight': float(self.huber_weight.get()) if self.use_huber_loss.get() else 0,
                'huber_delta': float(self.huber_delta.get()) if self.use_huber_loss.get() else 0.1,

                'use_l1': self.use_l1_loss.get(),
                'l1_weight': float(self.l1_weight.get()) if self.use_l1_loss.get() else 0,

                'use_ssim': self.use_ssim_loss.get(),
                'ssim_weight': float(self.ssim_weight.get()) if self.use_ssim_loss.get() else 0,

                'use_symmetry': self.use_symmetry_loss.get(),
                'symmetry_weight': float(self.symmetry_weight.get()) if self.use_symmetry_loss.get() else 0,

                'use_freq_consistency': self.use_freq_consistency.get(),
                'freq_consistency_weight': float(self.freq_consistency_weight.get()) if self.use_freq_consistency.get() else 0,
                'freq_consistency_type': self.freq_consistency_type.get(),

                'use_continuity': self.use_continuity_loss.get(),
                'continuity_weight': float(self.continuity_weight.get()) if self.use_continuity_loss.get() else 0,
                'continuity_type': self.continuity_type.get(),

                'use_multiscale': self.use_multiscale_loss.get(),
                'multiscale_weight': float(self.multiscale_weight.get()) if self.use_multiscale_loss.get() else 0,
            }

            # 保存到训练配置中
            self.app.training_config['custom_loss_config'] = loss_config

            messagebox.showinfo("成功", "损失函数配置已应用！\n训练时将使用自定义损失函数。")
            self.app.log_message("损失函数配置已更新")

        except ValueError as e:
            messagebox.showerror("错误", f"权重值格式错误: {e}")
        except Exception as e:
            messagebox.showerror("错误", f"应用配置失败: {e}")

    def reset_loss_config(self):
        """重置损失函数配置为默认值"""
        self.use_mse_loss.set(True)
        self.mse_weight.set("0.8")

        self.use_huber_loss.set(False)
        self.huber_weight.set("0.7")
        self.huber_delta.set("0.1")

        self.use_l1_loss.set(False)
        self.l1_weight.set("0.5")

        self.use_ssim_loss.set(False)
        self.ssim_weight.set("0.8")

        self.use_symmetry_loss.set(True)
        self.symmetry_weight.set("0.01")

        self.use_freq_consistency.set(False)
        self.freq_consistency_weight.set("0.02")
        self.freq_consistency_type.set("diff")

        self.use_continuity_loss.set(False)
        self.continuity_weight.set("0.02")
        self.continuity_type.set("standard")

        self.use_multiscale_loss.set(False)
        self.multiscale_weight.set("0.1")

        self.update_loss_config_preview()
        messagebox.showinfo("完成", "损失函数配置已重置为默认值")

    def load_original_preset(self):
        """加载Original预设配置"""
        self.use_mse_loss.set(True)
        self.use_huber_loss.set(False)
        self.use_l1_loss.set(False)
        self.use_ssim_loss.set(False)

        self.use_symmetry_loss.set(True)
        self.symmetry_weight.set("0.02")

        self.use_freq_consistency.set(False)
        self.use_continuity_loss.set(False)

        self.use_multiscale_loss.set(True)
        self.multiscale_weight.set("0.1")

        self.update_loss_config_preview()

    def load_enhanced_preset(self):
        """加载Enhanced预设配置"""
        self.use_mse_loss.set(False)

        self.use_huber_loss.set(True)
        self.huber_weight.set("0.7")
        self.huber_delta.set("0.1")

        self.use_l1_loss.set(False)
        self.use_ssim_loss.set(False)

        self.use_symmetry_loss.set(True)
        self.symmetry_weight.set("0.01")

        self.use_freq_consistency.set(True)
        self.freq_consistency_weight.set("0.02")
        self.freq_consistency_type.set("diff")

        self.use_continuity_loss.set(True)
        self.continuity_weight.set("0.02")
        self.continuity_type.set("standard")

        self.use_multiscale_loss.set(False)

        self.update_loss_config_preview()

    def load_robust_preset(self):
        """加载鲁棒训练预设配置"""
        self.use_mse_loss.set(False)

        self.use_huber_loss.set(True)
        self.huber_weight.set("0.8")
        self.huber_delta.set("0.2")

        self.use_l1_loss.set(True)
        self.l1_weight.set("0.1")
        self.use_ssim_loss.set(False)

        self.use_symmetry_loss.set(True)
        self.symmetry_weight.set("0.005")

        self.use_freq_consistency.set(True)
        self.freq_consistency_weight.set("0.01")
        self.freq_consistency_type.set("correlation")

        self.use_continuity_loss.set(False)
        self.use_multiscale_loss.set(False)

        self.update_loss_config_preview()

    def load_highfreq_preset(self):
        """加载高频信息保持预设配置"""
        self.use_mse_loss.set(True)
        self.use_huber_loss.set(False)
        self.use_l1_loss.set(False)
        self.use_ssim_loss.set(False)

        self.use_symmetry_loss.set(True)
        self.symmetry_weight.set("0.005")

        self.use_freq_consistency.set(True)
        self.freq_consistency_weight.set("0.005")
        self.freq_consistency_type.set("local")

        self.use_continuity_loss.set(True)
        self.continuity_weight.set("0.005")
        self.continuity_type.set("adaptive")

        self.use_multiscale_loss.set(False)

        self.update_loss_config_preview()

    def load_smooth_preset(self):
        """加载平滑优化预设配置"""
        self.use_mse_loss.set(True)
        self.mse_weight.set("0.6")

        self.use_huber_loss.set(False)
        self.use_l1_loss.set(False)
        self.use_ssim_loss.set(False)

        self.use_symmetry_loss.set(True)
        self.symmetry_weight.set("0.02")

        self.use_freq_consistency.set(True)
        self.freq_consistency_weight.set("0.05")
        self.freq_consistency_type.set("diff")

        self.use_continuity_loss.set(True)
        self.continuity_weight.set("0.05")
        self.continuity_type.set("standard")

        self.use_multiscale_loss.set(True)
        self.multiscale_weight.set("0.1")

        self.update_loss_config_preview()

    def load_perceptual_preset(self):
        """加载感知优化预设配置 (包含SSIM)"""
        self.use_mse_loss.set(False)

        self.use_huber_loss.set(True)
        self.huber_weight.set("0.1")
        self.huber_delta.set("0.1")

        self.use_l1_loss.set(True)
        self.l1_weight.set("0.1")

        self.use_ssim_loss.set(True)
        self.ssim_weight.set("0.8")

        self.use_symmetry_loss.set(True)
        self.symmetry_weight.set("0.02")

        self.use_freq_consistency.set(False)
        self.use_continuity_loss.set(False)
        self.use_multiscale_loss.set(False)

        self.update_loss_config_preview()
