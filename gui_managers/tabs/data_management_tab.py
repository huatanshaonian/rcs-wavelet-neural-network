import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import sys
import os
import torch
import gc

# 尝试导入项目模块
try:
    from training import create_data_config
except ImportError:
    # 简单的回退，防止导入失败导致崩溃（虽然在完整环境中应该能导入）
    def create_data_config(use_log_preprocessing=False):
        return {'preprocessing': {}}

class DataManagementTab(ttk.Frame):
    """
    数据管理标签页
    负责数据的加载、预处理配置、缓存管理和系统资源管理。
    """
    
    def __init__(self, notebook, app):
        """
        初始化数据管理标签页
        
        参数:
            notebook: 父容器 (ttk.Notebook)
            app: 主应用程序实例 (RCSWaveletGUI)，用于访问共享状态和配置
        """
        super().__init__(notebook)
        self.app = app
        
        # 引用主应用的字体
        self.font_small = getattr(app, 'font_small', None)
        
        # 初始化UI变量
        # 注意：部分变量如果是全局共享的（如ae_freq_config），则直接引用app中的
        self.params_path_var = tk.StringVar(value=self.app.data_config.get('params_file', ''))
        self.rcs_dir_var = tk.StringVar(value=self.app.data_config.get('rcs_data_dir', ''))
        self.model_start_var = tk.StringVar(value="1")
        self.model_end_var = tk.StringVar(value="100")
        
        # 预处理相关变量
        self.use_log_preprocessing = tk.BooleanVar(value=False)
        self.log_epsilon_var = tk.StringVar(value="1e-10")
        self.normalize_after_log = tk.BooleanVar(value=True)
        
        # 构建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 数据配置组
        config_group = ttk.LabelFrame(main_frame, text="数据配置")
        config_group.pack(fill=tk.X, pady=(0, 10))

        # 参数文件路径
        ttk.Label(config_group, text="参数文件:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.params_path_entry = ttk.Entry(config_group, textvariable=self.params_path_var, width=50)
        self.params_path_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(config_group, text="浏览", command=self.browse_params_file).grid(
            row=0, column=2, padx=5, pady=5)

        # RCS数据目录
        ttk.Label(config_group, text="RCS数据目录:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.rcs_dir_entry = ttk.Entry(config_group, textvariable=self.rcs_dir_var, width=50)
        self.rcs_dir_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(config_group, text="浏览", command=self.browse_rcs_dir).grid(
            row=1, column=2, padx=5, pady=5)

        # 模型ID范围
        ttk.Label(config_group, text="模型ID范围:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        range_frame = ttk.Frame(config_group)
        range_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(range_frame, text="从:").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.model_start_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_frame, text="到:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(range_frame, textvariable=self.model_end_var, width=5).pack(side=tk.LEFT, padx=2)

        # 频率配置
        # 注意：这里使用 app.ae_freq_config，因为这个配置可能被其他模块共享
        ttk.Label(config_group, text="频率配置:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5)
        freq_config_frame = ttk.Frame(config_group)
        freq_config_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        if hasattr(self.app, 'ae_freq_config'):
            freq_combo = ttk.Combobox(freq_config_frame, textvariable=self.app.ae_freq_config,
                                     values=["2freq", "3freq"], state="readonly", width=10)
            freq_combo.pack(side=tk.LEFT)
            ttk.Label(freq_config_frame, text="(2freq: 1.5GHz+3GHz, 3freq: +6GHz)",
                     font=self.font_small).pack(side=tk.LEFT, padx=(10, 0))
        else:
            ttk.Label(freq_config_frame, text="频率配置变量未找到", foreground="red").pack(side=tk.LEFT)

        # 操作按钮
        button_frame = ttk.Frame(config_group)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="加载数据", command=self.load_data,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="数据预览", command=self.preview_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="数据统计", command=self.show_data_stats).pack(side=tk.LEFT, padx=5)

        # 数据预处理配置组
        preprocessing_group = ttk.LabelFrame(main_frame, text="数据预处理")
        preprocessing_group.pack(fill=tk.X, pady=(10, 10))

        preprocessing_frame = ttk.Frame(preprocessing_group)
        preprocessing_frame.pack(fill=tk.X, padx=5, pady=5)

        # 对数预处理选项
        # 将此变量暴露给 app，以便保存模型时使用
        self.app.use_log_preprocessing = self.use_log_preprocessing 
        
        ttk.Checkbutton(preprocessing_frame, text="启用对数预处理",
                       variable=self.use_log_preprocessing,
                       command=self.on_preprocessing_change).pack(side=tk.LEFT)

        # 预处理参数
        params_frame = ttk.Frame(preprocessing_frame)
        params_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(params_frame, text="ε值:").grid(row=0, column=0, sticky=tk.W, pady=2)
        # 将此变量暴露给 app
        self.app.log_epsilon_var = self.log_epsilon_var
        self.log_epsilon_entry = ttk.Entry(params_frame, textvariable=self.log_epsilon_var, width=10)
        self.log_epsilon_entry.grid(row=0, column=1, padx=5, pady=2)
        self.log_epsilon_entry.configure(state=tk.DISABLED)

        # 将此变量暴露给 app
        self.app.normalize_after_log = self.normalize_after_log
        self.normalize_checkbox = ttk.Checkbutton(params_frame, text="对数后标准化",
                                                variable=self.normalize_after_log)
        self.normalize_checkbox.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.normalize_checkbox.configure(state=tk.DISABLED)

        # 预处理说明
        info_frame = ttk.Frame(preprocessing_group)
        info_frame.pack(fill=tk.X, padx=5, pady=2)

        info_text = "对数预处理将RCS数据转换为log10域，有助于处理大动态范围数据。建议在训练前启用以改善收敛性能。"
        ttk.Label(info_frame, text=info_text, font=self.font_small,
                 foreground="gray").pack(side=tk.LEFT)

        # 缓存管理组
        cache_group = ttk.LabelFrame(main_frame, text="数据缓存管理")
        cache_group.pack(fill=tk.X, pady=(10, 0))

        cache_frame = ttk.Frame(cache_group)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)

        # 缓存控制按钮
        ttk.Button(cache_frame, text="查看缓存信息", command=self.show_cache_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(cache_frame, text="清除所有缓存", command=self.clear_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(cache_frame, text="强制重新读取", command=self.force_reload_data).pack(side=tk.LEFT, padx=5)

        # 缓存说明
        cache_info_label = ttk.Label(cache_group,
                                   text="缓存功能可以避免重复的CSV文件读取，大幅提高数据加载速度。\n当参数文件或RCS数据发生变化时，缓存会自动更新。",
                                   font=self.font_small)
        cache_info_label.pack(padx=5, pady=(0, 5))

        # 系统管理组
        system_group = ttk.LabelFrame(main_frame, text="系统管理")
        system_group.pack(fill=tk.X, pady=(10, 0))

        system_frame = ttk.Frame(system_group)
        system_frame.pack(fill=tk.X, padx=5, pady=5)

        # 系统管理按钮
        ttk.Button(system_frame, text="重置CUDA", command=self.reset_cuda_manually).pack(side=tk.LEFT, padx=5)
        ttk.Button(system_frame, text="检查CUDA状态", command=self.check_cuda_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(system_frame, text="清理GPU内存", command=self.clean_gpu_memory).pack(side=tk.LEFT, padx=5)

        # 系统说明
        system_info_label = ttk.Label(system_group,
                                    text="CUDA重置功能可以解决GPU内存错误和训练启动问题。\n建议在遇到CUDA错误时使用重置功能。",
                                    font=self.font_small)
        system_info_label.pack(padx=5, pady=(0, 5))

        # 数据信息显示
        info_group = ttk.LabelFrame(main_frame, text="数据信息")
        info_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.data_info_text = scrolledtext.ScrolledText(info_group, height=15)
        self.data_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 将data_info_text暴露给app，以便日志重定向可以写入
        self.app.data_info_text = self.data_info_text

    def browse_params_file(self):
        """浏览参数文件"""
        filename = filedialog.askopenfilename(
            title="选择参数文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.params_path_var.set(filename)
            self.app.data_config['params_file'] = filename

    def browse_rcs_dir(self):
        """浏览RCS数据目录"""
        dirname = filedialog.askdirectory(title="选择RCS数据目录")
        if dirname:
            self.rcs_dir_var.set(dirname)
            self.app.data_config['rcs_data_dir'] = dirname

    def load_data(self):
        """加载数据"""
        try:
            self.app.status_var.set("正在加载数据...")
            self.app.root.update()

            # 更新数据配置
            self.app.data_config['params_file'] = self.params_path_var.get()
            self.app.data_config['rcs_data_dir'] = self.rcs_dir_var.get()

            start_id = int(self.model_start_var.get())
            end_id = int(self.model_end_var.get())
            self.app.data_config['model_ids'] = [f"{i:03d}" for i in range(start_id, end_id + 1)]

            # 根据频率配置更新frequencies列表
            freq_config = self.app.ae_freq_config.get()
            self.app.log_message(f"🔍 检查频率配置变量: {freq_config}")
            if freq_config == "3freq":
                self.app.data_config['frequencies'] = ['1.5G', '3G', '6G']
                self.app.log_message("✓ 频率配置: 3频率 (1.5GHz + 3GHz + 6GHz)")
            else:
                self.app.data_config['frequencies'] = ['1.5G', '3G']
                self.app.log_message("✓ 频率配置: 2频率 (1.5GHz + 3GHz)")
            self.app.log_message(f"📋 实际传递给缓存管理器的frequencies: {self.app.data_config['frequencies']}")

            # 使用缓存加载数据
            self.app.log_message("开始加载数据（支持缓存加速）...")
            self.app.param_data, self.app.rcs_data = self.app.cache_manager.load_data_with_cache(
                params_file=self.app.data_config['params_file'],
                rcs_data_dir=self.app.data_config['rcs_data_dir'],
                model_ids=self.app.data_config['model_ids'],
                frequencies=self.app.data_config['frequencies']
            )

            # 保存原始RCS数据副本，确保AutoEncoder训练使用线性域数据
            self.app._original_rcs_data = self.app.rcs_data.copy()
            self.app.log_message("已保存原始RCS数据副本，供AutoEncoder使用")

            self.app.data_loaded = True
            self.app.log_message("数据加载成功！")
            self.app.log_message(f"参数数据形状: {self.app.param_data.shape}")
            self.app.log_message(f"RCS数据形状: {self.app.rcs_data.shape}")

            # 验证频率配置匹配性（如果已加载模型）
            if hasattr(self.app, 'ae_system') and self.app.ae_system is not None:
                config_info = self.app.ae_system.get('config_info', {})
                model_num_freq = config_info.get('num_frequencies', None)
                model_freq_labels = config_info.get('frequency_labels', [])
                data_num_freq = self.app.rcs_data.shape[-1]

                if model_num_freq is not None:
                    self.app.log_message(f"检查频率配置: 模型={model_num_freq}频, 数据={data_num_freq}频")
                    if model_num_freq != data_num_freq:
                        warning_msg = (
                            f"⚠️ 频率配置不匹配！\n\n"
                            f"已加载模型: {model_num_freq}频 {model_freq_labels}\n"
                            f"当前数据: {data_num_freq}频\n\n"
                            f"建议重新加载匹配的模型或数据！"
                        )
                        self.app.log_message(f"❌ {warning_msg}")
                        messagebox.showwarning("频率配置不匹配", warning_msg)
                    else:
                        self.app.log_message(f"✅ 频率配置匹配：{data_num_freq}频")

            # 更新AutoEncoder扩展界面的模型选择列表
            if hasattr(self.app, 'ae_extension') and self.app.ae_extension is not None:
                self.app.ae_extension._update_model_selection()
                self.app.log_message("已更新小波分析模型选择列表")

            # 同步更新 ae_system 中的数据引用
            if hasattr(self.app, 'ae_system') and self.app.ae_system is not None:
                self.app.ae_system['rcs_data'] = self.app.rcs_data
                self.app.ae_system['param_data'] = self.app.param_data
                self.app.log_message("已同步更新AutoEncoder系统中的数据引用")

            self.app.status_var.set("数据加载完成")

        except Exception as e:
            self.app.log_message(f"数据加载失败: {str(e)}")
            self.app.status_var.set("数据加载失败")
            messagebox.showerror("错误", f"数据加载失败:\n{str(e)}")

    def preview_data(self):
        """预览数据"""
        if not self.app.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 显示参数数据预览
        preview_text = "=== 参数数据预览 ===\n"
        preview_text += f"数据形状: {self.app.param_data.shape}\n"
        preview_text += f"前5个样本:\n{self.app.param_data[:5]}\n\n"

        preview_text += "=== RCS数据预览 ===\n"
        preview_text += f"数据形状: {self.app.rcs_data.shape}\n"

        # 原始数据统计
        first_sample = self.app.rcs_data[0]
        preview_text += f"原始线性数据 - 第一个样本:\n"
        preview_text += f"  1.5GHz - 范围: [{np.min(first_sample[:,:,0]):.6e}, {np.max(first_sample[:,:,0]):.6e}]\n"
        preview_text += f"  3GHz - 范围: [{np.min(first_sample[:,:,1]):.6e}, {np.max(first_sample[:,:,1]):.6e}]\n"

        # 如果启用了对数预处理，显示对数化后的数据
        if self.use_log_preprocessing.get():
            epsilon = float(self.log_epsilon_var.get()) if self.log_epsilon_var.get() else 1e-10

            # 计算对数化数据 (转换为分贝值: 10 * log10)
            rcs_db_sample = 10 * np.log10(np.maximum(first_sample, epsilon))
            preview_text += f"\n对数化数据 (dB) - 第一个样本:\n"
            preview_text += f"  1.5GHz - 范围: [{np.min(rcs_db_sample[:,:,0]):.1f}, {np.max(rcs_db_sample[:,:,0]):.1f}] dB\n"
            preview_text += f"  3GHz - 范围: [{np.min(rcs_db_sample[:,:,1]):.1f}, {np.max(rcs_db_sample[:,:,1]):.1f}] dB\n"

            # 如果启用了标准化，显示标准化后的数据
            if self.normalize_after_log.get():
                # 计算全局统计用于标准化
                all_rcs_db = 10 * np.log10(np.maximum(self.app.rcs_data, epsilon))
                global_mean = np.mean(all_rcs_db)
                global_std = np.std(all_rcs_db)

                normalized_sample = (rcs_db_sample - global_mean) / global_std
                preview_text += f"\n标准化后数据 (μ=0, σ=1) - 第一个样本:\n"
                preview_text += f"  1.5GHz - 范围: [{np.min(normalized_sample[:,:,0]):.3f}, {np.max(normalized_sample[:,:,0]):.3f}]\n"
                preview_text += f"  3GHz - 范围: [{np.min(normalized_sample[:,:,1]):.3f}, {np.max(normalized_sample[:,:,1]):.3f}]\n"
                preview_text += f"  全局统计: 均值={global_mean:.1f} dB, 标准差={global_std:.1f} dB\n"
        else:
            preview_text += f"\n提示: 启用对数预处理以查看预处理后的数据范围\n"

        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(tk.END, preview_text)

    def show_data_stats(self):
        """显示数据统计"""
        if not self.app.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        stats_text = "=== 详细数据统计 ===\n\n"

        # 参数统计
        stats_text += "参数数据统计:\n"
        for i in range(self.app.param_data.shape[1]):
            param_col = self.app.param_data[:, i]
            stats_text += f"参数 {i+1}: 均值={np.mean(param_col):.4f}, "
            stats_text += f"标准差={np.std(param_col):.4f}, "
            stats_text += f"范围=[{np.min(param_col):.4f}, {np.max(param_col):.4f}]\n"

        stats_text += "\n原始RCS数据统计 (线性值):\n"
        for freq_idx, freq_name in enumerate(['1.5GHz', '3GHz']):
            freq_data = self.app.rcs_data[:, :, :, freq_idx]
            stats_text += f"{freq_name}: 均值={np.mean(freq_data):.6e}, "
            stats_text += f"标准差={np.std(freq_data):.6e}, "
            stats_text += f"范围=[{np.min(freq_data):.6e}, {np.max(freq_data):.6e}]\n"

        # 如果启用了对数预处理，显示对数化后的统计
        if self.use_log_preprocessing.get():
            epsilon = float(self.log_epsilon_var.get()) if self.log_epsilon_var.get() else 1e-10

            stats_text += f"\n对数化RCS数据统计 (dB, ε={epsilon}):\n"
            # 转换为分贝值: 10 * log10
            rcs_db_data = 10 * np.log10(np.maximum(self.app.rcs_data, epsilon))

            for freq_idx, freq_name in enumerate(['1.5GHz', '3GHz']):
                freq_db_data = rcs_db_data[:, :, :, freq_idx]
                stats_text += f"{freq_name}: 均值={np.mean(freq_db_data):.1f} dB, "
                stats_text += f"标准差={np.std(freq_db_data):.1f} dB, "
                stats_text += f"范围=[{np.min(freq_db_data):.1f}, {np.max(freq_db_data):.1f}] dB\n"

            # 全局对数统计
            global_db_mean = np.mean(rcs_db_data)
            global_db_std = np.std(rcs_db_data)
            stats_text += f"全局dB统计: 均值={global_db_mean:.1f} dB, 标准差={global_db_std:.1f} dB\n"

            # 如果启用了标准化，显示标准化后的统计
            if self.normalize_after_log.get():
                normalized_data = (rcs_db_data - global_db_mean) / global_db_std
                stats_text += f"\n标准化后数据统计 (μ=0, σ=1):\n"

                for freq_idx, freq_name in enumerate(['1.5GHz', '3GHz']):
                    freq_norm_data = normalized_data[:, :, :, freq_idx]
                    stats_text += f"{freq_name}: 均值={np.mean(freq_norm_data):.3f}, "
                    stats_text += f"标准差={np.std(freq_norm_data):.3f}, "
                    stats_text += f"范围=[{np.min(freq_norm_data):.3f}, {np.max(freq_norm_data):.3f}]\n"

                # 数据动态范围比较
                original_range = np.max(self.app.rcs_data) - np.min(self.app.rcs_data)
                db_range = np.max(rcs_db_data) - np.min(rcs_db_data)
                norm_range = np.max(normalized_data) - np.min(normalized_data)

                stats_text += f"\n数据动态范围对比:\n"
                stats_text += f"原始数据 (线性): {original_range:.6e}\n"
                stats_text += f"对数化后 (dB): {db_range:.1f} dB\n"
                stats_text += f"标准化后 (无量纲): {norm_range:.3f}\n"
                stats_text += f"动态范围压缩比: {original_range/norm_range:.2e}\n"

        else:
            stats_text += f"\n提示: 启用对数预处理以查看预处理后的详细统计信息\n"

        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(tk.END, stats_text)

    def show_cache_info(self):
        """显示缓存信息"""
        try:
            # 创建新窗口显示缓存信息
            cache_window = tk.Toplevel(self.app.root)
            cache_window.title("缓存信息")
            cache_window.geometry("800x600")
            cache_window.resizable(True, True)

            # 创建文本区域
            cache_text = scrolledtext.ScrolledText(cache_window, wrap=tk.WORD)
            cache_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 重定向输出到文本区域
            original_stdout = sys.stdout

            class CacheInfoRedirector:
                def __init__(self, text_widget):
                    self.text_widget = text_widget
                    self.content = ""

                def write(self, message):
                    self.content += message
                    self.text_widget.insert(tk.END, message)
                    self.text_widget.see(tk.END)
                    cache_window.update()

                def flush(self):
                    pass

            redirector = CacheInfoRedirector(cache_text)
            sys.stdout = redirector

            try:
                self.app.cache_manager.list_cache_info()
            finally:
                sys.stdout = original_stdout

            # 添加关闭按钮
            button_frame = ttk.Frame(cache_window)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            ttk.Button(button_frame, text="关闭", command=cache_window.destroy).pack(side=tk.RIGHT)

        except Exception as e:
            messagebox.showerror("错误", f"显示缓存信息失败:\n{str(e)}")

    def clear_cache(self):
        """清除所有缓存"""
        try:
            # 确认对话框
            result = messagebox.askyesno(
                "确认清除",
                "确定要清除所有数据缓存吗？\n这将删除所有已保存的缓存文件，下次加载数据时需要重新从CSV文件读取。"
            )

            if result:
                self.app.log_message("正在清除数据缓存...")
                self.app.cache_manager.clear_cache()
                self.app.log_message("✅ 缓存清除完成")
                messagebox.showinfo("完成", "所有缓存已清除")

        except Exception as e:
            error_msg = f"清除缓存失败: {str(e)}"
            self.app.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def force_reload_data(self):
        """强制重新读取数据（忽略缓存）"""
        if not self.params_path_var.get() or not self.rcs_dir_var.get():
            messagebox.showwarning("警告", "请先配置数据路径")
            return

        try:
            self.app.log_message("强制重新读取数据（忽略缓存）...")
            self.app.status_var.set("正在重新读取数据...")
            self.app.root.update()

            # 更新数据配置
            self.app.data_config['params_file'] = self.params_path_var.get()
            self.app.data_config['rcs_data_dir'] = self.rcs_dir_var.get()

            start_id = int(self.model_start_var.get())
            end_id = int(self.model_end_var.get())
            self.app.data_config['model_ids'] = [f"{i:03d}" for i in range(start_id, end_id + 1)]

            # 根据频率配置更新frequencies列表
            freq_config = self.app.ae_freq_config.get()
            self.app.log_message(f"🔍 检查频率配置变量: {freq_config}")
            if freq_config == "3freq":
                self.app.data_config['frequencies'] = ['1.5G', '3G', '6G']
                self.app.log_message("✓ 频率配置: 3频率 (1.5GHz + 3GHz + 6GHz)")
            else:
                self.app.data_config['frequencies'] = ['1.5G', '3G']
                self.app.log_message("✓ 频率配置: 2频率 (1.5GHz + 3GHz)")
            self.app.log_message(f"📋 实际传递给缓存管理器的frequencies: {self.app.data_config['frequencies']}")

            # 强制重新读取（force_reload=True）
            self.app.param_data, self.app.rcs_data = self.app.cache_manager.load_data_with_cache(
                params_file=self.app.data_config['params_file'],
                rcs_data_dir=self.app.data_config['rcs_data_dir'],
                model_ids=self.app.data_config['model_ids'],
                frequencies=self.app.data_config['frequencies'],
                force_reload=True  # 强制重新读取
            )

            # 保存原始RCS数据副本，确保AutoEncoder训练使用线性域数据
            self.app._original_rcs_data = self.app.rcs_data.copy()
            self.app.log_message("已保存原始RCS数据副本，供AutoEncoder使用")

            self.app.data_loaded = True
            self.app.log_message("✅ 数据重新读取完成！")
            self.app.log_message(f"参数数据形状: {self.app.param_data.shape}")
            self.app.log_message(f"RCS数据形状: {self.app.rcs_data.shape}")

            # 验证频率配置匹配性（如果已加载模型）
            if hasattr(self.app, 'ae_system') and self.app.ae_system is not None:
                config_info = self.app.ae_system.get('config_info', {})
                model_num_freq = config_info.get('num_frequencies', None)
                model_freq_labels = config_info.get('frequency_labels', [])
                data_num_freq = self.app.rcs_data.shape[-1]

                if model_num_freq is not None:
                    self.app.log_message(f"检查频率配置: 模型={model_num_freq}频, 数据={data_num_freq}频")
                    if model_num_freq != data_num_freq:
                        warning_msg = (
                            f"⚠️ 频率配置不匹配！\n\n"
                            f"已加载模型: {model_num_freq}频 {model_freq_labels}\n"
                            f"当前数据: {data_num_freq}频\n\n"
                            f"建议重新加载匹配的模型或数据！"
                        )
                        self.app.log_message(f"❌ {warning_msg}")
                        messagebox.showwarning("频率配置不匹配", warning_msg)
                    else:
                        self.app.log_message(f"✅ 频率配置匹配：{data_num_freq}频")

            # 更新AutoEncoder扩展界面的模型选择列表
            if hasattr(self.app, 'ae_extension') and self.app.ae_extension is not None:
                self.app.ae_extension._update_model_selection()
                self.app.log_message("已更新小波分析模型选择列表")

            # 同步更新 ae_system 中的数据引用
            if hasattr(self.app, 'ae_system') and self.app.ae_system is not None:
                self.app.ae_system['rcs_data'] = self.app.rcs_data
                self.app.ae_system['param_data'] = self.app.param_data
                self.app.log_message("已同步更新AutoEncoder系统中的数据引用")

            self.app.status_var.set("数据重新读取完成")

        except Exception as e:
            error_msg = f"强制重新读取数据失败: {str(e)}"
            self.app.log_message(f"❌ {error_msg}")
            self.app.status_var.set("数据读取失败")
            messagebox.showerror("错误", error_msg)

    def on_preprocessing_change(self):
        """预处理选项变化时的回调函数"""
        enabled = self.use_log_preprocessing.get()

        # 控制预处理参数的启用状态
        state = tk.NORMAL if enabled else tk.DISABLED
        self.log_epsilon_entry.configure(state=state)
        self.normalize_checkbox.configure(state=state)

        # 更新数据配置
        self.update_data_config()

        if enabled:
            self.app.log_message("已启用对数预处理 - 推荐用于大动态范围RCS数据")
        else:
            self.app.log_message("已禁用对数预处理 - 使用原始线性RCS数据")

    def update_data_config(self):
        """更新数据配置以包含预处理选项"""
        use_log = self.use_log_preprocessing.get()
        epsilon = float(self.log_epsilon_var.get()) if self.log_epsilon_var.get() else 1e-10
        normalize = self.normalize_after_log.get()

        self.app.data_config = create_data_config(use_log_preprocessing=use_log)
        self.app.data_config['preprocessing'].update({
            'log_epsilon': epsilon,
            'normalize_after_log': normalize
        })

        self.app.log_message(f"数据配置已更新: 对数预处理={use_log}, ε={epsilon}, 标准化={normalize}")

    def reset_cuda_manually(self):
        """手动重置CUDA环境"""
        try:
            self.app.log_message("🔧 开始手动重置CUDA环境...")

            if not torch.cuda.is_available():
                messagebox.showinfo("信息", "CUDA不可用，无需重置")
                return

            # 1. 清理所有CUDA缓存
            self.app.log_message("  清理CUDA缓存...")
            torch.cuda.empty_cache()

            # 2. 重置峰值内存统计
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
                self.app.log_message("  重置内存统计...")

            # 3. 同步所有CUDA操作
            torch.cuda.synchronize()
            self.app.log_message("  同步CUDA操作...")

            # 4. 强制垃圾回收
            gc.collect()
            self.app.log_message("  执行垃圾回收...")

            # 5. 重新初始化随机种子
            try:
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                self.app.log_message("  重置CUDA随机种子...")
            except RuntimeError as seed_error:
                self.app.log_message(f"  随机种子重置失败: {seed_error}")

            # 6. 测试CUDA功能
            try:
                test_tensor = torch.tensor([1.0], device='cuda')
                test_result = test_tensor + 1.0
                del test_tensor, test_result
                self.app.log_message("  CUDA功能测试通过...")
            except RuntimeError as test_error:
                self.app.log_message(f"  CUDA测试失败: {test_error}")
                raise test_error

            self.app.log_message("✅ CUDA环境重置完成！")
            messagebox.showinfo("成功", "CUDA环境已成功重置！\n现在可以安全地开始训练。 সন")

        except Exception as e:
            error_msg = f"CUDA重置失败: {str(e)}"
            self.app.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", f"{error_msg}\n\n建议：\n1. 重启程序\n2. 使用CPU模式训练")

    def check_cuda_status(self):
        """检查CUDA状态并显示详细信息"""
        try:
            self.app.log_message("🔍 检查CUDA状态...")

            if not torch.cuda.is_available():
                status_info = "CUDA状态: 不可用\n建议使用CPU模式训练"
                self.app.log_message("❌ CUDA不可用")
                messagebox.showinfo("CUDA状态", status_info)
                return

            # 获取设备信息
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            # 获取内存信息
            properties = torch.cuda.get_device_properties(current_device)
            total_memory = properties.total_memory
            allocated_memory = torch.cuda.memory_allocated(current_device)
            cached_memory = torch.cuda.memory_reserved(current_device)

            # 计算内存使用率
            memory_usage = (allocated_memory / total_memory) * 100
            cache_usage = (cached_memory / total_memory) * 100

            status_info = f"""CUDA状态: 可用 ✅

设备信息:
• 设备数量: {device_count}
• 当前设备: {current_device}
• 设备名称: {device_name}

内存信息:
• 总内存: {total_memory//1024//1024:,} MB
• 已分配: {allocated_memory//1024//1024:,} MB ({memory_usage:.1f}%)
• 缓存: {cached_memory//1024//1024:,} MB ({cache_usage:.1f}%)
• 可用: {(total_memory-cached_memory)//1024//1024:,} MB

计算能力: {properties.major}.{properties.minor}
多处理器: {properties.multi_processor_count}"""

            self.app.log_message("✅ CUDA状态检查完成")
            messagebox.showinfo("CUDA状态详情", status_info)

        except Exception as e:
            error_msg = f"CUDA状态检查失败: {str(e)}"
            self.app.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def clean_gpu_memory(self):
        """清理GPU内存"""
        try:
            self.app.log_message("🧹 开始清理GPU内存...")

            if not torch.cuda.is_available():
                messagebox.showinfo("信息", "CUDA不可用，无需清理GPU内存")
                return

            # 记录清理前的内存使用
            before_allocated = torch.cuda.memory_allocated()
            before_cached = torch.cuda.memory_reserved()

            self.app.log_message(f"  清理前: 已分配 {before_allocated//1024//1024}MB, 缓存 {before_cached//1024//1024}MB")

            # 清理缓存
            torch.cuda.empty_cache()

            # 垃圾回收
            gc.collect()

            # 再次清理
            torch.cuda.empty_cache()

            # 记录清理后的内存使用
            after_allocated = torch.cuda.memory_allocated()
            after_cached = torch.cuda.memory_reserved()

            freed_allocated = before_allocated - after_allocated
            freed_cached = before_cached - after_cached

            result_msg = f"""GPU内存清理完成 ✅

清理结果:
• 释放已分配内存: {freed_allocated//1024//1024} MB
• 释放缓存内存: {freed_cached//1024//1024} MB

当前状态:
• 已分配: {after_allocated//1024//1024} MB
• 缓存: {after_cached//1024//1024} MB"""

            self.app.log_message("✅ GPU内存清理完成")
            self.app.log_message(f"  释放内存: {freed_cached//1024//1024}MB")
            messagebox.showinfo("清理完成", result_msg)

        except Exception as e:
            error_msg = f"GPU内存清理失败: {str(e)}"
            self.app.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)
