import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinter.font as tkFont
import threading
import torch
import numpy as np
import os
import sys # 某些方法中用到，例如load_model中的map_location='cpu'

# 导入项目模块（需要确保这些在环境中可用）
# RCSWaveletGUI中导入的模块：
# from wavelet_network import create_model, create_loss_function
# from configurable_loss import create_loss_function as create_configurable_loss
# from training import (CrossValidationTrainer, RCSDataLoader,
#                      create_training_config, create_data_config, RCSDataset)
# from evaluation import RCSEvaluator, evaluate_model_with_visualizations
# from data_cache import create_cache_manager
# from modern_wavelet_network import get_available_networks, get_network_info, get_available_losses

# 为了避免在每个Tab中重复导入大量模块，这里采取更简洁的引用方式
# 或者直接让app在需要时提供这些函数/常量
# 对于 create_model, create_training_config, MODERN_INTERFACE_AVAILABLE 等，假定它们可以通过app传递或在app中调用

class TrainingTab(ttk.Frame):
    """
    模型训练标签页
    负责模型训练配置、控制、进度显示和模型管理。
    """

    def __init__(self, notebook, app):
        """
        初始化模型训练标签页。

        参数:
            notebook: 父容器 (ttk.Notebook)
            app: 主应用程序实例 (RCSWaveletGUI)，用于访问共享状态和配置。
        """
        super().__init__(notebook)
        self.app = app

        # 引用主应用的字体
        self.font_small = getattr(app, 'font_small', None)

        # 训练UI变量
        self.batch_size_var = tk.StringVar(value=str(self.app.training_config['batch_size']))
        self.lr_var = tk.StringVar(value=str(self.app.training_config['learning_rate']))
        self.min_lr_var = tk.StringVar(value=str(self.app.training_config.get('min_lr', 2e-5)))
        self.restart_period_var = tk.StringVar(value=str(self.app.training_config.get('restart_period', 100)))
        self.epochs_var = tk.StringVar(value=str(self.app.training_config['epochs']))
        self.weight_decay_var = tk.StringVar(value=str(self.app.training_config['weight_decay']))
        self.patience_var = tk.StringVar(value=str(self.app.training_config['early_stopping_patience']))
        self.lr_scheduler_var = tk.StringVar(value=self.app.training_config.get('lr_scheduler', 'cosine_restart'))
        self.scheduler_info_var = tk.StringVar(value=self._get_scheduler_info(self.lr_scheduler_var.get()))

        self.available_wavelets = {
            'Daubechies': ['db2', 'db4', 'db8', 'db10'],
            'Biorthogonal': ['bior1.1', 'bior2.2', 'bior2.4', 'bior2.6'],
            'Coiflets': ['coif2', 'coif4', 'coif6'],
            'Others': ['haar', 'dmey', 'sym4', 'sym8']
        }
        self.current_wavelets = ['db4', 'db4', 'bior2.2', 'bior2.2'] # 默认值
        self.wavelet_vars = []

        self.use_cross_validation = tk.BooleanVar(value=True)
        self.save_checkpoints = tk.BooleanVar(value=True)
        self.model_type = tk.StringVar(value="enhanced")

        self.progress_var = tk.DoubleVar()
        self.current_epoch_var = tk.StringVar(value="等待开始...")

        # 构建界面
        self.create_widgets()
        self._update_network_options()
        self._on_network_selection_changed() # 初始化网络信息显示

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 训练配置组
        config_group = ttk.LabelFrame(main_frame, text="训练配置")
        config_group.pack(fill=tk.X, pady=(0, 10))

        # 配置参数
        config_frame = ttk.Frame(config_group)
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # 左侧配置
        left_config = ttk.Frame(config_frame)
        left_config.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Label(left_config, text="批次大小:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(left_config, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(left_config, text="初始学习率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        lr_entry = ttk.Entry(left_config, textvariable=self.lr_var, width=10)
        lr_entry.grid(row=1, column=1, padx=5, pady=2)

        # 学习率快捷按钮
        lr_preset_frame = ttk.Frame(left_config)
        lr_preset_frame.grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(lr_preset_frame, text="快捷:", font=("Arial", 8)).pack(side=tk.LEFT)
        for lr_val in [0.001, 0.003, 0.005]:
            ttk.Button(lr_preset_frame, text=f"{lr_val}",
                      command=lambda v=lr_val: self.lr_var.set(str(v)),
                      width=5).pack(side=tk.LEFT, padx=1)

        ttk.Label(left_config, text="最低学习率:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(left_config, textvariable=self.min_lr_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(left_config, text="(eta_min, 推荐: 1e-5~5e-5)", font=("Arial", 8), foreground="gray").grid(row=2, column=2, sticky=tk.W, pady=2)

        ttk.Label(left_config, text="重启周期:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(left_config, textvariable=self.restart_period_var, width=10).grid(row=3, column=1, padx=5, pady=2)

        # 重启周期快捷按钮
        restart_preset_frame = ttk.Frame(left_config)
        restart_preset_frame.grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(restart_preset_frame, text="快捷:", font=("Arial", 8)).pack(side=tk.LEFT)
        for period_val in [50, 100, 150, 200]:
            ttk.Button(restart_preset_frame, text=f"{period_val}",
                      command=lambda v=period_val: self.restart_period_var.set(str(v)),
                      width=4).pack(side=tk.LEFT, padx=1)

        ttk.Label(left_config, text="训练轮数:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(left_config, textvariable=self.epochs_var, width=10).grid(row=4, column=1, padx=5, pady=2)

        # 右侧配置
        right_config = ttk.Frame(config_frame)
        right_config.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(right_config, text="权重衰减:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(right_config, textvariable=self.weight_decay_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(right_config, text="早停耐心:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(right_config, textvariable=self.patience_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # 学习率调度策略选择
        ttk.Label(right_config, text="LR调度策略:").grid(row=2, column=0, sticky=tk.W, pady=2)
        scheduler_combo = ttk.Combobox(right_config, textvariable=self.lr_scheduler_var,
                                     values=['cosine_restart', 'cosine_simple', 'adaptive'],
                                     state='readonly', width=12)
        scheduler_combo.grid(row=2, column=1, padx=5, pady=2)

        # 策略说明标签
        self.scheduler_info_label = ttk.Label(right_config, textvariable=self.scheduler_info_var, font=("Arial", 8),
                 foreground="gray", wraplength=200)
        self.scheduler_info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # 绑定策略选择事件
        scheduler_combo.bind('<<ComboboxSelected>>', self._on_scheduler_changed)

        # 小波配置区域
        wavelet_group = ttk.LabelFrame(config_group, text="小波配置")
        wavelet_group.pack(fill=tk.X, padx=5, pady=5)

        # 小波配置网格
        wavelet_frame = ttk.Frame(wavelet_group)
        wavelet_frame.pack(fill=tk.X, padx=5, pady=5)

        # 为4个尺度创建小波选择器
        ttk.Label(wavelet_frame, text="小波配置 (4个尺度):").grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=2)

        self.wavelet_combos = []

        all_wavelets = []
        for wavelets in self.available_wavelets.values():
            all_wavelets.extend(wavelets)

        for i in range(4):
            row = 1 + i // 2
            col = (i % 2) * 2

            ttk.Label(wavelet_frame, text=f"尺度{i+1}:").grid(row=row, column=col, sticky=tk.W, pady=2, padx=(0, 5))

            wavelet_var = tk.StringVar(value=self.current_wavelets[i])
            self.wavelet_vars.append(wavelet_var)

            combo = ttk.Combobox(wavelet_frame, textvariable=wavelet_var, values=all_wavelets,
                               width=12, state="readonly")
            combo.grid(row=row, column=col+1, pady=2, padx=(0, 15))
            self.wavelet_combos.append(combo)

        # 预设配置按钮
        preset_frame = ttk.Frame(wavelet_group)
        preset_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(preset_frame, text="预设配置:").pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="默认混合", command=self.set_default_wavelets).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="全DB4", command=self.set_db4_wavelets).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="全双正交", command=self.set_bior_wavelets).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="递增复杂度", command=self.set_progressive_wavelets).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="边缘检测", command=self.set_edge_wavelets).pack(side=tk.LEFT, padx=2)

        # 训练选项
        options_frame = ttk.Frame(config_group)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(options_frame, text="使用交叉验证", variable=self.use_cross_validation).pack(side=tk.LEFT)
        ttk.Checkbutton(options_frame, text="保存检查点", variable=self.save_checkpoints).pack(side=tk.LEFT, padx=20)

        # 网络架构选择
        arch_frame = ttk.Frame(config_group)
        arch_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(arch_frame, text="网络架构:").pack(side=tk.LEFT)
        self.arch_combo = ttk.Combobox(arch_frame, textvariable=self.model_type, width=15, state="readonly")

        # 初始化网络选项
        # self._update_network_options() # 在__init__中调用
        self.arch_combo.pack(side=tk.LEFT, padx=10)

        # 绑定选择变化事件
        self.arch_combo.bind("<<ComboboxSelected>>", self._on_network_selection_changed)

        # 网络信息显示
        self.network_info_label = ttk.Label(arch_frame, text="", font=("Arial", 8))
        self.network_info_label.pack(side=tk.LEFT, padx=10)

        # 初始化网络信息显示
        # self._on_network_selection_changed() # 在__init__中调用

        # 损失函数配置提示
        loss_info_frame = ttk.Frame(arch_frame)
        loss_info_frame.pack(side=tk.LEFT, padx=(20,0))
        ttk.Label(loss_info_frame, text="💡损失函数:", font=("Arial", 9)).pack(side=tk.LEFT)
        ttk.Label(loss_info_frame, text="请在'损失函数配置'页面设置",
                 font=("Arial", 8), foreground="blue").pack(side=tk.LEFT, padx=(5,0))

        # 训练控制按钮
        control_frame = ttk.Frame(config_group)
        control_frame.pack(fill=tk.X, padx=5, pady=10)

        self.train_button = ttk.Button(control_frame, text="开始训练", command=self.start_training,
                                      style="Accent.TButton")
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="停止训练", command=self.stop_training,
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="保存模型", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="测试日志", command=self.test_logging).pack(side=tk.LEFT, padx=5)

        # 训练进度和日志
        progress_group = ttk.LabelFrame(main_frame, text="训练进度")
        progress_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 进度条
        self.progress_bar = ttk.Progressbar(progress_group, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # 当前状态
        ttk.Label(progress_group, textvariable=self.current_epoch_var).pack(pady=2)

        # 训练日志
        self.training_log = scrolledtext.ScrolledText(progress_group, height=10)
        self.training_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 将 training_log 暴露给app，以便日志重定向可以写入
        self.app.training_log = self.training_log


    def start_training(self):
        """开始训练"""
        if not self.app.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 更新训练配置
        try:
            self.app.training_config['batch_size'] = int(self.batch_size_var.get())
            self.app.training_config['learning_rate'] = float(self.lr_var.get())
            self.app.training_config['min_lr'] = float(self.min_lr_var.get())
            self.app.training_config['epochs'] = int(self.epochs_var.get())
            self.app.training_config['weight_decay'] = float(self.weight_decay_var.get())
            self.app.training_config['early_stopping_patience'] = int(self.patience_var.get())
            self.app.training_config['restart_period'] = int(self.restart_period_var.get())
            self.app.training_config['lr_scheduler'] = self.lr_scheduler_var.get()

            # 添加小波配置
            self.app.training_config['wavelet_config'] = self.get_current_wavelet_config()
            self.app.log_message(f"使用小波配置: {self.app.training_config['wavelet_config']}")

            # 更新数据配置以包含预处理选项 (通过app的data_tab进行更新)
            if hasattr(self.app, 'data_tab') and hasattr(self.app.data_tab, 'update_data_config'):
                self.app.data_tab.update_data_config()
            else:
                self.app.log_message("警告: 无法通过app.data_tab更新数据预处理配置。请确保DataManagementTab已正确加载。")


        except ValueError as e:
            messagebox.showerror("错误", f"配置参数格式错误: {str(e)}")
            return

        # 重置停止标志
        self.app.stop_training_flag = False

        # CUDA预检查和初始化
        self._initialize_cuda_safely()

        # 设置全局随机种子以保证训练的可重现性
        self._set_random_seeds(42)

        # 禁用训练按钮，启用停止按钮
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # 清空日志
        self.training_log.delete(1.0, tk.END)

        # 在新线程中开始训练
        self.app.training_thread = threading.Thread(target=self._train_model, daemon=True)
        self.app.training_thread.start()

    def _train_model(self):
        """训练模型（在后台线程中运行）"""
        return self.app.training_manager._train_model(self.app, self.progress_var, self.current_epoch_var)

    def _training_finished(self):
        """训练完成后的UI更新"""
        return self.app.training_manager._training_finished(self.app)

    def _set_random_seeds(self, seed=42):
        """设置全局随机种子以保证训练的可重现性"""
        # 注意：这里需要确保app.training_manager._set_random_seeds可以访问全局的torch, np等
        return self.app.training_manager._set_random_seeds(seed=seed)

    def _initialize_cuda_safely(self):
        """安全初始化CUDA环境"""
        # 注意：这里需要确保app.training_manager._initialize_cuda_safely可以访问全局的torch
        return self.app.training_manager._initialize_cuda_safely()

    def stop_training(self):
        """停止训练"""
        self.app.stop_training_flag = True
        self.app.log_message("训练停止请求已发送，等待当前epoch完成...")

        # 禁用停止按钮防止重复点击
        self.stop_button.config(state=tk.DISABLED)

        # 如果训练线程存在，等待其完成
        if self.app.training_thread and self.app.training_thread.is_alive():
            # 启动一个监控线程来等待训练线程结束
            monitor_thread = threading.Thread(target=self._monitor_training_stop, daemon=True)
            monitor_thread.start()

    def _monitor_training_stop(self):
        """监控训练停止过程"""
        if self.app.training_thread:
            self.app.training_thread.join()  # 等待训练线程结束

        # 在主线程中更新UI
        self.app.root.after(0, self._on_training_stopped)

    def _on_training_stopped(self):
        """训练停止后的UI更新"""
        self.app.log_message("训练已停止")
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)  # 重新启用停止按钮
        self.app.status_var.set("训练已停止")
        self.app.stop_training_flag = False  # 重置停止标志

    def _get_scheduler_info(self, scheduler_type):
        """获取调度器信息"""
        # 注意：app.scheduler_descriptions应该在主app中初始化
        return self.app.scheduler_descriptions.get(scheduler_type, '')

    def _on_scheduler_changed(self, event=None):
        """调度器选择改变回调"""
        scheduler_type = self.lr_scheduler_var.get()
        self.scheduler_info_var.set(self._get_scheduler_info(scheduler_type))

    def _update_network_options(self):
        """更新网络架构选项列表"""
        # 需要访问主app中的 MODERN_INTERFACE_AVAILABLE, get_available_networks, get_network_info
        MODERN_INTERFACE_AVAILABLE = getattr(self.app, 'MODERN_INTERFACE_AVAILABLE', False)
        get_available_networks = getattr(self.app, 'get_available_networks', lambda: {})
        
        if MODERN_INTERFACE_AVAILABLE:
            try:
                # 获取所有可用网络
                available_networks = get_available_networks()
                network_names = list(available_networks.keys())
                self.arch_combo['values'] = network_names

                # 如果当前选择的网络不在列表中，选择第一个
                current = self.model_type.get()
                if current not in network_names and network_names:
                    self.model_type.set(network_names[0])

            except Exception as e:
                # 如果现代接口失败，回退到传统选项
                self.app.log_message(f"现代网络接口更新失败: {e}")
                self.arch_combo['values'] = ['original', 'enhanced']
                if self.model_type.get() not in ['original', 'enhanced']:
                    self.model_type.set('enhanced')
        else:
            # 使用传统网络选项
            self.arch_combo['values'] = ['original', 'enhanced']
            if self.model_type.get() not in ['original', 'enhanced']:
                self.model_type.set('enhanced')

    def _on_network_selection_changed(self, event=None):
        """网络架构选择改变回调"""
        selected_network = self.model_type.get()
        MODERN_INTERFACE_AVAILABLE = getattr(self.app, 'MODERN_INTERFACE_AVAILABLE', False)
        get_network_info = getattr(self.app, 'get_network_info', lambda x: {})

        if MODERN_INTERFACE_AVAILABLE:
            try:
                # 获取网络详细信息
                info = get_network_info(selected_network)
                info_text = f"{info.get('description', '无描述')} | 参数: {info.get('parameters', {}).get('total', 0):,}"
                self.network_info_label.config(text=info_text)
            except Exception as e:
                self.app.log_message(f"获取网络信息失败: {e}")
                self.network_info_label.config(text="信息获取失败")
        else:
            # 传统网络信息
            if selected_network == 'original':
                self.network_info_label.config(text="传统小波RCS网络 | 参数: ~1.7M")
            elif selected_network == 'enhanced':
                self.network_info_label.config(text="增强版小波RCS网络 | 参数: ~60M")
            else:
                self.network_info_label.config(text="")

    def test_logging(self):
        """测试日志系统"""
        self.app.log_message("=== 日志系统测试开始 ===")
        self.app.log_message("这是print输出测试")
        self.app.log_message("模拟数据处理中...")

        import time
        time.sleep(0.5)

        self.app.log_message("处理完成")
        self.app.log_message("=== 日志系统测试结束 ===")

    def save_model(self):
        """保存模型"""
        if not self.app.model_trained or self.app.current_model is None:
            messagebox.showwarning("警告", "没有可保存的模型")
            return

        filename = filedialog.asksaveasfilename(
            title="保存模型",
            defaultextension=".pth",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )

        if filename:
            try:
                # 创建完整的checkpoint，包含preprocessing_stats
                # 注意：use_log_preprocessing是tkinter变量，需要.get()获取值
                # 这里需要从app中获取 use_log_preprocessing
                use_log_output = getattr(self.app, 'use_log_preprocessing', tk.BooleanVar(value=False)).get()
                
                checkpoint = {
                    'model_state_dict': self.app.current_model.state_dict(),
                    'preprocessing_stats': getattr(self.app, 'preprocessing_stats', None),
                    'use_log_output': use_log_output,
                    'epoch': getattr(self.app, 'current_epoch', 0),
                    'val_loss': getattr(self.app, 'best_val_loss', 0.0)
                }
                torch.save(checkpoint, filename)

                if hasattr(self.app, 'preprocessing_stats') and self.app.preprocessing_stats:
                    self.app.log_message(f"模型已保存到: {filename} (包含preprocessing_stats)")
                    messagebox.showinfo("成功", "模型保存成功 (包含预处理统计信息)")
                else:
                    self.app.log_message(f"模型已保存到: {filename} (警告: 无preprocessing_stats)")
                    messagebox.showinfo("成功", "模型保存成功 (但缺少预处理统计信息)")
            except Exception as e:
                messagebox.showerror("错误", f"模型保存失败: {str(e)}")

    # 小波预设配置方法
    def set_default_wavelets(self):
        """设置默认混合小波配置"""
        wavelets = ['db4', 'db4', 'bior2.2', 'bior2.2']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.app.log_message("已设置默认混合小波配置: ['db4', 'db4', 'bior2.2', 'bior2.2']")

    def set_db4_wavelets(self):
        """设置全DB4小波配置"""
        wavelets = ['db4', 'db4', 'db4', 'db4']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.app.log_message("已设置全DB4小波配置: ['db4', 'db4', 'db4', 'db4']")

    def set_bior_wavelets(self):
        """设置全双正交小波配置"""
        wavelets = ['bior2.2', 'bior2.2', 'bior2.4', 'bior2.6']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.app.log_message("已设置全双正交小波配置: ['bior2.2', 'bior2.2', 'bior2.4', 'bior2.6']")

    def set_progressive_wavelets(self):
        """设置递增复杂度小波配置"""
        wavelets = ['db2', 'db4', 'db8', 'db10']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.app.log_message("已设置递增复杂度小波配置: ['db2', 'db4', 'db8', 'db10']")

    def set_edge_wavelets(self):
        """设置边缘检测优化小波配置"""
        wavelets = ['haar', 'db2', 'db4', 'bior2.2']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.app.log_message("已设置边缘检测优化小波配置: ['haar', 'db2', 'db4', 'bior2.2']")

    def get_current_wavelet_config(self):
        """获取当前小波配置"""
        return [var.get() for var in self.wavelet_vars]

    def load_model(self):
        """加载模型"""
        filename = filedialog.askopenfilename(
            title="加载模型",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )

        if filename:
            try:
                checkpoint = torch.load(filename, map_location='cpu')

                # create_model, create_loss_function, create_configurable_loss are imported in main gui.py
                create_model_func = getattr(self.app, 'create_model', None)
                if create_model_func is None:
                    raise ImportError("create_model function not found in app context.")

                # 兼容旧格式和新格式checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 新格式：包含preprocessing_stats
                    self.app.model_params['wavelet_config'] = self.get_current_wavelet_config()
                    # 从app的data_tab获取 use_log_preprocessing_var, log_epsilon_var, normalize_after_log_var
                    # 如果data_tab不存在，则使用默认值
                    use_log_output = getattr(self.app, 'use_log_preprocessing', tk.BooleanVar(value=False)).get()

                    self.app.model_params['use_log_output'] = checkpoint.get('use_log_output', use_log_output)
                    self.app.current_model = create_model_func(**self.app.model_params)
                    self.app.current_model.load_state_dict(checkpoint['model_state_dict'])
                    self.app.preprocessing_stats = checkpoint.get('preprocessing_stats')
                    self.app.log_message(f"模型已从 {filename} 加载 (新格式)")
                    if self.app.preprocessing_stats:
                        self.app.log_message(f"  预处理统计: mean={self.app.preprocessing_stats['mean']:.2f} dB, std={self.app.preprocessing_stats['std']:.2f} dB")
                else:
                    # 旧格式：只有state_dict
                    self.app.model_params['wavelet_config'] = self.get_current_wavelet_config()
                    use_log_output = getattr(self.app, 'use_log_preprocessing', tk.BooleanVar(value=False)).get()
                    self.app.model_params['use_log_output'] = use_log_output
                    self.app.current_model = create_model_func(**self.app.model_params)
                    self.app.current_model.load_state_dict(checkpoint)
                    self.app.preprocessing_stats = None
                    self.app.log_message(f"模型已从 {filename} 加载 (旧格式)")
                    self.app.log_message("  警告: 旧格式checkpoint无preprocessing_stats，预测可能不准确")

                self.app.model_trained = True
                self.app.log_message(f"注意: 使用当前界面的小波配置 {self.app.model_params['wavelet_config']}")
                self.app.log_message("如果与保存时的小波配置不同，可能导致加载错误")
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
