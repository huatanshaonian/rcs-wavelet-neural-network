"""
RCS小波神经网络图形用户界面

提供直观的GUI界面用于:
1. 数据加载和预处理
2. 模型训练和监控
3. 预测结果可视化
4. 模型评估和对比
5. 参数配置和管理

基于tkinter构建，提供完整的工作流程界面

作者: RCS Wavelet Network Project
版本: 1.0
"""

# Unicode字符支持 ✨
try:
    from unicode_fix import fix_unicode_output
    fix_unicode_output()
except ImportError:
    pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinter.font as tkFont
import os
import threading
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# 修复matplotlib字体问题
def setup_matplotlib_font():
    """设置matplotlib字体，修复负号显示问题"""
    # 修复负号显示
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置中文字体
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    matplotlib.rcParams['font.family'] = ['sans-serif']

    # 直接优先加入常用中文字体，避免在启动时遍历系统全部字体文件
    existing_fonts = list(matplotlib.rcParams.get('font.sans-serif', []))
    preferred_fonts = [font for font in chinese_fonts if font not in existing_fonts]
    matplotlib.rcParams['font.sans-serif'] = preferred_fonts + existing_fonts

    # 设置字体大小
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 9
    matplotlib.rcParams['ytick.labelsize'] = 9
    matplotlib.rcParams['legend.fontsize'] = 9
    matplotlib.rcParams['figure.titlesize'] = 12

# 应用字体设置
setup_matplotlib_font()
import json
from datetime import datetime
import sys
import torch

# GUI管理器模块
from gui_managers.managers import StatisticsManager, VisualizationManager, TrainingManager, EvaluationManager, ReconstructionManager

# 导入项目模块
try:
    import rcs_data_reader as rdr
    import rcs_visual as rv
    from wavelet_network import create_model, create_loss_function
    from configurable_loss import create_loss_function as create_configurable_loss
    from training import (CrossValidationTrainer, RCSDataLoader,
                         create_training_config, create_data_config, RCSDataset)
    from evaluation import RCSEvaluator, evaluate_model_with_visualizations
    from data_cache import create_cache_manager

    # 导入现代化的网络接口
    try:
        from modern_wavelet_network import get_available_networks, get_network_info, get_available_losses
        MODERN_INTERFACE_AVAILABLE = True
    except ImportError:
        MODERN_INTERFACE_AVAILABLE = False
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有模块文件都在当前目录下")
    # 如果导入失败，创建空的替代函数避免NameError
    def create_data_config(use_log_preprocessing=False):
        return {
            'params_file': "",
            'rcs_data_dir': "",
            'model_ids': [],
            'frequencies': ['1.5G', '3G'],
            'preprocessing': {
                'use_log': use_log_preprocessing,
                'log_epsilon': 1e-10
            }
        }
    def create_training_config():
        return {
            'batch_size': 8,
            'learning_rate': 1e-3,
            'min_lr': 1e-5,
            'weight_decay': 1e-4,
            'epochs': 200,
            'early_stopping_patience': 50,
            'loss_weights': {
                'mse': 1.0,
                'symmetry': 0.02,
                'multiscale': 0.1
            },
            'memory_optimization': {
                'gradient_accumulation': True,
                'mixed_precision': True,
                'pin_memory': True,
                'empty_cache_frequency': 10
            }
        }
    def create_cache_manager():
        return None
    def create_model():
        return None
    def create_loss_function():
        return None
    def create_configurable_loss():
        return None
    class RCSDataset:
        pass
    class CrossValidationTrainer:
        pass
    class RCSDataLoader:
        pass
    class RCSEvaluator:
        pass
    def evaluate_model_with_visualizations(*args, **kwargs):
        pass
    def get_available_networks():
        return []
    def get_network_info():
        return {}
    def get_available_losses():
        return []
    MODERN_INTERFACE_AVAILABLE = False


class RCSWaveletGUI:
    """
    RCS小波网络主界面类
    """

    def __init__(self, root):
        """
        初始化GUI界面

        参数:
            root: tkinter根窗口
        """
        self.root = root
        self.root.title("RCS小波神经网络预测系统 v1.0")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # 设置字体 (使用系统默认字体避免字体问题)
        try:
            self.font_large = tkFont.Font(family="Microsoft YaHei", size=12, weight="bold")
            self.font_medium = tkFont.Font(family="Microsoft YaHei", size=10)
            self.font_small = tkFont.Font(family="Microsoft YaHei", size=9)
        except:
            # 如果字体设置失败，使用默认字体
            self.font_large = tkFont.Font(size=12, weight="bold")
            self.font_medium = tkFont.Font(size=10)
            self.font_small = tkFont.Font(size=9)

        # 状态变量
        self.data_loaded = False
        self.model_trained = False
        self.current_model = None
        self.training_history = {}
        self.evaluation_results = {}
        self.stop_training_flag = False  # 训练停止标志

        # AutoEncoder相关状态
        self.ae_system = None
        self.ae_training_history = {}
        self.ae_trained = False

        # 学习率调度策略信息
        self.scheduler_descriptions = {
            'cosine_restart': '余弦退火+重启：周期性重置LR',
            'cosine_simple': '余弦退火：单调递减到最小值',
            'adaptive': '自适应：根据验证损失调整'
        }
        self.training_thread = None

        # 配置变量
        self.data_config = create_data_config()
        self.training_config = create_training_config()
        self.model_params = {'input_dim': 9, 'hidden_dims': [128, 256], 'wavelet_config': None}

        # 传统损失函数变量（向后兼容）
        self.loss_type = tk.StringVar(value="improved")

        # 设置日志系统
        self.setup_logging()

        # 初始化数据缓存管理器
        self.cache_manager = create_cache_manager()

        # 初始化界面
        self.init_autoencoder_vars()
        self.create_widgets()
        self.setup_layout()

        # 初始化功能管理器
        self.statistics_manager = StatisticsManager(self)
        self.visualization_manager = VisualizationManager(self)
        self.training_manager = TrainingManager(self)
        self.evaluation_manager = EvaluationManager(self)
        self.reconstruction_manager = ReconstructionManager(self)

        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def init_autoencoder_vars(self):
        """初始化AutoEncoder配置变量"""

        # 频率配置
        self.ae_freq_config = tk.StringVar(value="2freq")  # 2freq 或 3freq

        # 模型配置
        self.ae_latent_dim = tk.StringVar(value="32")
        self.ae_dropout_rate = tk.StringVar(value="0.2")
        self.ae_wavelet_type = tk.StringVar(value="db4")
        self.ae_architecture_type = tk.StringVar(value="CNN")  # 架构类型: CNN或MLP

        # 训练配置
        self.ae_batch_size = tk.StringVar(value="16")
        self.ae_learning_rate = tk.StringVar(value="1e-3")
        self.ae_epochs_stage1 = tk.StringVar(value="100")  # AE预训练轮数
        self.ae_epochs_stage2 = tk.StringVar(value="50")   # 参数映射训练轮数
        self.ae_epochs_stage3 = tk.StringVar(value="20")   # 端到端微调轮数

        # 优化器配置
        self.ae_optimizer_type = tk.StringVar(value="adam")  # adam/adamw/sgd - 优化算法选择
        self.ae_weight_decay = tk.StringVar(value="1e-4")    # L2正则化，防止过拟合
        self.ae_momentum = tk.StringVar(value="0.9")         # SGD动量参数，加速收敛

        # 数据划分
        self.ae_validation_split = tk.StringVar(value="0.2")  # 验证集比例，用于早停和模型选择

        # 学习率调度配置 (复用项目标准配置)
        self.ae_lr_scheduler = tk.StringVar(value="constant")
        self.ae_min_lr = tk.StringVar(value="1e-5")
        self.ae_restart_period = tk.StringVar(value="50")

        # 早停配置 (分阶段可配置)
        self.ae_patience_stage1 = tk.StringVar(value="10")  # 阶段1早停耐心值
        self.ae_patience_stage2 = tk.StringVar(value="10")  # 阶段2早停耐心值
        self.ae_patience_stage3 = tk.StringVar(value="5")   # 阶段3早停耐心值
        self.ae_patience_e2e = tk.StringVar(value="15")     # 端到端早停耐心值

        # 数据预处理配置
        # 预处理选项已移至数据管理页面，此处不再需要相关变量

        # 训练模式
        self.ae_training_mode = tk.StringVar(value="三阶段训练")  # 三阶段训练 / 端到端训练 / 仅Stage 1

        # 损失函数配置复用
        self.ae_use_custom_loss = tk.BooleanVar(value=False)  # 是否使用自定义损失函数

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

    def setup_logging(self):
        """设置日志系统和输出重定向"""
        from datetime import datetime
        import time

        # 创建logs目录
        os.makedirs('logs', exist_ok=True)

        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"logs/rcs_wavelet_{timestamp}.log"

        # 打开日志文件
        self.log_file = open(self.log_filename, 'w', encoding='utf-8')

        # 创建输出重定向类
        class OutputRedirector:
            def __init__(self, gui, output_type):
                self.gui = gui
                self.output_type = output_type
                self.original = sys.stdout if output_type == 'stdout' else sys.stderr
                self.buffer = []
                self.last_update = 0
                self.update_interval = 0.1  # 100ms更新一次GUI

            def write(self, text):
                # 保持原始输出
                self.original.write(text)
                self.original.flush()

                # 发送到日志文件和缓存
                if text.strip():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_line = f"[{timestamp}] {text.strip()}"

                    # 写入日志文件
                    self.gui.log_file.write(log_line + '\n')
                    self.gui.log_file.flush()

                    # 添加到缓存
                    self.buffer.append(text.strip())

                    # 控制GUI更新频率
                    current_time = time.time()
                    if current_time - self.last_update >= self.update_interval:
                        self._flush_to_gui()
                        self.last_update = current_time

            def _flush_to_gui(self):
                """批量更新GUI"""
                if self.buffer:
                    # 合并缓存中的所有消息
                    combined_text = '\n'.join(self.buffer)
                    self.gui.root.after(0, self.gui.add_to_gui_log, combined_text)
                    self.buffer.clear()

            def flush(self):
                self.original.flush()
                self._flush_to_gui()  # 确保剩余消息也被显示

        # 保存原始输出流
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # 设置重定向
        sys.stdout = OutputRedirector(self, 'stdout')
        sys.stderr = OutputRedirector(self, 'stderr')

        # 记录启动信息
        print(f"RCS小波神经网络系统启动 - 日志文件: {self.log_filename}")

    def add_to_gui_log(self, text):
        """添加文本到GUI日志区域"""
        if hasattr(self, 'training_log'):
            self.training_log.insert(tk.END, text + '\n')
            self.training_log.see(tk.END)

        if hasattr(self, 'data_info_text'):
            self.data_info_text.insert(tk.END, text + '\n')
            self.data_info_text.see(tk.END)

    def restore_output(self):
        """恢复原始输出流"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if hasattr(self, 'log_file'):
            self.log_file.close()

    def create_widgets(self):
        """创建界面组件"""

        # 创建主笔记本组件
        self.notebook = ttk.Notebook(self.root)

        # 标签页1: 数据管理
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="数据管理")
        self.create_data_tab()

        # 标签页2: 损失函数配置
        self.loss_config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.loss_config_frame, text="损失函数配置")
        self.create_loss_config_tab()

        # 标签页3: AutoEncoder配置
        self.autoencoder_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.autoencoder_frame, text="AutoEncoder")
        self.create_autoencoder_tab()

        # 标签页4: 模型训练
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="模型训练")
        self.create_training_tab()

        # 标签页5: 模型评估
        self.evaluation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_frame, text="模型评估")
        self.create_evaluation_tab()

        # 标签页6: RCS预测
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="RCS预测")
        self.create_prediction_tab()

        # 标签页7: 可视化
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="可视化")
        self.create_visualization_tab()

    def create_data_tab(self):
        """创建数据管理标签页"""

        # 主框架
        main_frame = ttk.Frame(self.data_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 数据配置组
        config_group = ttk.LabelFrame(main_frame, text="数据配置")
        config_group.pack(fill=tk.X, pady=(0, 10))

        # 参数文件路径
        ttk.Label(config_group, text="参数文件:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.params_path_var = tk.StringVar(value=self.data_config['params_file'])
        self.params_path_entry = ttk.Entry(config_group, textvariable=self.params_path_var, width=50)
        self.params_path_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(config_group, text="浏览", command=self.browse_params_file).grid(
            row=0, column=2, padx=5, pady=5)

        # RCS数据目录
        ttk.Label(config_group, text="RCS数据目录:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.rcs_dir_var = tk.StringVar(value=self.data_config['rcs_data_dir'])
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
        self.model_start_var = tk.StringVar(value="1")
        ttk.Entry(range_frame, textvariable=self.model_start_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_frame, text="到:").pack(side=tk.LEFT, padx=(10, 0))
        self.model_end_var = tk.StringVar(value="100")
        ttk.Entry(range_frame, textvariable=self.model_end_var, width=5).pack(side=tk.LEFT, padx=2)

        # 频率配置
        ttk.Label(config_group, text="频率配置:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5)
        freq_config_frame = ttk.Frame(config_group)
        freq_config_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        freq_combo = ttk.Combobox(freq_config_frame, textvariable=self.ae_freq_config,
                                 values=["2freq", "3freq"], state="readonly", width=10)
        freq_combo.pack(side=tk.LEFT)
        ttk.Label(freq_config_frame, text="(2freq: 1.5GHz+3GHz, 3freq: +6GHz)",
                 font=self.font_small).pack(side=tk.LEFT, padx=(10, 0))

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
        self.use_log_preprocessing = tk.BooleanVar(value=False)
        ttk.Checkbutton(preprocessing_frame, text="启用对数预处理",
                       variable=self.use_log_preprocessing,
                       command=self.on_preprocessing_change).pack(side=tk.LEFT)

        # 预处理参数
        params_frame = ttk.Frame(preprocessing_frame)
        params_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(params_frame, text="ε值:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.log_epsilon_var = tk.StringVar(value="1e-10")
        self.log_epsilon_entry = ttk.Entry(params_frame, textvariable=self.log_epsilon_var, width=10)
        self.log_epsilon_entry.grid(row=0, column=1, padx=5, pady=2)
        self.log_epsilon_entry.configure(state=tk.DISABLED)

        self.normalize_after_log = tk.BooleanVar(value=True)
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

    def create_autoencoder_tab(self):
        """创建AutoEncoder配置标签页

        ⚠️ 此方法已移除！实际界面由 gui_autoencoder_extension.py 提供
        ==========================================
        此标签页的内容在 main.py 启动时会被 AutoEncoderExtension 覆盖

        如需修改AutoEncoder界面，请修改：
        - gui_autoencoder_extension.py: AutoEncoderExtension类（主界面）
        - wavelet_gui_helper.py: 小波分析辅助函数
        ==========================================
        """
        # 创建占位框架，将被extension覆盖
        placeholder = ttk.Frame(self.autoencoder_frame)
        placeholder.pack(fill=tk.BOTH, expand=True)

        info_label = ttk.Label(
            placeholder,
            text="AutoEncoder界面正在加载...\n请稍候，界面将由扩展模块提供",
            font=self.font_medium,
            justify=tk.CENTER
        )
        info_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def create_training_tab(self):
        """创建模型训练标签页"""

        # 主框架
        main_frame = ttk.Frame(self.training_frame)
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
        self.batch_size_var = tk.StringVar(value=str(self.training_config['batch_size']))
        ttk.Entry(left_config, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(left_config, text="初始学习率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.StringVar(value=str(self.training_config['learning_rate']))
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
        self.min_lr_var = tk.StringVar(value=str(self.training_config.get('min_lr', 2e-5)))
        min_lr_entry = ttk.Entry(left_config, textvariable=self.min_lr_var, width=10)
        min_lr_entry.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(left_config, text="(eta_min, 推荐: 1e-5~5e-5)", font=("Arial", 8), foreground="gray").grid(row=2, column=2, sticky=tk.W, pady=2)

        ttk.Label(left_config, text="重启周期:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.restart_period_var = tk.StringVar(value=str(self.training_config.get('restart_period', 100)))
        restart_entry = ttk.Entry(left_config, textvariable=self.restart_period_var, width=10)
        restart_entry.grid(row=3, column=1, padx=5, pady=2)

        # 重启周期快捷按钮
        restart_preset_frame = ttk.Frame(left_config)
        restart_preset_frame.grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(restart_preset_frame, text="快捷:", font=("Arial", 8)).pack(side=tk.LEFT)
        for period_val in [50, 100, 150, 200]:
            ttk.Button(restart_preset_frame, text=f"{period_val}",
                      command=lambda v=period_val: self.restart_period_var.set(str(v)),
                      width=4).pack(side=tk.LEFT, padx=1)

        ttk.Label(left_config, text="训练轮数:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value=str(self.training_config['epochs']))
        ttk.Entry(left_config, textvariable=self.epochs_var, width=10).grid(row=4, column=1, padx=5, pady=2)

        # 右侧配置
        right_config = ttk.Frame(config_frame)
        right_config.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(right_config, text="权重衰减:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.weight_decay_var = tk.StringVar(value=str(self.training_config['weight_decay']))
        ttk.Entry(right_config, textvariable=self.weight_decay_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(right_config, text="早停耐心:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.patience_var = tk.StringVar(value=str(self.training_config['early_stopping_patience']))
        ttk.Entry(right_config, textvariable=self.patience_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # 学习率调度策略选择
        ttk.Label(right_config, text="LR调度策略:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.lr_scheduler_var = tk.StringVar(value=self.training_config.get('lr_scheduler', 'cosine_restart'))
        scheduler_combo = ttk.Combobox(right_config, textvariable=self.lr_scheduler_var,
                                     values=['cosine_restart', 'cosine_simple', 'adaptive'],
                                     state='readonly', width=12)
        scheduler_combo.grid(row=2, column=1, padx=5, pady=2)

        # 策略说明标签
        self.scheduler_info_var = tk.StringVar(value=self._get_scheduler_info('cosine_restart'))
        ttk.Label(right_config, textvariable=self.scheduler_info_var, font=("Arial", 8),
                 foreground="gray", wraplength=200).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # 绑定策略选择事件
        scheduler_combo.bind('<<ComboboxSelected>>', self._on_scheduler_changed)

        # 小波配置区域
        wavelet_group = ttk.LabelFrame(config_group, text="小波配置")
        wavelet_group.pack(fill=tk.X, padx=5, pady=5)

        # 小波配置网格
        wavelet_frame = ttk.Frame(wavelet_group)
        wavelet_frame.pack(fill=tk.X, padx=5, pady=5)

        # 小波类型选项
        self.available_wavelets = {
            'Daubechies': ['db2', 'db4', 'db8', 'db10'],
            'Biorthogonal': ['bior1.1', 'bior2.2', 'bior2.4', 'bior2.6'],
            'Coiflets': ['coif2', 'coif4', 'coif6'],
            'Others': ['haar', 'dmey', 'sym4', 'sym8']
        }

        # 当前小波配置 (默认值)
        self.current_wavelets = ['db4', 'db4', 'bior2.2', 'bior2.2']

        # 为4个尺度创建小波选择器
        ttk.Label(wavelet_frame, text="小波配置 (4个尺度):").grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=2)

        self.wavelet_vars = []
        self.wavelet_combos = []

        # 所有可用小波的扁平列表
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

        self.use_cross_validation = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="使用交叉验证", variable=self.use_cross_validation).pack(side=tk.LEFT)

        self.save_checkpoints = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="保存检查点", variable=self.save_checkpoints).pack(side=tk.LEFT, padx=20)

        # 网络架构选择
        arch_frame = ttk.Frame(config_group)
        arch_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(arch_frame, text="网络架构:").pack(side=tk.LEFT)
        self.model_type = tk.StringVar(value="enhanced")
        self.arch_combo = ttk.Combobox(arch_frame, textvariable=self.model_type, width=15, state="readonly")

        # 初始化网络选项
        self._update_network_options()
        self.arch_combo.pack(side=tk.LEFT, padx=10)

        # 绑定选择变化事件
        self.arch_combo.bind("<<ComboboxSelected>>", self._on_network_selection_changed)

        # 网络信息显示
        self.network_info_label = ttk.Label(arch_frame, text="", font=("Arial", 8))
        self.network_info_label.pack(side=tk.LEFT, padx=10)

        # 初始化网络信息显示
        self._on_network_selection_changed()

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
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_group, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # 当前状态
        self.current_epoch_var = tk.StringVar(value="等待开始...")
        ttk.Label(progress_group, textvariable=self.current_epoch_var).pack(pady=2)

        # 训练日志
        self.training_log = scrolledtext.ScrolledText(progress_group, height=10)
        self.training_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_evaluation_tab(self):
        """创建模型评估标签页"""

        # 主框架
        main_frame = ttk.Frame(self.evaluation_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 评估控制组
        control_group = ttk.LabelFrame(main_frame, text="评估控制")
        control_group.pack(fill=tk.X, pady=(0, 10))

        control_frame = ttk.Frame(control_group)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="开始评估", command=self.start_evaluation,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="生成报告", command=self.generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="导出结果", command=self.export_results).pack(side=tk.LEFT, padx=5)

        # 评估结果显示
        results_group = ttk.LabelFrame(main_frame, text="评估结果")
        results_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 创建评估结果的树形视图
        self.eval_tree = ttk.Treeview(results_group, columns=("指标", "1.5GHz", "3GHz", "总体"), show="tree headings")
        self.eval_tree.heading("#0", text="评估类别")
        self.eval_tree.heading("指标", text="指标")
        self.eval_tree.heading("1.5GHz", text="1.5GHz")
        self.eval_tree.heading("3GHz", text="3GHz")
        self.eval_tree.heading("总体", text="总体")

        # 设置列宽
        self.eval_tree.column("#0", width=150)
        self.eval_tree.column("指标", width=100)
        self.eval_tree.column("1.5GHz", width=100)
        self.eval_tree.column("3GHz", width=100)
        self.eval_tree.column("总体", width=100)

        # 添加滚动条
        eval_scrollbar = ttk.Scrollbar(results_group, orient=tk.VERTICAL, command=self.eval_tree.yview)
        self.eval_tree.configure(yscrollcommand=eval_scrollbar.set)

        # 打包
        self.eval_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        eval_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)

    def create_loss_config_tab(self):
        """创建损失函数配置标签页"""

        # 主框架
        main_frame = ttk.Frame(self.loss_config_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 损失函数配置变量初始化
        self.init_loss_config_vars()

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

    def create_prediction_tab(self):
        """创建RCS预测标签页"""

        # 主框架
        main_frame = ttk.Frame(self.prediction_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 参数输入组
        input_group = ttk.LabelFrame(main_frame, text="飞行器参数输入")
        input_group.pack(fill=tk.X, pady=(0, 10))

        # 创建参数输入网格
        self.param_vars = []
        param_frame = ttk.Frame(input_group)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        for i in range(9):
            row = i // 3
            col = i % 3

            ttk.Label(param_frame, text=f"参数 {i+1}:").grid(
                row=row*2, column=col*2, sticky=tk.W, padx=5, pady=2)

            var = tk.StringVar(value="0.0")
            self.param_vars.append(var)
            ttk.Entry(param_frame, textvariable=var, width=15).grid(
                row=row*2+1, column=col*2, padx=5, pady=2)

        # 预测控制按钮
        control_frame = ttk.Frame(input_group)
        control_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(control_frame, text="载入参数模板", command=self.load_param_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="随机生成参数", command=self.generate_random_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="执行预测", command=self.make_prediction,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        # 预测结果显示
        result_group = ttk.LabelFrame(main_frame, text="预测结果")
        result_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 创建matplotlib图形
        self.pred_fig = Figure(figsize=(12, 6), dpi=80)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, result_group)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加工具栏
        pred_toolbar = NavigationToolbar2Tk(self.pred_canvas, result_group)
        pred_toolbar.update()

    def create_visualization_tab(self):
        """创建可视化标签页"""

        # 主框架
        main_frame = ttk.Frame(self.visualization_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 可视化控制组
        control_group = ttk.LabelFrame(main_frame, text="可视化控制")
        control_group.pack(fill=tk.X, pady=(0, 10))

        control_frame = ttk.Frame(control_group)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 模型选择
        ttk.Label(control_frame, text="模型ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.vis_model_var = tk.StringVar(value="001")
        ttk.Entry(control_frame, textvariable=self.vis_model_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # 频率选择
        ttk.Label(control_frame, text="频率:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.vis_freq_var = tk.StringVar(value="1.5G")
        freq_combo = ttk.Combobox(control_frame, textvariable=self.vis_freq_var,
                                 values=["1.5G", "3G", "6G"], state="readonly", width=8)
        freq_combo.grid(row=0, column=3, padx=5, pady=2)

        # 保存图片按钮
        ttk.Button(control_frame, text="💾 保存图片", command=self.save_current_visualization,
                  width=12).grid(row=0, column=4, padx=5, pady=2)

        # 可视化类型选择
        ttk.Label(control_frame, text="图表类型:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.vis_type_var = tk.StringVar(value="2D热图")
        type_combo = ttk.Combobox(control_frame, textvariable=self.vis_type_var,
                                 values=["2D热图", "3D表面图", "球坐标图", "对比图", "小波系数对比", "差值分析", "相关性分析",
                                        "训练历史", "统计对比", "AE隐空间分析", "AE重建质量", "AE参数映射", "AE训练进度"],
                                 state="readonly", width=12)
        type_combo.grid(row=1, column=1, padx=5, pady=2)

        # 生成按钮
        ttk.Button(control_frame, text="生成图表", command=self.generate_visualization,
                  style="Accent.TButton").grid(row=1, column=3, padx=5, pady=2)

        # 图表显示区域
        chart_group = ttk.LabelFrame(main_frame, text="图表显示")
        chart_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 创建matplotlib图形
        self.vis_fig = Figure(figsize=(12, 8), dpi=80)
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, chart_group)
        self.vis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加工具栏
        vis_toolbar = NavigationToolbar2Tk(self.vis_canvas, chart_group)
        vis_toolbar.update()

    def setup_layout(self):
        """设置布局"""
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 设置样式
        style = ttk.Style()
        style.configure("Accent.TButton")

    # ======= 数据管理功能 =======

    def browse_params_file(self):
        """浏览参数文件"""
        filename = filedialog.askopenfilename(
            title="选择参数文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.params_path_var.set(filename)
            self.data_config['params_file'] = filename

    def browse_rcs_dir(self):
        """浏览RCS数据目录"""
        dirname = filedialog.askdirectory(title="选择RCS数据目录")
        if dirname:
            self.rcs_dir_var.set(dirname)
            self.data_config['rcs_data_dir'] = dirname

    def load_data(self):
        """加载数据"""
        try:
            self.status_var.set("正在加载数据...")
            self.root.update()

            # 更新数据配置
            self.data_config['params_file'] = self.params_path_var.get()
            self.data_config['rcs_data_dir'] = self.rcs_dir_var.get()

            start_id = int(self.model_start_var.get())
            end_id = int(self.model_end_var.get())
            self.data_config['model_ids'] = [f"{i:03d}" for i in range(start_id, end_id + 1)]

            # 根据频率配置更新frequencies列表
            freq_config = self.ae_freq_config.get()
            self.log_message(f"🔍 检查频率配置变量: {freq_config}")
            if freq_config == "3freq":
                self.data_config['frequencies'] = ['1.5G', '3G', '6G']
                self.log_message("✓ 频率配置: 3频率 (1.5GHz + 3GHz + 6GHz)")
            else:
                self.data_config['frequencies'] = ['1.5G', '3G']
                self.log_message("✓ 频率配置: 2频率 (1.5GHz + 3GHz)")
            self.log_message(f"📋 实际传递给缓存管理器的frequencies: {self.data_config['frequencies']}")

            # 使用缓存加载数据
            self.log_message("开始加载数据（支持缓存加速）...")
            self.param_data, self.rcs_data = self.cache_manager.load_data_with_cache(
                params_file=self.data_config['params_file'],
                rcs_data_dir=self.data_config['rcs_data_dir'],
                model_ids=self.data_config['model_ids'],
                frequencies=self.data_config['frequencies']
            )

            # 保存原始RCS数据副本，确保AutoEncoder训练使用线性域数据
            self._original_rcs_data = self.rcs_data.copy()
            self.log_message("已保存原始RCS数据副本，供AutoEncoder使用")

            self.data_loaded = True
            self.log_message("数据加载成功！")
            self.log_message(f"参数数据形状: {self.param_data.shape}")
            self.log_message(f"RCS数据形状: {self.rcs_data.shape}")

            # 验证频率配置匹配性（如果已加载模型）
            if hasattr(self, 'ae_system') and self.ae_system is not None:
                config_info = self.ae_system.get('config_info', {})
                model_num_freq = config_info.get('num_frequencies', None)
                model_freq_labels = config_info.get('frequency_labels', [])
                data_num_freq = self.rcs_data.shape[-1]

                if model_num_freq is not None:
                    self.log_message(f"检查频率配置: 模型={model_num_freq}频, 数据={data_num_freq}频")
                    if model_num_freq != data_num_freq:
                        warning_msg = (
                            f"⚠️ 频率配置不匹配！\n\n"
                            f"已加载模型: {model_num_freq}频 {model_freq_labels}\n"
                            f"当前数据: {data_num_freq}频\n\n"
                            f"建议重新加载匹配的模型或数据！"
                        )
                        self.log_message(f"❌ {warning_msg}")
                        messagebox.showwarning("频率配置不匹配", warning_msg)
                    else:
                        self.log_message(f"✅ 频率配置匹配：{data_num_freq}频")

            # 更新AutoEncoder扩展界面的模型选择列表
            if hasattr(self, 'ae_extension') and self.ae_extension is not None:
                self.ae_extension._update_model_selection()
                self.log_message("已更新小波分析模型选择列表")

            self.status_var.set("数据加载完成")

        except Exception as e:
            self.log_message(f"数据加载失败: {str(e)}")
            self.status_var.set("数据加载失败")
            messagebox.showerror("错误", f"数据加载失败:\n{str(e)}")

    def preview_data(self):
        """预览数据"""
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 显示参数数据预览
        preview_text = "=== 参数数据预览 ===\n"
        preview_text += f"数据形状: {self.param_data.shape}\n"
        preview_text += f"前5个样本:\n{self.param_data[:5]}\n\n"

        preview_text += "=== RCS数据预览 ===\n"
        preview_text += f"数据形状: {self.rcs_data.shape}\n"

        # 原始数据统计
        first_sample = self.rcs_data[0]
        preview_text += f"原始线性数据 - 第一个样本:\n"
        preview_text += f"  1.5GHz - 范围: [{np.min(first_sample[:,:,0]):.6e}, {np.max(first_sample[:,:,0]):.6e}]\n"
        preview_text += f"  3GHz - 范围: [{np.min(first_sample[:,:,1]):.6e}, {np.max(first_sample[:,:,1]):.6e}]\n"

        # 如果启用了对数预处理，显示对数化后的数据
        if hasattr(self, 'use_log_preprocessing') and self.use_log_preprocessing.get():
            epsilon = float(self.log_epsilon_var.get()) if self.log_epsilon_var.get() else 1e-10

            # 计算对数化数据 (转换为分贝值: 10 * log10)
            rcs_db_sample = 10 * np.log10(np.maximum(first_sample, epsilon))
            preview_text += f"\n对数化数据 (dB) - 第一个样本:\n"
            preview_text += f"  1.5GHz - 范围: [{np.min(rcs_db_sample[:,:,0]):.1f}, {np.max(rcs_db_sample[:,:,0]):.1f}] dB\n"
            preview_text += f"  3GHz - 范围: [{np.min(rcs_db_sample[:,:,1]):.1f}, {np.max(rcs_db_sample[:,:,1]):.1f}] dB\n"

            # 如果启用了标准化，显示标准化后的数据
            if self.normalize_after_log.get():
                # 计算全局统计用于标准化
                all_rcs_db = 10 * np.log10(np.maximum(self.rcs_data, epsilon))
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
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        stats_text = "=== 详细数据统计 ===\n\n"

        # 参数统计
        stats_text += "参数数据统计:\n"
        for i in range(self.param_data.shape[1]):
            param_col = self.param_data[:, i]
            stats_text += f"参数 {i+1}: 均值={np.mean(param_col):.4f}, "
            stats_text += f"标准差={np.std(param_col):.4f}, "
            stats_text += f"范围=[{np.min(param_col):.4f}, {np.max(param_col):.4f}]\n"

        stats_text += "\n原始RCS数据统计 (线性值):\n"
        for freq_idx, freq_name in enumerate(['1.5GHz', '3GHz']):
            freq_data = self.rcs_data[:, :, :, freq_idx]
            stats_text += f"{freq_name}: 均值={np.mean(freq_data):.6e}, "
            stats_text += f"标准差={np.std(freq_data):.6e}, "
            stats_text += f"范围=[{np.min(freq_data):.6e}, {np.max(freq_data):.6e}]\n"

        # 如果启用了对数预处理，显示对数化后的统计
        if hasattr(self, 'use_log_preprocessing') and self.use_log_preprocessing.get():
            epsilon = float(self.log_epsilon_var.get()) if self.log_epsilon_var.get() else 1e-10

            stats_text += f"\n对数化RCS数据统计 (dB, ε={epsilon}):\n"
            # 转换为分贝值: 10 * log10
            rcs_db_data = 10 * np.log10(np.maximum(self.rcs_data, epsilon))

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
                original_range = np.max(self.rcs_data) - np.min(self.rcs_data)
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

    # ======= 缓存管理功能 =======

    def show_cache_info(self):
        """显示缓存信息"""
        try:
            # 创建新窗口显示缓存信息
            cache_window = tk.Toplevel(self.root)
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
                self.cache_manager.list_cache_info()
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
                self.log_message("正在清除数据缓存...")
                self.cache_manager.clear_cache()
                self.log_message("✅ 缓存清除完成")
                messagebox.showinfo("完成", "所有缓存已清除")

        except Exception as e:
            error_msg = f"清除缓存失败: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def force_reload_data(self):
        """强制重新读取数据（忽略缓存）"""
        if not self.params_path_var.get() or not self.rcs_dir_var.get():
            messagebox.showwarning("警告", "请先配置数据路径")
            return

        try:
            self.log_message("强制重新读取数据（忽略缓存）...")
            self.status_var.set("正在重新读取数据...")
            self.root.update()

            # 更新数据配置
            self.data_config['params_file'] = self.params_path_var.get()
            self.data_config['rcs_data_dir'] = self.rcs_dir_var.get()

            start_id = int(self.model_start_var.get())
            end_id = int(self.model_end_var.get())
            self.data_config['model_ids'] = [f"{i:03d}" for i in range(start_id, end_id + 1)]

            # 根据频率配置更新frequencies列表
            freq_config = self.ae_freq_config.get()
            self.log_message(f"🔍 检查频率配置变量: {freq_config}")
            if freq_config == "3freq":
                self.data_config['frequencies'] = ['1.5G', '3G', '6G']
                self.log_message("✓ 频率配置: 3频率 (1.5GHz + 3GHz + 6GHz)")
            else:
                self.data_config['frequencies'] = ['1.5G', '3G']
                self.log_message("✓ 频率配置: 2频率 (1.5GHz + 3GHz)")
            self.log_message(f"📋 实际传递给缓存管理器的frequencies: {self.data_config['frequencies']}")

            # 强制重新读取（force_reload=True）
            self.param_data, self.rcs_data = self.cache_manager.load_data_with_cache(
                params_file=self.data_config['params_file'],
                rcs_data_dir=self.data_config['rcs_data_dir'],
                model_ids=self.data_config['model_ids'],
                frequencies=self.data_config['frequencies'],
                force_reload=True  # 强制重新读取
            )

            # 保存原始RCS数据副本，确保AutoEncoder训练使用线性域数据
            self._original_rcs_data = self.rcs_data.copy()
            self.log_message("已保存原始RCS数据副本，供AutoEncoder使用")

            self.data_loaded = True
            self.log_message("✅ 数据重新读取完成！")
            self.log_message(f"参数数据形状: {self.param_data.shape}")
            self.log_message(f"RCS数据形状: {self.rcs_data.shape}")

            # 验证频率配置匹配性（如果已加载模型）
            if hasattr(self, 'ae_system') and self.ae_system is not None:
                config_info = self.ae_system.get('config_info', {})
                model_num_freq = config_info.get('num_frequencies', None)
                model_freq_labels = config_info.get('frequency_labels', [])
                data_num_freq = self.rcs_data.shape[-1]

                if model_num_freq is not None:
                    self.log_message(f"检查频率配置: 模型={model_num_freq}频, 数据={data_num_freq}频")
                    if model_num_freq != data_num_freq:
                        warning_msg = (
                            f"⚠️ 频率配置不匹配！\n\n"
                            f"已加载模型: {model_num_freq}频 {model_freq_labels}\n"
                            f"当前数据: {data_num_freq}频\n\n"
                            f"建议重新加载匹配的模型或数据！"
                        )
                        self.log_message(f"❌ {warning_msg}")
                        messagebox.showwarning("频率配置不匹配", warning_msg)
                    else:
                        self.log_message(f"✅ 频率配置匹配：{data_num_freq}频")

            # 更新AutoEncoder扩展界面的模型选择列表
            if hasattr(self, 'ae_extension') and self.ae_extension is not None:
                self.ae_extension._update_model_selection()
                self.log_message("已更新小波分析模型选择列表")

            self.status_var.set("数据重新读取完成")

        except Exception as e:
            error_msg = f"强制重新读取数据失败: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            self.status_var.set("数据读取失败")
            messagebox.showerror("错误", error_msg)

    # ======= 系统管理功能 =======

    def reset_cuda_manually(self):
        """手动重置CUDA环境"""
        try:
            import torch
            import gc

            self.log_message("🔧 开始手动重置CUDA环境...")

            if not torch.cuda.is_available():
                messagebox.showinfo("信息", "CUDA不可用，无需重置")
                return

            # 1. 清理所有CUDA缓存
            self.log_message("  清理CUDA缓存...")
            torch.cuda.empty_cache()

            # 2. 重置峰值内存统计
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
                self.log_message("  重置内存统计...")

            # 3. 同步所有CUDA操作
            torch.cuda.synchronize()
            self.log_message("  同步CUDA操作...")

            # 4. 强制垃圾回收
            gc.collect()
            self.log_message("  执行垃圾回收...")

            # 5. 重新初始化随机种子
            try:
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                self.log_message("  重置CUDA随机种子...")
            except RuntimeError as seed_error:
                self.log_message(f"  随机种子重置失败: {seed_error}")

            # 6. 测试CUDA功能
            try:
                test_tensor = torch.tensor([1.0], device='cuda')
                test_result = test_tensor + 1.0
                del test_tensor, test_result
                self.log_message("  CUDA功能测试通过...")
            except RuntimeError as test_error:
                self.log_message(f"  CUDA测试失败: {test_error}")
                raise test_error

            self.log_message("✅ CUDA环境重置完成！")
            messagebox.showinfo("成功", "CUDA环境已成功重置！\n现在可以安全地开始训练。")

        except Exception as e:
            error_msg = f"CUDA重置失败: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", f"{error_msg}\n\n建议：\n1. 重启程序\n2. 使用CPU模式训练")

    def check_cuda_status(self):
        """检查CUDA状态并显示详细信息"""
        try:
            import torch

            self.log_message("🔍 检查CUDA状态...")

            if not torch.cuda.is_available():
                status_info = "CUDA状态: 不可用\n建议使用CPU模式训练"
                self.log_message("❌ CUDA不可用")
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

            self.log_message("✅ CUDA状态检查完成")
            messagebox.showinfo("CUDA状态详情", status_info)

        except Exception as e:
            error_msg = f"CUDA状态检查失败: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def clean_gpu_memory(self):
        """清理GPU内存"""
        try:
            import torch
            import gc

            self.log_message("🧹 开始清理GPU内存...")

            if not torch.cuda.is_available():
                messagebox.showinfo("信息", "CUDA不可用，无需清理GPU内存")
                return

            # 记录清理前的内存使用
            before_allocated = torch.cuda.memory_allocated()
            before_cached = torch.cuda.memory_reserved()

            self.log_message(f"  清理前: 已分配 {before_allocated//1024//1024}MB, 缓存 {before_cached//1024//1024}MB")

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

            self.log_message("✅ GPU内存清理完成")
            self.log_message(f"  释放内存: {freed_cached//1024//1024}MB")
            messagebox.showinfo("清理完成", result_msg)

        except Exception as e:
            error_msg = f"GPU内存清理失败: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    # ======= 训练功能 =======

    def start_training(self):
        """开始训练"""
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 更新训练配置
        try:
            self.training_config['batch_size'] = int(self.batch_size_var.get())
            self.training_config['learning_rate'] = float(self.lr_var.get())
            self.training_config['min_lr'] = float(self.min_lr_var.get())
            self.training_config['epochs'] = int(self.epochs_var.get())
            self.training_config['weight_decay'] = float(self.weight_decay_var.get())
            self.training_config['early_stopping_patience'] = int(self.patience_var.get())
            self.training_config['restart_period'] = int(self.restart_period_var.get())
            self.training_config['lr_scheduler'] = self.lr_scheduler_var.get()

            # 添加小波配置
            self.training_config['wavelet_config'] = self.get_current_wavelet_config()
            self.log_message(f"使用小波配置: {self.training_config['wavelet_config']}")

            # 更新数据配置以包含预处理选项
            self.update_data_config()

        except ValueError as e:
            messagebox.showerror("错误", f"配置参数格式错误: {str(e)}")
            return

        # 重置停止标志
        self.stop_training_flag = False

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
        self.training_thread = threading.Thread(target=self._train_model, daemon=True)
        self.training_thread.start()

    def _train_model(self):
        """训练模型（在后台线程中运行）"""
        return self.training_manager._train_model()

    def _training_finished(self):
        """训练完成后的UI更新"""
        return self.training_manager._training_finished()

    def _set_random_seeds(self, seed=42):
        """设置全局随机种子以保证训练的可重现性"""
        return self.training_manager._set_random_seeds(seed=seed)

    def _initialize_cuda_safely(self):
        """安全初始化CUDA环境"""
        return self.training_manager._initialize_cuda_safely()

    def stop_training(self):
        """停止训练"""
        self.stop_training_flag = True
        self.log_message("训练停止请求已发送，等待当前epoch完成...")

        # 禁用停止按钮防止重复点击
        self.stop_button.config(state=tk.DISABLED)

        # 如果训练线程存在，等待其完成
        if self.training_thread and self.training_thread.is_alive():
            # 启动一个监控线程来等待训练线程结束
            monitor_thread = threading.Thread(target=self._monitor_training_stop, daemon=True)
            monitor_thread.start()

    def _monitor_training_stop(self):
        """监控训练停止过程"""
        if self.training_thread:
            self.training_thread.join()  # 等待训练线程结束

        # 在主线程中更新UI
        self.root.after(0, self._on_training_stopped)

    def _on_training_stopped(self):
        """训练停止后的UI更新"""
        self.log_message("训练已停止")
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)  # 重新启用停止按钮
        self.status_var.set("训练已停止")
        self.stop_training_flag = False  # 重置停止标志

    def _get_scheduler_info(self, scheduler_type):
        """获取调度器信息"""
        return self.scheduler_descriptions.get(scheduler_type, '')

    def _on_scheduler_changed(self, event=None):
        """调度器选择改变回调"""
        scheduler_type = self.lr_scheduler_var.get()
        self.scheduler_info_var.set(self._get_scheduler_info(scheduler_type))

    def _update_network_options(self):
        """更新网络架构选项列表"""
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
                print(f"现代网络接口更新失败: {e}")
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

        if MODERN_INTERFACE_AVAILABLE:
            try:
                # 获取网络详细信息
                info = get_network_info(selected_network)
                info_text = f"{info.get('description', '无描述')} | 参数: {info.get('parameters', {}).get('total', 0):,}"
                self.network_info_label.config(text=info_text)
            except Exception as e:
                print(f"获取网络信息失败: {e}")
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
        print("=== 日志系统测试开始 ===")
        print("这是print输出测试")
        print("模拟数据处理中...")

        import time
        time.sleep(0.5)

        print("处理完成")
        print("=== 日志系统测试结束 ===")

    def save_model(self):
        """保存模型"""
        if not self.model_trained or self.current_model is None:
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
                use_log_output = self.use_log_preprocessing.get() if hasattr(self, 'use_log_preprocessing') else False
                checkpoint = {
                    'model_state_dict': self.current_model.state_dict(),
                    'preprocessing_stats': getattr(self, 'preprocessing_stats', None),
                    'use_log_output': use_log_output,
                    'epoch': getattr(self, 'current_epoch', 0),
                    'val_loss': getattr(self, 'best_val_loss', 0.0)
                }
                torch.save(checkpoint, filename)

                if hasattr(self, 'preprocessing_stats') and self.preprocessing_stats:
                    self.log_message(f"模型已保存到: {filename} (包含preprocessing_stats)")
                    messagebox.showinfo("成功", "模型保存成功 (包含预处理统计信息)")
                else:
                    self.log_message(f"模型已保存到: {filename} (警告: 无preprocessing_stats)")
                    messagebox.showinfo("成功", "模型保存成功 (但缺少预处理统计信息)")
            except Exception as e:
                messagebox.showerror("错误", f"模型保存失败: {str(e)}")

    # 小波预设配置方法
    def set_default_wavelets(self):
        """设置默认混合小波配置"""
        wavelets = ['db4', 'db4', 'bior2.2', 'bior2.2']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.log_message("已设置默认混合小波配置: ['db4', 'db4', 'bior2.2', 'bior2.2']")

    def set_db4_wavelets(self):
        """设置全DB4小波配置"""
        wavelets = ['db4', 'db4', 'db4', 'db4']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.log_message("已设置全DB4小波配置: ['db4', 'db4', 'db4', 'db4']")

    def set_bior_wavelets(self):
        """设置全双正交小波配置"""
        wavelets = ['bior2.2', 'bior2.2', 'bior2.4', 'bior2.6']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.log_message("已设置全双正交小波配置: ['bior2.2', 'bior2.2', 'bior2.4', 'bior2.6']")

    def set_progressive_wavelets(self):
        """设置递增复杂度小波配置"""
        wavelets = ['db2', 'db4', 'db8', 'db10']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.log_message("已设置递增复杂度小波配置: ['db2', 'db4', 'db8', 'db10']")

    def set_edge_wavelets(self):
        """设置边缘检测优化小波配置"""
        wavelets = ['haar', 'db2', 'db4', 'bior2.2']
        for i, var in enumerate(self.wavelet_vars):
            var.set(wavelets[i])
        self.log_message("已设置边缘检测优化小波配置: ['haar', 'db2', 'db4', 'bior2.2']")

    def get_current_wavelet_config(self):
        """获取当前小波配置"""
        return [var.get() for var in self.wavelet_vars]

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
            self.log_message("已启用对数预处理 - 推荐用于大动态范围RCS数据")
        else:
            self.log_message("已禁用对数预处理 - 使用原始线性RCS数据")

    def update_data_config(self):
        """更新数据配置以包含预处理选项"""
        use_log = self.use_log_preprocessing.get()
        epsilon = float(self.log_epsilon_var.get()) if self.log_epsilon_var.get() else 1e-10
        normalize = self.normalize_after_log.get()

        self.data_config = create_data_config(use_log_preprocessing=use_log)
        self.data_config['preprocessing'].update({
            'log_epsilon': epsilon,
            'normalize_after_log': normalize
        })

        self.log_message(f"数据配置已更新: 对数预处理={use_log}, ε={epsilon}, 标准化={normalize}")

    def load_model(self):
        """加载模型"""
        filename = filedialog.askopenfilename(
            title="加载模型",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )

        if filename:
            try:
                checkpoint = torch.load(filename, map_location='cpu')

                # 兼容旧格式和新格式checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 新格式：包含preprocessing_stats
                    self.model_params['wavelet_config'] = self.get_current_wavelet_config()
                    self.model_params['use_log_output'] = checkpoint.get('use_log_output', self.use_log_preprocessing.get())
                    self.current_model = create_model(**self.model_params)
                    self.current_model.load_state_dict(checkpoint['model_state_dict'])
                    self.preprocessing_stats = checkpoint.get('preprocessing_stats')
                    self.log_message(f"模型已从 {filename} 加载 (新格式)")
                    if self.preprocessing_stats:
                        self.log_message(f"  预处理统计: mean={self.preprocessing_stats['mean']:.2f} dB, std={self.preprocessing_stats['std']:.2f} dB")
                else:
                    # 旧格式：只有state_dict
                    self.model_params['wavelet_config'] = self.get_current_wavelet_config()
                    self.model_params['use_log_output'] = self.use_log_preprocessing.get()
                    self.current_model = create_model(**self.model_params)
                    self.current_model.load_state_dict(checkpoint)
                    self.preprocessing_stats = None
                    self.log_message(f"模型已从 {filename} 加载 (旧格式)")
                    self.log_message("  警告: 旧格式checkpoint无preprocessing_stats，预测可能不准确")

                self.model_trained = True
                self.log_message(f"注意: 使用当前界面的小波配置 {self.model_params['wavelet_config']}")
                self.log_message("如果与保存时的小波配置不同，可能导致加载错误")
                messagebox.showinfo("成功", "模型加载成功")
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    # ======= 评估功能 =======

    def start_evaluation(self):
        """开始评估（支持AutoEncoder和传统网络）"""
        # 检查是否有训练好的模型（传统网络或AutoEncoder）
        has_traditional_model = self.model_trained and self.current_model is not None
        has_ae_model = hasattr(self, 'ae_system') and self.ae_system is not None

        if not has_traditional_model and not has_ae_model:
            messagebox.showwarning("警告", "请先训练或加载模型（传统网络或AutoEncoder）")
            return

        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            # 根据模型类型选择评估路径
            if has_ae_model:
                self.log_message("🔬 开始AutoEncoder模型评估...")
                self._evaluate_autoencoder_model()
            else:
                self.log_message("🔬 开始传统网络模型评估...")
                self._evaluate_traditional_model()

            messagebox.showinfo("成功", "模型评估完成")

        except Exception as e:
            messagebox.showerror("错误", f"评估失败: {str(e)}")

    def _evaluate_traditional_model(self):
        """评估传统网络模型"""
        return self.evaluation_manager._evaluate_traditional_model()

    def _reconstruct_rcs(self, input_data=None, input_type='auto', model_ids=None, return_latents=False, return_wavelet_coeffs=False):
        """
        统一的RCS重建函数 - 支持多种输入方式

        根据训练模式自动选择重建路径：
        - Three-Stage模式: 参数 → ParameterMapper → Decoder → RCS
        - Stage1-Only模式: RCS → Encoder → Decoder → RCS

        Args:
            input_data: 输入数据（可选）
                - 如果input_type='params': np.ndarray [N, 9] 设计参数
                - 如果input_type='rcs': np.ndarray [N, H, W, C] RCS数据
                - 如果input_type='auto': 根据training_mode自动推断
            input_type: 输入类型
                - 'auto': 根据training_mode自动判断（默认）
                - 'params': 从参数重建（需要Three-Stage模式）
                - 'rcs': 从RCS重建（Stage1-Only模式）
                - 'model_ids': 从模型ID列表重建
            model_ids: 模型ID列表（当input_type='model_ids'时使用）
                - 可以是字符串列表 ['001', '002'] 或整数列表 [0, 1]
            return_latents: 是否返回隐空间表示（默认False）
            return_wavelet_coeffs: 是否返回小波系数（仅Wavelet模式，默认False）

        Returns:
            dict: {
                'reconstructed_rcs': np.ndarray [N, 91, 91, num_freq] 重建的RCS（线性域）
                'latents': np.ndarray [N, latent_dim] 隐空间表示（如果return_latents=True）
                'original_wavelet_coeffs': np.ndarray [N, 49, 49, 8] 原始小波系数（如果return_wavelet_coeffs=True且mode='wavelet'）
                'reconstructed_wavelet_coeffs': np.ndarray [N, 49, 49, 8] 重建小波系数（如果return_wavelet_coeffs=True且mode='wavelet'）
                'input_type_used': str 实际使用的输入类型
                'training_mode': str 训练模式
            }
        """
        return self.reconstruction_manager._reconstruct_rcs(
            input_data=input_data,
            input_type=input_type,
            model_ids=model_ids,
            return_latents=return_latents,
            return_wavelet_coeffs=return_wavelet_coeffs
        )

    def _evaluate_autoencoder_model(self):
        """评估AutoEncoder模型 - 使用统一重建函数"""
        return self.evaluation_manager._evaluate_autoencoder_model()

    def _update_evaluation_display(self):
        """更新评估结果显示（支持AutoEncoder和传统网络）"""
        return self.evaluation_manager._update_evaluation_display()

    def _display_autoencoder_results(self, results):
        """显示AutoEncoder评估结果"""
        return self.evaluation_manager._display_autoencoder_results(results)

    def _display_traditional_results(self, results):
        """显示传统网络评估结果"""
        return self.evaluation_manager._display_traditional_results(results)

    def generate_report(self):
        """生成评估报告"""
        if not self.evaluation_results:
            messagebox.showwarning("警告", "请先进行模型评估")
            return

        # 选择保存位置
        filename = filedialog.asksaveasfilename(
            title="保存评估报告",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                evaluator = RCSEvaluator(self.current_model)
                evaluator.evaluation_results = self.evaluation_results
                report = evaluator.generate_evaluation_report(filename)
                messagebox.showinfo("成功", f"评估报告已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"报告生成失败: {str(e)}")

    def export_results(self):
        """导出评估结果"""
        if not self.evaluation_results:
            messagebox.showwarning("警告", "请先进行模型评估")
            return

        filename = filedialog.asksaveasfilename(
            title="导出评估结果",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=str)
                messagebox.showinfo("成功", f"评估结果已导出到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"结果导出失败: {str(e)}")

    # ======= 预测功能 =======

    def load_param_template(self):
        """加载参数模板"""
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 使用第一个样本作为模板
        template_params = self.param_data[0]
        for i, var in enumerate(self.param_vars):
            var.set(f"{template_params[i]:.6f}")

    def generate_random_params(self):
        """生成随机参数"""
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 基于已有数据的分布生成随机参数
        for i, var in enumerate(self.param_vars):
            param_col = self.param_data[:, i]
            mean = np.mean(param_col)
            std = np.std(param_col)
            random_val = np.random.normal(mean, std)
            var.set(f"{random_val:.6f}")

    def make_prediction(self):
        """执行RCS预测"""
        if not self.model_trained or self.current_model is None:
            messagebox.showwarning("警告", "请先训练或加载模型")
            return

        try:
            # 获取输入参数
            params = []
            for var in self.param_vars:
                params.append(float(var.get()))

            params = np.array(params).reshape(1, -1)

            # 标准化参数 (使用训练时的scaler)
            if hasattr(self, 'param_data'):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(self.param_data)
                params_scaled = scaler.transform(params)
            else:
                params_scaled = params

            # 执行预测
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.current_model.to(device)
            self.current_model.eval()

            with torch.no_grad():
                params_tensor = torch.tensor(params_scaled, dtype=torch.float32).to(device)
                prediction = self.current_model(params_tensor)
                prediction = prediction.cpu().numpy()[0]  # [91, 91, 2]

            # 可视化预测结果
            self._plot_prediction_results(prediction)

        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")

    def _plot_prediction_results(self, prediction):
        """绘制预测结果"""
        self.pred_fig.clear()

        # 创建子图
        ax1 = self.pred_fig.add_subplot(1, 2, 1)
        ax2 = self.pred_fig.add_subplot(1, 2, 2)

        # 定义角度范围 (基于实际数据)
        phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
        theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°

        # 绘制1.5GHz结果
        im1 = ax1.imshow(prediction[:, :, 0], cmap='jet', aspect='equal',
                        extent=[phi_range[0], phi_range[1], theta_range[1], theta_range[0]])
        ax1.set_title('1.5GHz RCS预测')
        ax1.set_xlabel('φ (方位角, 度)')
        ax1.set_ylabel('θ (俯仰角, 度)')
        self.pred_fig.colorbar(im1, ax=ax1)

        # 绘制3GHz结果
        im2 = ax2.imshow(prediction[:, :, 1], cmap='jet', aspect='equal',
                        extent=[phi_range[0], phi_range[1], theta_range[1], theta_range[0]])
        ax2.set_title('3GHz RCS预测')
        ax2.set_xlabel('φ (方位角, 度)')
        ax2.set_ylabel('θ (俯仰角, 度)')
        self.pred_fig.colorbar(im2, ax=ax2)

        self.pred_fig.tight_layout()
        self.pred_canvas.draw()

    # ======= 可视化功能 =======

    def generate_visualization(self):
        """生成可视化图表（支持AutoEncoder和传统网络）"""
        try:
            chart_type = self.vis_type_var.get()

            # 检查模型可用性
            has_traditional_model = self.model_trained and self.current_model is not None
            has_ae_model = hasattr(self, 'ae_system') and self.ae_system is not None

            # 分类处理：需要model_id的图表 vs 全局统计图表 vs AutoEncoder特定图表
            if chart_type in ["训练历史", "统计对比"]:
                # 全局统计图表 - 不需要model_id
                if chart_type == "训练历史":
                    self._plot_training_history()
                elif chart_type == "统计对比":
                    self._plot_global_statistics_comparison()
            elif chart_type in ["AE隐空间分析", "AE重建质量", "AE参数映射", "AE训练进度"]:
                # AutoEncoder特定图表
                if not has_ae_model:
                    messagebox.showwarning("警告", "AutoEncoder图表需要先训练或加载AutoEncoder模型")
                    return
                self._plot_autoencoder_visualization(chart_type)
            elif chart_type in ["2D热图", "3D表面图", "球坐标图"]:
                # 这些图表始终显示原始RCS数据，不使用模型预测
                model_id = self.vis_model_var.get()
                if not model_id:
                    messagebox.showwarning("警告", "请输入模型ID")
                    return

                freq = self.vis_freq_var.get()

                if chart_type == "2D热图":
                    self._plot_2d_heatmap(model_id, freq)
                elif chart_type == "3D表面图":
                    self._plot_3d_surface(model_id, freq)
                elif chart_type == "球坐标图":
                    self._plot_spherical(model_id, freq)
            else:
                # 对比图、差值分析、相关性分析 - 需要模型预测
                if not has_traditional_model and not has_ae_model:
                    messagebox.showwarning("警告", "请先训练或加载模型")
                    return

                model_id = self.vis_model_var.get()
                if not model_id:
                    messagebox.showwarning("警告", "请输入模型ID")
                    return

                freq = self.vis_freq_var.get()

                if chart_type == "对比图":
                    if has_ae_model:
                        self._plot_ae_comparison()
                    else:
                        self._plot_comparison(model_id)
                elif chart_type == "小波系数对比":
                    if has_ae_model:
                        self._plot_wavelet_coefficients_comparison()
                    else:
                        messagebox.showwarning("警告", "小波系数对比功能需要AutoEncoder模型")
                elif chart_type == "差值分析":
                    self._plot_difference_analysis(model_id)
                elif chart_type == "相关性分析":
                    self._plot_correlation_analysis(model_id)

        except Exception as e:
            messagebox.showerror("错误", f"图表生成失败: {str(e)}")

    def _plot_2d_heatmap(self, model_id, freq):
        """绘制2D热图"""
        return self.visualization_manager._plot_2d_heatmap(model_id, freq)

    def _plot_3d_surface(self, model_id, freq):
        """绘制3D表面图"""
        return self.visualization_manager._plot_3d_surface(model_id, freq)

    def _plot_spherical(self, model_id, freq):
        """绘制球坐标图"""
        return self.visualization_manager._plot_spherical(model_id, freq)

    def _plot_comparison(self, model_id):
        """绘制原始RCS vs 神经网络预测RCS对比图"""
        return self.visualization_manager._plot_comparison(model_id)

    def _plot_difference_analysis(self, model_id):
        """绘制差值分析图（原始RCS - 预测RCS）"""
        return self.visualization_manager._plot_difference_analysis(model_id)

    def _plot_correlation_analysis(self, model_id):
        """绘制相关性分析图"""
        return self.visualization_manager._plot_correlation_analysis(model_id)

    def _plot_training_history(self):
        """绘制训练历史图（对交叉验证，分别保存每折到results文件夹，GUI显示最佳折）"""
        return self.visualization_manager._plot_training_history()

    def _save_fold_plot(self, fold_data, fold_idx, results_dir):
        """保存单个折的训练历史图表"""
        return self.visualization_manager._save_fold_plot(fold_data, fold_idx, results_dir)

    def _display_fold_in_gui(self, fold_data, fold_idx):
        """在GUI中显示指定折的训练历史"""
        return self.visualization_manager._display_fold_in_gui(fold_data, fold_idx)

    def _display_simple_training_history(self):
        """显示简单训练模式的历史（非交叉验证）"""
        return self.visualization_manager._display_simple_training_history()

    def _plot_global_statistics_comparison(self):
        """改进的全局统计对比分析 - 委托给StatisticsManager处理"""
        self.statistics_manager.plot_global_statistics_comparison()

    def _save_scatter_plots(self, all_actual_1_5g, all_predicted_1_5g, all_actual_3g, all_predicted_3g, results_dir):
        """保存散点图到文件 - 委托给StatisticsManager处理"""
        self.statistics_manager._save_scatter_plots(all_actual_1_5g, all_predicted_1_5g, all_actual_3g, all_predicted_3g, results_dir)

    # ======= 辅助功能 =======

    def log_message(self, message, level='INFO'):
        """记录日志消息 - 现在直接使用print输出，会被自动捕获"""
        print(message)

    # ======= 损失函数配置方法 =======

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
            self.training_config['custom_loss_config'] = loss_config

            messagebox.showinfo("成功", "损失函数配置已应用！\n训练时将使用自定义损失函数。")
            self.log_message("损失函数配置已更新")

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
        self.mse_weight.set("1.0")

        self.use_huber_loss.set(False)
        self.use_l1_loss.set(False)

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
        self.mse_weight.set("0.9")

        self.use_huber_loss.set(False)
        self.use_l1_loss.set(False)

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

    def on_closing(self):
        """窗口关闭事件处理"""
        try:
            # 记录关闭日志
            print("RCS小波神经网络系统关闭")

            # 停止正在进行的训练
            if hasattr(self, 'training_thread') and self.training_thread and self.training_thread.is_alive():
                self.stop_training_flag = True
                print("正在停止训练...")

            # 恢复输出流
            self.restore_output()

            # 销毁窗口
            self.root.destroy()

        except Exception as e:
            print(f"关闭时发生错误: {e}")
            self.root.destroy()


    # ==================== AutoEncoder功能函数 ====================

    def update_ae_status(self):
        """更新AutoEncoder系统状态显示"""
        try:
            status_info = []
            status_info.append("=== AutoEncoder系统状态 ===")

            # 频率配置信息
            freq_config = self.ae_freq_config.get()
            freq_info = "1.5GHz+3GHz" if freq_config == "2freq" else "1.5GHz+3GHz+6GHz"
            status_info.append(f"频率配置: {freq_config} ({freq_info})")

            # 模型配置信息
            status_info.append(f"隐空间维度: {self.ae_latent_dim.get()}")
            status_info.append(f"小波类型: {self.ae_wavelet_type.get()}")
            status_info.append(f"Dropout率: {self.ae_dropout_rate.get()}")

            # 系统状态
            if self.ae_system is None:
                status_info.append("系统状态: 未创建")
            else:
                status_info.append("系统状态: 已创建")
                # 显示模型信息 (兼容不同模型格式)
                model_info = self.ae_system['autoencoder'].get_model_info()

                # 获取参数量 (兼容两种格式)
                if 'parameters' in model_info and 'total' in model_info['parameters']:
                    # WaveletAutoEncoder格式
                    total_params = model_info['parameters']['total']
                elif 'total_parameters' in model_info:
                    # DirectAutoEncoder格式
                    total_params = model_info['total_parameters']
                else:
                    total_params = 0

                status_info.append(f"模型参数量: {total_params:,}")

                # 压缩比 (可能不存在于直接模式)
                if 'compression_ratio' in model_info:
                    status_info.append(f"压缩比: {model_info['compression_ratio']}")

            if self.ae_trained:
                status_info.append("训练状态: 已训练")
            else:
                status_info.append("训练状态: 未训练")

            # 更新显示
            self.ae_status_text.delete(1.0, tk.END)
            self.ae_status_text.insert(tk.END, "\n".join(status_info))

        except Exception as e:
            print(f"更新AE状态失败: {e}")

    def get_ae_session_timestamp(self):
        """获取当前AE会话的时间戳（固定，除非重新创建系统）"""
        if not hasattr(self, 'ae_session_timestamp') or self.ae_session_timestamp is None:
            # 如果还没有会话时间戳，生成一个新的
            self.ae_session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.ae_session_timestamp

    def ae_log(self, message):
        """添加AutoEncoder日志信息"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}"

            # 总是输出到控制台，确保用户能看到
            print(log_message)

            # 检查组件是否存在且有效
            if not hasattr(self, 'ae_log_text') or self.ae_log_text is None:
                return

            # 检查组件是否还存在（没有被销毁）
            if not self.ae_log_text.winfo_exists():
                return

            # 输出到GUI日志组件
            self.ae_log_text.insert(tk.END, log_message + "\n")
            self.ae_log_text.see(tk.END)
            self.root.update_idletasks()

        except Exception as e:
            print(f"添加AE日志失败: {e}")

    def create_ae_system(self):
        """创建AutoEncoder系统"""
        try:
            if not self.data_loaded:
                messagebox.showwarning("警告", "请先在数据管理页面加载数据!")
                return

            # 检查已加载的数据
            if not hasattr(self, 'rcs_data') or self.rcs_data is None:
                messagebox.showwarning("警告", "RCS数据未加载，请在数据管理页面加载数据!")
                return

            if not hasattr(self, 'param_data') or self.param_data is None:
                messagebox.showwarning("警告", "参数数据未加载，请在数据管理页面加载数据!")
                return

            # 重置并生成本次会话的时间戳（用于命名一致性）
            self.ae_session_timestamp = None  # 清空旧的时间戳
            session_ts = self.get_ae_session_timestamp()  # 生成新的时间戳
            self.ae_log(f"🕐 会话时间戳: {session_ts}")

            self.ae_log("📊 检测到已加载的数据:")
            self.ae_log(f"  RCS数据形状: {self.rcs_data.shape}")
            self.ae_log(f"  参数数据形状: {self.param_data.shape}")

            # 自动检测频率配置
            detected_freq = self.rcs_data.shape[-1] if len(self.rcs_data.shape) == 4 else 2
            if detected_freq == 2:
                auto_freq_config = "2freq"
                freq_desc = "1.5GHz+3GHz"
            elif detected_freq == 3:
                auto_freq_config = "3freq"
                freq_desc = "1.5GHz+3GHz+6GHz"
            else:
                auto_freq_config = "2freq"  # 默认
                freq_desc = f"{detected_freq}频率"

            # 更新频率配置（如果与检测结果不同）
            current_config = self.ae_freq_config.get()
            if current_config != auto_freq_config:
                self.ae_log(f"⚠️ 自动调整频率配置: {current_config} → {auto_freq_config}")
                self.ae_freq_config.set(auto_freq_config)

            self.ae_log("🚀 开始创建AutoEncoder系统...")

            # 导入AutoEncoder模块
            try:
                import sys
                sys.path.append('autoencoder')
                from autoencoder.utils.frequency_config import create_autoencoder_system

                # 获取配置参数
                freq_config = self.ae_freq_config.get()
                latent_dim = int(self.ae_latent_dim.get())
                dropout_rate = float(self.ae_dropout_rate.get())
                wavelet_type = self.ae_wavelet_type.get()

                # 获取mode和architecture参数（如果存在）
                mode = self.ae_mode.get() if hasattr(self, 'ae_mode') else 'wavelet'
                architecture = self.ae_architecture_type.get().lower() if hasattr(self, 'ae_architecture_type') else 'cnn'

                # 移除重复的预处理配置，直接使用数据管理的预处理结果
                normalize = True  # 数据管理页面已经处理过标准化

                # 创建系统
                self.ae_system = create_autoencoder_system(
                    config_name=freq_config,
                    latent_dim=latent_dim,
                    dropout_rate=dropout_rate,
                    wavelet=wavelet_type,
                    normalize=normalize,
                    mode=mode,
                    architecture=architecture
                )

                # 存储数据引用，便于训练使用
                self.ae_system['rcs_data'] = self.rcs_data

                self.ae_system['param_data'] = self.param_data

                self.ae_log(f"✅ AutoEncoder系统创建成功!")
                self.ae_log(f"  📊 配置: {freq_config}")
                self.ae_log(f"  🔧 模式: {mode}")
                self.ae_log(f"  🏗️ 架构: {architecture.upper()}")
                self.ae_log(f"  🎯 隐空间维度: {latent_dim}")
                self.ae_log(f"  📊 频率信息: {self.ae_system['config_info'].get('num_frequencies')}频 {self.ae_system['config_info'].get('frequency_labels', [])}")
                self.ae_log(f"  📈 模型参数量: {self.ae_system['autoencoder'].get_parameter_count()['total']:,}")

                # 更新状态
                self.update_ae_status()

                messagebox.showinfo("成功",
                    f"AutoEncoder系统创建成功!\n\n"
                    f"模式: {mode}\n"
                    f"架构: {architecture.upper()}\n"
                    f"频率: {freq_config}")

            except ImportError as e:
                error_msg = f"导入AutoEncoder模块失败: {e}"
                self.ae_log(f"❌ {error_msg}")
                messagebox.showerror("错误", error_msg)

        except Exception as e:
            error_msg = f"创建AutoEncoder系统失败: {e}"
            self.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def start_ae_training(self):
        """开始AutoEncoder训练 (使用统一配置管理器)"""
        return self.training_manager.start_ae_training()

    def stop_ae_training(self):
        """停止AutoEncoder训练"""
        return self.training_manager.stop_ae_training()

    def save_ae_model(self):
        """保存AutoEncoder模型"""
        try:
            if self.ae_system is None:
                messagebox.showwarning("警告", "没有可保存的模型!")
                return

            filename = filedialog.asksaveasfilename(
                title="保存AutoEncoder模型",
                defaultextension=".pth",
                filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
            )

            if filename:
                # 保存模型状态（包含完整配置信息）
                import torch

                # 构建完整配置（包含系统创建所需的所有参数）
                complete_config = self.ae_system['config_info'].copy()
                complete_config.update({
                    'config_name': self.ae_freq_config.get(),
                    'latent_dim': int(self.ae_latent_dim.get()),
                    'dropout_rate': float(self.ae_dropout_rate.get()),
                    'wavelet': self.ae_wavelet_type.get(),
                    'normalize': self.ae_normalize.get(),  # 从GUI读取
                    'db_transform': self.ae_db_transform.get(),  # 从GUI读取
                    'mode': self.ae_system.get('mode', 'wavelet'),  # 从系统字典获取
                    'architecture': self.ae_system.get('architecture', 'cnn')  # 从系统字典获取
                })

                # 保存data_adapter统计信息（用于inverse_adapt还原数据）
                data_adapter = self.ae_system.get('data_adapter', None)
                adapter_stats = {}
                if data_adapter and hasattr(data_adapter, 'data_stats'):
                    adapter_stats = data_adapter.data_stats.copy()
                    # 将numpy数组转换为列表以便保存
                    if 'mean' in adapter_stats:
                        adapter_stats['mean'] = adapter_stats['mean'].tolist()
                    if 'std' in adapter_stats:
                        adapter_stats['std'] = adapter_stats['std'].tolist()

                # 保存训练模式信息
                training_mode = 'three_stage'  # 默认
                if self.ae_training_history and 'training_mode' in self.ae_training_history:
                    training_mode = self.ae_training_history['training_mode']

                model_state = {
                    'autoencoder': self.ae_system['autoencoder'].state_dict(),
                    'parameter_mapper': self.ae_system['parameter_mapper'].state_dict(),
                    'config': complete_config,
                    'adapter_stats': adapter_stats,  # 保存统计信息
                    'training_history': self.ae_training_history,
                    'training_mode': training_mode  # 保存训练模式
                }

                torch.save(model_state, filename)
                self.ae_log(f"💾 模型保存成功: {filename}")
                self.ae_log(f"  保存配置: mode={complete_config['mode']}, arch={complete_config['architecture']}")
                self.ae_log(f"  频率配置: {complete_config.get('num_frequencies', 'N/A')}频 {complete_config.get('frequency_labels', [])}")
                messagebox.showinfo("成功",
                    f"模型已保存到: {filename}\n\n"
                    f"频率配置: {complete_config.get('num_frequencies')}频 {complete_config.get('frequency_labels', [])}")

        except Exception as e:
            error_msg = f"保存模型失败: {e}"
            self.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def load_ae_model(self):
        """加载AutoEncoder模型 (自动重建系统)"""
        try:
            filename = filedialog.askopenfilename(
                title="加载AutoEncoder模型",
                filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
            )

            if filename:
                self.ae_log(f"正在加载模型: {filename}")

                import torch
                checkpoint = torch.load(filename, map_location='cpu')

                # 从检查点提取配置信息
                if 'config' not in checkpoint:
                    self.ae_log("❌ 检查点缺少config信息，无法自动重建系统")
                    messagebox.showerror("错误", "模型文件缺少配置信息，请使用旧版方式先创建系统再加载")
                    return

                config = checkpoint['config']

                # 提取系统创建所需参数
                freq_config = config.get('config_name', '2freq')
                latent_dim = config.get('latent_dim', 256)
                dropout_rate = config.get('dropout_rate', 0.2)
                wavelet_type = config.get('wavelet', 'db4')
                normalize = config.get('normalize', True)
                mode = config.get('mode', 'wavelet')  # 默认小波模式

                # 智能检测架构类型（从config或state_dict推断）
                architecture = config.get('architecture', None)

                if architecture is None:
                    # 旧版模型没有保存architecture，需要从state_dict推断
                    state_dict_keys = list(checkpoint['autoencoder'].keys())

                    self.ae_log("🔍 检测旧版模型架构...")
                    self.ae_log(f"  State dict前5个键: {state_dict_keys[:5]}")

                    # 检测MLP特征：encoder.1是大型Linear层（输入维度>1000）
                    if 'encoder.1.weight' in state_dict_keys:
                        shape = checkpoint['autoencoder']['encoder.1.weight'].shape
                        self.ae_log(f"  encoder.1.weight形状: {shape}")

                        if len(shape) == 2 and shape[1] > 1000:  # 注意：shape[1]是输入维度
                            architecture = 'mlp'
                            self.ae_log("⚠️ 检测到旧版MLP模型（encoder.1输入维度={})".format(shape[1]))
                        else:
                            architecture = 'cnn'
                            self.ae_log("⚠️ 检测到旧版CNN模型")

                    # 检测Enhanced CNN特征：multi_scale模块
                    elif any('multi_scale' in key for key in state_dict_keys):
                        architecture = 'enhanced_cnn'
                        self.ae_log("⚠️ 检测到Enhanced CNN模型")

                    # 检测标准CNN/Direct特征
                    elif 'encoder.0.weight' in state_dict_keys:
                        shape = checkpoint['autoencoder']['encoder.0.weight'].shape
                        if len(shape) == 4:  # Conv2d
                            architecture = 'cnn'
                            self.ae_log("⚠️ 检测到标准CNN模型")
                        else:
                            architecture = 'cnn'
                            self.ae_log("⚠️ 无法确定架构，默认CNN")
                    else:
                        architecture = 'cnn'  # 默认CNN
                        self.ae_log("⚠️ 无法确定架构，默认使用CNN")

                self.ae_log(f"📋 从检查点读取配置:")
                self.ae_log(f"  频率配置: {freq_config}")
                self.ae_log(f"  隐空间维度: {latent_dim}")
                self.ae_log(f"  模式: {mode}")
                self.ae_log(f"  架构: {architecture}")

                # 导入AutoEncoder模块
                import sys
                sys.path.append('autoencoder')
                from autoencoder.utils.frequency_config import create_autoencoder_system

                # 自动创建系统
                self.ae_log("🔧 正在自动重建AutoEncoder系统...")
                self.ae_system = create_autoencoder_system(
                    config_name=freq_config,
                    latent_dim=latent_dim,
                    dropout_rate=dropout_rate,
                    wavelet=wavelet_type,
                    normalize=normalize,
                    mode=mode,
                    architecture=architecture
                )

                # 加载模型权重
                self.ae_system['autoencoder'].load_state_dict(checkpoint['autoencoder'])
                self.ae_system['parameter_mapper'].load_state_dict(checkpoint['parameter_mapper'])

                # 恢复data_adapter统计信息
                if 'adapter_stats' in checkpoint and checkpoint['adapter_stats']:
                    import numpy as np
                    adapter_stats = checkpoint['adapter_stats'].copy()
                    # 将列表转换回numpy数组
                    if 'mean' in adapter_stats:
                        adapter_stats['mean'] = np.array(adapter_stats['mean'])
                    if 'std' in adapter_stats:
                        adapter_stats['std'] = np.array(adapter_stats['std'])
                    self.ae_system['data_adapter'].data_stats = adapter_stats
                    self.ae_log(f"✅ 已恢复data_adapter统计信息")
                else:
                    self.ae_log(f"⚠️ 模型文件不包含adapter统计信息（可能是旧版模型）")

                # 更新GUI中的预处理选项
                db_transform = config.get('db_transform', False)
                self.ae_normalize.set(normalize)
                self.ae_db_transform.set(db_transform)
                self.ae_system['data_adapter'].normalize = normalize
                self.ae_system['data_adapter'].db_transform = db_transform
                self.ae_log(f"🔧 数据预处理: 标准化={normalize}, dB变换={db_transform}")

                # 如果有数据，也加载到系统中
                if hasattr(self, 'rcs_data') and self.rcs_data is not None:
                    self.ae_system['rcs_data'] = self.rcs_data
                if hasattr(self, 'param_data') and self.param_data is not None:
                    # 修复键名错误：应为 'param_data'，否则重建函数在three_stage模式下取数失败
                    self.ae_system['param_data'] = self.param_data

                if 'training_history' in checkpoint:
                    self.ae_training_history = checkpoint['training_history']

                # 识别训练模式
                training_mode = checkpoint.get('training_mode', 'three_stage')  # 默认为三阶段
                training_mode_display = {
                    'stage1_only': 'Stage 1 Only (仅重建)',
                    'three_stage': '完整三阶段'
                }.get(training_mode, training_mode)

                # 重置会话时间戳（加载模型算作新会话）
                self.ae_session_timestamp = None
                session_ts = self.get_ae_session_timestamp()

                self.ae_trained = True

                # 验证频率配置匹配性
                model_num_freq = self.ae_system['config_info'].get('num_frequencies', 'unknown')
                model_freq_labels = self.ae_system['config_info'].get('frequency_labels', [])

                self.ae_log(f"✅ 模型加载成功: {filename}")
                self.ae_log(f"  系统已自动重建，无需手动创建")
                self.ae_log(f"  模型频率配置: {model_num_freq}频 {model_freq_labels}")
                self.ae_log(f"  训练模式: {training_mode_display}")
                self.ae_log(f"🕐 新会话时间戳: {session_ts}")

                # 如果是Stage 1 Only模式，给出提示
                if training_mode == 'stage1_only':
                    self.ae_log(f"💡 提示: 该模型仅训练了AutoEncoder重建，不能从参数预测RCS")
                    self.ae_log(f"  评估方式: 直接从RCS数据测试重建能力")
                else:
                    self.ae_log(f"  评估方式: 从参数预测RCS")

                # 存储training_mode到系统中供评估使用
                self.ae_system['training_mode'] = training_mode

                # 检查是否已加载数据
                if hasattr(self, 'rcs_data') and self.rcs_data is not None:
                    data_num_freq = self.rcs_data.shape[-1]
                    self.ae_log(f"  当前数据频率数: {data_num_freq}")

                    # 频率不匹配警告
                    if model_num_freq != data_num_freq:
                        warning_msg = (
                            f"⚠️ 频率配置不匹配！\n\n"
                            f"模型频率: {model_num_freq}频 {model_freq_labels}\n"
                            f"数据频率: {data_num_freq}频\n\n"
                            f"请重新加载匹配的数据！\n"
                            f"数据管理页面 → 加载RCS数据 → 选择{model_num_freq}频数据"
                        )
                        self.ae_log(f"❌ {warning_msg}")
                        messagebox.showwarning("频率配置不匹配", warning_msg)
                else:
                    self.ae_log(f"  尚未加载数据，请在数据管理页面加载{model_num_freq}频数据")
                    messagebox.showinfo("提示",
                        f"模型已加载！\n\n"
                        f"模型频率配置: {model_num_freq}频 {model_freq_labels}\n\n"
                        f"请前往数据管理页面加载匹配的{model_num_freq}频数据")

                self.update_ae_status()

                messagebox.showinfo("成功",
                    f"模型已加载并自动重建系统!\n\n"
                    f"文件: {filename}\n"
                    f"模式: {mode}\n"
                    f"架构: {architecture}\n"
                    f"频率: {freq_config} ({model_num_freq}频)")

        except Exception as e:
            error_msg = f"加载模型失败: {e}"
            self.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)
            import traceback
            traceback.print_exc()

    def save_ae_params(self):
        """保存AutoEncoder训练参数配置"""
        try:
            filename = filedialog.asksaveasfilename(
                title="保存AutoEncoder参数配置",
                defaultextension=".json",
                filetypes=[("JSON配置", "*.json"), ("所有文件", "*.*")]
            )

            if filename:
                import json

                # 收集所有参数
                params = {
                    'freq_config': self.ae_freq_config.get(),
                    'latent_dim': self.ae_latent_dim.get(),
                    'dropout_rate': self.ae_dropout_rate.get(),
                    'wavelet_type': self.ae_wavelet_type.get(),
                    'architecture_type': self.ae_architecture_type.get(),
                    'mode': self.ae_mode.get() if hasattr(self, 'ae_mode') else 'wavelet',

                    # 训练参数
                    'batch_size': self.ae_batch_size.get(),
                    'learning_rate': self.ae_learning_rate.get(),
                    'epochs_stage1': self.ae_epochs_stage1.get(),
                    'epochs_stage2': self.ae_epochs_stage2.get(),
                    'epochs_stage3': self.ae_epochs_stage3.get(),

                    # 学习率调度
                    'lr_scheduler': self.ae_lr_scheduler.get(),
                    'min_lr': self.ae_min_lr.get(),
                    'restart_period': self.ae_restart_period.get(),

                    # 早停参数
                    'patience_stage1': self.ae_patience_stage1.get(),
                    'patience_stage2': self.ae_patience_stage2.get(),
                    'patience_stage3': self.ae_patience_stage3.get(),
                    'patience_e2e': self.ae_patience_e2e.get(),

                    # 训练模式和损失函数
                    'training_mode': self.ae_training_mode.get(),
                    'use_custom_loss': self.ae_use_custom_loss.get()
                }

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=4, ensure_ascii=False)

                self.ae_log(f"💾 参数配置保存成功: {filename}")
                messagebox.showinfo("成功", f"参数配置已保存到:\n{filename}")

        except Exception as e:
            error_msg = f"保存参数配置失败: {e}"
            self.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def load_ae_params(self):
        """加载AutoEncoder训练参数配置"""
        try:
            filename = filedialog.askopenfilename(
                title="加载AutoEncoder参数配置",
                filetypes=[("JSON配置", "*.json"), ("所有文件", "*.*")]
            )

            if filename:
                import json

                with open(filename, 'r', encoding='utf-8') as f:
                    params = json.load(f)

                # 应用参数
                self.ae_freq_config.set(params.get('freq_config', '2freq'))
                self.ae_latent_dim.set(params.get('latent_dim', '256'))
                self.ae_dropout_rate.set(params.get('dropout_rate', '0.2'))
                self.ae_wavelet_type.set(params.get('wavelet_type', 'db4'))
                self.ae_architecture_type.set(params.get('architecture_type', 'CNN'))
                if hasattr(self, 'ae_mode'):
                    self.ae_mode.set(params.get('mode', 'wavelet'))

                # 训练参数
                self.ae_batch_size.set(params.get('batch_size', '16'))
                self.ae_learning_rate.set(params.get('learning_rate', '1e-3'))
                self.ae_epochs_stage1.set(params.get('epochs_stage1', '100'))
                self.ae_epochs_stage2.set(params.get('epochs_stage2', '50'))
                self.ae_epochs_stage3.set(params.get('epochs_stage3', '20'))

                # 学习率调度
                self.ae_lr_scheduler.set(params.get('lr_scheduler', 'constant'))
                self.ae_min_lr.set(params.get('min_lr', '1e-5'))
                self.ae_restart_period.set(params.get('restart_period', '50'))

                # 早停参数
                self.ae_patience_stage1.set(params.get('patience_stage1', '10'))
                self.ae_patience_stage2.set(params.get('patience_stage2', '10'))
                self.ae_patience_stage3.set(params.get('patience_stage3', '5'))
                self.ae_patience_e2e.set(params.get('patience_e2e', '15'))

                # 训练模式和损失函数
                self.ae_training_mode.set(params.get('training_mode', '三阶段训练'))
                self.ae_use_custom_loss.set(params.get('use_custom_loss', False))

                self.ae_log(f"📂 参数配置加载成功: {filename}")
                self.ae_log(f"  模式: {params.get('mode', 'wavelet')}")
                self.ae_log(f"  架构: {params.get('architecture_type', 'CNN')}")
                self.ae_log(f"  隐空间维度: {params.get('latent_dim', '256')}")
                self.ae_log(f"  批次大小: {params.get('batch_size', '16')}")

                messagebox.showinfo("成功",
                    f"参数配置已加载!\n\n"
                    f"文件: {filename}\n"
                    f"模式: {params.get('mode', 'wavelet')}\n"
                    f"架构: {params.get('architecture_type', 'CNN')}")

        except Exception as e:
            error_msg = f"加载参数配置失败: {e}"
            self.ae_log(f"❌ {error_msg}")
            messagebox.showerror("错误", error_msg)

    def _run_three_stage_training(self, rcs_data, param_data, batch_size, learning_rate,
                                epochs_stage1, epochs_stage2, epochs_stage3):
        """执行三阶段训练"""
        return self.training_manager._run_three_stage_training(rcs_data, param_data, batch_size, learning_rate, epochs_stage1, epochs_stage2, epochs_stage3)

    def _run_end_to_end_training(self, rcs_data, param_data, batch_size, learning_rate, total_epochs):
        """执行端到端训练"""
        return self.training_manager._run_end_to_end_training(rcs_data, param_data, batch_size, learning_rate, total_epochs)

    def _train_autoencoder_stage1(self, rcs_data, batch_size, learning_rate, epochs):
        """阶段1: AutoEncoder预训练"""
        return self.training_manager._train_autoencoder_stage1(rcs_data, batch_size, learning_rate, epochs)

    def _train_parameter_mapping_stage2(self, rcs_data, param_data, batch_size, learning_rate, epochs):
        """阶段2: 参数映射训练"""
        return self.training_manager._train_parameter_mapping_stage2(rcs_data, param_data, batch_size, learning_rate, epochs)

    def _train_end_to_end_stage3(self, rcs_data, param_data, batch_size, learning_rate, epochs):
        """阶段3: 端到端微调"""
        return self.training_manager._train_end_to_end_stage3(rcs_data, param_data, batch_size, learning_rate, epochs)

    def _train_full_end_to_end(self, rcs_data, param_data, batch_size, learning_rate, total_epochs):
        """完整端到端训练"""
        return self.training_manager._train_full_end_to_end(rcs_data, param_data, batch_size, learning_rate, total_epochs)

    def _run_three_stage_training_v2(self, rcs_data, param_data, training_config):
        """执行三阶段训练 v2 (使用统一配置管理器)"""
        return self.training_manager._run_three_stage_training_v2(rcs_data, param_data, training_config)

    def _run_end_to_end_training_v2(self, rcs_data, param_data, training_config):
        """执行端到端训练 v2 (使用统一配置管理器)"""
        return self.training_manager._run_end_to_end_training_v2(rcs_data, param_data, training_config)

    def _train_autoencoder_stage1_v2(self, rcs_data, training_config):
        """阶段1: AutoEncoder预训练 v2 (使用统一配置)"""
        return self.training_manager._train_autoencoder_stage1_v2(rcs_data, training_config)

    def _train_parameter_mapping_stage2_v2(self, rcs_data, param_data, training_config):
        """阶段2: 参数映射训练 v2 (使用统一配置)"""
        return self.training_manager._train_parameter_mapping_stage2_v2(rcs_data, param_data, training_config)

    def _train_end_to_end_stage3_v2(self, rcs_data, param_data, training_config):
        """阶段3: 端到端微调 v2 (使用统一配置)"""
        return self.training_manager._train_end_to_end_stage3_v2(rcs_data, param_data, training_config)

    def _train_full_end_to_end_v2(self, rcs_data, param_data, training_config, total_epochs):
        """完整端到端训练 v2 (使用统一配置)"""
        return self.training_manager._train_full_end_to_end_v2(rcs_data, param_data, training_config, total_epochs)

    def _open_loss_config_for_ae(self):
        """为AutoEncoder打开损失函数配置页面"""
        return self.training_manager._open_loss_config_for_ae()

    def _create_ae_training_config(self):
        """创建AutoEncoder训练配置 (复用项目配置管理器)"""
        return self.training_manager._create_ae_training_config()

    def _create_ae_optimizer_and_scheduler(self, model_params, training_config):
        """创建AutoEncoder优化器和学习率调度器 (复用项目标准)"""
        return self.training_manager._create_ae_optimizer_and_scheduler(model_params, training_config)

    def _create_ae_loss_function(self, training_config):
        """创建AutoEncoder损失函数 (用于阶段1重建任务)"""
        return self.training_manager._create_ae_loss_function(training_config)

    def _create_end_to_end_loss_function(self, training_config):
        """创建端到端损失函数 (用于阶段3 RCS预测，与其他网络相同)"""
        return self.training_manager._create_end_to_end_loss_function(training_config)

    def _ae_step_scheduler(self, scheduler, scheduler_type, val_loss=None):
        """AutoEncoder学习率调度器步进 (复用项目调度逻辑)"""
        return self.training_manager._ae_step_scheduler(scheduler, scheduler_type, val_loss=val_loss)

    def _ae_log_training_progress(self, epoch, total_epochs, train_loss, val_loss, lr, stage_name):
        """AutoEncoder训练进度日志 (统一格式)"""
        return self.training_manager._ae_log_training_progress(epoch, total_epochs, train_loss, val_loss, lr, stage_name)

    def _plot_autoencoder_visualization(self, chart_type):
        """绘制AutoEncoder特定可视化图表"""
        return self.visualization_manager._plot_autoencoder_visualization(chart_type)

    def _plot_ae_latent_space(self):
        """绘制AutoEncoder隐空间分析"""
        return self.visualization_manager._plot_ae_latent_space()

    def _plot_ae_reconstruction_quality(self):
        """绘制AutoEncoder重建质量分析 - 使用统一重建函数"""
        return self.visualization_manager._plot_ae_reconstruction_quality()

    def _plot_ae_parameter_mapping(self):
        """绘制AutoEncoder参数映射分析"""
        return self.visualization_manager._plot_ae_parameter_mapping()

    def _plot_ae_training_progress_vis(self):
        """绘制AutoEncoder训练进度可视化"""
        return self.visualization_manager._plot_ae_training_progress_vis()

    def _plot_autoencoder_prediction_visualization(self, chart_type, freq):
        """使用AutoEncoder进行预测可视化"""
        return self.visualization_manager._plot_autoencoder_prediction_visualization(chart_type, freq)

    def _plot_ae_2d_heatmap(self, freq):
        """绘制AutoEncoder预测的2D热图 - 支持模型未加载时显示原始数据"""
        return self.visualization_manager._plot_ae_2d_heatmap(freq)

    def _plot_original_rcs_fallback(self, freq):
        """当AutoEncoder模型未加载时，显示原始RCS数据作为替代"""
        return self.visualization_manager._plot_original_rcs_fallback(freq)

    def _plot_ae_comparison(self):
        """绘制AutoEncoder对比图：原图、重构图、残差图 - 使用统一重建函数"""
        return self.visualization_manager._plot_ae_comparison()

    def _plot_wavelet_coefficients_comparison(self):
        """绘制小波系数对比图：原始vs重建的4个通道（LL, LH, HL, HH）"""
        return self.visualization_manager._plot_wavelet_coefficients_comparison()

    def save_current_visualization(self):
        """保存当前显示的可视化图表到results文件夹"""
        return self.visualization_manager.save_current_visualization()

def main():
    """主函数"""
    # 创建根窗口
    root = tk.Tk()

    # 设置主题
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except:
        pass  # 如果主题文件不存在，使用默认主题

    # 创建应用
    app = RCSWaveletGUI(root)

    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()
