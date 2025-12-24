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
# GUI标签页模块
from gui_managers.tabs.data_management_tab import DataManagementTab
from gui_managers.tabs.training_tab import TrainingTab
from gui_managers.tabs.visualization_tab import VisualizationTab
from gui_managers.tabs.loss_config_tab import LossConfigTab
from gui_managers.tabs.evaluation_tab import EvaluationTab
from gui_managers.tabs.prediction_tab import PredictionTab

# 导入项目模块
try:
    from wavelet_network import create_model, create_loss_function
    from autoencoder.utils.configurable_loss import create_loss_function as create_configurable_loss
    from training import (CrossValidationTrainer, RCSDataLoader,
                         create_training_config, create_data_config, RCSDataset)
    from evaluation import RCSEvaluator, evaluate_model_with_visualizations
    from autoencoder.utils.data_cache import create_cache_manager

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
        self.root.geometry("1600x1200")  # 增加默认尺寸
        self.root.minsize(1200, 900)
        
        # 尝试最大化窗口
        # try:
        #     self.root.state('zoomed')
        # except:
        #     pass

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
        self.ae_model_loaded = False  # 标记是否加载了模型
        self.ae_loaded_weights = None  # 保存加载的权重副本（用于继续训练）

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
        self.ae_activation = tk.StringVar(value="relu")  # 激活函数类型

        # Additive Dual-Branch专用配置
        self.ae_activation_encoder = tk.StringVar(value="relu")  # Encoder激活函数
        self.ae_activation_high = tk.StringVar(value="sin")  # 高频Decoder激活函数
        self.ae_activation_smooth = tk.StringVar(value="tanh")  # 低频Decoder激活函数
        self.ae_learnable_weights = tk.BooleanVar(value=False)  # 是否使用可学习权重
        self.ae_alpha_high = tk.StringVar(value="0.5")  # 高频权重（固定权重模式）
        self.ae_alpha_smooth = tk.StringVar(value="0.5")  # 低频权重（固定权重模式）

        # 参数映射器配置
        self.ae_mapper_activation = tk.StringVar(value="auto")  # auto表示与AutoEncoder相同
        self.ae_mapper_use_adaptive = tk.BooleanVar(value=True)  # 默认使用自适应层

        # 训练配置
        self.ae_batch_size = tk.StringVar(value="16")
        self.ae_learning_rate = tk.StringVar(value="1e-3")
        self.ae_epochs_stage1 = tk.StringVar(value="100")  # AE预训练轮数
        self.ae_epochs_stage2 = tk.StringVar(value="50")   # 参数映射训练轮数
        self.ae_epochs_stage3 = tk.StringVar(value="20")   # 端到端微调轮数
        self.ae_epochs_joint = tk.StringVar(value="200")   # 联合训练轮数

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
        # 多阶段学习率调度器参数
        self.ae_num_lr_stages = tk.StringVar(value="3")  # 阶段数量
        self.ae_lr_decay_factor = tk.StringVar(value="0.1")  # 学习率衰减因子
        self.ae_patience_multiplier = tk.StringVar(value="2.0")  # Patience倍增因子

        # 早停配置 (分阶段可配置)
        self.ae_patience_stage1 = tk.StringVar(value="10")  # 阶段1早停耐心值
        self.ae_patience_stage2 = tk.StringVar(value="10")  # 阶段2早停耐心值
        self.ae_patience_stage3 = tk.StringVar(value="5")   # 阶段3早停耐心值
        self.ae_patience_e2e = tk.StringVar(value="15")     # 端到端早停耐心值
        self.ae_patience_joint = tk.StringVar(value="50")   # 联合训练早停耐心值

        # 联合训练损失权重配置
        self.ae_alpha_recon = tk.StringVar(value="0.3")       # RCS重建损失权重
        self.ae_beta_consistency = tk.StringVar(value="0.5")  # 一致性损失权重
        self.ae_gamma_param_recon = tk.StringVar(value="1.0") # 参数重建损失权重（最重要）

        # 数据预处理配置
        # 预处理选项已移至数据管理页面，此处不再需要相关变量

        # 训练模式
        self.ae_training_mode = tk.StringVar(value="三阶段训练")  # 三阶段训练 / 端到端训练 / 仅Stage 1

        # 损失函数配置复用
        self.ae_use_custom_loss = tk.BooleanVar(value=False)  # 是否使用自定义损失函数

        # 梯度监控配置 (新增)
        self.ae_gradient_monitoring = tk.BooleanVar(value=False)  # 是否启用梯度监控 (默认关闭以提高性能)

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

    def setup_layout(self):
        """设置布局"""
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 设置样式
        style = ttk.Style()
        style.configure("Accent.TButton")

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

        # 标签页8: 批量实验
        self.batch_experiment_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_experiment_frame, text="批量实验")
        self.create_batch_experiment_tab()

    def create_data_tab(self):
        """创建数据管理标签页"""
        
        # 使用新重构的标签页类
        # 代码已迁移至 gui_managers/tabs/data_management_tab.py
        self.data_tab = DataManagementTab(self.data_frame, self)
        self.data_tab.pack(fill=tk.BOTH, expand=True)
        return

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

    def create_loss_config_tab(self):
        """创建损失配置标签页"""

        # 使用新重构的标签页类
        # 代码已迁移至 gui_managers/tabs/loss_config_tab.py
        self.loss_config_tab = LossConfigTab(self.loss_config_frame, self)
        self.loss_config_tab.pack(fill=tk.BOTH, expand=True)
        return

    def create_training_tab(self):
        """创建模型训练标签页"""

        # 使用新重构的标签页类
        # 代码已迁移至 gui_managers/tabs/training_tab.py
        self.training_tab = TrainingTab(self.training_frame, self)
        self.training_tab.pack(fill=tk.BOTH, expand=True)
        return

    def create_evaluation_tab(self):
        """创建评估标签页"""

        # 使用新重构的标签页类
        # 代码已迁移至 gui_managers/tabs/evaluation_tab.py
        self.evaluation_tab = EvaluationTab(self.evaluation_frame, self)
        self.evaluation_tab.pack(fill=tk.BOTH, expand=True)
        return

    def create_prediction_tab(self):
        """创建预测标签页"""

        # 使用新重构的标签页类
        # 代码已迁移至 gui_managers/tabs/prediction_tab.py
        self.prediction_tab = PredictionTab(self.prediction_frame, self)
        self.prediction_tab.pack(fill=tk.BOTH, expand=True)
        return

    def create_visualization_tab(self):
        """创建可视化标签页"""

        # 使用新重构的标签页类
        # 代码已迁移至 gui_managers/tabs/visualization_tab.py
        self.visualization_tab = VisualizationTab(self.visualization_frame, self)
        self.visualization_tab.pack(fill=tk.BOTH, expand=True)
        return

    def create_batch_experiment_tab(self):
        """创建批量实验标签页

        ⚠️ 此方法已移除！实际界面由 gui_batch_experiment_extension.py 提供
        ==========================================
        此标签页的内容在 main.py 启动时会被 BatchExperimentExtension 覆盖

        如需修改批量实验界面，请修改：
        - gui_batch_experiment_extension.py: BatchExperimentExtension类
        ==========================================
        """
        # 创建占位框架，将被extension覆盖
        placeholder = ttk.Frame(self.batch_experiment_frame)
        placeholder.pack(fill=tk.BOTH, expand=True)

        info_label = ttk.Label(
            placeholder,
            text="批量实验界面正在加载...\n请稍候，界面将由扩展模块提供",
            font=self.font_medium,
            justify=tk.CENTER
        )
        info_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)



    # ======= 系统管理功能 (已迁移至 DataManagementTab) =======
    # CUDA管理方法已迁移至 gui_managers/tabs/data_management_tab.py

    # ======= 训练功能 (已迁移至 TrainingTab) =======
    # 所有训练相关方法已迁移至 gui_managers/tabs/training_tab.py

    # ======= 评估功能 (已迁移至 EvaluationTab) =======
    # 所有评估相关方法已迁移至 gui_managers/tabs/evaluation_tab.py

    def _reconstruct_rcs(self, input_data=None, input_type='auto', model_ids=None, return_latents=False, return_wavelet_coeffs=False):
        """统一的RCS重建函数 (委托给ReconstructionManager)"""
        return self.reconstruction_manager._reconstruct_rcs(
            input_data=input_data,
            input_type=input_type,
            model_ids=model_ids,
            return_latents=return_latents,
            return_wavelet_coeffs=return_wavelet_coeffs
        )

    # ======= 预测功能 (已迁移至 PredictionTab) =======
    # 所有预测相关方法已迁移至 gui_managers/tabs/prediction_tab.py

    # ======= 可视化功能 (已迁移至 VisualizationTab) =======
    # 所有可视化相关方法已迁移至 gui_managers/tabs/visualization_tab.py

    # ======= 辅助功能 =======

    def log_message(self, message, level='INFO'):
        """记录日志消息 - 现在直接使用print输出，会被自动捕获"""
        print(message)

    # ======= 损失函数配置方法 (已迁移至 LossConfigTab) =======
    # 所有损失配置相关方法已迁移至 gui_managers/tabs/loss_config_tab.py

    def on_closing(self):
        """窗口关闭事件处理"""
        # 弹出确认对话框
        if messagebox.askyesno("确认退出", "确定要退出程序吗？"):
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
        """更新AutoEncoder系统状态显示（紧凑格式）"""
        try:
            status_info = []

            # 如果系统已创建，显示完整信息
            if self.ae_system is not None:
                config = self.ae_system.get('config_info', {})
                model = self.ae_system['autoencoder']

                # 提取关键参数
                mode = config.get('mode', self.ae_mode.get())
                architecture = config.get('architecture', self.ae_architecture.get())
                activation = config.get('activation', self.ae_activation.get())
                latent_dim = config.get('latent_dim', self.ae_latent_dim.get())
                num_freq = config.get('num_frequencies', 2)
                wavelet = config.get('wavelet', self.ae_wavelet_type.get())
                dropout = config.get('dropout_rate', self.ae_dropout_rate.get())

                # 获取参数量
                param_count = model.get_parameter_count()
                total_params = param_count.get('total', 0)
                if total_params >= 1_000_000:
                    params_str = f"{total_params/1_000_000:.2f}M"
                elif total_params >= 1_000:
                    params_str = f"{total_params/1_000:.1f}K"
                else:
                    params_str = f"{total_params}"

                # 数据预处理参数
                data_adapter = self.ae_system.get('data_adapter')
                if data_adapter:
                    normalize = getattr(data_adapter, 'normalize', False)
                    use_db = getattr(data_adapter, 'use_db', False)
                    norm_method = getattr(data_adapter, 'normalization_method', 'z_score')
                    preprocess_str = []
                    if normalize:
                        preprocess_str.append(f"标准化({norm_method})")
                    if use_db:
                        preprocess_str.append("dB")
                    preprocess = "+".join(preprocess_str) if preprocess_str else "无"
                else:
                    preprocess = "无"

                # 训练模式
                training_mode = self.ae_system.get('training_mode', 'N/A')
                mode_display = {
                    'stage1_only': '仅Stage1', 
                    'three_stage': '3阶段',
                    'joint_training': '联合',
                    'stage2_only': '仅Stage2',
                    'end_to_end': '端到端'
                }.get(training_mode, training_mode)
                trained_status = "已训练" if self.ae_trained else "未训练"

                # 第1行：网络架构（紧凑格式）
                freq_labels = config.get('frequency_labels', ['1.5GHz', '3.0GHz'])
                freq_str = '+'.join(freq_labels)
                status_info.append(f"【网络】{mode.upper()}-{architecture.upper()} | 激活:{activation} | 隐空间:{latent_dim}D | 参数:{params_str} | 频率:{num_freq}f({freq_str}) | 小波:{wavelet} | Dropout:{dropout}")

                # 第2行：数据和训练状态（紧凑格式）
                status_info.append(f"【状态】预处理:{preprocess} | 训练:{trained_status}({mode_display}) | 系统:✓")

                # 如果有训练历史，显示最佳Loss
                if hasattr(self, 'ae_training_history') and self.ae_training_history:
                    stage_histories = self.ae_training_history.get('stage_histories', {})
                    best_losses = []
                    for stage_name in ['stage1', 'stage2', 'stage3']:
                        if stage_name in stage_histories:
                            best_loss = stage_histories[stage_name].get('best_val_loss', None)
                            if isinstance(best_loss, float):
                                best_losses.append(f"{stage_name.upper()}:{best_loss:.6f}")
                    if best_losses:
                        status_info.append(f"【性能】最佳Loss: {' | '.join(best_losses)}")

            else:
                # 系统未创建，显示配置信息（单行）
                mode = self.ae_mode.get()
                architecture = self.ae_architecture.get()
                activation = self.ae_activation.get()
                freq_config = self.ae_freq_config.get()
                latent_dim = self.ae_latent_dim.get()
                wavelet = self.ae_wavelet_type.get()
                dropout = self.ae_dropout_rate.get()

                status_info.append(f"【配置】模式:{mode} | 架构:{architecture} | 激活:{activation} | 频率:{freq_config} | 隐空间:{latent_dim}D | 小波:{wavelet} | Dropout:{dropout} | 系统:✗")
                status_info.append("💡 点击'创建AutoEncoder系统'开始")

            # 更新显示
            self.ae_status_text.delete(1.0, tk.END)
            self.ae_status_text.insert(tk.END, "\n".join(status_info))

        except Exception as e:
            print(f"更新AE状态失败: {e}")

    def _print_latent_space_statistics(self, rcs_data):
        """
        打印隐空间统计信息

        Args:
            rcs_data: RCS数据 [N, 91, 91, num_freq]
        """
        try:
            import torch
            import numpy as np

            # 获取组件
            autoencoder = self.ae_system['autoencoder']
            wavelet_transform = self.ae_system.get('wavelet_transform', None)
            data_adapter = self.ae_system.get('data_adapter', None)
            mode = self.ae_system.get('mode', 'wavelet')

            # 设置设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            autoencoder.to(device)
            autoencoder.eval()

            self.ae_log("\n" + "="*60)
            self.ae_log("📊 隐空间统计分析")
            self.ae_log("="*60)

            # 数据预处理（与训练时保持一致）
            with torch.no_grad():
                if mode == 'wavelet':
                    # 先小波变换，再预处理
                    rcs_tensor = torch.FloatTensor(rcs_data)
                    wavelet_coeffs = wavelet_transform.forward_transform(rcs_tensor)
                    input_data = data_adapter.adapt_rcs_data(wavelet_coeffs.cpu().numpy())
                else:
                    # 直接预处理RCS
                    input_data = data_adapter.adapt_rcs_data(rcs_data)

                # 获取隐空间表示
                input_data = input_data.to(device)
                _, latent_space = autoencoder(input_data)
                latent_space = latent_space.cpu().numpy()

            # 打印前5个样本的完整隐空间数据
            num_samples_to_show = min(5, latent_space.shape[0])
            self.ae_log(f"\n【前{num_samples_to_show}个样本的隐空间表示】")
            self.ae_log(f"隐空间维度: {latent_space.shape[1]}")
            self.ae_log("-" * 60)

            for i in range(num_samples_to_show):
                self.ae_log(f"\n样本 #{i}:")
                # 将隐空间向量格式化为多行显示
                latent_vec = latent_space[i]
                # 每行显示10个数值
                for j in range(0, len(latent_vec), 10):
                    chunk = latent_vec[j:j+10]
                    chunk_str = ', '.join([f'{val:8.4f}' for val in chunk])
                    self.ae_log(f"  [{j:3d}-{min(j+9, len(latent_vec)-1):3d}]: {chunk_str}")

            # 计算每个维度的统计信息
            self.ae_log("\n" + "-" * 60)
            self.ae_log("【隐空间每个维度的统计信息】")
            self.ae_log("-" * 60)

            dim_means = np.mean(latent_space, axis=0)
            dim_stds = np.std(latent_space, axis=0)

            self.ae_log(f"总样本数: {latent_space.shape[0]}")
            self.ae_log(f"隐空间维度: {latent_space.shape[1]}")
            self.ae_log(f"\n每个维度的均值（前20维）:")
            for i in range(min(20, len(dim_means))):
                self.ae_log(f"  维度 {i:3d}: mean = {dim_means[i]:8.4f}, std = {dim_stds[i]:7.4f}")

            if len(dim_means) > 20:
                self.ae_log(f"  ... (共{len(dim_means)}维，仅显示前20维)")

            # 整体统计
            self.ae_log(f"\n【隐空间整体统计】")
            self.ae_log(f"  全局均值: {np.mean(latent_space):8.4f}")
            self.ae_log(f"  全局标准差: {np.std(latent_space):8.4f}")
            self.ae_log(f"  最小值: {np.min(latent_space):8.4f}")
            self.ae_log(f"  最大值: {np.max(latent_space):8.4f}")
            self.ae_log(f"  数值范围: [{np.min(latent_space):8.4f}, {np.max(latent_space):8.4f}]")

            # 维度间的统计
            self.ae_log(f"\n【维度间统计】")
            self.ae_log(f"  各维度均值的均值: {np.mean(dim_means):8.4f}")
            self.ae_log(f"  各维度均值的标准差: {np.std(dim_means):8.4f}")
            self.ae_log(f"  各维度标准差的均值: {np.mean(dim_stds):8.4f}")
            self.ae_log(f"  各维度标准差的标准差: {np.std(dim_stds):8.4f}")

            self.ae_log("="*60 + "\n")

        except Exception as e:
            self.ae_log(f"⚠️ 打印隐空间统计信息失败: {e}")
            import traceback
            self.ae_log(traceback.format_exc())

    def _update_status_bar_with_model_info(self):
        """更新状态栏显示网络参数信息"""
        try:
            if self.ae_system is None:
                self.status_var.set("就绪")
                return

            # 获取配置信息
            config = self.ae_system.get('config_info', {})
            model = self.ae_system['autoencoder']
            model_info = model.get_model_info()

            # 提取关键参数
            mode = config.get('mode', 'N/A')
            architecture = config.get('architecture', 'N/A')
            activation = config.get('activation', 'relu')
            latent_dim = config.get('latent_dim', 'N/A')
            num_freq = config.get('num_frequencies', 'N/A')
            wavelet = config.get('wavelet', 'N/A')
            dropout = config.get('dropout_rate', 'N/A')

            # 获取参数量
            param_count = model.get_parameter_count()
            total_params = param_count.get('total', 0)

            # 格式化参数量（k/M单位）
            if total_params >= 1_000_000:
                params_str = f"{total_params/1_000_000:.2f}M"
            elif total_params >= 1_000:
                params_str = f"{total_params/1_000:.1f}K"
            else:
                params_str = f"{total_params}"

            # 获取训练模式显示
            training_mode = self.ae_system.get('training_mode', 'N/A')
            mode_map = {
                'stage1_only': '仅Stage1',
                'three_stage': '3阶段',
                'joint_training': '联合',
                'stage2_only': '仅Stage2',
                'end_to_end': '端到端'
            }
            mode_display = mode_map.get(training_mode, training_mode)

            # 构建状态栏信息
            status_text = (
                f"网络: {mode.upper()}-{architecture.upper()} | "
                f"模式: {mode_display} | "
                f"激活: {activation} | "
                f"隐空间: {latent_dim}D | "
                f"频率: {num_freq}freq | "
                f"小波: {wavelet} | "
                f"参数量: {params_str}"
            )

            # 更新状态栏
            self.status_var.set(status_text)

        except Exception as e:
            print(f"更新状态栏失败: {e}")
            self.status_var.set("网络已创建")

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

            # 如果批量实验日志文件存在，同时写入文件
            if hasattr(self, 'batch_experiment_log_file') and self.batch_experiment_log_file is not None:
                try:
                    self.batch_experiment_log_file.write(log_message + "\n")
                    self.batch_experiment_log_file.flush()  # 立即刷新到磁盘
                except Exception as e:
                    print(f"写入批量实验日志失败: {e}")

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

    # ⚠️ 已废弃：create_ae_system() 函数已被删除
    # 原因：缺少 mapper_activation 等参数，导致ParameterMapper配置不正确
    # 替代方案：使用 gui_autoencoder_extension.py 中的 create_current_system()
    # 该函数已完整实现所有参数传递，包括mapper配置

    def start_ae_training(self):
        """开始AutoEncoder训练 (使用统一配置管理器)"""
        return self.training_manager.start_ae_training()

    def resume_ae_training(self):
        """继续训练AutoEncoder（从加载的权重继续）"""
        return self.training_manager.resume_ae_training()

    def stop_ae_training(self):
        """停止AutoEncoder训练"""
        return self.training_manager.stop_ae_training()

    def save_ae_model(self):
        """保存AutoEncoder模型"""
        try:
            if self.ae_system is None:
                messagebox.showwarning("警告", "没有可保存的模型!")
                return

            # 生成预设文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            mode = self.ae_system.get('mode', 'wavelet')
            architecture = self.ae_system.get('architecture', 'cnn')

            # 获取激活函数
            activation = self.ae_activation.get() if hasattr(self, 'ae_activation') else 'relu'

            # 获取预处理方式
            normalize = self.ae_normalize.get()
            db_transform = self.ae_db_transform.get()

            # 构建预处理标签
            if normalize and db_transform:
                preprocess = "norm_db"
            elif normalize:
                preprocess = "norm"
            elif db_transform:
                preprocess = "db"
            else:
                preprocess = "raw"

            # 格式: {mode}_{architecture}_{activation}_{preprocess}_{timestamp}.pth
            suggested_filename = f"{mode}_{architecture}_{activation}_{preprocess}_{timestamp}.pth"

            filename = filedialog.asksaveasfilename(
                title="保存AutoEncoder模型",
                defaultextension=".pth",
                initialfile=suggested_filename,
                filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
            )

            if filename:
                # ===== 使用统一的保存函数（autoencoder/utils/model_io.py）=====
                from autoencoder.utils.model_io import save_ae_model_to_file

                # 构建配置覆盖（补充GUI特有的字段）
                config_override = {
                    'config_name': self.ae_freq_config.get(),
                    'latent_dim': int(self.ae_latent_dim.get()),
                    'dropout_rate': float(self.ae_dropout_rate.get()),
                    'wavelet': self.ae_wavelet_type.get(),
                    'normalize': self.ae_normalize.get(),
                    'db_transform': self.ae_db_transform.get(),
                    'normalization_method': self.ae_normalization_method.get() if hasattr(self, 'ae_normalization_method') else 'none',
                    'mode': self.ae_system.get('mode', 'wavelet'),
                    'architecture': self.ae_system.get('architecture', 'cnn'),
                    'activation': activation
                }

                # 获取训练模式
                training_mode = 'three_stage'  # 默认
                if self.ae_training_history and 'training_mode' in self.ae_training_history:
                    training_mode = self.ae_training_history['training_mode']

                # 调用统一保存函数
                save_ae_model_to_file(
                    ae_system=self.ae_system,
                    model_path=filename,
                    training_history=self.ae_training_history,
                    training_mode=training_mode,
                    save_json_config=True,
                    config_override=config_override
                )
                # ===== 统一保存函数调用结束 =====

                self.ae_log(f"💾 模型保存成功: {filename}")

                # 获取配置信息用于日志
                import os
                config_filename = os.path.splitext(filename)[0] + "_config.json"
                saved_mode = config_override['mode']
                saved_arch = config_override['architecture']
                saved_nfreq = self.ae_system['config_info'].get('num_frequencies', 'N/A')
                saved_freq_labels = self.ae_system['config_info'].get('frequency_labels', [])

                self.ae_log(f"📄 配置文件保存成功: {config_filename}")
                self.ae_log(f"  保存配置: mode={saved_mode}, arch={saved_arch}")
                self.ae_log(f"  频率配置: {saved_nfreq}频 {saved_freq_labels}")

                messagebox.showinfo("成功",
                    f"模型已保存到: {filename}\n"
                    f"配置文件: {config_filename}\n\n"
                    f"频率配置: {saved_nfreq}频 {saved_freq_labels}")

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

                # ===== 使用统一的加载函数（autoencoder/utils/model_io.py）=====
                from autoencoder.utils.model_io import (
                    load_ae_model_from_file,
                    validate_model_config,
                    restore_data_adapter_stats
                )

                try:
                    checkpoint, config, training_history = load_ae_model_from_file(
                        model_path=filename,
                        device='cpu'
                    )
                except (FileNotFoundError, KeyError, RuntimeError) as e:
                    self.ae_log(f"❌ 加载模型失败: {e}")
                    messagebox.showerror("错误", f"模型文件加载失败:\n{e}")
                    return
                # ===== 统一加载函数调用结束 =====

                # 验证配置完整性
                is_valid, error_msg = validate_model_config(config)
                if not is_valid:
                    self.ae_log(f"❌ 模型配置无效: {error_msg}")
                    messagebox.showerror("错误", f"模型配置无效:\n{error_msg}")
                    return

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
                activation = config.get('activation', 'relu')  # 从config获取，默认relu
                self.ae_log(f"  激活函数: {activation}")

                # ⚠️ 处理旧版模型的向后兼容：正确推断normalization_method
                # 旧版模型只有 normalize (bool)，新版有 normalization_method (str)
                normalization_method = config.get('normalization_method', None)
                db_transform = config.get('db_transform', False)

                if normalization_method is None:
                    # 旧版模型：根据normalize推断normalization_method
                    if normalize:
                        normalization_method = 'zscore'  # 旧版默认使用zscore
                        self.ae_log(f"  ⚠️ 旧版模型，从normalize={normalize}推断 normalization_method='zscore'")
                    else:
                        normalization_method = 'none'
                        self.ae_log(f"  ⚠️ 旧版模型，从normalize={normalize}推断 normalization_method='none'")
                else:
                    self.ae_log(f"  标准化方法: {normalization_method}")

                # 读取ParameterMapper配置（如果存在）
                mapper_config = config.get('mapper_config', None)
                if mapper_config:
                    self.ae_log(f"  检测到Mapper配置:")
                    self.ae_log(f"    - 隐空间维度: {mapper_config.get('latent_dim', 'N/A')}")
                    self.ae_log(f"    - 隐藏层维度: {mapper_config.get('hidden_dims', 'N/A')}")
                    self.ae_log(f"    - 激活函数: {mapper_config.get('activation', 'N/A')}")
                    self.ae_log(f"    - 自适应层: {mapper_config.get('use_adaptive', False)}")
                    mapper_activation = mapper_config.get('activation', None)
                    mapper_use_adaptive = mapper_config.get('use_adaptive', False)
                    mapper_hidden_dims = mapper_config.get('hidden_dims', None)
                else:
                    # 旧版模型：使用默认mapper配置
                    self.ae_log("  ⚠️ 未检测到Mapper配置，使用默认值")
                    mapper_activation = None
                    mapper_use_adaptive = False
                    mapper_hidden_dims = None

                # 读取Additive Dual-Branch特有参数（如果存在）
                learnable_weights = config.get('learnable_weights', False)
                alpha_high = config.get('alpha_high', 0.5)
                alpha_smooth = config.get('alpha_smooth', 0.5)
                activation_encoder = config.get('activation_encoder', None)
                activation_high = config.get('activation_high', None)
                activation_smooth = config.get('activation_smooth', None)
                enforce_nonnegative_rcs = config.get('enforce_nonnegative_rcs', False)
                use_channel_attention = config.get('use_channel_attention', False)

                if 'additive_dual_branch' in architecture:
                    self.ae_log(f"  检测到Additive Dual-Branch架构配置:")
                    self.ae_log(f"    - learnable_weights: {learnable_weights}")
                    self.ae_log(f"    - alpha_high: {alpha_high}")
                    self.ae_log(f"    - alpha_smooth: {alpha_smooth}")
                    self.ae_log(f"    - activation_encoder: {activation_encoder}")
                    self.ae_log(f"    - activation_high: {activation_high}")
                    self.ae_log(f"    - activation_smooth: {activation_smooth}")

                self.ae_system = create_autoencoder_system(
                    config_name=freq_config,
                    latent_dim=latent_dim,
                    dropout_rate=dropout_rate,
                    wavelet=wavelet_type,
                    normalize=normalize,
                    mode=mode,
                    architecture=architecture,
                    activation=activation,
                    db_transform=db_transform,
                    normalization_method=normalization_method,
                    mapper_activation=mapper_activation,
                    mapper_use_adaptive=mapper_use_adaptive,
                    mapper_hidden_dims=mapper_hidden_dims,
                    # Additive Dual-Branch特有参数
                    activation_encoder=activation_encoder,
                    activation_high=activation_high,
                    activation_smooth=activation_smooth,
                    learnable_weights=learnable_weights,
                    alpha_high=alpha_high,
                    alpha_smooth=alpha_smooth,
                    enforce_nonnegative_rcs=enforce_nonnegative_rcs,
                    use_channel_attention=use_channel_attention
                )

                # 加载模型权重
                self.ae_system['autoencoder'].load_state_dict(checkpoint['autoencoder'])
                self.ae_system['parameter_mapper'].load_state_dict(checkpoint['parameter_mapper'])

                # 恢复data_adapter统计信息（使用统一函数）
                if 'adapter_stats' in checkpoint and checkpoint['adapter_stats']:
                    data_adapter = self.ae_system.get('data_adapter', None)
                    if data_adapter:
                        restore_data_adapter_stats(data_adapter, checkpoint['adapter_stats'])
                        self.ae_log(f"✅ 已恢复data_adapter统计信息")
                    else:
                        self.ae_log(f"⚠️ 系统中没有data_adapter，跳过统计信息恢复")
                else:
                    self.ae_log(f"⚠️ 模型文件不包含adapter统计信息（可能是旧版模型）")

                # 恢复param_scaler统计信息
                if 'param_scaler_stats' in checkpoint and checkpoint['param_scaler_stats']:
                    from sklearn.preprocessing import StandardScaler
                    import numpy as np

                    scaler_stats = checkpoint['param_scaler_stats']
                    if scaler_stats['mean'] is not None and scaler_stats['scale'] is not None:
                        # 创建新的StandardScaler并恢复统计信息
                        param_scaler = StandardScaler()
                        param_scaler.mean_ = np.array(scaler_stats['mean'])
                        param_scaler.scale_ = np.array(scaler_stats['scale'])
                        param_scaler.n_features_in_ = scaler_stats['n_features_in']

                        # 保存到ae_system
                        self.ae_system['param_scaler'] = param_scaler
                        self.ae_log(f"✅ 已恢复param_scaler统计信息 (参数数量: {scaler_stats['n_features_in']})")
                    else:
                        self.ae_log(f"⚠️ param_scaler统计信息不完整")
                else:
                    self.ae_log(f"⚠️ 模型文件不包含param_scaler统计信息（可能是旧版模型，参数未标准化）")

                # 恢复Loss归一化系数（如果存在）
                if 'loss_normalization_factor' in checkpoint:
                    loss_normalization_factor = checkpoint['loss_normalization_factor']
                    self.ae_system['loss_normalization_factor'] = loss_normalization_factor
                    self.ae_log(f"✅ 已恢复Loss归一化系数: {loss_normalization_factor:.6f}")
                else:
                    # 旧版模型没有Loss归一化系数，默认为1.0
                    self.ae_system['loss_normalization_factor'] = 1.0
                    self.ae_log(f"⚠️ 模型文件不包含Loss归一化系数（旧版模型），默认设为1.0")

                # ✅ 恢复所有GUI配置选项（让用户能看到模型的完整配置）
                self.ae_log(f"📋 恢复GUI配置选项...")

                # 1. 预处理选项
                self.ae_normalize.set(normalize)
                self.ae_db_transform.set(db_transform)
                self.ae_system['data_adapter'].normalize = normalize
                self.ae_system['data_adapter'].db_transform = db_transform
                self.ae_log(f"  数据预处理: 标准化={normalize}, dB变换={db_transform}")

                # 2. 模型配置选项
                self.ae_freq_config.set(freq_config)
                self.ae_latent_dim.set(str(latent_dim))
                self.ae_dropout_rate.set(str(dropout_rate))
                self.ae_wavelet_type.set(wavelet_type)
                self.ae_log(f"  模型配置: 频率={freq_config}, 隐空间={latent_dim}, Dropout={dropout_rate}, 小波={wavelet_type}")

                # 3. 架构和激活函数（需要大小写转换）
                # 架构类型转换：cnn → CNN, mlp → MLP, enhanced_cnn → Enhanced_CNN等
                architecture_display = architecture.replace('_', ' ').title().replace(' ', '_')
                if architecture_display.lower() == 'cnn':
                    architecture_display = 'CNN'
                elif architecture_display.lower() == 'mlp':
                    architecture_display = 'MLP'
                elif architecture_display.lower() == 'enhanced_cnn':
                    architecture_display = 'Enhanced_CNN'
                elif architecture_display.lower() == 'deep_cnn':
                    architecture_display = 'Deep_CNN'

                self.ae_architecture_type.set(architecture_display)
                self.ae_activation.set(activation)
                self.ae_log(f"  架构类型: {architecture_display}, 激活函数: {activation}")

                # 恢复Mapper配置选项到GUI界面
                if mapper_config:
                    mapper_act = mapper_config.get('activation', 'relu')
                    mapper_adaptive = mapper_config.get('use_adaptive', False)

                    if hasattr(self, 'ae_mapper_activation'):
                        self.ae_mapper_activation.set(mapper_act)
                    if hasattr(self, 'ae_mapper_use_adaptive'):
                        self.ae_mapper_use_adaptive.set(mapper_adaptive)

                    self.ae_log(f"  Mapper GUI选项已更新: 激活={mapper_act}, 自适应={mapper_adaptive}")

                # 4. 模式选项（ae_mode由ae_extension初始化到main_gui）
                if hasattr(self, 'ae_mode'):
                    self.ae_mode.set(mode)
                    self.ae_log(f"  运行模式: {mode}")
                else:
                    self.ae_log(f"  ⚠️ ae_mode未定义（ae_extension可能未加载），跳过模式恢复")

                self.ae_log(f"✅ GUI配置选项已全部恢复")

                # ⚠️ 后处理选项控制逻辑
                # 根据模型训练时是否使用dB变换来决定是否启用后处理分贝转换
                self.ae_log(f"🔍 调试: hasattr(ae_extension)={hasattr(self, 'ae_extension')}, ae_extension={getattr(self, 'ae_extension', None)}")

                # 尝试从所有可能的位置获取 ae_extension
                ae_extension = None
                if hasattr(self, 'ae_extension') and self.ae_extension is not None:
                    ae_extension = self.ae_extension
                    self.ae_log(f"🔍 调试: 从self找到ae_extension")
                else:
                    # 尝试从全局查找（如果main()中设置失败，但对象还在）
                    import sys
                    main_module = sys.modules.get('__main__')
                    if main_module and hasattr(main_module, 'app') and hasattr(main_module.app, 'ae_extension'):
                        ae_extension = main_module.app.ae_extension
                        self.ae_extension = ae_extension  # 补救措施：设置到self
                        self.ae_log(f"🔍 调试: 从main找到ae_extension并设置到self")

                if ae_extension is not None:
                    self.ae_log(f"🔍 调试: 找到ae_extension，准备配置后处理选项")
                    try:
                        postprocess_checkbox = ae_extension.postprocess_abs_db_checkbox
                        postprocess_help_label = ae_extension.postprocess_help_label

                        if db_transform:
                            # 模型训练时已使用dB变换，禁用后处理选项
                            postprocess_checkbox.config(state='disabled')
                            self.ae_postprocess_abs_db.set(False)
                            postprocess_help_label.config(
                                text="   • 模型已使用分贝训练，数据已在对数空间，无需再次转换",
                                foreground="orange"
                            )
                            self.ae_log(f"🔒 后处理分贝转换: 已禁用（模型训练时已使用dB变换）")
                        else:
                            # 模型训练时未使用dB变换，启用后处理选项
                            postprocess_checkbox.config(state='normal')
                            self.ae_postprocess_abs_db.set(False)  # 默认不勾选
                            postprocess_help_label.config(
                                text="   • 仅在模型线性训练时可用，用于消除负值影响",
                                foreground="gray"
                            )
                            self.ae_log(f"🔓 后处理分贝转换: 已启用（模型线性训练，可选择后处理转分贝）")
                    except Exception as e:
                        self.ae_log(f"⚠️ 调试: 配置后处理选项时出错: {e}")
                else:
                    self.ae_log(f"⚠️ 调试: 未找到ae_extension，无法配置后处理选项")

                # 如果有数据，也加载到系统中
                if hasattr(self, 'rcs_data') and self.rcs_data is not None:
                    self.ae_system['rcs_data'] = self.rcs_data
                if hasattr(self, 'param_data') and self.param_data is not None:
                    # 修复键名错误：应为 'param_data'，否则重建函数在three_stage模式下取数失败
                    self.ae_system['param_data'] = self.param_data

                # 设置训练历史（来自load_ae_model_from_file的返回值）
                if training_history:
                    self.ae_training_history = training_history

                    # ✅ 打印训练历史中的最佳loss信息和数据集划分
                    self.ae_log("📊 训练历史信息:")
                    stage_histories = training_history.get('stage_histories', {})

                    # 首先显示数据集划分（从Stage 1获取，因为所有阶段使用相同的划分）
                    if 'stage1' in stage_histories:
                        stage1 = stage_histories['stage1']
                        train_indices = stage1.get('train_indices', [])
                        val_indices = stage1.get('val_indices', [])

                        # 如果旧模型没保存indices，但有数据，可以重新计算（种子固定，结果相同）
                        if (not train_indices or not val_indices) and hasattr(self, 'rcs_data') and self.rcs_data is not None:
                            self.ae_log(f"💡 模型未保存数据集划分信息（旧版本模型）")
                            self.ae_log(f"   正在根据固定种子(42)重新计算...")

                            import torch
                            from torch.utils.data import TensorDataset, random_split

                            # 使用相同的逻辑重新计算
                            total_samples = len(self.rcs_data)
                            train_size = int(total_samples * 0.8)
                            val_size = total_samples - train_size

                            # 创建临时dataset用于获取indices
                            temp_tensor = torch.zeros(total_samples, 1)  # 占位tensor
                            temp_dataset = TensorDataset(temp_tensor)
                            generator = torch.Generator().manual_seed(42)
                            train_dataset, val_dataset = random_split(temp_dataset, [train_size, val_size], generator=generator)

                            train_indices = list(train_dataset.indices)
                            val_indices = list(val_dataset.indices)
                            self.ae_log(f"✅ 重新计算完成（基于当前{total_samples}个样本）")

                        if train_indices and val_indices:
                            self.ae_log(f"📋 数据集划分:")
                            self.ae_log(f"  训练集: {len(train_indices)} 样本 - {sorted(train_indices)[:20]}{'...' if len(train_indices) > 20 else ''}")
                            self.ae_log(f"  验证集: {len(val_indices)} 样本 - {sorted(val_indices)[:20]}{'...' if len(val_indices) > 20 else ''}")
                            if len(train_indices) > 20 or len(val_indices) > 20:
                                self.ae_log(f"  (仅显示前20个标号)")
                        else:
                            self.ae_log(f"⚠️ 无法显示数据集划分（模型未保存且未加载数据）")
                    if 'stage1' in stage_histories:
                        stage1 = stage_histories['stage1']
                        best_loss = stage1.get('best_val_loss', 'N/A')
                        best_epoch = stage1.get('best_epoch', 'N/A')
                        if isinstance(best_loss, float):
                            self.ae_log(f"  Stage 1: 最佳Loss={best_loss:.6f} @ Epoch {best_epoch}")
                        else:
                            self.ae_log(f"  Stage 1: 最佳Loss={best_loss} @ Epoch {best_epoch}")

                    if 'stage2' in stage_histories:
                        stage2 = stage_histories['stage2']
                        best_loss = stage2.get('best_val_loss', 'N/A')
                        best_epoch = stage2.get('best_epoch', 'N/A')
                        if isinstance(best_loss, float):
                            self.ae_log(f"  Stage 2: 最佳Loss={best_loss:.6f} @ Epoch {best_epoch}")
                        else:
                            self.ae_log(f"  Stage 2: 最佳Loss={best_loss} @ Epoch {best_epoch}")

                    if 'stage3' in stage_histories:
                        stage3 = stage_histories['stage3']
                        best_loss = stage3.get('best_val_loss', 'N/A')
                        best_epoch = stage3.get('best_epoch', 'N/A')
                        if isinstance(best_loss, float):
                            self.ae_log(f"  Stage 3: 最佳Loss={best_loss:.6f} @ Epoch {best_epoch}")
                        else:
                            self.ae_log(f"  Stage 3: 最佳Loss={best_loss} @ Epoch {best_epoch}")

                    if not stage_histories:
                        self.ae_log("  暂无训练历史")

                    # ✅ 恢复训练配置到GUI（用于继续训练）
                    if 'training_config' in training_history:
                        saved_config = training_history['training_config']
                        self.ae_log(f"📋 恢复训练配置:")

                        # 恢复学习率参数
                        if 'learning_rate' in saved_config:
                            self.ae_learning_rate.set(saved_config['learning_rate'])
                            self.ae_log(f"  学习率: {saved_config['learning_rate']}")

                        if 'min_lr' in saved_config:
                            self.ae_min_lr.set(saved_config['min_lr'])
                            self.ae_log(f"  最小LR: {saved_config['min_lr']}")

                        if 'lr_scheduler' in saved_config:
                            self.ae_lr_scheduler.set(saved_config['lr_scheduler'])
                            self.ae_log(f"  LR调度器: {saved_config['lr_scheduler']}")

                        # 恢复其他训练参数
                        if 'batch_size' in saved_config:
                            self.ae_batch_size.set(saved_config['batch_size'])

                        if 'restart_period' in saved_config:
                            self.ae_restart_period.set(saved_config['restart_period'])

                        # 恢复patience参数
                        if 'patience' in saved_config:
                            patience_dict = saved_config['patience']
                            if 'stage1' in patience_dict:
                                self.ae_patience_stage1.set(patience_dict['stage1'])
                            if 'stage2' in patience_dict:
                                self.ae_patience_stage2.set(patience_dict['stage2'])
                            if 'stage3' in patience_dict:
                                self.ae_patience_stage3.set(patience_dict['stage3'])

                        # 恢复epochs参数
                        if 'epochs' in saved_config:
                            epochs_dict = saved_config['epochs']
                            if 'stage1' in epochs_dict:
                                self.ae_epochs_stage1.set(epochs_dict['stage1'])
                            if 'stage2' in epochs_dict:
                                self.ae_epochs_stage2.set(epochs_dict['stage2'])
                            if 'stage3' in epochs_dict:
                                self.ae_epochs_stage3.set(epochs_dict['stage3'])

                        self.ae_log(f"✅ 训练配置已恢复，可修改学习率后继续训练")
                    else:
                        self.ae_log(f"⚠️ 模型未包含训练配置（可能是旧版模型）")
                else:
                    self.ae_training_history = None

                # 识别训练模式
                training_mode = checkpoint.get('training_mode', 'three_stage')  # 默认为三阶段
                training_mode_display = {
                    'stage1_only': 'Stage 1 Only (仅重建)',
                    'three_stage': '完整三阶段',
                    'joint_training': '联合训练'
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
                    self.ae_log(f"  💡 可使用'继续训练'完成Stage 2/3，或'开始训练'选择其他模式")
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

                # 更新状态栏显示网络参数
                self._update_status_bar_with_model_info()

                # ✅ 保存加载的权重副本（用于"继续训练"功能）
                from copy import deepcopy
                self.ae_loaded_weights = {
                    'autoencoder': deepcopy(self.ae_system['autoencoder'].state_dict()),
                    'parameter_mapper': deepcopy(self.ae_system['parameter_mapper'].state_dict())
                }
                self.ae_model_loaded = True
                self.ae_log("✅ 已保存模型权重副本，可使用'继续训练'功能")

                # 📊 打印隐空间统计信息（如果有数据）
                if hasattr(self, 'rcs_data') and self.rcs_data is not None:
                    self._print_latent_space_statistics(self.rcs_data)

                # 启用"继续训练"按钮
                if hasattr(self, 'ae_extension') and self.ae_extension:
                    self.ae_extension.continue_training_btn.config(state='normal')
                    self.ae_log("✅ '继续训练'按钮已启用")

                messagebox.showinfo("成功",
                    f"模型已加载并自动重建系统!\n\n"
                    f"文件: {filename}\n"
                    f"模式: {mode}\n"
                    f"架构: {architecture}\n"
                    f"频率: {freq_config} ({model_num_freq}频)\n\n"
                    f"💡 提示:\n"
                    f"• 点击'开始训练'：重新初始化权重训练\n"
                    f"• 点击'继续训练'：从加载的权重继续训练")

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

    def _create_stage_loss_function(self, training_config, stage='stage1'):
        """创建AutoEncoder损失函数 (通用阶段损失函数创建)"""
        return self.training_manager._create_stage_loss_function(training_config, stage=stage)

    def _ae_step_scheduler(self, scheduler, scheduler_type, val_loss=None):
        """AutoEncoder学习率调度器步进 (复用项目调度逻辑)"""
        return self.training_manager._ae_step_scheduler(scheduler, scheduler_type, val_loss=val_loss)

    def _ae_log_training_progress(self, epoch, total_epochs, train_loss, val_loss, lr, stage_name):
        """AutoEncoder训练进度日志 (统一格式)"""
        return self.training_manager._ae_log_training_progress(epoch, total_epochs, train_loss, val_loss, lr, stage_name)

    def _plot_autoencoder_visualization(self, chart_type):
        """绘制AutoEncoder特定可视化图表 (已迁移)"""
        pass

    def _plot_ae_latent_space(self):
        """绘制AutoEncoder隐空间分析 (已迁移)"""
        pass

    def _plot_ae_reconstruction_quality(self):
        """绘制AutoEncoder重建质量分析 - 使用统一重建函数 (已迁移)"""
        pass

    def _plot_ae_parameter_mapping(self):
        """绘制AutoEncoder参数映射分析 (已迁移)"""
        pass

    def _plot_ae_training_progress_vis(self):
        """绘制AutoEncoder训练进度可视化 (已迁移)"""
        pass

    def _plot_autoencoder_prediction_visualization(self, chart_type, freq):
        """使用AutoEncoder进行预测可视化 (已迁移)"""
        pass

    def _plot_ae_2d_heatmap(self, freq):
        """绘制AutoEncoder预测的2D热图 - 支持模型未加载时显示原始数据 (已迁移)"""
        pass

    def _plot_original_rcs_fallback(self, freq):
        """当AutoEncoder模型未加载时，显示原始RCS数据作为替代 (已迁移)"""
        pass

    def _plot_ae_comparison(self):
        """绘制AutoEncoder对比图：原图、重构图、残差图 - 使用统一重建函数 (已迁移)"""
        pass

    def _plot_attention_weights(self):
        """绘制通道注意力权重历史折线图 (已迁移)"""
        pass

    def _plot_wavelet_coefficients_comparison(self):
        """绘制小波系数对比图：原始vs重建的4个通道（LL, LH, HL, HH） (已迁移)"""
        pass

    def save_current_visualization(self):
        """保存当前显示的可视化图表到results文件夹 (已迁移)"""
        pass

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

    # 初始化AutoEncoder扩展（如果可用）
    try:
        from gui_managers.extensions.gui_autoencoder_extension import AutoEncoderExtension
        app.ae_extension = AutoEncoderExtension(app)
        app.ae_extension.extend_autoencoder_tab()
        print("✓ AutoEncoder扩展已加载")
    except ImportError:
        print("⚠ AutoEncoderExtension未找到")
    except Exception as e:
        print(f"⚠ AutoEncoder扩展初始化失败: {str(e)}")

    # 初始化批量实验扩展（如果可用）
    try:
        from gui_managers.extensions.gui_batch_experiment_extension import BatchExperimentExtension
        batch_extension = BatchExperimentExtension(app)
        batch_extension.extend_batch_experiment_tab()
        print("✓ 批量实验扩展已加载")
    except ImportError:
        print("⚠ BatchExperimentExtension未找到")
    except Exception as e:
        print(f"⚠ 批量实验扩展初始化失败: {str(e)}")

    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()
