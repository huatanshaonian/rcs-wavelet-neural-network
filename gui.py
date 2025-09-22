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

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinter.font as tkFont
import os
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import json
from datetime import datetime
import sys

# 导入项目模块
try:
    import rcs_data_reader as rdr
    import rcs_visual as rv
    from wavelet_network import create_model, create_loss_function
    from training import (CrossValidationTrainer, RCSDataLoader,
                         create_training_config, create_data_config, RCSDataset)
    from evaluation import RCSEvaluator, evaluate_model_with_visualizations
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有模块文件都在当前目录下")


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

        # 配置变量
        self.data_config = create_data_config()
        self.training_config = create_training_config()
        self.model_params = {'input_dim': 9, 'hidden_dims': [128, 256]}

        # 初始化界面
        self.create_widgets()
        self.setup_layout()

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_widgets(self):
        """创建界面组件"""

        # 创建主笔记本组件
        self.notebook = ttk.Notebook(self.root)

        # 标签页1: 数据管理
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="数据管理")
        self.create_data_tab()

        # 标签页2: 模型训练
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="模型训练")
        self.create_training_tab()

        # 标签页3: 模型评估
        self.evaluation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_frame, text="模型评估")
        self.create_evaluation_tab()

        # 标签页4: RCS预测
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="RCS预测")
        self.create_prediction_tab()

        # 标签页5: 可视化
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

        # 操作按钮
        button_frame = ttk.Frame(config_group)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="加载数据", command=self.load_data,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="数据预览", command=self.preview_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="数据统计", command=self.show_data_stats).pack(side=tk.LEFT, padx=5)

        # 数据信息显示
        info_group = ttk.LabelFrame(main_frame, text="数据信息")
        info_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.data_info_text = scrolledtext.ScrolledText(info_group, height=15)
        self.data_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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

        ttk.Label(left_config, text="学习率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.StringVar(value=str(self.training_config['learning_rate']))
        ttk.Entry(left_config, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(left_config, text="训练轮数:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value=str(self.training_config['epochs']))
        ttk.Entry(left_config, textvariable=self.epochs_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # 右侧配置
        right_config = ttk.Frame(config_frame)
        right_config.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(right_config, text="权重衰减:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.weight_decay_var = tk.StringVar(value=str(self.training_config['weight_decay']))
        ttk.Entry(right_config, textvariable=self.weight_decay_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(right_config, text="早停耐心:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.patience_var = tk.StringVar(value=str(self.training_config['early_stopping_patience']))
        ttk.Entry(right_config, textvariable=self.patience_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # 训练选项
        options_frame = ttk.Frame(config_group)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        self.use_cross_validation = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="使用交叉验证", variable=self.use_cross_validation).pack(side=tk.LEFT)

        self.save_checkpoints = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="保存检查点", variable=self.save_checkpoints).pack(side=tk.LEFT, padx=20)

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
                                 values=["1.5G", "3G"], state="readonly", width=8)
        freq_combo.grid(row=0, column=3, padx=5, pady=2)

        # 可视化类型选择
        ttk.Label(control_frame, text="图表类型:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.vis_type_var = tk.StringVar(value="2D热图")
        type_combo = ttk.Combobox(control_frame, textvariable=self.vis_type_var,
                                 values=["2D热图", "3D表面图", "球坐标图", "对比图"], state="readonly", width=12)
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

            # 加载数据
            data_loader = RCSDataLoader(self.data_config)
            self.param_data, self.rcs_data = data_loader.load_data()

            self.data_loaded = True
            self.log_message("数据加载成功！")
            self.log_message(f"参数数据形状: {self.param_data.shape}")
            self.log_message(f"RCS数据形状: {self.rcs_data.shape}")

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
        preview_text += f"第一个样本的统计信息:\n"
        first_sample = self.rcs_data[0]
        preview_text += f"1.5GHz - 范围: [{np.min(first_sample[:,:,0]):.4f}, {np.max(first_sample[:,:,0]):.4f}]\n"
        preview_text += f"3GHz - 范围: [{np.min(first_sample[:,:,1]):.4f}, {np.max(first_sample[:,:,1]):.4f}]\n"

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

        stats_text += "\nRCS数据统计:\n"
        for freq_idx, freq_name in enumerate(['1.5GHz', '3GHz']):
            freq_data = self.rcs_data[:, :, :, freq_idx]
            stats_text += f"{freq_name}: 均值={np.mean(freq_data):.6f}, "
            stats_text += f"标准差={np.std(freq_data):.6f}, "
            stats_text += f"范围=[{np.min(freq_data):.6f}, {np.max(freq_data):.6f}]\n"

        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(tk.END, stats_text)

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
            self.training_config['epochs'] = int(self.epochs_var.get())
            self.training_config['weight_decay'] = float(self.weight_decay_var.get())
            self.training_config['early_stopping_patience'] = int(self.patience_var.get())
        except ValueError as e:
            messagebox.showerror("错误", f"配置参数格式错误: {str(e)}")
            return

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
        try:
            self.log_message("开始训练...")

            # 创建数据集
            dataset = RCSDataset(self.param_data, self.rcs_data, augment=True)

            if self.use_cross_validation.get():
                # 交叉验证训练
                trainer = CrossValidationTrainer(
                    self.model_params,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

                results = trainer.cross_validate(dataset, self.training_config)
                self.log_message(f"交叉验证完成，平均得分: {results['mean_score']:.4f}")

                # 加载最佳模型
                best_fold = results['best_fold']
                self.current_model = create_model(**self.model_params)
                self.current_model.load_state_dict(
                    torch.load(f'checkpoints/best_model_fold_{best_fold}.pth')
                )

            else:
                # 简单训练
                self.log_message("简单训练模式暂未实现")

            self.model_trained = True
            self.log_message("训练完成！")

        except Exception as e:
            self.log_message(f"训练失败: {str(e)}")

        finally:
            # 重新启用按钮
            self.root.after(0, self._training_finished)

    def _training_finished(self):
        """训练完成后的UI更新"""
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("训练完成" if self.model_trained else "训练失败")

    def stop_training(self):
        """停止训练"""
        # 这里可以实现训练停止逻辑
        self.log_message("训练停止请求已发送")

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
                torch.save(self.current_model.state_dict(), filename)
                self.log_message(f"模型已保存到: {filename}")
                messagebox.showinfo("成功", "模型保存成功")
            except Exception as e:
                messagebox.showerror("错误", f"模型保存失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        filename = filedialog.askopenfilename(
            title="加载模型",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.current_model = create_model(**self.model_params)
                self.current_model.load_state_dict(torch.load(filename, map_location='cpu'))
                self.model_trained = True
                self.log_message(f"模型已从 {filename} 加载")
                messagebox.showinfo("成功", "模型加载成功")
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    # ======= 评估功能 =======

    def start_evaluation(self):
        """开始评估"""
        if not self.model_trained or self.current_model is None:
            messagebox.showwarning("警告", "请先训练或加载模型")
            return

        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            # 创建测试数据集
            test_dataset = RCSDataset(self.param_data[-20:], self.rcs_data[-20:], augment=False)

            # 创建评估器
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            evaluator = RCSEvaluator(self.current_model, device)

            # 执行评估
            self.evaluation_results = evaluator.evaluate_dataset(test_dataset)

            # 更新评估结果显示
            self._update_evaluation_display()

            messagebox.showinfo("成功", "模型评估完成")

        except Exception as e:
            messagebox.showerror("错误", f"评估失败: {str(e)}")

    def _update_evaluation_display(self):
        """更新评估结果显示"""
        # 清空现有内容
        for item in self.eval_tree.get_children():
            self.eval_tree.delete(item)

        results = self.evaluation_results

        # 添加回归指标
        reg_node = self.eval_tree.insert("", "end", text="回归指标")
        metrics = results['regression_metrics']
        self.eval_tree.insert(reg_node, "end", values=("RMSE", "", "", f"{metrics['rmse']:.4f}"))
        self.eval_tree.insert(reg_node, "end", values=("R²", "", "", f"{metrics['r2']:.4f}"))
        self.eval_tree.insert(reg_node, "end", values=("相关系数", "", "", f"{metrics['correlation']:.4f}"))

        # 添加频率指标
        freq_node = self.eval_tree.insert("", "end", text="频率指标")
        freq_metrics = results['frequency_metrics']
        for metric in ['rmse', 'correlation', 'r2']:
            self.eval_tree.insert(freq_node, "end",
                                values=(metric.upper(),
                                       f"{freq_metrics['1.5GHz'][metric]:.4f}",
                                       f"{freq_metrics['3GHz'][metric]:.4f}", ""))

        # 添加物理一致性
        phys_node = self.eval_tree.insert("", "end", text="物理一致性")
        phys_metrics = results['physics_consistency']
        self.eval_tree.insert(phys_node, "end",
                            values=("对称性得分", "", "", f"{phys_metrics['symmetry_score']:.4f}"))

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

        # 绘制1.5GHz结果
        im1 = ax1.imshow(prediction[:, :, 0], cmap='jet', aspect='auto')
        ax1.set_title('1.5GHz RCS预测')
        ax1.set_xlabel('θ (俯仰角)')
        ax1.set_ylabel('φ (偏航角)')
        self.pred_fig.colorbar(im1, ax=ax1)

        # 绘制3GHz结果
        im2 = ax2.imshow(prediction[:, :, 1], cmap='jet', aspect='auto')
        ax2.set_title('3GHz RCS预测')
        ax2.set_xlabel('θ (俯仰角)')
        ax2.set_ylabel('φ (偏航角)')
        self.pred_fig.colorbar(im2, ax=ax2)

        self.pred_fig.tight_layout()
        self.pred_canvas.draw()

    # ======= 可视化功能 =======

    def generate_visualization(self):
        """生成可视化图表"""
        try:
            model_id = self.vis_model_var.get()
            freq = self.vis_freq_var.get()
            chart_type = self.vis_type_var.get()

            if chart_type == "2D热图":
                self._plot_2d_heatmap(model_id, freq)
            elif chart_type == "3D表面图":
                self._plot_3d_surface(model_id, freq)
            elif chart_type == "球坐标图":
                self._plot_spherical(model_id, freq)
            elif chart_type == "对比图":
                self._plot_comparison(model_id)

        except Exception as e:
            messagebox.showerror("错误", f"图表生成失败: {str(e)}")

    def _plot_2d_heatmap(self, model_id, freq):
        """绘制2D热图"""
        self.vis_fig.clear()

        try:
            # 使用现有的可视化函数
            data = rv.get_rcs_matrix(model_id, freq, self.data_config['rcs_data_dir'])

            ax = self.vis_fig.add_subplot(1, 1, 1)
            im = ax.imshow(data['rcs_db'], cmap='jet', aspect='auto')
            ax.set_title(f'模型 {model_id} - {freq} RCS分布')
            ax.set_xlabel('θ (俯仰角)')
            ax.set_ylabel('φ (偏航角)')
            self.vis_fig.colorbar(im, ax=ax, label='RCS (dB)')

            self.vis_fig.tight_layout()
            self.vis_canvas.draw()

        except Exception as e:
            self.log_message(f"无法生成2D热图: {str(e)}")

    def _plot_3d_surface(self, model_id, freq):
        """绘制3D表面图"""
        # 实现3D表面图
        self.log_message("3D表面图功能待实现")

    def _plot_spherical(self, model_id, freq):
        """绘制球坐标图"""
        # 实现球坐标图
        self.log_message("球坐标图功能待实现")

    def _plot_comparison(self, model_id):
        """绘制对比图"""
        # 实现对比图
        self.log_message("对比图功能待实现")

    # ======= 辅助功能 =======

    def log_message(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        if hasattr(self, 'training_log'):
            self.training_log.insert(tk.END, log_entry)
            self.training_log.see(tk.END)

        if hasattr(self, 'data_info_text'):
            self.data_info_text.insert(tk.END, log_entry)
            self.data_info_text.see(tk.END)

        print(log_entry.strip())  # 同时输出到控制台


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