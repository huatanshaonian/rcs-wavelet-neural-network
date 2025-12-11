import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class PredictionTab(ttk.Frame):
    """
    RCS预测标签页
    负责飞行器参数输入和RCS预测及结果展示。
    """

    def __init__(self, notebook, app):
        """
        初始化预测标签页。

        参数:
            notebook: 父容器 (ttk.Notebook)
            app: 主应用程序实例 (RCSWaveletGUI)，用于访问共享状态和配置。
        """
        super().__init__(notebook)
        self.app = app

        # 引用主应用的字体
        self.font_small = getattr(app, 'font_small', None)
        self.font_medium = getattr(app, 'font_medium', None)

        self.param_vars = []
        self.pred_model_type_var = tk.StringVar(value="传统网络")
        
        # 初始化Matplotlib图形
        self.pred_fig = Figure(figsize=(12, 6), dpi=80)

        self.create_widgets()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 参数输入组
        input_group = ttk.LabelFrame(main_frame, text="飞行器参数输入")
        input_group.pack(fill=tk.X, pady=(0, 10))

        # 创建参数输入网格
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

        # 模型选择
        ttk.Label(control_frame, text="使用模型:").pack(side=tk.LEFT, padx=(20, 5))
        model_combo = ttk.Combobox(control_frame, textvariable=self.pred_model_type_var,
                                   values=["传统网络", "AutoEncoder"], state="readonly", width=12)
        model_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="执行预测", command=self.make_prediction,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        # 预测结果显示
        result_group = ttk.LabelFrame(main_frame, text="预测结果")
        result_group.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 创建matplotlib图形 (已经在__init__中)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, result_group)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加工具栏
        self.pred_toolbar = NavigationToolbar2Tk(self.pred_canvas, result_group)
        self.pred_toolbar.update()

    def load_param_template(self):
        """加载参数模板"""
        if not self.app.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 使用第一个样本作为模板
        template_params = self.app.param_data[0]
        for i, var in enumerate(self.param_vars):
            var.set(f"{template_params[i]:.6f}")

    def generate_random_params(self):
        """生成随机参数"""
        if not self.app.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 基于已有数据的分布生成随机参数
        for i, var in enumerate(self.param_vars):
            param_col = self.app.param_data[:, i]
            mean = np.mean(param_col)
            std = np.std(param_col)
            random_val = np.random.normal(mean, std)
            var.set(f"{random_val:.6f}")

    def make_prediction(self):
        """执行RCS预测（支持传统网络和AutoEncoder）"""
        # 检查选择的模型类型
        model_type = self.pred_model_type_var.get()

        # 检查模型是否可用
        if model_type == "传统网络":
            if not self.app.model_trained or self.app.current_model is None:
                messagebox.showwarning("警告", "请先训练或加载传统神经网络模型")
                return
        elif model_type == "AutoEncoder":
            if not hasattr(self.app, 'ae_system') or self.app.ae_system is None:
                messagebox.showwarning("警告", "请先训练或加载AutoEncoder模型")
                return

        try:
            # 获取输入参数
            params = []
            for var in self.param_vars:
                params.append(float(var.get()))

            params = np.array(params).reshape(1, -1)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            if model_type == "传统网络":
                # 传统网络需要标准化参数
                if hasattr(self.app, 'param_data'):
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.fit(self.app.param_data)
                    params_scaled = scaler.transform(params)
                else:
                    params_scaled = params
                # 使用传统神经网络预测
                self.app.current_model.to(device)
                self.app.current_model.eval()

                with torch.no_grad():
                    params_tensor = torch.tensor(params_scaled, dtype=torch.float32).to(device)
                    prediction = self.app.current_model(params_tensor)
                    prediction = prediction.cpu().numpy()[0]  # [91, 91, 2]

                # 可视化预测结果
                self._plot_prediction_results(prediction, model_type="传统网络")

            elif model_type == "AutoEncoder":
                # 使用AutoEncoder预测
                # ⚠️ 注意：AutoEncoder的ParameterMapper在训练时接收的是**未标准化的原始参数**
                # 因此预测时也必须使用未标准化的参数，与训练时保持一致
                autoencoder = self.app.ae_system['autoencoder']
                parameter_mapper = self.app.ae_system['parameter_mapper']
                wavelet_transform = self.app.ae_system.get('wavelet_transform', None)
                data_adapter = self.app.ae_system.get('data_adapter', None)
                mode = self.app.ae_system.get('mode', 'wavelet')
                config_info = self.app.ae_system.get('config_info', {})
                num_frequencies = config_info.get('num_frequencies', 2)

                autoencoder.to(device).eval()
                parameter_mapper.to(device).eval()

                with torch.no_grad():
                    # Step 1: 参数 → 隐空间（使用未标准化的原始参数）
                    params_tensor = torch.tensor(params, dtype=torch.float32).to(device)
                    predicted_latents = parameter_mapper(params_tensor)

                    # Step 2: 隐空间 → 重建输出
                    predicted_output = autoencoder.decode(predicted_latents)

                    # Step 3: 逆变换到原始RCS空间
                    if mode == 'wavelet':
                        # 小波模式：标准化小波系数 → 逆标准化 → 逆小波变换 → RCS
                        if data_adapter:
                            predicted_coeffs_np = data_adapter.inverse_adapt(predicted_output)
                            predicted_coeffs = torch.FloatTensor(predicted_coeffs_np).to(device)
                        else:
                            predicted_coeffs = predicted_output

                        prediction = wavelet_transform.inverse_transform(predicted_coeffs)
                    else:
                        # 直接模式：标准化RCS → 逆标准化 → RCS
                        if data_adapter:
                            prediction_np = data_adapter.inverse_adapt(predicted_output)
                            prediction = torch.FloatTensor(prediction_np).to(device)
                        else:
                            prediction = predicted_output

                    prediction = prediction.cpu().numpy()[0]  # [91, 91, num_freq]

                # 可视化预测结果
                self._plot_prediction_results(prediction, model_type="AutoEncoder", num_frequencies=num_frequencies)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"预测失败: {str(e)}")

    def _plot_prediction_results(self, prediction, model_type="传统网络", num_frequencies=2):
        """绘制预测结果（支持2freq和3freq）"""
        self.pred_fig.clear()

        # 获取字号缩放因子
        # 尝试从VisualizationTab获取，如果不可用则默认为1.0
        try:
            if hasattr(self.app, 'visualization_tab'):
                fontsize_scale = self.app.visualization_tab.fontsize_scale_var.get()
            else:
                fontsize_scale = 1.0
            fontsize_scale = max(0.5, min(3.0, fontsize_scale))
        except:
            fontsize_scale = 1.0

        # 定义角度范围 (基于实际数据)
        phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
        theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°

        # 定义频率标签
        freq_labels = ['1.5GHz', '3GHz', '6GHz']

        # 根据频率数量创建子图
        if num_frequencies == 2:
            # 2频率：1.5GHz 和 3GHz
            ax1 = self.pred_fig.add_subplot(1, 2, 1)
            ax2 = self.pred_fig.add_subplot(1, 2, 2)
            axes = [ax1, ax2]
        elif num_frequencies == 3:
            # 3频率：1.5GHz, 3GHz 和 6GHz
            ax1 = self.pred_fig.add_subplot(1, 3, 1)
            ax2 = self.pred_fig.add_subplot(1, 3, 2)
            ax3 = self.pred_fig.add_subplot(1, 3, 3)
            axes = [ax1, ax2, ax3]
        else:
            messagebox.showerror("错误", f"不支持的频率数量: {num_frequencies}")
            return

        # 绘制每个频率的结果
        for idx, ax in enumerate(axes):
            # ⚠️ 重要：prediction是线性值，需要转换为dB显示
            # AutoEncoder的inverse_adapt输出是线性域RCS
            prediction_db = 10 * np.log10(np.clip(prediction[:, :, idx], 1e-10, None))

            im = ax.imshow(prediction_db, cmap='jet', aspect='equal',
                          extent=[phi_range[0], phi_range[1], theta_range[1], theta_range[0]])
            ax.set_title(f'{freq_labels[idx]} RCS预测 ({model_type})',
                        fontsize=int(20*fontsize_scale), fontweight='bold')
            ax.set_xlabel('φ (方位角, 度)', fontsize=int(16*fontsize_scale), fontweight='bold')
            ax.set_ylabel('θ (俯仰角, 度)', fontsize=int(16*fontsize_scale), fontweight='bold')
            ax.tick_params(axis='both', labelsize=int(14*fontsize_scale))

            cbar = self.pred_fig.colorbar(im, ax=ax)
            cbar.set_label('RCS (dB)', fontsize=int(16*fontsize_scale), fontweight='bold')
            cbar.ax.tick_params(labelsize=int(14*fontsize_scale))

        self.pred_fig.tight_layout()
        self.pred_canvas.draw()
