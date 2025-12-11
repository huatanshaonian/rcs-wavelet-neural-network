import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from evaluation import RCSEvaluator

class EvaluationTab(ttk.Frame):
    """
    模型评估标签页
    负责模型性能评估、报告生成和结果导出。
    """

    def __init__(self, notebook, app):
        """
        初始化模型评估标签页。

        参数:
            notebook: 父容器 (ttk.Notebook)
            app: 主应用程序实例 (RCSWaveletGUI)，用于访问共享状态和配置。
        """
        super().__init__(notebook)
        self.app = app

        self.create_widgets()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self)
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
        ttk.Button(control_frame, text="详细报告", command=self.save_detailed_evaluation_report).pack(side=tk.LEFT, padx=5)

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
        
        # 将eval_tree暴露给app，以便EvaluationManager使用
        self.app.eval_tree = self.eval_tree

    def start_evaluation(self):
        """开始评估（支持AutoEncoder和传统网络）"""
        # 检查是否有训练好的模型（传统网络或AutoEncoder）
        has_traditional_model = self.app.model_trained and self.app.current_model is not None
        has_ae_model = hasattr(self.app, 'ae_system') and self.app.ae_system is not None

        if not has_traditional_model and not has_ae_model:
            messagebox.showwarning("警告", "请先训练或加载模型（传统网络或AutoEncoder）")
            return

        if not self.app.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            # 根据模型类型选择评估路径
            if has_ae_model:
                self.app.log_message("🔬 开始AutoEncoder模型评估...")
                self._evaluate_autoencoder_model()
            else:
                self.app.log_message("🔬 开始传统网络模型评估...")
                self._evaluate_traditional_model()

            messagebox.showinfo("成功", "模型评估完成")

        except Exception as e:
            messagebox.showerror("错误", f"评估失败: {str(e)}")

    def _evaluate_traditional_model(self):
        """评估传统网络模型"""
        return self.app.evaluation_manager._evaluate_traditional_model(tree=self.eval_tree)

    def _evaluate_autoencoder_model(self):
        """评估AutoEncoder模型 - 使用统一重建函数"""
        return self.app.evaluation_manager._evaluate_autoencoder_model(tree=self.eval_tree)

    def save_detailed_evaluation_report(self):
        """保存详细评估报告（AutoEncoder专用）"""
        return self.app.evaluation_manager.save_detailed_evaluation_report()

    def generate_report(self):
        """生成评估报告"""
        if not self.app.evaluation_results:
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
                evaluator = RCSEvaluator(self.app.current_model)
                evaluator.evaluation_results = self.app.evaluation_results
                report = evaluator.generate_evaluation_report(filename)
                messagebox.showinfo("成功", f"评估报告已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"报告生成失败: {str(e)}")

    def export_results(self):
        """导出评估结果"""
        if not self.app.evaluation_results:
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
                    json.dump(self.app.evaluation_results, f, indent=2, ensure_ascii=False, default=str)
                messagebox.showinfo("成功", f"评估结果已导出到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"结果导出失败: {str(e)}")
