"""
批量实验管理器

支持批量对比训练不同配置的AutoEncoder模型，自动保存结果和生成对比图表。

主要功能：
1. 从基准配置生成实验配置列表
2. 批量执行训练
3. 结果追踪和保存
4. 生成对比图表和报告

使用示例：
    manager = BatchExperimentManager(
        base_config=base_config,
        compare_dimensions={'activation': ['relu', 'sin', 'gelu']},
        experiment_name='activation_comparison'
    )
    manager.run_batch_experiments(training_func, rcs_data, param_data)
"""

import os
import json
import time
import csv
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class ExperimentResult:
    """单个实验的结果记录"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_id = config.get('experiment_id', 'unknown')

        # 训练历史
        self.stage1_train_loss = []
        self.stage1_val_loss = []
        self.stage2_train_loss = []
        self.stage2_val_loss = []
        self.stage3_train_loss = []
        self.stage3_val_loss = []

        # 完整的AE训练历史（包含stage_histories，用于绘图）
        self.ae_training_history = None

        # 最终指标
        self.final_metrics = {}

        # 训练时间
        self.training_time = 0.0
        self.start_time = None
        self.end_time = None

        # 模型路径
        self.model_path = ""

        # 状态
        self.status = "pending"  # pending, running, completed, failed
        self.error_message = ""

    def start(self):
        """开始训练"""
        self.status = "running"
        self.start_time = time.time()

    def complete(self, success: bool = True, error: str = ""):
        """完成训练"""
        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time
        if success:
            self.status = "completed"
        else:
            self.status = "failed"
            self.error_message = error

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON保存）"""
        return {
            'experiment_id': self.experiment_id,
            'config': self.config,
            'training_time': self.training_time,
            'final_metrics': self.final_metrics,
            'status': self.status,
            'error_message': self.error_message,
            'model_path': self.model_path
        }


class BatchExperimentManager:
    """批量实验管理器"""

    def __init__(self,
                 base_config: Dict[str, Any],
                 compare_dimensions: Dict[str, List[Any]],
                 experiment_name: str = "batch_experiment",
                 save_dir: str = "batch_experiments"):
        """
        初始化批量实验管理器

        Args:
            base_config: 基准配置（固定参数）
            compare_dimensions: 要对比的维度和值
                例如: {'activation': ['relu', 'sin', 'gelu'],
                      'normalization_method': ['zscore', 'minmax']}
            experiment_name: 实验名称
            save_dir: 保存根目录
        """
        self.base_config = base_config
        self.compare_dimensions = compare_dimensions
        self.experiment_name = experiment_name

        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            save_dir,
            f"{experiment_name}_{timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 创建子目录
        self.models_dir = os.path.join(self.experiment_dir, "models")
        self.plots_dir = os.path.join(self.experiment_dir, "comparison_plots")
        self.visualizations_dir = os.path.join(self.experiment_dir, "visualizations")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

        # 生成实验配置列表
        self.experiment_configs = self._generate_configs()
        self.results: List[ExperimentResult] = []

        # 保存实验总配置
        self._save_experiment_config()

        # 回调函数
        self.on_experiment_start = None    # 回调: (index, total, config)
        self.on_experiment_complete = None  # 回调: (index, total, result)
        self.on_batch_complete = None      # 回调: (results)

        # 日志文件（可选，用于记录批量实验输出）
        self.log_file = None

    def _log(self, message: str):
        """统一的日志输出方法，同时输出到控制台和日志文件"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # 输出到控制台
        print(log_message)

        # 如果日志文件存在，同时写入文件
        if self.log_file is not None:
            try:
                self.log_file.write(log_message + "\n")
                self.log_file.flush()
            except Exception as e:
                print(f"写入日志文件失败: {e}")

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """生成所有实验配置"""
        configs = []

        # 提取维度名称和值
        dimension_names = list(self.compare_dimensions.keys())
        dimension_values = list(self.compare_dimensions.values())

        # 生成笛卡尔积（所有组合）
        for combination in product(*dimension_values):
            config = self.base_config.copy()

            # 构建实验ID
            id_parts = []
            for dim_name, value in zip(dimension_names, combination):
                config[dim_name] = value

                # 处理不同类型的值，生成合法的文件名
                if isinstance(value, bool):
                    value_str = str(value).lower()
                elif isinstance(value, dict):
                    # 字典类型：提取关键信息生成简洁字符串
                    # 例如 {'normalization_method': 'zscore', 'db_transform': False} -> 'zscore_nodb'
                    if 'normalization_method' in value and 'db_transform' in value:
                        norm_method = value['normalization_method']
                        db_flag = 'db' if value['db_transform'] else 'nodb'
                        value_str = f"{norm_method}_{db_flag}"
                    else:
                        # 通用处理：key1-val1_key2-val2
                        parts = [f"{k}-{v}" for k, v in value.items()]
                        value_str = '_'.join(parts)
                else:
                    value_str = str(value)

                id_parts.append(f"{value_str}")

            config['experiment_id'] = '_'.join(id_parts)
            configs.append(config)

        return configs

    def _save_experiment_config(self):
        """保存实验总配置"""
        config_data = {
            'experiment_name': self.experiment_name,
            'created_time': datetime.now().isoformat(),
            'base_config': self.base_config,
            'compare_dimensions': self.compare_dimensions,
            'total_experiments': len(self.experiment_configs),
            'experiment_configs': self.experiment_configs
        }

        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    def run_batch_experiments(self,
                             training_func: Callable,
                             rcs_data: np.ndarray,
                             param_data: np.ndarray,
                             ae_system_creator: Callable,
                             save_models: bool = True,
                             evaluator_func: Optional[Callable] = None,
                             visualizer_func: Optional[Callable] = None,
                             n_samples_visualize: int = 3) -> List[ExperimentResult]:
        """
        执行批量实验

        Args:
            training_func: 训练函数，签名: (rcs_data, param_data, config, ae_system) -> training_history
            rcs_data: RCS数据
            param_data: 参数数据
            ae_system_creator: AutoEncoder系统创建函数，签名: (config) -> ae_system
            save_models: 是否保存每个实验的模型
            evaluator_func: 评估函数，签名: (ae_system, rcs_data, param_data, indices) -> evaluation_results
            visualizer_func: 可视化函数，签名: (ae_system, rcs_data, param_data, indices, save_dir, experiment_id) -> None
            n_samples_visualize: 每个数据集（train/test）可视化的样本数量

        Returns:
            实验结果列表
        """
        total = len(self.experiment_configs)
        self.results = []

        self._log("")
        self._log("="*80)
        self._log(f"开始批量实验: {self.experiment_name}")
        self._log(f"实验目录: {self.experiment_dir}")
        self._log(f"实验数量: {total}")
        self._log("="*80)
        self._log("")

        for idx, config in enumerate(self.experiment_configs, 1):
            result = ExperimentResult(config)
            experiment_id = config['experiment_id']

            self._log("")
            self._log(f"[{idx}/{total}] 开始实验: {experiment_id}")
            self._log(f"配置: {self._format_config_diff(config)}")

            # 回调：实验开始
            if self.on_experiment_start:
                self.on_experiment_start(idx, total, config)

            result.start()

            try:
                # 创建AutoEncoder系统
                ae_system = ae_system_creator(config)

                # 执行训练
                training_history = training_func(
                    rcs_data=rcs_data,
                    param_data=param_data,
                    training_config=config,
                    ae_system=ae_system
                )

                # 记录训练历史
                if training_history:
                    result.stage1_train_loss = training_history.get('stage1_train_loss', [])
                    result.stage1_val_loss = training_history.get('stage1_val_loss', [])
                    result.stage2_train_loss = training_history.get('stage2_train_loss', [])
                    result.stage2_val_loss = training_history.get('stage2_val_loss', [])
                    result.stage3_train_loss = training_history.get('stage3_train_loss', [])
                    result.stage3_val_loss = training_history.get('stage3_val_loss', [])
                    result.final_metrics = training_history.get('final_metrics', {})

                    # 保存完整的ae_training_history（用于绘制训练进度图）
                    result.ae_training_history = training_history

                # 保存模型
                if save_models and ae_system:
                    model_filename = f"{experiment_id}_model.pth"
                    model_path = os.path.join(self.models_dir, model_filename)
                    self._save_model(ae_system, model_path, config)
                    result.model_path = model_path
                    self._log(f"  ✓ 模型已保存: {model_filename}")

                # 评估模型（如果提供了评估函数）
                if evaluator_func and ae_system:
                    self._log(f"  正在评估模型...")
                    try:
                        # 划分训练集和测试集（80/20）
                        n_total = len(rcs_data)
                        n_train = int(n_total * 0.8)
                        train_indices = list(range(n_train))
                        test_indices = list(range(n_train, n_total))

                        # 在训练集和测试集上评估
                        train_eval = evaluator_func(ae_system, rcs_data, param_data, train_indices)
                        test_eval = evaluator_func(ae_system, rcs_data, param_data, test_indices)

                        # 保存评估结果
                        eval_result = {
                            'train': train_eval,
                            'test': test_eval
                        }
                        result.evaluation_results = eval_result

                        eval_path = os.path.join(self.models_dir, f"{experiment_id}_evaluation.json")
                        with open(eval_path, 'w', encoding='utf-8') as f:
                            json.dump(eval_result, f, indent=2, ensure_ascii=False)

                        self._log(f"  ✓ 评估完成: Train MSE={train_eval.get('mse', 'N/A'):.6f}, "
                                  f"Test MSE={test_eval.get('mse', 'N/A'):.6f}")

                    except Exception as e:
                        self._log(f"  ⚠ 评估失败: {str(e)}")

                # 生成可视化（如果提供了可视化函数）
                if visualizer_func and ae_system:
                    self._log(f"  正在生成可视化图表...")
                    try:
                        # 创建实验的可视化目录
                        vis_dir = os.path.join(self.visualizations_dir, experiment_id)
                        os.makedirs(vis_dir, exist_ok=True)

                        # 从训练集和测试集各选择n个样本进行可视化
                        n_total = len(rcs_data)
                        n_train = int(n_total * 0.8)

                        # 随机选择样本
                        import random
                        train_sample_indices = random.sample(range(n_train), min(n_samples_visualize, n_train))
                        test_sample_indices = random.sample(range(n_train, n_total),
                                                           min(n_samples_visualize, n_total - n_train))

                        # 训练集样本可视化
                        visualizer_func(ae_system, rcs_data, param_data,
                                      train_sample_indices, vis_dir, f"{experiment_id}_train")

                        # 测试集样本可视化
                        visualizer_func(ae_system, rcs_data, param_data,
                                      test_sample_indices, vis_dir, f"{experiment_id}_test")

                        self._log(f"  ✓ 可视化完成: {len(train_sample_indices)}个训练样本 + "
                                  f"{len(test_sample_indices)}个测试样本")

                    except Exception as e:
                        self._log(f"  ⚠ 可视化失败: {str(e)}")

                # 生成训练进度图（使用统一绘图函数）
                if training_history and result.ae_training_history:
                    self._log(f"  正在生成训练进度图...")
                    try:
                        from autoencoder.utils.plotting import plot_ae_training_progress

                        # 创建训练日志目录（如果不存在）
                        training_logs_dir = os.path.join(self.base_dir, 'training_logs')
                        os.makedirs(training_logs_dir, exist_ok=True)

                        # 保存训练进度图
                        progress_plot_path = os.path.join(training_logs_dir, f"{experiment_id}_training_progress.png")
                        plot_ae_training_progress(
                            ae_training_history=result.ae_training_history,
                            fontsize_scale=1.0,
                            save_path=progress_plot_path,
                            use_log_scale=True,
                            show_best_epoch=True
                        )

                        self._log(f"  ✓ 训练进度图已保存: {experiment_id}_training_progress.png")

                    except Exception as e:
                        self._log(f"  ⚠ 训练进度图生成失败: {str(e)}")

                result.complete(success=True)
                self._log(f"✓ 实验完成，耗时: {result.training_time:.1f}秒")

                # 显示关键指标
                if result.final_metrics:
                    self._log(f"  训练指标: {result.final_metrics}")
                if hasattr(result, 'evaluation_results') and result.evaluation_results:
                    test_metrics = result.evaluation_results.get('test', {})
                    self._log(f"  测试指标: {test_metrics}")

            except Exception as e:
                result.complete(success=False, error=str(e))
                self._log(f"✗ 实验失败: {str(e)}")

            self.results.append(result)

            # 回调：实验完成
            if self.on_experiment_complete:
                self.on_experiment_complete(idx, total, result)

            # 保存中间结果
            self._save_results_summary()

        self._log("")
        self._log("="*80)
        self._log(f"批量实验完成！")
        self._log(f"成功: {sum(1 for r in self.results if r.status == 'completed')}/{total}")
        self._log(f"失败: {sum(1 for r in self.results if r.status == 'failed')}/{total}")
        self._log("="*80)
        self._log("")

        # 生成对比图表
        self.generate_comparison_plots()

        # 保存最终结果
        self._save_results_summary()
        self._save_detailed_results()

        # 回调：批量完成
        if self.on_batch_complete:
            self.on_batch_complete(self.results)

        return self.results

    def _format_config_diff(self, config: Dict[str, Any]) -> str:
        """格式化配置差异（只显示变化的参数）"""
        diff_parts = []
        for key in self.compare_dimensions.keys():
            value = config.get(key, 'N/A')
            diff_parts.append(f"{key}={value}")
        return ", ".join(diff_parts)

    def _save_model(self, ae_system: Dict, model_path: str, config: Dict):
        """保存模型和配置（使用统一保存函数）"""
        # ===== 使用统一的保存函数（autoencoder/utils/model_io.py）=====
        from autoencoder.utils.model_io import save_ae_model_to_file

        # 获取训练模式
        training_mode = config.get('training_mode', 'three_stage')

        # 调用统一保存函数
        # 注意：批量实验不保存详细训练历史（training_history=None）
        save_ae_model_to_file(
            ae_system=ae_system,
            model_path=model_path,
            training_history=None,  # 批量实验不保存详细训练历史
            training_mode=training_mode,
            save_json_config=True,
            config_override=config  # 直接传递实验配置
        )
        # ===== 统一保存函数调用结束 =====

    def _save_results_summary(self):
        """保存结果汇总表（CSV格式）"""
        csv_path = os.path.join(self.experiment_dir, "results_summary.csv")

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 表头
            header = ['experiment_id', 'status', 'training_time']

            # 添加变化维度列
            for dim in self.compare_dimensions.keys():
                header.append(dim)

            # 添加指标列
            if self.results and self.results[0].final_metrics:
                metric_keys = list(self.results[0].final_metrics.keys())
                header.extend(metric_keys)

            header.append('error_message')
            writer.writerow(header)

            # 数据行
            for result in self.results:
                row = [
                    result.experiment_id,
                    result.status,
                    f"{result.training_time:.2f}"
                ]

                # 变化维度值
                for dim in self.compare_dimensions.keys():
                    row.append(result.config.get(dim, 'N/A'))

                # 指标值
                if result.final_metrics:
                    for key in metric_keys:
                        value = result.final_metrics.get(key, 'N/A')
                        if isinstance(value, float):
                            row.append(f"{value:.6f}")
                        else:
                            row.append(str(value))

                row.append(result.error_message)
                writer.writerow(row)

        print(f"✓ 结果汇总已保存: results_summary.csv")

    def _save_detailed_results(self):
        """保存详细结果（JSON格式）"""
        results_data = {
            'experiment_name': self.experiment_name,
            'experiment_dir': self.experiment_dir,
            'total_experiments': len(self.results),
            'successful_experiments': sum(1 for r in self.results if r.status == 'completed'),
            'failed_experiments': sum(1 for r in self.results if r.status == 'failed'),
            'results': [r.to_dict() for r in self.results]
        }

        json_path = os.path.join(self.experiment_dir, "detailed_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"✓ 详细结果已保存: detailed_results.json")

    def generate_comparison_plots(self):
        """生成对比图表"""
        if not self.results:
            print("没有结果可供绘图")
            return

        # 只绘制成功的实验
        successful_results = [r for r in self.results if r.status == 'completed']
        if not successful_results:
            print("没有成功的实验可供绘图")
            return

        print("\n生成对比图表...")

        # 1. 训练曲线对比
        self._plot_loss_curves(successful_results)

        # 2. 指标柱状图对比
        self._plot_metrics_bar(successful_results)

        # 3. 综合性能雷达图
        if len(successful_results) <= 8:  # 雷达图适合少量对比
            self._plot_radar_chart(successful_results)

        # 4. 误差分布箱线图（新增）
        self._plot_error_distribution_box(successful_results)

        # 5. 收敛速度对比（新增）
        self._plot_convergence_comparison(successful_results)

        # 6. 训练时间对比（新增）
        self._plot_training_time_comparison(successful_results)

        print("✓ 对比图表已生成")

    def _plot_loss_curves(self, results: List[ExperimentResult]):
        """绘制训练曲线对比"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        stages = [
            ('stage1', 'Stage 1: AutoEncoder预训练'),
            ('stage2', 'Stage 2: 参数映射器训练'),
            ('stage3', 'Stage 3: 端到端微调')
        ]

        for idx, (stage_name, stage_title) in enumerate(stages):
            # 训练损失
            ax_train = fig.add_subplot(gs[idx, 0])
            ax_train.set_title(f"{stage_title} - Train Loss")
            ax_train.set_xlabel("Epoch")
            ax_train.set_ylabel("Loss")
            ax_train.grid(True, alpha=0.3)

            # 验证损失
            ax_val = fig.add_subplot(gs[idx, 1])
            ax_val.set_title(f"{stage_title} - Val Loss")
            ax_val.set_xlabel("Epoch")
            ax_val.set_ylabel("Loss")
            ax_val.grid(True, alpha=0.3)

            for result in results:
                train_loss = getattr(result, f'{stage_name}_train_loss', [])
                val_loss = getattr(result, f'{stage_name}_val_loss', [])

                if train_loss:
                    ax_train.plot(train_loss, label=result.experiment_id, alpha=0.8)
                if val_loss:
                    ax_val.plot(val_loss, label=result.experiment_id, alpha=0.8)

            ax_train.legend(fontsize=8)
            ax_val.legend(fontsize=8)

        plt.savefig(os.path.join(self.plots_dir, "loss_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_metrics_bar(self, results: List[ExperimentResult]):
        """绘制指标柱状图"""
        if not results[0].final_metrics:
            return

        metric_names = list(results[0].final_metrics.keys())
        experiment_ids = [r.experiment_id for r in results]

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            values = [r.final_metrics.get(metric_name, 0) for r in results]

            bars = ax.bar(range(len(values)), values, alpha=0.7)
            ax.set_xlabel("Experiment")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} Comparison")
            ax.set_xticks(range(len(experiment_ids)))
            ax.set_xticklabels(experiment_ids, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            # 标注数值
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "metrics_bar.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_radar_chart(self, results: List[ExperimentResult]):
        """绘制综合性能雷达图"""
        if not results[0].final_metrics:
            return

        metric_names = list(results[0].final_metrics.keys())
        n_metrics = len(metric_names)

        # 归一化指标（越小越好的指标取倒数）
        all_values = []
        for metric_name in metric_names:
            values = [r.final_metrics.get(metric_name, 0) for r in results]
            all_values.append(values)

        # 归一化到[0, 1]
        normalized_values = []
        for values in all_values:
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                norm = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                norm = [1.0] * len(values)
            # 反转（损失越小越好）
            norm = [1 - v for v in norm]
            normalized_values.append(norm)

        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for idx, result in enumerate(results):
            values = [normalized_values[i][idx] for i in range(n_metrics)]
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2, label=result.experiment_id, alpha=0.7)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.set_title("Comprehensive Performance Comparison\n(Normalized, Higher is Better)",
                    fontsize=12, pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "radar_chart.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_error_distribution_box(self, results: List[ExperimentResult]):
        """绘制误差分布箱线图"""
        # 需要有evaluation_results
        results_with_eval = [r for r in results if hasattr(r, 'evaluation_results') and r.evaluation_results]
        if not results_with_eval:
            return

        # 提取测试集MSE
        experiment_ids = [r.experiment_id for r in results_with_eval]
        test_mses = [r.evaluation_results['test'].get('mse', 0) for r in results_with_eval]

        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制箱线图（这里只显示单点，实际应用中如果有多次运行可以显示分布）
        positions = range(len(experiment_ids))
        ax.bar(positions, test_mses, alpha=0.7, color='steelblue')

        ax.set_xlabel("Experiment", fontsize=12)
        ax.set_ylabel("Test MSE", fontsize=12)
        ax.set_title("Test Error Distribution Comparison", fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(experiment_ids, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 标注数值
        for i, (pos, mse) in enumerate(zip(positions, test_mses)):
            ax.text(pos, mse, f'{mse:.6f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "error_distribution_box.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_convergence_comparison(self, results: List[ExperimentResult]):
        """绘制收敛速度对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        stages = [
            ('stage1', 'Stage 1: AutoEncoder'),
            ('stage2', 'Stage 2: Parameter Mapper'),
            ('stage3', 'Stage 3: End-to-End')
        ]

        for idx, (stage_name, stage_title) in enumerate(stages):
            ax = axes[idx]
            ax.set_title(stage_title, fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Validation Loss", fontsize=10)
            ax.grid(True, alpha=0.3)

            for result in results:
                val_loss = getattr(result, f'{stage_name}_val_loss', [])
                if val_loss:
                    ax.plot(val_loss, label=result.experiment_id, alpha=0.8, linewidth=2)

            ax.legend(fontsize=8, loc='upper right')

            # 标注收敛点（假设收敛为最低点）
            for result in results:
                val_loss = getattr(result, f'{stage_name}_val_loss', [])
                if val_loss:
                    min_idx = np.argmin(val_loss)
                    min_val = val_loss[min_idx]
                    ax.plot(min_idx, min_val, 'o', markersize=6)

        plt.suptitle("Convergence Speed Comparison", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "convergence_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_training_time_comparison(self, results: List[ExperimentResult]):
        """绘制训练时间对比"""
        experiment_ids = [r.experiment_id for r in results]
        training_times = [r.training_time / 60 for r in results]  # 转换为分钟

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(range(len(experiment_ids)), training_times, alpha=0.7,
                     color='coral', edgecolor='darkred', linewidth=1.5)

        ax.set_xlabel("Experiment", fontsize=12)
        ax.set_ylabel("Training Time (minutes)", fontsize=12)
        ax.set_title("Training Time Comparison", fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(experiment_ids)))
        ax.set_xticklabels(experiment_ids, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 标注数值
        for i, (bar, time_min) in enumerate(zip(bars, training_times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_min:.1f}m',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 添加平均线
        avg_time = np.mean(training_times)
        ax.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Average: {avg_time:.1f}m')
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "training_time_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def get_best_config(self, metric: str = 'mse', lower_is_better: bool = True) -> Optional[ExperimentResult]:
        """
        获取最佳配置

        Args:
            metric: 评价指标名称
            lower_is_better: True表示指标越小越好，False表示越大越好

        Returns:
            最佳实验结果
        """
        successful_results = [r for r in self.results if r.status == 'completed']
        if not successful_results:
            return None

        valid_results = [r for r in successful_results if metric in r.final_metrics]
        if not valid_results:
            return None

        if lower_is_better:
            best_result = min(valid_results, key=lambda r: r.final_metrics[metric])
        else:
            best_result = max(valid_results, key=lambda r: r.final_metrics[metric])

        return best_result


if __name__ == "__main__":
    """测试批量实验管理器"""
    print("=" * 80)
    print("批量实验管理器测试")
    print("=" * 80)

    # 示例配置
    base_config = {
        'mode': 'wavelet',
        'architecture': 'cnn',
        'latent_dim': 256,
        'epochs': {'stage1': 50, 'stage2': 30, 'stage3': 20},
        'batch_size': 8,
        'learning_rate': 1e-3,
    }

    compare_dimensions = {
        'activation': ['relu', 'sin', 'gelu'],
        'normalization_method': ['zscore', 'minmax']
    }

    # 创建管理器
    manager = BatchExperimentManager(
        base_config=base_config,
        compare_dimensions=compare_dimensions,
        experiment_name='test_experiment',
        save_dir='test_batch_experiments'
    )

    print(f"\n生成的实验配置数量: {len(manager.experiment_configs)}")
    print("\n实验配置列表:")
    for i, config in enumerate(manager.experiment_configs, 1):
        print(f"  {i}. {config['experiment_id']}")
        print(f"     activation={config['activation']}, normalization={config['normalization_method']}")

    print("\n实验目录结构:")
    print(f"  根目录: {manager.experiment_dir}")
    print(f"  模型目录: {manager.models_dir}")
    print(f"  日志目录: {manager.logs_dir}")
    print(f"  图表目录: {manager.plots_dir}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
