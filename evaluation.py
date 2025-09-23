"""
RCS预测模型评估模块

提供全面的模型评估功能:
1. 双频RCS预测精度评估
2. 物理一致性检查
3. 频率交叉验证
4. 可视化对比
5. 参数敏感性分析
6. 统计指标计算

评估指标:
- RMSE (Root Mean Square Error)
- 相关系数 (Correlation Coefficient)
- 物理一致性得分
- 频率比例一致性
- 角度域高频保持度

作者: RCS Wavelet Network Project
版本: 1.0
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import json
import pandas as pd
from datetime import datetime
import warnings

# 导入项目模块
import rcs_visual as rv
from wavelet_network import TriDimensionalRCSNet
from training import RCSDataset

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class RCSEvaluator:
    """
    RCS模型评估器

    提供全面的模型性能评估功能
    """

    def __init__(self, model: TriDimensionalRCSNet, device: str = 'cuda',
                 use_log_output: bool = False, preprocessing_stats: Dict = None):
        """
        初始化评估器

        参数:
            model: 训练好的RCS预测模型
            device: 计算设备
            use_log_output: 模型是否输出对数域数据
            preprocessing_stats: 预处理统计信息 (mean, std用于反标准化)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # 对数域相关配置
        self.use_log_output = use_log_output
        self.preprocessing_stats = preprocessing_stats or {}

        # 评估结果存储
        self.evaluation_results = {}

    def evaluate_dataset(self, test_dataset: RCSDataset) -> Dict:
        """
        评估整个测试数据集

        参数:
            test_dataset: 测试数据集

        返回:
            评估结果字典
        """
        print("开始数据集评估...")

        # 获取预测结果
        predictions, targets, parameters = self._get_predictions(test_dataset)

        # 计算各项评估指标
        results = {}

        # 1. 基础回归指标
        results['regression_metrics'] = self._calculate_regression_metrics(predictions, targets)

        # 2. 双频独立评估
        results['frequency_metrics'] = self._evaluate_frequencies(predictions, targets)

        # 3. 物理一致性评估
        results['physics_consistency'] = self._evaluate_physics_consistency(predictions, targets)

        # 4. 频率交叉验证
        results['frequency_cross_validation'] = self._frequency_cross_validation(predictions, targets)

        # 5. 角度域分析
        results['angular_analysis'] = self._angular_domain_analysis(predictions, targets)

        # 6. 参数敏感性分析
        results['parameter_sensitivity'] = self._parameter_sensitivity_analysis(
            parameters, predictions, targets
        )

        self.evaluation_results = results

        print("数据集评估完成")
        return results

    def _get_predictions(self, dataset: RCSDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取模型预测结果

        返回:
            (predictions, targets, parameters): 预测值、真实值、参数
        """
        predictions_list = []
        targets_list = []
        parameters_list = []

        with torch.no_grad():
            for i in range(len(dataset)):
                params, target = dataset[i]
                params = params.unsqueeze(0).to(self.device)

                prediction = self.model(params)

                predictions_list.append(prediction.cpu().numpy())
                targets_list.append(target.numpy())
                parameters_list.append(dataset.parameters[i].numpy())

        predictions = np.concatenate(predictions_list, axis=0)
        targets = np.array(targets_list)
        parameters = np.array(parameters_list)

        return predictions, targets, parameters

    def _calculate_regression_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        计算基础回归评估指标

        如果use_log_output=True, 会先转回线性域再计算指标
        """
        # 如果模型输出是对数域(已标准化的dB值), 需要转回线性域
        if self.use_log_output and self.preprocessing_stats:
            # 反标准化: x = x_std * std + mean (恢复到dB域)
            mean = self.preprocessing_stats.get('mean', 0)
            std = self.preprocessing_stats.get('std', 1)

            pred_db = predictions * std + mean
            target_db = targets * std + mean

            # dB转线性值: rcs = 10^(dB/10)
            pred_linear = np.power(10, pred_db / 10)
            target_linear = np.power(10, target_db / 10)

            predictions = pred_linear
            targets = target_linear

        # 展平数据以计算总体指标
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)

        # 移除NaN值
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        metrics = {
            'rmse': np.sqrt(mean_squared_error(target_valid, pred_valid)),
            'mae': np.mean(np.abs(target_valid - pred_valid)),
            'r2': r2_score(target_valid, pred_valid),
            'correlation': stats.pearsonr(target_valid, pred_valid)[0],
            'relative_error': np.mean(np.abs((target_valid - pred_valid) / (target_valid + 1e-8))),
            'valid_samples': len(pred_valid),
            'total_samples': len(pred_flat)
        }

        return metrics

    def _evaluate_frequencies(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        评估每个频率的独立性能
        """
        freq_metrics = {}

        # 频率标签
        freq_names = ['1.5GHz', '3GHz']

        for freq_idx, freq_name in enumerate(freq_names):
            pred_freq = predictions[:, :, :, freq_idx]
            target_freq = targets[:, :, :, freq_idx]

            # 展平并移除NaN
            pred_flat = pred_freq.reshape(-1)
            target_flat = target_freq.reshape(-1)

            valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
            pred_valid = pred_flat[valid_mask]
            target_valid = target_flat[valid_mask]

            freq_metrics[freq_name] = {
                'rmse': np.sqrt(mean_squared_error(target_valid, pred_valid)),
                'correlation': stats.pearsonr(target_valid, pred_valid)[0],
                'r2': r2_score(target_valid, pred_valid),
                'mean_prediction': np.mean(pred_valid),
                'mean_target': np.mean(target_valid),
                'std_prediction': np.std(pred_valid),
                'std_target': np.std(target_valid)
            }

        return freq_metrics

    def _evaluate_physics_consistency(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        评估物理一致性约束

        检查φ=0°平面对称性: σ(φ,θ,f) = σ(-φ,θ,f)
        """
        symmetry_scores = []
        center_phi = 45  # φ=0°对应的索引

        for sample_idx in range(predictions.shape[0]):
            for freq_idx in range(predictions.shape[3]):
                sample_pred = predictions[sample_idx, :, :, freq_idx]

                # 计算对称性误差
                symmetry_error = 0.0
                count = 0

                for i in range(1, center_phi + 1):
                    left_idx = center_phi - i
                    right_idx = center_phi + i

                    if right_idx < 91:
                        left_values = sample_pred[left_idx, :]
                        right_values = sample_pred[right_idx, :]

                        # 计算对称误差
                        diff = np.abs(left_values - right_values)
                        symmetry_error += np.mean(diff)
                        count += 1

                if count > 0:
                    symmetry_scores.append(symmetry_error / count)

        physics_metrics = {
            'symmetry_error_mean': np.mean(symmetry_scores),
            'symmetry_error_std': np.std(symmetry_scores),
            'symmetry_score': 1.0 / (1.0 + np.mean(symmetry_scores)),  # 归一化得分
            'samples_evaluated': len(symmetry_scores)
        }

        return physics_metrics

    def _frequency_cross_validation(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        频率交叉验证

        使用一个频率预测另一个频率的合理性
        """
        # 简化的频率关系模型 (基于散射理论)
        freq_ratio = 3.0 / 1.5  # 频率比

        # 使用1.5GHz预测3GHz
        pred_15g = predictions[:, :, :, 0]
        pred_3g = predictions[:, :, :, 1]
        target_3g = targets[:, :, :, 1]

        # 简单的频率关系模型
        predicted_3g_from_15g = pred_15g * np.log(freq_ratio)

        # 计算交叉验证误差
        cross_val_error = np.mean(np.abs(predicted_3g_from_15g - target_3g))
        direct_pred_error = np.mean(np.abs(pred_3g - target_3g))

        cross_val_metrics = {
            'cross_frequency_error': cross_val_error,
            'direct_prediction_error': direct_pred_error,
            'frequency_consistency_ratio': cross_val_error / direct_pred_error,
            'frequency_correlation': stats.pearsonr(
                pred_15g.reshape(-1), pred_3g.reshape(-1)
            )[0]
        }

        return cross_val_metrics

    def _angular_domain_analysis(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        角度域分析

        评估不同角度区域的预测精度
        """
        angular_metrics = {}

        # 定义角度区域
        regions = {
            'forward_sector': (40, 50, 0, 91),     # 前向扇区 (θ=90°±5°)
            'side_sector': (0, 20, 0, 91),         # 侧向扇区 (θ<60°)
            'rear_sector': (70, 91, 0, 91),        # 后向扇区 (θ>120°)
            'center_phi': (0, 91, 40, 50)          # 中心φ区域 (φ=0°±5°)
        }

        for region_name, (theta_start, theta_end, phi_start, phi_end) in regions.items():
            # 提取区域数据
            pred_region = predictions[:, phi_start:phi_end, theta_start:theta_end, :]
            target_region = targets[:, phi_start:phi_end, theta_start:theta_end, :]

            # 计算区域指标
            pred_flat = pred_region.reshape(-1)
            target_flat = target_region.reshape(-1)

            valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
            if np.sum(valid_mask) > 0:
                pred_valid = pred_flat[valid_mask]
                target_valid = target_flat[valid_mask]

                angular_metrics[region_name] = {
                    'rmse': np.sqrt(mean_squared_error(target_valid, pred_valid)),
                    'correlation': stats.pearsonr(target_valid, pred_valid)[0],
                    'relative_error': np.mean(np.abs((target_valid - pred_valid) / (target_valid + 1e-8)))
                }

        return angular_metrics

    def _parameter_sensitivity_analysis(self, parameters: np.ndarray,
                                      predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        参数敏感性分析

        分析不同飞行器参数对RCS预测的影响
        """
        sensitivity_metrics = {}

        # 计算每个参数与预测误差的相关性
        pred_errors = np.mean(np.abs(predictions - targets), axis=(1, 2, 3))

        for param_idx in range(parameters.shape[1]):
            param_values = parameters[:, param_idx]

            # 计算参数与误差的相关性
            correlation, p_value = stats.pearsonr(param_values, pred_errors)

            sensitivity_metrics[f'parameter_{param_idx+1}'] = {
                'error_correlation': correlation,
                'p_value': p_value,
                'param_range': (np.min(param_values), np.max(param_values)),
                'param_std': np.std(param_values)
            }

        return sensitivity_metrics

    def visualize_predictions(self, test_dataset: RCSDataset, sample_indices: List[int] = None,
                            save_path: str = None):
        """
        可视化预测结果

        参数:
            test_dataset: 测试数据集
            sample_indices: 要可视化的样本索引
            save_path: 保存路径
        """
        if sample_indices is None:
            sample_indices = [0, 1, 2]  # 默认显示前3个样本

        predictions, targets, _ = self._get_predictions(test_dataset)

        for idx in sample_indices:
            if idx >= len(predictions):
                continue

            self._plot_sample_comparison(predictions[idx], targets[idx], idx, save_path)

    def _plot_sample_comparison(self, prediction: np.ndarray, target: np.ndarray,
                               sample_idx: int, save_path: str = None):
        """
        绘制单个样本的预测对比
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        freq_names = ['1.5GHz', '3GHz']

        # 定义角度范围 (基于实际数据)
        phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
        theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°
        extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

        for freq_idx, freq_name in enumerate(freq_names):
            # 真实值
            im1 = axes[freq_idx, 0].imshow(target[:, :, freq_idx], cmap='jet', aspect='equal', extent=extent)
            axes[freq_idx, 0].set_title(f'真实RCS - {freq_name}')
            axes[freq_idx, 0].set_xlabel('φ (方位角, 度)')
            axes[freq_idx, 0].set_ylabel('θ (俯仰角, 度)')
            plt.colorbar(im1, ax=axes[freq_idx, 0])

            # 预测值
            im2 = axes[freq_idx, 1].imshow(prediction[:, :, freq_idx], cmap='jet', aspect='equal', extent=extent)
            axes[freq_idx, 1].set_title(f'预测RCS - {freq_name}')
            axes[freq_idx, 1].set_xlabel('φ (方位角, 度)')
            axes[freq_idx, 1].set_ylabel('θ (俯仰角, 度)')
            plt.colorbar(im2, ax=axes[freq_idx, 1])

            # 误差图
            error = np.abs(prediction[:, :, freq_idx] - target[:, :, freq_idx])
            im3 = axes[freq_idx, 2].imshow(error, cmap='Reds', aspect='equal', extent=extent)
            axes[freq_idx, 2].set_title(f'绝对误差 - {freq_name}')
            axes[freq_idx, 2].set_xlabel('φ (方位角, 度)')
            axes[freq_idx, 2].set_ylabel('θ (俯仰角, 度)')
            plt.colorbar(im3, ax=axes[freq_idx, 2])

            # 散点图对比
            pred_flat = prediction[:, :, freq_idx].reshape(-1)
            target_flat = target[:, :, freq_idx].reshape(-1)

            valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
            pred_valid = pred_flat[valid_mask]
            target_valid = target_flat[valid_mask]

            axes[freq_idx, 3].scatter(target_valid, pred_valid, alpha=0.5, s=1)
            axes[freq_idx, 3].plot([target_valid.min(), target_valid.max()],
                                 [target_valid.min(), target_valid.max()], 'r--')
            axes[freq_idx, 3].set_title(f'预测vs真实 - {freq_name}')
            axes[freq_idx, 3].set_xlabel('真实RCS')
            axes[freq_idx, 3].set_ylabel('预测RCS')

            # 计算相关系数
            corr = stats.pearsonr(target_valid, pred_valid)[0]
            axes[freq_idx, 3].text(0.05, 0.95, f'R = {corr:.3f}',
                                  transform=axes[freq_idx, 3].transAxes,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(f'样本 {sample_idx} 预测结果对比', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/sample_{sample_idx}_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

    def generate_evaluation_report(self, save_path: str = None) -> str:
        """
        生成评估报告

        参数:
            save_path: 报告保存路径

        返回:
            报告内容字符串
        """
        if not self.evaluation_results:
            raise ValueError("请先运行 evaluate_dataset() 方法")

        report = self._create_detailed_report()

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report

    def _create_detailed_report(self) -> str:
        """
        创建详细的评估报告
        """
        results = self.evaluation_results

        report = f"""
# RCS小波网络评估报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 总体回归性能

- **RMSE**: {results['regression_metrics']['rmse']:.4f}
- **MAE**: {results['regression_metrics']['mae']:.4f}
- **R²**: {results['regression_metrics']['r2']:.4f}
- **相关系数**: {results['regression_metrics']['correlation']:.4f}
- **相对误差**: {results['regression_metrics']['relative_error']:.4f}
- **有效样本数**: {results['regression_metrics']['valid_samples']}/{results['regression_metrics']['total_samples']}

## 2. 双频独立性能

### 1.5GHz频率:
- RMSE: {results['frequency_metrics']['1.5GHz']['rmse']:.4f}
- 相关系数: {results['frequency_metrics']['1.5GHz']['correlation']:.4f}
- R²: {results['frequency_metrics']['1.5GHz']['r2']:.4f}

### 3GHz频率:
- RMSE: {results['frequency_metrics']['3GHz']['rmse']:.4f}
- 相关系数: {results['frequency_metrics']['3GHz']['correlation']:.4f}
- R²: {results['frequency_metrics']['3GHz']['r2']:.4f}

## 3. 物理一致性评估

- **对称性误差均值**: {results['physics_consistency']['symmetry_error_mean']:.4f}
- **对称性得分**: {results['physics_consistency']['symmetry_score']:.4f}
- **评估样本数**: {results['physics_consistency']['samples_evaluated']}

## 4. 频率交叉验证

- **频率一致性比率**: {results['frequency_cross_validation']['frequency_consistency_ratio']:.4f}
- **频率间相关性**: {results['frequency_cross_validation']['frequency_correlation']:.4f}

## 5. 角度域分析

"""

        # 添加角度域分析结果
        for region, metrics in results['angular_analysis'].items():
            report += f"### {region}:\n"
            report += f"- RMSE: {metrics['rmse']:.4f}\n"
            report += f"- 相关系数: {metrics['correlation']:.4f}\n"
            report += f"- 相对误差: {metrics['relative_error']:.4f}\n\n"

        report += "## 6. 参数敏感性分析\n\n"

        # 添加参数敏感性分析结果
        for param, metrics in results['parameter_sensitivity'].items():
            report += f"### {param}:\n"
            report += f"- 误差相关性: {metrics['error_correlation']:.4f}\n"
            report += f"- p值: {metrics['p_value']:.4f}\n"
            report += f"- 参数范围: [{metrics['param_range'][0]:.3f}, {metrics['param_range'][1]:.3f}]\n\n"

        return report


def load_model_for_evaluation(model_path: str, model_params: Dict, device: str = 'cuda') -> TriDimensionalRCSNet:
    """
    加载训练好的模型用于评估

    参数:
        model_path: 模型权重文件路径
        model_params: 模型参数
        device: 计算设备

    返回:
        加载的模型
    """
    from wavelet_network import create_model

    model = create_model(**model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_model_with_visualizations(model_path: str, test_dataset: RCSDataset,
                                     model_params: Dict, output_dir: str = 'evaluation_results'):
    """
    完整的模型评估流程，包含可视化

    参数:
        model_path: 模型权重文件路径
        test_dataset: 测试数据集
        model_params: 模型参数
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_for_evaluation(model_path, model_params, device)

    # 创建评估器
    evaluator = RCSEvaluator(model, device)

    # 执行评估
    print("开始模型评估...")
    results = evaluator.evaluate_dataset(test_dataset)

    # 生成报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    report = evaluator.generate_evaluation_report(report_path)
    print(f"评估报告已保存到: {report_path}")

    # 可视化预测结果
    vis_path = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_path, exist_ok=True)
    evaluator.visualize_predictions(test_dataset, sample_indices=[0, 1, 2, 3, 4], save_path=vis_path)

    # 保存结果为JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为可序列化类型
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"评估结果已保存到: {results_path}")

    return results


def convert_to_serializable(obj):
    """
    转换对象为JSON可序列化格式
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


if __name__ == "__main__":
    # 评估模块测试
    print("RCS评估模块测试")

    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建示例数据进行测试
    from wavelet_network import create_model

    model = create_model()
    evaluator = RCSEvaluator(model, device)

    print("评估模块初始化完成")
    print(f"模型信息: {model.get_model_info()}")

    # 创建测试数据
    test_params = np.random.randn(10, 9)
    test_rcs = np.random.randn(10, 91, 91, 2)
    test_dataset = RCSDataset(test_params, test_rcs)

    print("测试数据创建完成")
    print("评估模块准备就绪")