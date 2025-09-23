"""
RCS小波神经网络主程序

提供多种运行模式:
1. GUI模式: 图形界面操作
2. 训练模式: 命令行训练
3. 评估模式: 模型评估
4. 预测模式: 单次预测
5. 批处理模式: 批量处理

使用说明:
python main.py --mode gui                    # 启动GUI界面
python main.py --mode train                  # 命令行训练
python main.py --mode evaluate               # 模型评估
python main.py --mode predict                # 单次预测
python main.py --mode batch                  # 批量处理

作者: RCS Wavelet Network Project
版本: 1.0
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
import warnings

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
try:
    import rcs_data_reader as rdr
    import rcs_visual as rv
    from wavelet_network import create_model, create_loss_function, TriDimensionalRCSNet
    from training import (CrossValidationTrainer, RCSDataLoader,
                         create_training_config, create_data_config, RCSDataset)
    from evaluation import RCSEvaluator, evaluate_model_with_visualizations
    from gui import RCSWaveletGUI
except ImportError as e:
    print(f"模块导入错误: {e}")
    print("请确保所有依赖模块都已正确安装")
    sys.exit(1)

warnings.filterwarnings('ignore')


class RCSWaveletApp:
    """
    RCS小波网络应用主类
    """

    def __init__(self):
        """初始化应用"""
        self.config = self.load_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        # 创建必要的目录
        self.create_directories()

    def load_config(self):
        """加载配置文件"""
        config_file = 'config.json'

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"配置文件已加载: {config_file}")
                return config
            except Exception as e:
                print(f"配置文件加载失败: {e}")

        # 使用默认配置
        print("使用默认配置")
        return self.create_default_config()

    def create_default_config(self):
        """创建默认配置"""
        config = {
            "data": {
                "params_file": r"..\parameter\parameters_sorted.csv",
                "rcs_data_dir": r"..\parameter\csv_output",
                "model_ids": [f"{i:03d}" for i in range(1, 101)],
                "frequencies": ["1.5G", "3G"]
            },
            "model": {
                "input_dim": 9,
                "hidden_dims": [128, 256],
                "dropout_rate": 0.2
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 200,
                "early_stopping_patience": 20,
                "use_cross_validation": True,
                "n_folds": 5,
                "loss_weights": {
                    "mse": 1.0,
                    "symmetry": 0.1,
                    "multiscale": 0.2
                }
            },
            "evaluation": {
                "test_split": 0.2,
                "metrics": ["rmse", "r2", "correlation", "physics_consistency"],
                "visualize_samples": 5
            },
            "output": {
                "model_dir": "models",
                "results_dir": "results",
                "logs_dir": "logs",
                "visualizations_dir": "visualizations"
            }
        }

        # 保存默认配置
        self.save_config(config)
        return config

    def save_config(self, config):
        """保存配置到文件"""
        try:
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print("配置文件已保存: config.json")
        except Exception as e:
            print(f"配置文件保存失败: {e}")

    def create_directories(self):
        """创建必要的目录"""
        dirs = [
            self.config['output']['model_dir'],
            self.config['output']['results_dir'],
            self.config['output']['logs_dir'],
            self.config['output']['visualizations_dir'],
            'checkpoints'
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def run_gui(self):
        """运行GUI模式"""
        print("启动GUI界面...")

        try:
            # 彻底修复tkinter环境冲突
            import os
            import sys

            # 1. 完全清理冲突的环境变量
            env_vars_to_clear = [
                'TCL_LIBRARY', 'TK_LIBRARY', 'TCLLIBPATH',
                'TCLLIBPATH', 'TIX_LIBRARY'
            ]
            for var in env_vars_to_clear:
                if var in os.environ:
                    print(f"清理环境变量: {var}")
                    del os.environ[var]

            # 2. 强制使用当前Python环境的Tcl/Tk
            python_dir = os.path.dirname(sys.executable)

            # 设置正确的Tcl/Tk路径
            tcl_lib_paths = [
                os.path.join(python_dir, "tcl", "tcl8.6"),
                os.path.join(python_dir, "lib", "tcl8.6"),
                os.path.join(python_dir, "Library", "lib", "tcl8.6"),
                os.path.join(os.path.dirname(python_dir), "Library", "lib", "tcl8.6")
            ]

            tk_lib_paths = [
                os.path.join(python_dir, "tcl", "tk8.6"),
                os.path.join(python_dir, "lib", "tk8.6"),
                os.path.join(python_dir, "Library", "lib", "tk8.6"),
                os.path.join(os.path.dirname(python_dir), "Library", "lib", "tk8.6")
            ]

            # 设置TCL_LIBRARY
            for tcl_path in tcl_lib_paths:
                if os.path.exists(tcl_path):
                    os.environ['TCL_LIBRARY'] = tcl_path
                    print(f"设置TCL_LIBRARY: {tcl_path}")
                    break

            # 设置TK_LIBRARY
            for tk_path in tk_lib_paths:
                if os.path.exists(tk_path):
                    os.environ['TK_LIBRARY'] = tk_path
                    print(f"设置TK_LIBRARY: {tk_path}")
                    break

            # 3. 导入tkinter
            import tkinter as tk
            print("tkinter导入成功")

            # 4. 创建GUI
            root = tk.Tk()
            app = RCSWaveletGUI(root)
            print("GUI创建成功，启动主循环...")
            root.mainloop()
        except ImportError:
            print("错误: tkinter未安装，无法启动GUI")
            print("建议使用命令行模式：")
            print("  python quick_test.py  # 功能测试")
            print("  python main.py --mode train  # 训练模型")
            return False
        except Exception as e:
            print(f"GUI启动失败: {e}")
            print("建议使用命令行模式：")
            print("  python quick_test.py  # 功能测试")
            print("  python main.py --mode train  # 训练模型")
            return False

        return True

    def run_training(self, args):
        """运行训练模式"""
        print("开始模型训练...")

        try:
            # 加载数据
            print("加载训练数据...")
            data_loader = RCSDataLoader(self.config['data'])
            param_data, rcs_data = data_loader.load_data()

            print(f"数据加载完成: 参数 {param_data.shape}, RCS {rcs_data.shape}")

            # 创建数据集
            dataset = RCSDataset(param_data, rcs_data, augment=True)

            # 配置训练参数
            training_config = self.config['training'].copy()
            if args.epochs:
                training_config['epochs'] = args.epochs
            if args.batch_size:
                training_config['batch_size'] = args.batch_size
            if args.learning_rate:
                training_config['learning_rate'] = args.learning_rate
            if args.patience:
                training_config['early_stopping_patience'] = args.patience

            # 开始训练
            if training_config['use_cross_validation']:
                print("使用交叉验证训练...")
                trainer = CrossValidationTrainer(
                    self.config['model'],
                    device=self.device,
                    n_folds=training_config['n_folds']
                )

                results = trainer.cross_validate(dataset, training_config)

                print(f"交叉验证完成:")
                print(f"平均得分: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
                print(f"最佳fold: {results['best_fold']}")

                # 保存结果
                results_file = os.path.join(
                    self.config['output']['results_dir'],
                    f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str)

                print(f"训练结果已保存到: {results_file}")

            else:
                print("使用简单训练模式...")
                from torch.utils.data import random_split, DataLoader as TorchDataLoader
                import torch.optim as optim

                # 分割数据集
                train_size = int(len(dataset) * 0.8)
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                print(f"数据分割: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

                # 创建数据加载器
                train_loader = TorchDataLoader(train_dataset,
                                             batch_size=training_config['batch_size'],
                                             shuffle=True)
                val_loader = TorchDataLoader(val_dataset,
                                           batch_size=training_config['batch_size'],
                                           shuffle=False)

                # 创建模型和训练器
                from training import ProgressiveTrainer
                model = create_model(**self.config['model'])
                trainer = ProgressiveTrainer(model, self.device)

                # 创建优化器和调度器
                optimizer = optim.Adam(model.parameters(),
                                     lr=training_config['learning_rate'],
                                     weight_decay=training_config['weight_decay'])

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10
                )

                # 创建损失函数
                loss_fn = create_loss_function(loss_weights=training_config.get('loss_weights'))

                # 训练循环
                best_val_loss = float('inf')
                patience_counter = 0

                for epoch in range(training_config['epochs']):
                    # 训练
                    train_losses = trainer.train_epoch(train_loader, optimizer, loss_fn,
                                                     epoch, training_config['epochs'])

                    # 验证
                    val_losses = trainer.validate_epoch(val_loader, loss_fn)

                    # 学习率调度
                    scheduler.step(val_losses['total'])

                    # 记录进度
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch+1}/{training_config['epochs']}: "
                              f"Train Loss: {train_losses['total']:.4f}, "
                              f"Val Loss: {val_losses['total']:.4f}")

                    # 早停检查
                    if val_losses['total'] < best_val_loss:
                        best_val_loss = val_losses['total']
                        patience_counter = 0

                        # 保存最佳模型
                        os.makedirs('checkpoints', exist_ok=True)
                        torch.save(model.state_dict(), 'checkpoints/best_model_simple.pth')
                    else:
                        patience_counter += 1

                    if patience_counter >= training_config['early_stopping_patience']:
                        print(f"早停于epoch {epoch+1}")
                        break

                print(f"简单训练完成！最佳验证损失: {best_val_loss:.4f}")
                print("模型已保存到: checkpoints/best_model_simple.pth")

        except Exception as e:
            print(f"训练失败: {e}")
            return False

        return True

    def run_evaluation(self, args):
        """运行评估模式"""
        print("开始模型评估...")

        if not args.model_path or not os.path.exists(args.model_path):
            print("错误: 请指定有效的模型路径")
            return False

        try:
            # 加载数据
            data_loader = RCSDataLoader(self.config['data'])
            param_data, rcs_data = data_loader.load_data()

            # 创建测试数据集
            test_size = int(len(param_data) * self.config['evaluation']['test_split'])
            test_dataset = RCSDataset(
                param_data[-test_size:],
                rcs_data[-test_size:],
                augment=False
            )

            # 执行评估
            output_dir = os.path.join(
                self.config['output']['results_dir'],
                f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            results = evaluate_model_with_visualizations(
                args.model_path,
                test_dataset,
                self.config['model'],
                output_dir
            )

            print(f"评估完成，结果保存到: {output_dir}")

        except Exception as e:
            print(f"评估失败: {e}")
            return False

        return True

    def run_prediction(self, args):
        """运行预测模式"""
        print("开始RCS预测...")

        if not args.model_path or not os.path.exists(args.model_path):
            print("错误: 请指定有效的模型路径")
            return False

        if not args.params:
            print("错误: 请指定输入参数")
            return False

        try:
            # 加载模型
            model = create_model(**self.config['model'])
            model.load_state_dict(torch.load(args.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            # 处理输入参数
            if isinstance(args.params, str):
                # 从文件加载参数
                if os.path.exists(args.params):
                    params = np.loadtxt(args.params, delimiter=',')
                else:
                    # 解析字符串参数
                    params = np.array([float(x) for x in args.params.split(',')])
            else:
                params = np.array(args.params)

            # 确保参数形状正确
            if params.ndim == 1:
                params = params.reshape(1, -1)

            # 执行预测
            with torch.no_grad():
                params_tensor = torch.tensor(params, dtype=torch.float32).to(self.device)
                prediction = model(params_tensor)
                prediction = prediction.cpu().numpy()

            # 保存预测结果
            output_file = args.output or f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

            np.savez(output_file,
                    prediction=prediction,
                    input_params=params,
                    frequencies=['1.5GHz', '3GHz'])

            print(f"预测完成，结果保存到: {output_file}")
            print(f"预测形状: {prediction.shape}")

            # 生成可视化（如果指定）
            if args.visualize:
                self._visualize_prediction(prediction[0], output_file.replace('.npz', ''))

        except Exception as e:
            print(f"预测失败: {e}")
            return False

        return True

    def run_batch_processing(self, args):
        """运行批处理模式"""
        print("开始批量处理...")

        if not args.input_dir or not os.path.exists(args.input_dir):
            print("错误: 请指定有效的输入目录")
            return False

        try:
            # 查找输入文件
            input_files = []
            for ext in ['*.csv', '*.txt', '*.npz']:
                import glob
                input_files.extend(glob.glob(os.path.join(args.input_dir, ext)))

            print(f"找到 {len(input_files)} 个输入文件")

            # 处理每个文件
            for file_path in input_files:
                print(f"处理文件: {file_path}")
                # 这里可以实现具体的批处理逻辑

            print("批量处理完成")

        except Exception as e:
            print(f"批量处理失败: {e}")
            return False

        return True

    def _visualize_prediction(self, prediction, output_prefix):
        """可视化预测结果"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 定义角度范围 (基于实际数据)
            phi_range = (-45.0, 45.0)  # φ范围: -45° 到 +45°
            theta_range = (45.0, 135.0)  # θ范围: 45° 到 135°
            extent = [phi_range[0], phi_range[1], theta_range[1], theta_range[0]]

            # 1.5GHz
            im1 = axes[0].imshow(prediction[:, :, 0], cmap='jet', aspect='equal', extent=extent)
            axes[0].set_title('1.5GHz RCS预测')
            axes[0].set_xlabel('φ (方位角, 度)')
            axes[0].set_ylabel('θ (俯仰角, 度)')
            plt.colorbar(im1, ax=axes[0])

            # 3GHz
            im2 = axes[1].imshow(prediction[:, :, 1], cmap='jet', aspect='equal', extent=extent)
            axes[1].set_title('3GHz RCS预测')
            axes[1].set_xlabel('φ (方位角, 度)')
            axes[1].set_ylabel('θ (俯仰角, 度)')
            plt.colorbar(im2, ax=axes[1])

            plt.tight_layout()

            vis_file = f"{output_prefix}_visualization.png"
            plt.savefig(vis_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"可视化结果保存到: {vis_file}")

        except Exception as e:
            print(f"可视化失败: {e}")


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="RCS小波神经网络预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode gui                              # 启动GUI界面
  python main.py --mode train --epochs 100              # 训练100轮
  python main.py --mode train --epochs 100 --patience 30 # 训练100轮，早停耐心30轮
  python main.py --mode evaluate --model models/best.pth # 评估模型
  python main.py --mode predict --model models/best.pth --params "1,2,3,4,5,6,7,8,9"
  python main.py --mode batch --input-dir data/         # 批量处理
        """
    )

    parser.add_argument('--mode',
                       choices=['gui', 'train', 'evaluate', 'predict', 'batch'],
                       default='gui',
                       help='运行模式 (默认: gui)')

    # 训练相关参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    parser.add_argument('--patience', type=int, help='早停耐心值，连续多少轮验证损失不改善就停止训练')

    # 评估和预测相关参数
    parser.add_argument('--model-path', help='模型文件路径')
    parser.add_argument('--params', help='输入参数 (逗号分隔或文件路径)')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')

    # 批处理相关参数
    parser.add_argument('--input-dir', help='输入目录路径')

    # 其他参数
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    return parser


def main():
    """主函数"""
    print("=" * 60)
    print("RCS小波神经网络预测系统 v1.0")
    print("作者: RCS Wavelet Network Project")
    print("=" * 60)

    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()

    # 创建应用实例
    app = RCSWaveletApp()

    # 如果指定了配置文件，重新加载配置
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                app.config = json.load(f)
            print(f"使用配置文件: {args.config}")
        except Exception as e:
            print(f"配置文件加载失败: {e}")

    # 根据模式运行相应功能
    success = False

    if args.mode == 'gui':
        success = app.run_gui()

    elif args.mode == 'train':
        success = app.run_training(args)

    elif args.mode == 'evaluate':
        success = app.run_evaluation(args)

    elif args.mode == 'predict':
        success = app.run_prediction(args)

    elif args.mode == 'batch':
        success = app.run_batch_processing(args)

    # 输出结果
    if success:
        print("\n程序执行成功完成")
        sys.exit(0)
    else:
        print("\n程序执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()