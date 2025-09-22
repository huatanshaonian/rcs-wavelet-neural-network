"""
RCS小波神经网络快速测试

验证所有核心功能是否正常工作
"""

import torch
import numpy as np
import os
from datetime import datetime

def test_network():
    """测试网络架构"""
    print("=" * 50)
    print("测试网络架构")
    print("=" * 50)

    from wavelet_network import create_model

    model = create_model()
    model_info = model.get_model_info()

    print(f"模型类型: {model_info['architecture']}")
    print(f"总参数数: {model_info['total_parameters']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"输入形状: {model_info['input_shape']}")
    print(f"输出形状: {model_info['output_shape']}")

    # 测试前向传播
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    test_input = torch.randn(4, 9).to(device)
    with torch.no_grad():
        output = model(test_input)

    print(f"前向传播测试成功!")
    print(f"输入: {test_input.shape}")
    print(f"输出: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print()

def test_loss_function():
    """测试损失函数"""
    print("=" * 50)
    print("测试损失函数")
    print("=" * 50)

    from wavelet_network import create_loss_function

    loss_fn = create_loss_function()

    # 创建测试数据
    pred = torch.randn(2, 91, 91, 2)
    target = torch.randn(2, 91, 91, 2)

    losses = loss_fn(pred, target)

    print("损失函数测试成功!")
    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")
    print()

def test_training_modules():
    """测试训练相关模块"""
    print("=" * 50)
    print("测试训练模块")
    print("=" * 50)

    from training import RCSDataset, create_training_config, create_data_config

    # 测试数据集
    params = np.random.randn(10, 9)
    rcs_data = np.random.randn(10, 91, 91, 2)
    dataset = RCSDataset(params, rcs_data)

    print(f"数据集创建成功! 长度: {len(dataset)}")

    sample_params, sample_rcs = dataset[0]
    print(f"样本参数形状: {sample_params.shape}")
    print(f"样本RCS形状: {sample_rcs.shape}")

    # 测试配置
    train_config = create_training_config()
    data_config = create_data_config()

    print("配置创建成功!")
    print(f"训练配置: {train_config}")
    print()

def test_evaluation():
    """测试评估模块"""
    print("=" * 50)
    print("测试评估模块")
    print("=" * 50)

    from evaluation import RCSEvaluator
    from wavelet_network import create_model
    from training import RCSDataset

    # 创建测试数据
    model = create_model()
    params = np.random.randn(5, 9)
    rcs_data = np.random.randn(5, 91, 91, 2)
    dataset = RCSDataset(params, rcs_data)

    evaluator = RCSEvaluator(model)

    print("评估器创建成功!")
    print("注意: 完整评估需要在训练后进行")
    print()

def test_data_compatibility():
    """测试数据兼容性"""
    print("=" * 50)
    print("测试数据兼容性")
    print("=" * 50)

    import rcs_data_reader as rdr
    import rcs_visual as rv

    # 测试现有模块导入
    print("现有模块导入成功!")

    # 尝试加载配置中的数据路径
    from training import create_data_config
    config = create_data_config()

    params_file = config['params_file']
    rcs_dir = config['rcs_data_dir']

    print(f"参数文件路径: {params_file}")
    print(f"RCS数据目录: {rcs_dir}")

    if os.path.exists(params_file):
        print("参数文件存在 - 可以进行真实数据训练!")
    else:
        print("参数文件不存在 - 请检查路径或使用模拟数据")

    if os.path.exists(rcs_dir):
        print("RCS数据目录存在 - 可以进行真实数据训练!")
    else:
        print("RCS数据目录不存在 - 请检查路径或使用模拟数据")
    print()

def test_model_prediction():
    """测试模型预测流程"""
    print("=" * 50)
    print("测试模型预测流程")
    print("=" * 50)

    from wavelet_network import create_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model().to(device)
    model.eval()

    # 模拟飞行器参数
    aircraft_params = np.array([
        [1.2, 0.8, 2.1, 1.5, 0.9, 1.8, 2.3, 1.1, 0.7],
        [0.9, 1.1, 1.8, 1.2, 1.0, 2.0, 2.1, 0.9, 0.8],
    ])

    print("输入飞行器参数:")
    for i, params in enumerate(aircraft_params):
        print(f"飞行器 {i+1}: {params}")

    # 执行预测
    with torch.no_grad():
        params_tensor = torch.tensor(aircraft_params, dtype=torch.float32).to(device)
        predictions = model(params_tensor)
        predictions = predictions.cpu().numpy()

    print(f"\n预测成功! 输出形状: {predictions.shape}")

    # 分析预测结果
    for i in range(len(aircraft_params)):
        pred = predictions[i]
        print(f"\n飞行器 {i+1} 预测结果:")
        print(f"1.5GHz - 最大值: {pred[:,:,0].max():.4f}, 平均值: {pred[:,:,0].mean():.4f}")
        print(f"3GHz   - 最大值: {pred[:,:,1].max():.4f}, 平均值: {pred[:,:,1].mean():.4f}")
    print()

def main():
    """主测试函数"""
    print("RCS小波神经网络系统 - 功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
    print()

    try:
        # 依次测试各个模块
        test_network()
        test_loss_function()
        test_training_modules()
        test_evaluation()
        test_data_compatibility()
        test_model_prediction()

        print("=" * 50)
        print("所有测试通过!")
        print("=" * 50)
        print("\n后续步骤:")
        print("1. 准备真实RCS数据")
        print("2. 运行训练: python main.py --mode train")
        print("3. 评估模型: python main.py --mode evaluate --model-path models/best.pth")
        print("4. 进行预测: python main.py --mode predict --model-path models/best.pth --params '1,2,3,4,5,6,7,8,9'")
        print("\n如果tkinter可用，可以运行GUI:")
        print("python main.py --mode gui")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()