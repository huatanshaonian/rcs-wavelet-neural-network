"""
深度诊断：为什么loss小但R²极负
"""
import numpy as np
import torch
from training import RCSDataLoader, RCSDataset
from wavelet_network import create_model, create_loss_function
from evaluation import RCSEvaluator
import glob

print("="*70)
print("深度诊断：loss vs R²矛盾分析")
print("="*70)

# 1. 加载数据
print("\n第1步：加载并检查训练数据")
data_config = {
    'params_file': '../parameter/parameters_sorted.csv',
    'rcs_data_dir': '../parameter/csv_output',
    'model_ids': [f"{i:03d}" for i in range(1, 101)],
    'frequencies': ['1.5G', '3G'],
    'preprocessing': {
        'use_log': True,
        'log_epsilon': 1e-10,
        'normalize_after_log': True
    }
}

data_loader = RCSDataLoader(data_config)
params, rcs_data = data_loader.load_data()
stats = data_loader.preprocessing_stats

print(f"训练数据（标准化RCS）:")
print(f"  形状: {rcs_data.shape}")
print(f"  范围: {rcs_data.min():.3f} ~ {rcs_data.max():.3f}")
print(f"  均值: {rcs_data.mean():.6f}")
print(f"  标准差: {rcs_data.std():.6f}")
print(f"预处理统计: mean={stats['mean']:.2f} dB, std={stats['std']:.2f} dB")

# 2. 加载模型
print("\n第2步：加载模型")
checkpoint_files = glob.glob("checkpoints/best_model_fold_*.pth")
if not checkpoint_files:
    checkpoint_files = glob.glob("models/*.pth")

if not checkpoint_files:
    print("ERROR: 未找到模型")
    exit(1)

checkpoint_path = checkpoint_files[0]
print(f"使用: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model = create_model(input_dim=9, use_log_output=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    saved_stats = checkpoint.get('preprocessing_stats')
    print(f"Checkpoint stats: {saved_stats}")
else:
    print("ERROR: 旧格式checkpoint")
    exit(1)

model.eval()

# 3. 测试训练loss计算
print("\n第3步：测试训练loss计算")
loss_fn = create_loss_function()

# 用一个batch测试
test_batch_size = 8
test_params = torch.FloatTensor(params[:test_batch_size])
test_targets = torch.FloatTensor(rcs_data[:test_batch_size])

with torch.no_grad():
    test_preds = model(test_params)
    losses = loss_fn(test_preds, test_targets)

print(f"训练loss计算（标准化域）:")
print(f"  pred范围: {test_preds.min().item():.3f} ~ {test_preds.max().item():.3f}")
print(f"  target范围: {test_targets.min().item():.3f} ~ {test_targets.max().item():.3f}")
print(f"  MSE loss: {losses['mse'].item():.6f}")
print(f"  Total loss: {losses['total'].item():.6f}")

# 4. 检查评估器的域转换
print("\n第4步：检查评估器的域转换")

evaluator = RCSEvaluator(
    model,
    'cpu',
    use_log_output=True,
    preprocessing_stats=saved_stats
)

# 创建测试数据集
test_dataset = RCSDataset(params[:20], rcs_data[:20], augment=False)

# 评估
eval_results = evaluator.evaluate_dataset(test_dataset)

print(f"评估结果:")
print(f"  RMSE: {eval_results['rmse']:.6f}")
print(f"  R²: {eval_results['r2']:.6f}")
print(f"  Correlation: {eval_results['correlation']:.6f}")

# 5. 手动验证域转换
print("\n第5步：手动验证预测和评估的域转换")

with torch.no_grad():
    pred_std = model(test_params[0:1]).numpy().squeeze()

target_std = rcs_data[0]

print(f"第一个样本（标准化域）:")
print(f"  pred范围: {pred_std.min():.3f} ~ {pred_std.max():.3f}")
print(f"  target范围: {target_std.min():.3f} ~ {target_std.max():.3f}")
print(f"  MSE(标准化): {np.mean((pred_std - target_std)**2):.6f}")

# 反标准化到dB
if saved_stats:
    mean = saved_stats['mean']
    std_val = saved_stats['std']
    pred_db = pred_std * std_val + mean
    target_db = target_std * std_val + mean

    print(f"反标准化到dB域:")
    print(f"  pred (dB): {pred_db.min():.2f} ~ {pred_db.max():.2f}")
    print(f"  target (dB): {target_db.min():.2f} ~ {target_db.max():.2f}")

# 转换到线性域
pred_linear = np.power(10, pred_db / 10)
target_linear = np.power(10, target_db / 10)

print(f"转换到线性域:")
print(f"  pred (线性): {pred_linear.min():.6e} ~ {pred_linear.max():.6e}")
print(f"  target (线性): {target_linear.min():.6e} ~ {target_linear.max():.6e}")

# 在线性域计算RMSE和R²
rmse_linear = np.sqrt(np.mean((pred_linear - target_linear)**2))
ss_res = np.sum((target_linear - pred_linear)**2)
ss_tot = np.sum((target_linear - target_linear.mean())**2)
r2_linear = 1 - ss_res / ss_tot

print(f"线性域指标:")
print(f"  RMSE: {rmse_linear:.6e}")
print(f"  R²: {r2_linear:.6f}")

# 6. 检查是否有数值溢出
print(f"\n第6步：检查数值溢出")
print(f"  10^(pred_db/10) 的范围检查:")
print(f"    pred_db范围: {pred_db.min():.2f} ~ {pred_db.max():.2f}")
print(f"    10^(min/10) = {10**(pred_db.min()/10):.6e}")
print(f"    10^(max/10) = {10**(pred_db.max()/10):.6e}")

if pred_db.max() > 50:
    print("  WARNING: pred_db过大，可能导致溢出")
if pred_db.min() < -100:
    print("  WARNING: pred_db过小，可能下溢为0")

# 7. 对比evaluation.py的逻辑
print(f"\n第7步：模拟evaluation.py的计算")
print("evaluation.py逻辑:")
print("  1. 网络输出pred_std (标准化)")
print("  2. 反标准化: pred_db = pred_std * std + mean")
print("  3. 转线性: pred_linear = 10^(pred_db/10)")
print("  4. 计算RMSE/R² (线性域)")

# 检查evaluation.py是否正确使用了preprocessing_stats
print(f"\nevaluation.py使用的stats:")
print(f"  evaluator.preprocessing_stats = {evaluator.preprocessing_stats}")
print(f"  evaluator.use_log_output = {evaluator.use_log_output}")

print("\n" + "="*70)
print("诊断完成")
print("="*70)