import torch
import os
import sys

# 指定的文件路径
file_path = r"G:\feko_data\wavelet\models\direct_additive_dual_branch_mlp_relu_raw_20251224_155231.pth"

def analyze_checkpoint_structure(path):
    if not os.path.exists(path):
        print(f"错误: 文件未找到 - {path}")
        return

    print(f"=== 分析模型文件: {os.path.basename(path)} ===\n")
    try:
        # 加载 checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # 1. 顶层结构
        print(f"文件类型: {type(checkpoint)}")
        if not isinstance(checkpoint, dict):
            print("警告: 文件不是标准的字典格式，可能是直接保存的模型对象。")
            return
            
        print(f"包含的顶层键 (Keys): {list(checkpoint.keys())}\n")

        # 2. 详细分析 Config (配置信息)
        if 'config' in checkpoint:
            print("--- [config] 模型配置信息 ---")
            config = checkpoint['config']
            for k, v in config.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sub_k, sub_v in v.items():
                        print(f"    {sub_k}: {sub_v}")
                else:
                    print(f"  {k}: {v}")
        else:
            print("警告: 未找到 'config' 键，无法自动重建模型结构。")
        print("")

        # 3. 分析 Autoencoder 权重
        if 'autoencoder' in checkpoint:
            ae_state = checkpoint['autoencoder']
            print(f"--- [autoencoder] 权重信息 ---")
            print(f"  层数/参数量: {len(ae_state)} 个张量")
            # 打印前3层名称示例
            print(f"  示例层名称: {list(ae_state.keys())[:3]} ...")
            
            # 检查是否有特定于 Additive Dual Branch 的参数
            special_params = [k for k in ae_state.keys() if 'alpha' in k or 'weight' in k and 'layer' not in k]
            if special_params:
                print(f"  发现特殊参数 (如可学习权重): {special_params[:5]}")
        print("")

        # 4. 分析 Parameter Mapper
        if 'parameter_mapper' in checkpoint:
            pm_state = checkpoint['parameter_mapper']
            print(f"--- [parameter_mapper] 权重信息 ---")
            print(f"  层数/参数量: {len(pm_state)} 个张量")
        print("")

        # 5. 分析 Adapter Stats (数据统计)
        if 'adapter_stats' in checkpoint:
            stats = checkpoint['adapter_stats']
            print(f"--- [adapter_stats] 数据预处理统计 ---")
            print(f"  包含统计量: {list(stats.keys())}")
            # 简单检查形状
            for k, v in stats.items():
                if hasattr(v, 'shape'):
                     print(f"    {k}: shape={v.shape}")
                elif isinstance(v, list):
                     print(f"    {k}: list length={len(v)}")
        
        # 6. 训练历史摘要
        if 'training_history' in checkpoint:
            hist = checkpoint['training_history']
            print(f"\n--- [training_history] 训练记录 ---")
            if 'stage_histories' in hist:
                print("  包含阶段:")
                for stage in hist['stage_histories']:
                    print(f"    - {stage}")

    except Exception as e:
        print(f"解析失败: {e}")

if __name__ == "__main__":
    analyze_checkpoint_structure(file_path)
