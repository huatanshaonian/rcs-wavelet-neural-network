import torch
import os
import sys

# 指定的文件路径
file_path = r"G:\feko_data\wavelet\models\direct_additive_dual_branch_mlp_relu_raw_20251224_155231.pth"

def analyze_checkpoint(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    print(f"Analyzing file: {path}")
    try:
        # 加载模型文件，映射到 CPU 以免显存不足
        checkpoint = torch.load(path, map_location='cpu')
        
        print(f"Data Type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print("\n--- Keys in Checkpoint ---")
            print(list(checkpoint.keys()))
            
            for key, value in checkpoint.items():
                print(f"\n[Key: {key}]")
                if isinstance(value, dict):
                    # 检查是否是 state_dict (通常包含 'weight' 或 'bias')
                    keys_list = list(value.keys())
                    if len(keys_list) > 0 and any(('weight' in k or 'bias' in k) for k in keys_list[:5]):
                        print(f"  Type: State Dict (Model or Optimizer)")
                        print(f"  Number of parameters/layers: {len(keys_list)}")
                        print(f"  First 5 keys: {keys_list[:5]}")
                    else:
                        print(f"  Type: Dictionary")
                        print(f"  Content: {value}")
                elif torch.is_tensor(value):
                    print(f"  Type: Tensor")
                    print(f"  Shape: {value.shape}")
                else:
                    print(f"  Type: {type(value)}")
                    print(f"  Value: {value}")
        else:
            print("\nThis file appears to be a direct model save (not a dict).")
            print(checkpoint)

    except Exception as e:
        print(f"Failed to load or analyze: {e}")

if __name__ == "__main__":
    analyze_checkpoint(file_path)
