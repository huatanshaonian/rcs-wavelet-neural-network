"""
批量为所有AutoEncoder模型应用动态适配功能
此脚本会修改以下模型文件：
- DirectMLPAutoEncoder
- Enhanced系列 (EnhancedWaveletAutoEncoder, EnhancedDirectAutoEncoder)
- Deep系列 (DeepWaveletAutoEncoder, DeepDirectAutoEncoder)
- Sine系列 (SinWaveletAutoEncoder, SinDirectAutoEncoder, SinWaveletMLPAutoEncoder, SinDirectMLPAutoEncoder)
"""

# 需要修改的模型列表
MODELS_TO_UPDATE = [
    {
        'file': 'autoencoder/models/mlp_autoencoder.py',
        'class': 'DirectMLPAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/enhanced_cnn_autoencoder.py',
        'class': 'EnhancedWaveletAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/enhanced_cnn_autoencoder.py',
        'class': 'EnhancedDirectAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/deep_autoencoder.py',
        'class': 'DeepWaveletAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/deep_autoencoder.py',
        'class': 'DeepDirectAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/sine_cnn_autoencoder.py',
        'class': 'SinWaveletAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/sine_cnn_autoencoder.py',
        'class': 'SinDirectAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/sine_mlp_autoencoder.py',
        'class': 'SinWaveletMLPAutoEncoder',
        'status': 'pending'
    },
    {
        'file': 'autoencoder/models/sine_mlp_autoencoder.py',
        'class': 'SinDirectMLPAutoEncoder',
        'status': 'pending'
    }
]

print("需要修改的模型：")
for i, model in enumerate(MODELS_TO_UPDATE, 1):
    print(f"{i}. {model['class']} ({model['file']})")

print(f"\n总计: {len(MODELS_TO_UPDATE)} 个模型类需要应用动态适配")
