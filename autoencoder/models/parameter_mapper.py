"""
参数映射网络模块
支持多种映射策略：MLP, RandomForest, XGBoost
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


class MLPMapper(nn.Module):
    """
    多层感知机参数映射器

    支持功能：
    1. 多种激活函数（relu/gelu/sin/swish/mish/tanh）
    2. 自适应隐藏层维度（根据param_dim→latent_dim自动计算）
    3. 固定隐藏层维度（向后兼容）
    """

    def __init__(self,
                 param_dim: int = 9,
                 latent_dim: int = 256,
                 hidden_dims: list = None,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 use_adaptive: bool = False,
                 max_ratio: int = 4):
        """
        Args:
            param_dim: 输入参数维度
            latent_dim: 输出隐空间维度
            hidden_dims: 固定隐藏层维度列表（如[64, 128]）。如果use_adaptive=True，此参数被忽略
            dropout_rate: Dropout比率
            activation: 激活函数类型
            use_adaptive: 是否使用自适应隐藏层维度
            max_ratio: 自适应模式下每级最大压缩比（默认4:1）
        """
        super().__init__()

        self.param_dim = param_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.use_adaptive = use_adaptive

        # 确定隐藏层维度
        if use_adaptive:
            # 使用自适应层计算
            from autoencoder.utils.adaptive_layers import calculate_intermediate_dims
            # ParameterMapper通常从小维度(9)映射到大维度(256)，需要渐进式扩展
            # 反转输入输出，计算扩展路径，然后反转回来
            if param_dim < latent_dim:
                # 从小到大：需要扩展
                self.hidden_dims = calculate_intermediate_dims(
                    latent_dim, param_dim, max_ratio=max_ratio, min_layers=2
                )
                self.hidden_dims = list(reversed(self.hidden_dims))  # 反转：从小到大
            else:
                # 从大到小：需要压缩
                self.hidden_dims = calculate_intermediate_dims(
                    param_dim, latent_dim, max_ratio=max_ratio, min_layers=2
                )
        else:
            # 使用固定隐藏层维度
            if hidden_dims is None:
                hidden_dims = [64, 128]  # 默认值
            self.hidden_dims = hidden_dims

        # 选择激活函数
        self.activation = self._create_activation(activation)

        # 构建网络
        layers = []
        input_dim = param_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._create_activation(activation),  # 每层独立创建激活函数
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, latent_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _create_activation(self, activation: str) -> nn.Module:
        """创建激活函数层"""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sin':
            from autoencoder.utils.activation_factory import SinActivation
            return SinActivation()
        elif activation == 'swish' or activation == 'silu':
            return nn.SiLU(inplace=True)
        elif activation == 'mish':
            try:
                return nn.Mish(inplace=True)
            except:
                from autoencoder.utils.activation_factory import MishActivation
                return MishActivation()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU(inplace=True)  # 默认ReLU

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.network(params)

    def get_parameter_count(self) -> int:
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        structure = [self.param_dim] + self.hidden_dims + [self.latent_dim]
        structure_str = ' → '.join(map(str, structure))

        return {
            'architecture': 'MLPMapper',
            'param_dim': self.param_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'structure': structure_str,
            'num_layers': len(structure) - 1,
            'activation': self.activation_name,
            'dropout_rate': self.dropout_rate,
            'use_adaptive': self.use_adaptive,
            'parameter_count': self.get_parameter_count()
        }


class RandomForestMapper:
    """
    随机森林参数映射器
    """

    def __init__(self,
                 param_dim: int = 9,
                 latent_dim: int = 256,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = 10,
                 random_state: int = 42):

        self.param_dim = param_dim
        self.latent_dim = latent_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # 为每个隐空间维度创建一个回归器
        self.regressors = []
        for i in range(latent_dim):
            regressor = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state + i,
                n_jobs=-1
            )
            self.regressors.append(regressor)

        self.is_fitted = False

    def fit(self, params: np.ndarray, latent: np.ndarray):
        """
        训练随机森林

        Args:
            params: [N, param_dim] 设计参数
            latent: [N, latent_dim] 隐空间表示
        """
        print(f"训练随机森林映射器: {params.shape} -> {latent.shape}")

        for i, regressor in enumerate(self.regressors):
            regressor.fit(params, latent[:, i])

        self.is_fitted = True
        print("随机森林训练完成")

    def predict(self, params: np.ndarray) -> np.ndarray:
        """
        预测隐空间表示

        Args:
            params: [N, param_dim] 设计参数

        Returns:
            latent: [N, latent_dim] 预测的隐空间表示
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        predictions = []
        for regressor in self.regressors:
            pred = regressor.predict(params)
            predictions.append(pred)

        return np.column_stack(predictions)

    def evaluate(self, params: np.ndarray, latent: np.ndarray) -> Dict[str, float]:
        """
        评估映射性能
        """
        pred_latent = self.predict(params)

        mse = mean_squared_error(latent, pred_latent)
        r2 = r2_score(latent, pred_latent)

        # 计算每个维度的MSE
        dim_mses = []
        for i in range(self.latent_dim):
            dim_mse = mean_squared_error(latent[:, i], pred_latent[:, i])
            dim_mses.append(dim_mse)

        return {
            'mse': mse,
            'r2_score': r2,
            'mean_dim_mse': np.mean(dim_mses),
            'std_dim_mse': np.std(dim_mses),
            'max_dim_mse': np.max(dim_mses),
            'min_dim_mse': np.min(dim_mses)
        }

    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'regressors': self.regressors,
            'param_dim': self.param_dim,
            'latent_dim': self.latent_dim,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"随机森林模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.regressors = model_data['regressors']
        self.param_dim = model_data['param_dim']
        self.latent_dim = model_data['latent_dim']
        self.n_estimators = model_data['n_estimators']
        self.max_depth = model_data['max_depth']
        self.is_fitted = model_data['is_fitted']
        print(f"随机森林模型已从 {filepath} 加载")


class HybridMapper:
    """
    混合映射器：结合深度学习和传统机器学习
    """

    def __init__(self,
                 param_dim: int = 9,
                 latent_dim: int = 256,
                 mlp_hidden_dims: list = [128, 256],
                 rf_n_estimators: int = 50):

        self.param_dim = param_dim
        self.latent_dim = latent_dim

        # MLP特征提取器
        self.mlp_feature_extractor = MLPMapper(
            param_dim=param_dim,
            latent_dim=128,  # 中间特征维度
            hidden_dims=mlp_hidden_dims,
            dropout_rate=0.1
        )

        # 随机森林映射器（使用MLP特征）
        self.rf_mapper = RandomForestMapper(
            param_dim=128,  # 使用MLP特征
            latent_dim=latent_dim,
            n_estimators=rf_n_estimators,
            max_depth=8
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp_feature_extractor.to(self.device)

    def extract_features(self, params: np.ndarray) -> np.ndarray:
        """使用MLP提取特征"""
        params_tensor = torch.FloatTensor(params).to(self.device)

        with torch.no_grad():
            features = self.mlp_feature_extractor(params_tensor)

        return features.cpu().numpy()

    def fit(self, params: np.ndarray, latent: np.ndarray, mlp_epochs: int = 100):
        """
        训练混合映射器

        Args:
            params: [N, param_dim] 设计参数
            latent: [N, latent_dim] 隐空间表示
            mlp_epochs: MLP训练轮数
        """
        # 1. 训练MLP特征提取器（可以使用无监督或自监督方法）
        print("训练MLP特征提取器...")
        # 这里简化处理，实际可以设计更复杂的训练策略

        # 2. 提取MLP特征
        mlp_features = self.extract_features(params)

        # 3. 训练随机森林
        self.rf_mapper.fit(mlp_features, latent)

        print("混合映射器训练完成")

    def predict(self, params: np.ndarray) -> np.ndarray:
        """预测隐空间表示"""
        mlp_features = self.extract_features(params)
        return self.rf_mapper.predict(mlp_features)

    def evaluate(self, params: np.ndarray, latent: np.ndarray) -> Dict[str, float]:
        """评估性能"""
        pred_latent = self.predict(params)

        mse = mean_squared_error(latent, pred_latent)
        r2 = r2_score(latent, pred_latent)

        return {
            'mse': mse,
            'r2_score': r2,
            'mapping_type': 'hybrid'
        }


class ParameterMapperFactory:
    """
    参数映射器工厂类
    """

    @staticmethod
    def create_mapper(mapper_type: str,
                     param_dim: int = 9,
                     latent_dim: int = 256,
                     **kwargs) -> Union[MLPMapper, RandomForestMapper, HybridMapper]:
        """
        创建参数映射器

        Args:
            mapper_type: 'mlp', 'random_forest', 'hybrid'
            param_dim: 参数维度
            latent_dim: 隐空间维度
            **kwargs: 其他参数
        """

        if mapper_type == 'mlp':
            return MLPMapper(
                param_dim=param_dim,
                latent_dim=latent_dim,
                hidden_dims=kwargs.get('hidden_dims', [64, 128]),
                dropout_rate=kwargs.get('dropout_rate', 0.3),
                activation=kwargs.get('activation', 'relu')
            )

        elif mapper_type == 'random_forest':
            return RandomForestMapper(
                param_dim=param_dim,
                latent_dim=latent_dim,
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42)
            )

        elif mapper_type == 'hybrid':
            return HybridMapper(
                param_dim=param_dim,
                latent_dim=latent_dim,
                mlp_hidden_dims=kwargs.get('mlp_hidden_dims', [128, 256]),
                rf_n_estimators=kwargs.get('rf_n_estimators', 50)
            )

        else:
            raise ValueError(f"Unsupported mapper type: {mapper_type}")


def test_parameter_mappers():
    """测试参数映射器"""
    print("=== 参数映射器测试 ===")

    # 创建测试数据
    n_samples = 100
    param_dim = 9
    latent_dim = 256

    # 生成模拟数据
    np.random.seed(42)
    params = np.random.randn(n_samples, param_dim)
    latent = np.random.randn(n_samples, latent_dim)

    print(f"测试数据: {params.shape} -> {latent.shape}")

    # 分割训练/测试数据
    split_idx = int(0.8 * n_samples)
    train_params, test_params = params[:split_idx], params[split_idx:]
    train_latent, test_latent = latent[:split_idx], latent[split_idx:]

    # 测试不同的映射器
    mapper_types = ['mlp', 'random_forest']

    for mapper_type in mapper_types:
        print(f"\n--- 测试 {mapper_type} 映射器 ---")

        try:
            mapper = ParameterMapperFactory.create_mapper(
                mapper_type=mapper_type,
                param_dim=param_dim,
                latent_dim=latent_dim
            )

            if mapper_type == 'mlp':
                # PyTorch模型测试
                params_tensor = torch.FloatTensor(test_params)
                with torch.no_grad():
                    pred_latent = mapper(params_tensor).numpy()

                mse = mean_squared_error(test_latent, pred_latent)
                print(f"MLP映射器 MSE (未训练): {mse:.6f}")

            else:
                # 训练和测试机器学习模型
                mapper.fit(train_params, train_latent)

                # 评估
                metrics = mapper.evaluate(test_params, test_latent)
                print(f"{mapper_type} 评估结果:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.6f}")

        except Exception as e:
            print(f"测试 {mapper_type} 时出错: {e}")

    print("\n参数映射器测试完成")


if __name__ == "__main__":
    test_parameter_mappers()