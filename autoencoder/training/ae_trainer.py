"""
AutoEncoder训练器
实现两阶段训练策略：
1. 无监督AE训练
2. 参数映射训练
3. 端到端微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import os
import warnings
from collections import defaultdict

from ..models.cnn_autoencoder import WaveletAutoEncoder
from ..models.parameter_mapper import ParameterMapperFactory
from ..utils.wavelet_transform import WaveletTransform


class AE_Trainer:
    """
    AutoEncoder训练器
    支持分阶段训练和端到端训练
    """

    def __init__(self,
                 autoencoder: WaveletAutoEncoder,
                 wavelet_transform: WaveletTransform,
                 device: Optional[torch.device] = None):
        """
        初始化训练器

        Args:
            autoencoder: 小波AutoEncoder模型
            wavelet_transform: 小波变换器
            device: 计算设备
        """
        self.autoencoder = autoencoder
        self.wavelet_transform = wavelet_transform
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 移动模型到设备
        self.autoencoder.to(self.device)

        # 参数映射器（后续设置）
        self.parameter_mapper = None
        self.mapper_type = None

        # 训练历史
        self.training_history = {
            'stage1_ae': defaultdict(list),      # 阶段1: AE训练
            'stage2_mapping': defaultdict(list), # 阶段2: 参数映射
            'stage3_e2e': defaultdict(list)      # 阶段3: 端到端
        }

        print(f"AE训练器初始化完成，使用设备: {self.device}")

    def stage1_train_autoencoder(self,
                                rcs_data: np.ndarray,
                                epochs: int = 100,
                                batch_size: int = 16,
                                learning_rate: float = 1e-3,
                                weight_decay: float = 1e-5,
                                validation_split: float = 0.2,
                                save_best: bool = True,
                                save_path: str = "models/ae_best.pth") -> Dict[str, List[float]]:
        """
        阶段1: 无监督AutoEncoder训练

        Args:
            rcs_data: [N, 91, 91, 2] RCS数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            validation_split: 验证集比例
            save_best: 是否保存最佳模型
            save_path: 模型保存路径

        Returns:
            training_history: 训练历史
        """
        print(f"\\n=== 阶段1: AutoEncoder无监督训练 ===")
        print(f"数据形状: {rcs_data.shape}")
        print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

        # 数据预处理: RCS → 小波系数
        print("执行小波变换预处理...")
        rcs_tensor = torch.FloatTensor(rcs_data)
        wavelet_data = self.wavelet_transform.forward_transform(rcs_tensor)
        print(f"小波系数形状: {wavelet_data.shape}")

        # 分割训练/验证集
        n_samples = wavelet_data.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_data = wavelet_data[train_indices]
        val_data = wavelet_data[val_indices]

        # 创建数据加载器
        train_dataset = TensorDataset(train_data, train_data)  # 自编码任务
        val_dataset = TensorDataset(val_data, val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 优化器和损失函数
        optimizer = optim.Adam(self.autoencoder.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        criterion = nn.MSELoss()

        # 训练循环
        best_val_loss = float('inf')
        history = defaultdict(list)

        for epoch in range(epochs):
            start_time = time.time()

            # 训练阶段
            self.autoencoder.train()
            train_loss = 0.0
            train_samples = 0

            for batch_wavelet, _ in train_loader:
                batch_wavelet = batch_wavelet.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                recon_wavelet, latent = self.autoencoder(batch_wavelet)

                # 计算损失
                loss = criterion(recon_wavelet, batch_wavelet)

                # 反向传播
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_wavelet.size(0)
                train_samples += batch_wavelet.size(0)

            avg_train_loss = train_loss / train_samples

            # 验证阶段
            self.autoencoder.eval()
            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for batch_wavelet, _ in val_loader:
                    batch_wavelet = batch_wavelet.to(self.device)

                    recon_wavelet, latent = self.autoencoder(batch_wavelet)
                    loss = criterion(recon_wavelet, batch_wavelet)

                    val_loss += loss.item() * batch_wavelet.size(0)
                    val_samples += batch_wavelet.size(0)

            avg_val_loss = val_loss / val_samples

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 保存最佳模型
            if save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': self.autoencoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': avg_val_loss,
                        'model_config': self.autoencoder.get_model_info()
                    }, save_path)

            # 打印进度
            epoch_time = time.time() - start_time
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
                      f"Time: {epoch_time:.2f}s")

        # 保存训练历史
        self.training_history['stage1_ae'] = history

        print(f"\\n阶段1训练完成！最佳验证损失: {best_val_loss:.6f}")
        return dict(history)

    def stage2_train_parameter_mapping(self,
                                      params_data: np.ndarray,
                                      rcs_data: np.ndarray,
                                      mapper_type: str = 'mlp',
                                      epochs: int = 50,
                                      batch_size: int = 32,
                                      learning_rate: float = 1e-4,
                                      validation_split: float = 0.2,
                                      **mapper_kwargs) -> Dict[str, Any]:
        """
        阶段2: 参数映射训练

        Args:
            params_data: [N, 9] 设计参数
            rcs_data: [N, 91, 91, 2] RCS数据
            mapper_type: 映射器类型 ('mlp', 'random_forest', 'hybrid')
            epochs: 训练轮数（仅对深度学习模型有效）
            batch_size: 批次大小
            learning_rate: 学习率
            validation_split: 验证集比例
            **mapper_kwargs: 映射器特定参数

        Returns:
            training_results: 训练结果
        """
        print(f"\\n=== 阶段2: 参数映射训练 ===")
        print(f"参数数据形状: {params_data.shape}")
        print(f"RCS数据形状: {rcs_data.shape}")
        print(f"映射器类型: {mapper_type}")

        # 1. 使用预训练的AE获取隐空间表示
        print("提取隐空间表示...")
        self.autoencoder.eval()

        rcs_tensor = torch.FloatTensor(rcs_data)
        wavelet_data = self.wavelet_transform.forward_transform(rcs_tensor)

        # 批量处理避免内存溢出
        latent_representations = []
        batch_size_extract = 32

        with torch.no_grad():
            for i in range(0, wavelet_data.shape[0], batch_size_extract):
                batch_wavelet = wavelet_data[i:i+batch_size_extract].to(self.device)
                _, latent = self.autoencoder(batch_wavelet)
                latent_representations.append(latent.cpu())

        target_latent = torch.cat(latent_representations, dim=0).numpy()
        print(f"隐空间表示形状: {target_latent.shape}")

        # 2. 分割训练/验证数据
        n_samples = params_data.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_params = params_data[train_indices]
        train_latent = target_latent[train_indices]
        val_params = params_data[val_indices]
        val_latent = target_latent[val_indices]

        # 3. 创建和训练映射器
        self.parameter_mapper = ParameterMapperFactory.create_mapper(
            mapper_type=mapper_type,
            param_dim=params_data.shape[1],
            latent_dim=target_latent.shape[1],
            **mapper_kwargs
        )

        self.mapper_type = mapper_type

        if mapper_type == 'mlp':
            # 深度学习训练
            history = self._train_mlp_mapper(
                train_params, train_latent, val_params, val_latent,
                epochs, batch_size, learning_rate
            )
        else:
            # 传统机器学习训练
            history = self._train_ml_mapper(
                train_params, train_latent, val_params, val_latent
            )

        self.training_history['stage2_mapping'] = history

        print(f"\\n阶段2训练完成！")
        return history

    def _train_mlp_mapper(self,
                         train_params: np.ndarray,
                         train_latent: np.ndarray,
                         val_params: np.ndarray,
                         val_latent: np.ndarray,
                         epochs: int,
                         batch_size: int,
                         learning_rate: float) -> Dict[str, List[float]]:
        """训练MLP映射器"""

        # 移动映射器到设备
        self.parameter_mapper.to(self.device)

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(train_params),
            torch.FloatTensor(train_latent)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_params),
            torch.FloatTensor(val_latent)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 优化器
        optimizer = optim.Adam(self.parameter_mapper.parameters(),
                              lr=learning_rate,
                              weight_decay=1e-5)

        criterion = nn.MSELoss()
        history = defaultdict(list)

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.parameter_mapper.train()
            train_loss = 0.0
            train_samples = 0

            for batch_params, batch_latent in train_loader:
                batch_params = batch_params.to(self.device)
                batch_latent = batch_latent.to(self.device)

                optimizer.zero_grad()

                pred_latent = self.parameter_mapper(batch_params)
                loss = criterion(pred_latent, batch_latent)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_params.size(0)
                train_samples += batch_params.size(0)

            avg_train_loss = train_loss / train_samples

            # 验证阶段
            self.parameter_mapper.eval()
            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for batch_params, batch_latent in val_loader:
                    batch_params = batch_params.to(self.device)
                    batch_latent = batch_latent.to(self.device)

                    pred_latent = self.parameter_mapper(batch_params)
                    loss = criterion(pred_latent, batch_latent)

                    val_loss += loss.item() * batch_params.size(0)
                    val_samples += batch_params.size(0)

            avg_val_loss = val_loss / val_samples

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}")

        return dict(history)

    def _train_ml_mapper(self,
                        train_params: np.ndarray,
                        train_latent: np.ndarray,
                        val_params: np.ndarray,
                        val_latent: np.ndarray) -> Dict[str, Any]:
        """训练传统机器学习映射器"""

        # 训练
        self.parameter_mapper.fit(train_params, train_latent)

        # 评估
        train_metrics = self.parameter_mapper.evaluate(train_params, train_latent)
        val_metrics = self.parameter_mapper.evaluate(val_params, val_latent)

        print(f"训练集评估: MSE={train_metrics['mse']:.6f}, R²={train_metrics['r2_score']:.4f}")
        print(f"验证集评估: MSE={val_metrics['mse']:.6f}, R²={val_metrics['r2_score']:.4f}")

        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'mapper_type': self.mapper_type
        }

    def stage3_end_to_end_training(self,
                                  params_data: np.ndarray,
                                  rcs_data: np.ndarray,
                                  epochs: int = 20,
                                  batch_size: int = 16,
                                  learning_rate: float = 1e-5,
                                  validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        阶段3: 端到端微调

        Args:
            params_data: [N, 9] 设计参数
            rcs_data: [N, 91, 91, 2] RCS数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            validation_split: 验证集比例

        Returns:
            training_history: 训练历史
        """
        print(f"\\n=== 阶段3: 端到端微调 ===")

        if self.parameter_mapper is None or self.mapper_type != 'mlp':
            warnings.warn("端到端训练仅支持MLP映射器，跳过此阶段")
            return {}

        # 数据预处理
        rcs_tensor = torch.FloatTensor(rcs_data)
        target_wavelet = self.wavelet_transform.forward_transform(rcs_tensor)

        # 分割数据
        n_samples = params_data.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_params = torch.FloatTensor(params_data[train_indices])
        train_target = target_wavelet[train_indices]
        val_params = torch.FloatTensor(params_data[val_indices])
        val_target = target_wavelet[val_indices]

        # 数据加载器
        train_dataset = TensorDataset(train_params, train_target)
        val_dataset = TensorDataset(val_params, val_target)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 联合优化器
        all_params = list(self.autoencoder.parameters()) + list(self.parameter_mapper.parameters())
        optimizer = optim.Adam(all_params, lr=learning_rate)

        criterion = nn.MSELoss()
        history = defaultdict(list)

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.autoencoder.train()
            self.parameter_mapper.train()

            train_loss = 0.0
            train_samples = 0

            for batch_params, batch_target in train_loader:
                batch_params = batch_params.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()

                # 端到端前向传播: 参数 → 隐空间 → 重建小波系数
                pred_latent = self.parameter_mapper(batch_params)
                pred_wavelet = self.autoencoder.decode(pred_latent)

                # 计算重建损失
                loss = criterion(pred_wavelet, batch_target)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_params.size(0)
                train_samples += batch_params.size(0)

            avg_train_loss = train_loss / train_samples

            # 验证阶段
            self.autoencoder.eval()
            self.parameter_mapper.eval()

            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for batch_params, batch_target in val_loader:
                    batch_params = batch_params.to(self.device)
                    batch_target = batch_target.to(self.device)

                    pred_latent = self.parameter_mapper(batch_params)
                    pred_wavelet = self.autoencoder.decode(pred_latent)

                    loss = criterion(pred_wavelet, batch_target)

                    val_loss += loss.item() * batch_params.size(0)
                    val_samples += batch_params.size(0)

            avg_val_loss = val_loss / val_samples

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            if epoch % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}")

        self.training_history['stage3_e2e'] = history

        print(f"\\n阶段3训练完成！")
        return dict(history)

    def predict_rcs_from_params(self, params: np.ndarray) -> np.ndarray:
        """
        从设计参数预测RCS

        Args:
            params: [N, 9] 设计参数

        Returns:
            pred_rcs: [N, 91, 91, 2] 预测的RCS
        """
        if self.parameter_mapper is None:
            raise ValueError("参数映射器尚未训练，请先执行阶段2训练")

        self.autoencoder.eval()

        params_tensor = torch.FloatTensor(params)

        with torch.no_grad():
            if self.mapper_type == 'mlp':
                # PyTorch模型
                self.parameter_mapper.eval()
                params_tensor = params_tensor.to(self.device)
                pred_latent = self.parameter_mapper(params_tensor)
            else:
                # 传统机器学习模型
                pred_latent = self.parameter_mapper.predict(params)
                pred_latent = torch.FloatTensor(pred_latent).to(self.device)

            # 解码为小波系数
            pred_wavelet = self.autoencoder.decode(pred_latent)

            # 逆小波变换得到RCS
            pred_rcs = self.wavelet_transform.inverse_transform(pred_wavelet.cpu())

        return pred_rcs.numpy()

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        summary = {
            'model_info': self.autoencoder.get_model_info(),
            'wavelet_info': self.wavelet_transform.get_transform_info(),
            'training_stages_completed': [],
            'training_history': self.training_history
        }

        if len(self.training_history['stage1_ae']) > 0:
            summary['training_stages_completed'].append('Stage1: AutoEncoder')

        if len(self.training_history['stage2_mapping']) > 0:
            summary['training_stages_completed'].append(f'Stage2: Parameter Mapping ({self.mapper_type})')

        if len(self.training_history['stage3_e2e']) > 0:
            summary['training_stages_completed'].append('Stage3: End-to-End')

        return summary


def test_ae_trainer():
    """测试AE训练器"""
    print("=== AE训练器测试 ===")

    # 创建模拟数据
    n_samples = 50
    rcs_data = np.random.randn(n_samples, 91, 91, 2) * 10
    params_data = np.random.randn(n_samples, 9)

    print(f"模拟数据: RCS {rcs_data.shape}, 参数 {params_data.shape}")

    # 创建模型和组件
    ae = WaveletAutoEncoder(latent_dim=128)  # 较小的隐空间便于测试
    wt = WaveletTransform(wavelet='db4')

    # 创建训练器
    trainer = AE_Trainer(ae, wt)

    # 测试阶段1: AE训练（短时间测试）
    print("\\n测试阶段1...")
    stage1_history = trainer.stage1_train_autoencoder(
        rcs_data=rcs_data,
        epochs=5,  # 短测试
        batch_size=8,
        save_best=False
    )

    print(f"阶段1完成，最终训练损失: {stage1_history['train_loss'][-1]:.6f}")

    # 测试阶段2: 参数映射
    print("\\n测试阶段2...")
    stage2_history = trainer.stage2_train_parameter_mapping(
        params_data=params_data,
        rcs_data=rcs_data,
        mapper_type='random_forest',  # 使用随机森林快速测试
        epochs=5
    )

    print("阶段2完成")

    # 测试预测
    print("\\n测试预测...")
    test_params = params_data[:5]  # 使用前5个样本测试
    pred_rcs = trainer.predict_rcs_from_params(test_params)

    print(f"预测RCS形状: {pred_rcs.shape}")
    print(f"原始RCS范围: [{rcs_data[:5].min():.3f}, {rcs_data[:5].max():.3f}]")
    print(f"预测RCS范围: [{pred_rcs.min():.3f}, {pred_rcs.max():.3f}]")

    # 训练总结
    summary = trainer.get_training_summary()
    print(f"\\n训练总结:")
    print(f"完成阶段: {summary['training_stages_completed']}")

    return True


if __name__ == "__main__":
    test_ae_trainer()