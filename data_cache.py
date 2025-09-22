#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RCS数据缓存系统

提供快速的数据读取和缓存功能，避免每次都从CSV重新读取
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import time


class RCSDataCache:
    """
    RCS数据缓存管理器

    功能:
    1. 自动检测数据变化
    2. 缓存已处理的数据
    3. 提供快速读取接口
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        初始化缓存管理器

        参数:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        return ""

    def _get_data_signature(self, params_file: str, rcs_data_dir: str,
                           model_ids: list, frequencies: list) -> str:
        """生成数据配置的签名"""
        # 参数文件哈希
        params_hash = self._get_file_hash(params_file)

        # RCS数据目录中相关文件的哈希
        rcs_files = []
        for model_id in model_ids:
            for freq in frequencies:
                rcs_file = os.path.join(rcs_data_dir, f"{model_id}_{freq}.csv")
                if os.path.exists(rcs_file):
                    rcs_files.append(self._get_file_hash(rcs_file))

        # 组合所有哈希值
        combined_hash = hashlib.md5()
        combined_hash.update(params_hash.encode())
        combined_hash.update("".join(rcs_files).encode())
        combined_hash.update("".join(model_ids).encode())
        combined_hash.update("".join(frequencies).encode())

        return combined_hash.hexdigest()

    def _get_cache_path(self, signature: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"rcs_data_{signature}.pkl")

    def load_data_with_cache(self, params_file: str, rcs_data_dir: str,
                            model_ids: list, frequencies: list,
                            force_reload: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用缓存加载数据

        参数:
            params_file: 参数文件路径
            rcs_data_dir: RCS数据目录
            model_ids: 模型ID列表
            frequencies: 频率列表
            force_reload: 强制重新读取CSV

        返回:
            (参数数据, RCS数据)
        """
        print("检查数据缓存...")

        # 生成数据签名
        signature = self._get_data_signature(params_file, rcs_data_dir, model_ids, frequencies)
        cache_path = self._get_cache_path(signature)

        print(f"数据签名: {signature[:16]}...")
        print(f"缓存路径: {cache_path}")

        # 检查是否有有效缓存
        if not force_reload and os.path.exists(cache_path):
            try:
                print("发现缓存文件，正在加载...")
                start_time = time.time()

                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                param_data = cache_data['param_data']
                rcs_data = cache_data['rcs_data']
                cache_info = cache_data['info']

                load_time = time.time() - start_time
                print(f"缓存加载完成: {load_time:.2f}秒")
                print(f"参数数据形状: {param_data.shape}")
                print(f"RCS数据形状: {rcs_data.shape}")
                print(f"缓存创建时间: {cache_info['created_at']}")

                return param_data, rcs_data

            except Exception as e:
                print(f"警告: 缓存文件损坏: {e}")
                print("将重新读取原始数据...")

        # 从CSV重新读取数据
        print("从CSV读取数据...")
        start_time = time.time()

        param_data, rcs_data = self._load_from_csv(
            params_file, rcs_data_dir, model_ids, frequencies
        )

        load_time = time.time() - start_time
        print(f"CSV读取完成: {load_time:.2f}秒")

        # 保存到缓存
        print("保存数据到缓存...")
        cache_data = {
            'param_data': param_data,
            'rcs_data': rcs_data,
            'info': {
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'signature': signature,
                'params_file': params_file,
                'rcs_data_dir': rcs_data_dir,
                'model_ids': model_ids,
                'frequencies': frequencies,
                'param_shape': param_data.shape,
                'rcs_shape': rcs_data.shape
            }
        }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"缓存已保存: {cache_path}")

        except Exception as e:
            print(f"警告: 缓存保存失败: {e}")

        return param_data, rcs_data

    def _load_from_csv(self, params_file: str, rcs_data_dir: str,
                      model_ids: list, frequencies: list) -> Tuple[np.ndarray, np.ndarray]:
        """从CSV文件加载数据"""
        import rcs_data_reader as rdr

        print("读取参数文件...")
        # 读取参数数据
        params_df = pd.read_csv(params_file)

        param_data = []
        rcs_data = []

        print(f"开始读取 {len(model_ids)} 个模型的RCS数据...")

        for i, model_id in enumerate(model_ids):
            try:
                # 获取参数 - 根据model_id作为索引（从0开始）
                model_index = int(model_id) - 1  # model_id从1开始，但索引从0开始
                if model_index < len(params_df):
                    model_params = params_df.iloc[model_index].values  # 获取整行参数
                    param_data.append(model_params.astype(np.float32))
                else:
                    print(f"警告: 模型 {model_id} 在参数文件中不存在")
                    continue

                # 读取RCS数据
                rcs_matrices = []
                for freq in frequencies:
                    try:
                        rcs_result = rdr.get_rcs_matrix(model_id, freq, rcs_data_dir)
                        if rcs_result is not None and isinstance(rcs_result, dict):
                            # 提取线性RCS矩阵
                            rcs_matrix = rcs_result['rcs_linear']
                            rcs_matrices.append(rcs_matrix)
                        else:
                            print(f"警告: 模型 {model_id} 频率 {freq} 数据缺失")
                            # 用零矩阵填充
                            rcs_matrices.append(np.zeros((91, 91)))
                    except Exception as e:
                        print(f"警告: 读取模型 {model_id} 频率 {freq} 数据失败: {e}")
                        # 用零矩阵填充
                        rcs_matrices.append(np.zeros((91, 91)))

                # 组合双频数据 [91, 91, 2]
                if len(rcs_matrices) == 2:
                    combined_rcs = np.stack(rcs_matrices, axis=-1)
                    rcs_data.append(combined_rcs)

                # 进度显示
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"已处理: {i+1}/{len(model_ids)} ({(i+1)/len(model_ids)*100:.1f}%)")

            except Exception as e:
                print(f"警告: 处理模型 {model_id} 时出错: {e}")
                continue

        param_data = np.array(param_data, dtype=np.float32)
        rcs_data = np.array(rcs_data, dtype=np.float32)

        print(f"数据读取完成:")
        print(f"  参数数据: {param_data.shape}")
        print(f"  RCS数据: {rcs_data.shape}")

        return param_data, rcs_data

    def clear_cache(self):
        """清除所有缓存文件"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('rcs_data_') and f.endswith('.pkl')]

        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            try:
                os.remove(cache_path)
                print(f"已删除缓存: {cache_file}")
            except Exception as e:
                print(f"删除缓存失败 {cache_file}: {e}")

        print(f"缓存清理完成，共删除 {len(cache_files)} 个文件")

    def list_cache_info(self):
        """列出所有缓存信息"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('rcs_data_') and f.endswith('.pkl')]

        if not cache_files:
            print("没有找到缓存文件")
            return

        print("缓存文件列表:")
        print("-" * 80)

        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                info = cache_data['info']
                file_size = os.path.getsize(cache_path) / 1024 / 1024  # MB

                print(f"文件: {cache_file}")
                print(f"  创建时间: {info['created_at']}")
                print(f"  参数形状: {info['param_shape']}")
                print(f"  RCS形状: {info['rcs_shape']}")
                print(f"  文件大小: {file_size:.1f} MB")
                print(f"  签名: {info['signature'][:16]}...")
                print()

            except Exception as e:
                print(f"读取缓存信息失败 {cache_file}: {e}")


def create_cache_manager() -> RCSDataCache:
    """创建缓存管理器实例"""
    return RCSDataCache()


if __name__ == "__main__":
    # 测试缓存功能
    cache_manager = create_cache_manager()
    cache_manager.list_cache_info()