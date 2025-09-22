"""
RCS 数据读取模块 - 增强版本
融合原始 rcs_data_reader.py 和 adaptive_rcs_reader.py 的所有功能

该模块包含从RCS POD项目中提取的所有数据读取函数，经过优化后可独立使用。
新增自适应功能，支持任意尺寸的RCS数据处理。

主要功能:
1. 加载设计参数数据 (CSV格式，支持多种编码)
2. 加载RCS数据 (CSV格式，支持NaN值填充和多种编码)
3. 自适应RCS矩阵读取 (支持任意尺寸)
4. 加载预测参数数据
5. 数据预处理和验证
6. RCS数据格式转换
7. 自适应可视化

作者: 融合版本 - 基于原始rcs_data_reader.py + adaptive功能
版本: 2.0 Enhanced
"""

import os
import numpy as np
import pandas as pd
import warnings


def load_parameters(params_file, verbose=True):
    """
    加载设计参数数据 - 增强版，添加详细诊断，并在读取时将NaN填充为相邻值的均值

    参数:
    params_file: 参数CSV文件路径
    verbose: 是否输出详细信息 (默认True)

    返回:
    param_data: numpy数组，包含参数值
    param_names: 参数名称列表
    """
    try:
        if verbose:
            print(f"正在从 {params_file} 加载参数数据...")

        # 读取CSV文件 - 尝试多种编码格式
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
        df = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(params_file, encoding=encoding)
                if verbose:
                    print(f"成功使用 {encoding} 编码读取参数文件")
                break
            except UnicodeDecodeError:
                if verbose:
                    print(f"使用 {encoding} 编码失败，尝试下一种编码...")
                continue

        if df is None:
            raise ValueError(f"无法使用任何编码格式读取文件 {params_file}")

        # 输出CSV文件的基本信息
        if verbose:
            print(f"CSV文件形状: {df.shape}")
            print(f"CSV文件列名: {list(df.columns)}")
            print("\n参数数据前5行:")
            print(df.head(5))
            print("\n参数数据类型:")
            print(df.dtypes)

        # 检查NaN值
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            if verbose:
                print(f"\n发现 {nan_count} 个NaN值，使用插值方法填充")
            # 对每一列使用插值填充NaN，使用前后数据的平均值
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            # 检查是否还有NaN值（边缘可能无法插值）
            remaining_nan = df.isna().sum().sum()
            if remaining_nan > 0:
                if verbose:
                    print(f"插值后仍有 {remaining_nan} 个NaN值，使用前向/后向填充")
                df = df.fillna(method='ffill').fillna(method='bfill')

        # 检查每列的唯一值数量
        if verbose:
            print("\n每列唯一值数量:")
            for col in df.columns:
                unique_values = df[col].unique()
                print(f"  {col}: {len(unique_values)} 个唯一值")

                # 检查是否为常数列
                if len(unique_values) <= 1:
                    print(f"  警告: 列 '{col}' 可能是常数列!")

                # 输出前几个唯一值用于检查
                if len(unique_values) <= 5:
                    print(f"  {col} 的所有唯一值: {unique_values}")
                else:
                    print(f"  {col} 的前5个唯一值: {unique_values[:5]}")

        # 获取参数名称
        param_names = df.columns.tolist()

        # 转换为numpy数组
        param_data = df.values

        # 检查数据中是否包含Inf
        inf_count = np.isinf(param_data).sum()
        if inf_count > 0:
            if verbose:
                print(f"\n警告: 参数数据中包含 {inf_count} 个Inf值")
            # 替换Inf值
            param_data = np.nan_to_num(param_data, nan=0.0, posinf=0.0, neginf=0.0)
            if verbose:
                print("已将Inf值替换为0")

        if verbose:
            print(f"\n成功加载了 {param_data.shape[0]} 个模型的 {param_data.shape[1]} 个参数")

        return param_data, param_names

    except Exception as e:
        print(f"加载参数数据时发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 返回空数据
        return np.array([]), []


def impute_rcs_nans(df, theta_col='Theta', phi_col='Phi', rcs_col='RCS(Total)', verbose=True):
    """
    对RCS数据中的NaN值使用相邻点的平均值进行填充 - 优化版

    参数:
    df: 包含RCS数据的DataFrame
    theta_col: 俯仰角列名 (默认'Theta')
    phi_col: 偏航角列名 (默认'Phi')
    rcs_col: RCS值列名 (默认'RCS(Total)')
    verbose: 是否输出详细信息 (默认True)

    返回:
    填充后的DataFrame
    """
    # 检查是否有NaN值
    nan_count = df[rcs_col].isna().sum()
    if nan_count == 0:
        return df  # 如果没有NaN值，直接返回原始数据

    if verbose:
        print(f"  发现 {nan_count} 个NaN值，使用相邻点的平均值填充")

    # 获取角度网格信息
    theta_values = np.sort(df[theta_col].unique())
    phi_values = np.sort(df[phi_col].unique())
    n_theta = len(theta_values)
    n_phi = len(phi_values)

    # 创建索引映射，便于后续查找
    theta_map = {theta: i for i, theta in enumerate(theta_values)}
    phi_map = {phi: i for i, phi in enumerate(phi_values)}

    # 将数据重塑为2D网格形式
    rcs_grid = np.full((n_phi, n_theta), np.nan)

    # 创建索引数组，加速数据填充
    theta_idx = np.array([theta_map[t] for t in df[theta_col]])
    phi_idx = np.array([phi_map[p] for p in df[phi_col]])

    # 填充已知数据 - 使用高效索引
    rcs_grid[phi_idx, theta_idx] = df[rcs_col].values

    # 创建用于卷积的核
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=float)

    # 存储原始NaN位置
    nan_mask = np.isnan(rcs_grid)
    original_nan_count = np.sum(nan_mask)

    # 使用卷积进行邻域平均 - 最多迭代5次
    for iteration in range(5):
        # 如果没有NaN值，提前结束
        if not np.any(nan_mask):
            break

        # 创建值累加器和计数累加器
        values_sum = np.zeros_like(rcs_grid)
        counts = np.zeros_like(rcs_grid)

        # 处理四个相邻方向
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # 创建移位后的数据和有效性掩码
            shifted_values = np.full_like(rcs_grid, np.nan)
            valid_i_min = max(0, -di)
            valid_i_max = min(n_phi, n_phi - di)
            valid_j_min = max(0, -dj)
            valid_j_max = min(n_theta, n_theta - dj)

            # 计算移位后的索引
            src_i_min = max(0, di)
            src_i_max = min(n_phi, n_phi + di)
            src_j_min = max(0, dj)
            src_j_max = min(n_theta, n_theta + dj)

            # 填充移位后的值
            shifted_values[valid_i_min:valid_i_max, valid_j_min:valid_j_max] = \
                rcs_grid[src_i_min:src_i_max, src_j_min:src_j_max]

            # 累加非NaN值和计数
            valid_values = ~np.isnan(shifted_values)
            values_sum += np.where(valid_values, shifted_values, 0)
            counts += valid_values.astype(int)

        # 计算平均值
        with np.errstate(divide='ignore', invalid='ignore'):
            averages = values_sum / counts

        # 只更新原始NaN位置
        fill_mask = nan_mask & ~np.isnan(averages)
        rcs_grid[fill_mask] = averages[fill_mask]

        # 更新NaN掩码
        nan_mask = np.isnan(rcs_grid)

        # 如果没有新填充的值，跳出循环
        current_nan_count = np.sum(nan_mask)
        if current_nan_count == original_nan_count - np.sum(fill_mask):
            if verbose:
                print(f"  第 {iteration + 1} 次迭代后没有新的填充值")
            break

        if verbose:
            print(f"  第 {iteration + 1} 次迭代: 填充了 {np.sum(fill_mask)} 个NaN值")
        original_nan_count = current_nan_count

    # 如果还有NaN，使用全局平均值填充
    remaining_nans = np.sum(nan_mask)
    if remaining_nans > 0:
        global_mean = np.nanmean(rcs_grid)
        if verbose:
            print(f"  使用全局平均值 {global_mean:.6f} 填充剩余的 {remaining_nans} 个NaN")
        rcs_grid[nan_mask] = global_mean

    # 将处理后的网格数据转回DataFrame
    for idx, row in df.iterrows():
        i = phi_map[row[phi_col]]
        j = theta_map[row[theta_col]]
        df.loc[idx, rcs_col] = rcs_grid[i, j]

    return df


def get_adaptive_rcs_matrix(model_id="001", freq_suffix="1.5G",
                           data_dir=r"F:\data\parameter\csv_output",
                           verbose=True, max_size_warning=500):
    """
    自适应RCS矩阵读取器 - 核心增强功能，支持任意尺寸

    参数:
    model_id: 模型编号
    freq_suffix: 频率后缀
    data_dir: 数据目录
    verbose: 详细输出
    max_size_warning: 矩阵尺寸警告阈值

    返回:
    dict: 包含矩阵数据和详细信息
    """

    # 构建文件路径
    data_file = os.path.join(data_dir, f"{model_id}_{freq_suffix}.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    if verbose:
        print(f"正在加载数据: {data_file}")

    # 读取数据 - 使用多编码支持
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    df = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(data_file, encoding=encoding)
            if verbose:
                print(f"  成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("无法使用任何编码格式读取文件")

    # 检查必需列
    required_cols = ['Theta', 'Phi', 'RCS(Total)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")

    # 数据基本信息
    total_points = len(df)
    if verbose:
        print(f"  原始数据点数: {total_points}")

    # 如果有NaN值，使用现有的填充函数处理
    if df.isna().sum().sum() > 0:
        df = impute_rcs_nans(df, verbose=verbose)

    # 提取角度信息
    theta_values = np.sort(df['Theta'].unique())
    phi_values = np.sort(df['Phi'].unique())

    n_theta = len(theta_values)
    n_phi = len(phi_values)

    # 尺寸检查和警告
    matrix_size = n_theta * n_phi
    if verbose:
        print(f"  检测到矩阵尺寸: {n_theta} × {n_phi} = {matrix_size} 点")
        print(f"  Theta范围: {theta_values.min():.1f}° - {theta_values.max():.1f}°")
        print(f"  Phi范围: {phi_values.min():.1f}° - {phi_values.max():.1f}°")

        if n_theta > 1:
            print(f"  Theta步长: 平均 {np.mean(np.diff(theta_values)):.1f}°")
        if n_phi > 1:
            print(f"  Phi步长: 平均 {np.mean(np.diff(phi_values)):.1f}°")

    # 性能警告
    if max(n_theta, n_phi) > max_size_warning:
        warnings.warn(f"矩阵尺寸较大 ({n_theta}×{n_phi})，可能影响性能")

    # 数据完整性检查
    expected_points = n_theta * n_phi
    if total_points != expected_points:
        if verbose:
            print(f"  警告: 数据点数量不匹配")
            print(f"    期望: {expected_points} 点 (完整网格)")
            print(f"    实际: {total_points} 点")
            print(f"    缺失: {expected_points - total_points} 点")

    # 创建矩阵
    rcs_matrix = np.full((n_theta, n_phi), np.nan)

    # 填充数据 - 使用索引映射提高效率
    theta_map = {theta: i for i, theta in enumerate(theta_values)}
    phi_map = {phi: i for i, phi in enumerate(phi_values)}

    filled_count = 0
    for _, row in df.iterrows():
        theta_idx = theta_map.get(row['Theta'])
        phi_idx = phi_map.get(row['Phi'])

        if theta_idx is not None and phi_idx is not None:
            rcs_matrix[theta_idx, phi_idx] = row['RCS(Total)']
            filled_count += 1

    # 数据填充验证
    if verbose:
        valid_points = np.sum(~np.isnan(rcs_matrix))
        print(f"  成功填充: {filled_count} 点")
        print(f"  矩阵中有效点: {valid_points} 点")

    # 转换为分贝
    rcs_db_matrix = 10 * np.log10(np.maximum(rcs_matrix, 1e-10))

    # 创建网格 - 处理1D数据的特殊情况
    if n_theta > 1 and n_phi > 1:
        phi_grid, theta_grid = np.meshgrid(phi_values, theta_values)
    else:
        # 处理1D数据的特殊情况
        if n_theta == 1:
            phi_grid = phi_values.reshape(1, -1)
            theta_grid = np.full_like(phi_grid, theta_values[0])
        else:  # n_phi == 1
            theta_grid = theta_values.reshape(-1, 1)
            phi_grid = np.full_like(theta_grid, phi_values[0])

    # 统计信息
    valid_mask = ~np.isnan(rcs_matrix)
    data_info = {
        'matrix_shape': rcs_matrix.shape,
        'total_data_points': total_points,
        'expected_points': expected_points,
        'valid_points': np.sum(valid_mask),
        'missing_points': expected_points - np.sum(valid_mask),
        'completeness_ratio': np.sum(valid_mask) / expected_points if expected_points > 0 else 0,
        'theta_count': n_theta,
        'phi_count': n_phi,
        'theta_range': (theta_values.min(), theta_values.max()),
        'phi_range': (phi_values.min(), phi_values.max()),
        'theta_step_mean': np.mean(np.diff(theta_values)) if n_theta > 1 else 0,
        'phi_step_mean': np.mean(np.diff(phi_values)) if n_phi > 1 else 0,
        'rcs_linear_range': (np.nanmin(rcs_matrix), np.nanmax(rcs_matrix)) if np.any(valid_mask) else (np.nan, np.nan),
        'rcs_db_range': (np.nanmin(rcs_db_matrix), np.nanmax(rcs_db_matrix)) if np.any(valid_mask) else (np.nan, np.nan),
        'rcs_linear_mean': np.nanmean(rcs_matrix) if np.any(valid_mask) else np.nan,
        'rcs_db_mean': np.nanmean(rcs_db_matrix) if np.any(valid_mask) else np.nan,
        'is_regular_grid': (total_points == expected_points),
        'is_1d_data': (n_theta == 1 or n_phi == 1),
        'data_density': np.sum(valid_mask) / (n_theta * n_phi)
    }

    if verbose:
        print(f"  数据完整性: {data_info['completeness_ratio']:.1%}")
        if data_info['is_1d_data']:
            print(f"  检测到1D数据")
        if not data_info['is_regular_grid']:
            print(f"  检测到不规则网格")

        if np.any(valid_mask):
            print(f"  RCS线性值范围: {data_info['rcs_linear_range'][0]:.6e} - {data_info['rcs_linear_range'][1]:.6e}")
            print(f"  RCS分贝值范围: {data_info['rcs_db_range'][0]:.1f} - {data_info['rcs_db_range'][1]:.1f} dB")

    return {
        'rcs_linear': rcs_matrix,
        'rcs_db': rcs_db_matrix,
        'theta_values': theta_values,
        'phi_values': phi_values,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid,
        'data_info': data_info
    }


# 保持向后兼容的别名
def get_rcs_matrix(model_id="001", freq_suffix="1.5G", data_dir=r"F:\data\parameter\csv_output"):
    """
    向后兼容的RCS矩阵获取函数
    实际调用 get_adaptive_rcs_matrix
    """
    return get_adaptive_rcs_matrix(model_id, freq_suffix, data_dir, verbose=True)


def load_single_rcs_data(data_dir, model_id, freq_suffix, verbose=True):
    """
    加载指定模型和频率的单个RCS数据文件 - 增强版
    使用自适应读取器，但保持原始API

    参数:
    data_dir: RCS数据目录
    model_id: 模型编号 (如 "001", "002" 等)
    freq_suffix: 频率后缀 (如 "1.5G", "3G")
    verbose: 是否输出详细信息 (默认True)

    返回:
    theta_values: Theta角度值
    phi_values: Phi角度值
    rcs_values: RCS值 (线性值，非分贝)
    """
    # 使用自适应读取器
    data = get_adaptive_rcs_matrix(model_id, freq_suffix, data_dir, verbose)

    # 提取需要的数据，转换为原始API格式
    theta_values = data['theta_values']
    phi_values = data['phi_values']
    rcs_values = data['rcs_linear'].flatten()

    return theta_values, phi_values, rcs_values


def load_rcs_data(rcs_dir, freq_suffix, num_models=100, verbose=True):
    """
    加载RCS数据，处理文件缺失的情况，使用相邻点的平均值填充NaN值，并添加详细的诊断输出
    保持原始API，内部使用增强的自适应读取

    参数:
    rcs_dir: RCS CSV文件目录
    freq_suffix: 频率后缀 (如 "1.5G" 或 "3G")
    num_models: 最大模型数量 (默认100)
    verbose: 是否输出详细信息 (默认True)

    返回:
    rcs_data: 形状为 [available_models, num_angles] 的numpy数组
    theta_values: 唯一的theta角度值
    phi_values: 唯一的phi角度值
    available_models: 成功加载的模型索引列表
    """
    if verbose:
        print(f"\n===== 加载 {freq_suffix} RCS数据 =====")
        print(f"搜索目录: {rcs_dir}")

    # 查找可用的模型文件
    available_models = []
    for i in range(1, num_models + 1):
        model_id = f"{i:03d}"
        file_path = os.path.join(rcs_dir, f"{model_id}_{freq_suffix}.csv")
        if os.path.exists(file_path):
            available_models.append(i)

    if not available_models:
        raise FileNotFoundError(f"在目录 {rcs_dir} 中找不到任何 *_{freq_suffix}.csv 文件")

    if verbose:
        print(f"找到 {len(available_models)} 个可用的 {freq_suffix} 模型文件")

    # 使用第一个可用文件获取角度信息
    first_model = f"{available_models[0]:03d}"
    try:
        first_data = get_adaptive_rcs_matrix(first_model, freq_suffix, rcs_dir, verbose=verbose)
        theta_values = first_data['theta_values']
        phi_values = first_data['phi_values']
        num_angles = len(theta_values) * len(phi_values)

        if verbose:
            print(f"角度信息: {len(theta_values)} × {len(phi_values)} = {num_angles} 个角度点")
    except Exception as e:
        raise ValueError(f"无法读取第一个模型文件 {first_model}: {e}")

    # 初始化RCS数据数组
    rcs_data = np.zeros((len(available_models), num_angles))

    # 循环读取每个可用模型的RCS数据
    if verbose:
        print("\n开始读取RCS数据:")

    successful_models = 0
    for idx, model_num in enumerate(available_models):
        model_id = f"{model_num:03d}"

        try:
            # 使用自适应读取器
            data = get_adaptive_rcs_matrix(model_id, freq_suffix, rcs_dir, verbose=False)
            rcs_values = data['rcs_linear'].flatten()

            if len(rcs_values) == num_angles:
                rcs_data[idx, :] = rcs_values
                successful_models += 1
                if verbose and (idx < 5 or idx % 20 == 0):
                    print(f"  成功加载模型 {model_id}")
            else:
                if verbose:
                    print(f"  警告: 模型 {model_id} 的数据大小不匹配")
                # 尝试适配
                if len(rcs_values) > num_angles:
                    rcs_data[idx, :] = rcs_values[:num_angles]
                else:
                    rcs_data[idx, :len(rcs_values)] = rcs_values
                    rcs_data[idx, len(rcs_values):] = 0

        except Exception as e:
            if verbose:
                print(f"  错误: 无法加载模型 {model_id}: {e}")
            rcs_data[idx, :] = 0

    if verbose:
        print(f"\n成功加载了 {successful_models}/{len(available_models)} 个模型的RCS数据")

    return rcs_data, theta_values, phi_values, available_models


def load_prediction_parameters(param_file, param_names, verbose=True):
    """
    加载要进行预测的设计参数

    参数:
    param_file: 参数文件路径
    param_names: 参数名称列表
    verbose: 是否输出详细信息 (默认True)

    返回:
    预测参数的Numpy数组
    """
    try:
        # 读取参数文件 - 支持多种编码
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
        df = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(param_file, encoding=encoding)
                if verbose:
                    print(f"成功使用 {encoding} 编码读取预测参数文件")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"无法使用任何编码格式读取文件 {param_file}")

        # 检查是否包含所有需要的参数
        missing_params = [p for p in param_names if p not in df.columns]
        if missing_params:
            if verbose:
                print(f"警告: 参数文件缺少以下参数: {missing_params}")
            # 如果参数文件使用不同名称，尝试按顺序匹配
            if len(df.columns) >= len(param_names):
                if verbose:
                    print("使用按列顺序匹配...")
                pred_params = df.iloc[:, :len(param_names)].values
            else:
                raise ValueError("参数不足，无法进行预测")
        else:
            # 按原始参数名称顺序提取参数
            pred_params = df[param_names].values

        if verbose:
            print(f"加载了 {pred_params.shape[0]} 组预测参数")
        return pred_params

    except Exception as e:
        print(f"加载预测参数时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def reshape_rcs_data(rcs_data, theta_values, phi_values):
    """
    将RCS数据从一维向量重塑为二维角度矩阵

    参数:
    rcs_data: 形状为 [num_angles] 的一维RCS数据
    theta_values: theta角度值
    phi_values: phi角度值

    返回:
    rcs_2d: 形状为 [len(theta_values), len(phi_values)] 的二维RCS数据
    """
    return rcs_data.reshape(len(theta_values), len(phi_values))


def convert_rcs_to_db(rcs_values):
    """
    将RCS值从线性值转换为分贝值

    参数:
    rcs_values: RCS线性值

    返回:
    RCS分贝值
    """
    # 确保没有负值或零值
    rcs_values = np.maximum(rcs_values, 1e-10)
    return 10 * np.log10(rcs_values)


def validate_rcs_data(rcs_data, verbose=True):
    """
    验证RCS数据的完整性和有效性

    参数:
    rcs_data: RCS数据数组
    verbose: 是否输出详细信息 (默认True)

    返回:
    validation_result: 验证结果字典
    """
    result = {
        'is_valid': True,
        'nan_count': 0,
        'inf_count': 0,
        'zero_count': 0,
        'negative_count': 0,
        'min_value': None,
        'max_value': None,
        'mean_value': None,
        'std_value': None
    }

    try:
        # 检查NaN值
        result['nan_count'] = np.isnan(rcs_data).sum()

        # 检查Inf值
        result['inf_count'] = np.isinf(rcs_data).sum()

        # 检查零值
        result['zero_count'] = (rcs_data == 0).sum()

        # 检查负值
        result['negative_count'] = (rcs_data < 0).sum()

        # 计算统计值
        valid_mask = ~(np.isnan(rcs_data) | np.isinf(rcs_data))
        if np.any(valid_mask):
            valid_data = rcs_data[valid_mask]
            result['min_value'] = np.min(valid_data)
            result['max_value'] = np.max(valid_data)
            result['mean_value'] = np.mean(valid_data)
            result['std_value'] = np.std(valid_data)

        # 判断数据是否有效
        if result['nan_count'] > 0 or result['inf_count'] > 0:
            result['is_valid'] = False

        if verbose:
            print("RCS数据验证结果:")
            print(f"  数据有效性: {'有效' if result['is_valid'] else '无效'}")
            print(f"  NaN值数量: {result['nan_count']}")
            print(f"  Inf值数量: {result['inf_count']}")
            print(f"  零值数量: {result['zero_count']}")
            print(f"  负值数量: {result['negative_count']}")
            if result['min_value'] is not None:
                print(f"  数据范围: {result['min_value']:.6e} - {result['max_value']:.6e}")
                print(f"  统计信息: 均值={result['mean_value']:.6e}, 标准差={result['std_value']:.6e}")

    except Exception as e:
        result['is_valid'] = False
        if verbose:
            print(f"验证过程中出错: {e}")

    return result


# 新增：自适应可视化函数
def adaptive_plot_2d(data, title_suffix="", figsize=None, save_path=None):
    """
    自适应2D绘图 - 根据数据尺寸调整显示
    """
    try:
        import matplotlib.pyplot as plt

        rcs_db = data['rcs_db']
        theta_values = data['theta_values']
        phi_values = data['phi_values']
        info = data['data_info']

        # 根据数据尺寸调整图像大小
        if figsize is None:
            aspect_ratio = len(phi_values) / len(theta_values)
            if info['is_1d_data']:
                figsize = (12, 4)
            else:
                figsize = (min(12, 6 * aspect_ratio), min(8, 6 / aspect_ratio))

        fig, ax = plt.subplots(figsize=figsize)

        if info['is_1d_data']:
            # 1D数据用线图
            if len(theta_values) == 1:
                ax.plot(phi_values, rcs_db.flatten(), 'o-', linewidth=2, markersize=4)
                ax.set_xlabel('Phi (degrees)')
                ax.set_title(f'RCS vs Phi at Theta={theta_values[0]:.1f}° {title_suffix}')
            else:
                ax.plot(theta_values, rcs_db.flatten(), 'o-', linewidth=2, markersize=4)
                ax.set_xlabel('Theta (degrees)')
                ax.set_title(f'RCS vs Theta at Phi={phi_values[0]:.1f}° {title_suffix}')
            ax.set_ylabel('RCS (dB)')
            ax.grid(True, alpha=0.3)
        else:
            # 2D数据用热图
            im = ax.imshow(rcs_db,
                          extent=[phi_values.min(), phi_values.max(),
                                 theta_values.max(), theta_values.min()],
                          aspect='auto', origin='upper', cmap='jet')

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('RCS (dB)')
            ax.set_xlabel('Phi - Azimuth (degrees)')
            ax.set_ylabel('Theta - Elevation (degrees)')
            ax.set_title(f'RCS Heatmap {info["matrix_shape"]} {title_suffix}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存: {save_path}")

        plt.show()
        return fig, ax

    except ImportError:
        print("警告: matplotlib未安装，无法进行可视化")
        return None, None


if __name__ == "__main__":
    """
    使用示例和测试代码
    """
    print("RCS数据读取模块 - 增强版本")
    print("=" * 50)

    # 示例：使用自适应读取器
    try:
        print("测试自适应RCS矩阵读取...")
        data = get_adaptive_rcs_matrix("001", "1.5G")

        print("\n数据信息:")
        info = data['data_info']
        print(f"  矩阵形状: {info['matrix_shape']}")
        print(f"  数据完整性: {info['completeness_ratio']:.1%}")
        print(f"  是否为1D数据: {info['is_1d_data']}")

        # 示例：验证数据
        validation = validate_rcs_data(data['rcs_linear'])

        print("\n✓ 增强版模块加载完成，所有功能可用!")

    except Exception as e:
        print(f"测试失败: {e}")

    print("\n主要函数:")
    print("- load_parameters(): 加载设计参数")
    print("- get_adaptive_rcs_matrix(): 自适应RCS矩阵读取 (新)")
    print("- load_rcs_data(): 批量加载RCS数据")
    print("- load_single_rcs_data(): 加载单个RCS文件")
    print("- load_prediction_parameters(): 加载预测参数")
    print("- impute_rcs_nans(): NaN值填充")
    print("- validate_rcs_data(): 数据验证")
    print("- convert_rcs_to_db(): 转换为分贝值")
    print("- reshape_rcs_data(): 数据重塑")
    print("- adaptive_plot_2d(): 自适应可视化 (新)")