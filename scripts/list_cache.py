"""
列出所有可用的缓存文件信息

用于快速查看缓存数据，帮助诊断脚本选择正确的缓存
"""

import os
import pickle
from datetime import datetime


def list_cache_files():
    """列出所有缓存文件的详细信息"""
    cache_dir = "cache"

    print("="*80)
    print("RCS Data Cache Information")
    print("="*80)
    print()

    if not os.path.exists(cache_dir):
        print(f"[ERROR] Cache directory '{cache_dir}' not found!")
        print("\nPlease load data in GUI first to create cache files.")
        return

    cache_files = [f for f in os.listdir(cache_dir)
                  if f.startswith('rcs_data_') and f.endswith('.pkl')]

    if not cache_files:
        print(f"[WARNING] No cache files found in '{cache_dir}/' directory")
        print("\nPlease load data in GUI first to create cache files.")
        return

    # 按修改时间排序
    cache_files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)

    print(f"Found {len(cache_files)} cache file(s):\n")

    for idx, cache_file in enumerate(cache_files, 1):
        cache_path = os.path.join(cache_dir, cache_file)

        try:
            # 读取缓存元信息
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            info = cache_data.get('info', {})
            param_data = cache_data.get('param_data')
            rcs_data = cache_data.get('rcs_data')

            # 文件大小
            file_size_mb = os.path.getsize(cache_path) / 1024 / 1024

            # 修改时间
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))

            print(f"[{idx}] {cache_file}")
            print(f"    Created at:    {info.get('created_at', 'Unknown')}")
            print(f"    Modified at:   {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    File size:     {file_size_mb:.2f} MB")

            if param_data is not None:
                print(f"    Param shape:   {param_data.shape}  (samples={param_data.shape[0]})")
            else:
                print(f"    Param shape:   {info.get('param_shape', 'Unknown')}")

            if rcs_data is not None:
                print(f"    RCS shape:     {rcs_data.shape}  (samples={rcs_data.shape[0]})")
                num_freq = rcs_data.shape[-1] if len(rcs_data.shape) >= 4 else 'Unknown'
                print(f"    Frequencies:   {num_freq}")
            else:
                print(f"    RCS shape:     {info.get('rcs_shape', 'Unknown')}")

            print(f"    Signature:     {info.get('signature', 'Unknown')[:16]}...")

            if idx == 1:
                print(f"    [LATEST] <- This will be used by diagnosis script")

            print()

        except Exception as e:
            print(f"[{idx}] {cache_file}")
            print(f"    [ERROR] Failed to read: {e}")
            print()

    print("="*80)
    print("\nTo use a specific cache for diagnosis:")
    print("  (The diagnosis script automatically uses the latest cache)")
    print()


if __name__ == "__main__":
    list_cache_files()
