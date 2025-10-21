#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
替换重建方法为委托调用
"""

METHODS = [
    ('_reconstruct_rcs', 2137),
]

def find_method_end(lines, start_idx):
    """找到方法结束位置"""
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        # 新的同级方法或主函数
        if line.startswith('    def ') or line.startswith('def main('):
            return i
    return len(lines)

def replace_method(lines, method_name, start_hint):
    """替换单个方法"""
    # 找到方法开始
    method_start = None
    for i in range(max(0, start_hint - 10), len(lines)):
        if f'    def {method_name}(' in lines[i]:
            method_start = i
            break

    if method_start is None:
        print(f"  [SKIP] {method_name} - not found")
        return lines, 0

    # 找到方法结束
    method_end = find_method_end(lines, method_start)

    # 提取签名 - 这个方法签名很长，可能跨越多行
    signature_lines = [lines[method_start]]
    i = method_start + 1
    # 继续读取直到找到完整的签名（以):结尾）
    while i < method_end and not signature_lines[-1].rstrip().endswith('):'):
        signature_lines.append(lines[i])
        i += 1

    # 查找文档字符串（紧跟在签名后面）
    docstring_lines = []
    if i < method_end and '"""' in lines[i]:
        docstring_lines.append(lines[i])
        i += 1
        while i < method_end:
            docstring_lines.append(lines[i])
            if '"""' in lines[i]:
                break
            i += 1

    # 提取参数 - 这个方法有多个参数
    import re
    full_signature = ''.join(signature_lines)
    # 匹配所有参数（除了self）
    params_match = re.search(r'\(self(?:,\s*([^)]*))?\)', full_signature, re.DOTALL)
    if params_match and params_match.group(1):
        # 清理参数（移除换行和多余空格）
        params = params_match.group(1).strip()
        params = re.sub(r'\s+', ' ', params)  # 压缩空格
        delegation = f"        return self.reconstruction_manager.{method_name}({params})\n"
    else:
        delegation = f"        return self.reconstruction_manager.{method_name}()\n"

    # 构建新方法
    new_method = signature_lines + docstring_lines + [delegation, '\n']

    # 替换
    removed = method_end - method_start
    new_lines = lines[:method_start] + new_method + lines[method_end:]

    print(f"  [OK] {method_name} ({removed} -> {len(new_method)} lines)")
    return new_lines, removed - len(new_method)

def main():
    print("Replacing reconstruction method...")

    with open('gui.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    original = len(lines)
    total_removed = 0

    for method_name, line_hint in METHODS:
        lines, removed = replace_method(lines, method_name, line_hint)
        total_removed += removed

    with open('gui.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    new_count = len(lines)
    print(f"\nDone!")
    print(f"  Original: {original} lines")
    print(f"  New: {new_count} lines")
    print(f"  Removed: {total_removed} ({100*total_removed/original:.1f}%)")

if __name__ == '__main__':
    main()
