#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
替换评估方法为委托调用
"""

METHODS = [
    ('_evaluate_traditional_model', 2131),
    ('_evaluate_autoencoder_model', 2413),
    ('_update_evaluation_display', 2535),
    ('_display_autoencoder_results', 2549),
    ('_display_traditional_results', 2577),
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

    # 提取签名和文档字符串
    signature_line = lines[method_start]

    # 查找文档字符串
    docstring_lines = []
    i = method_start + 1
    if i < method_end and '"""' in lines[i]:
        docstring_lines.append(lines[i])
        i += 1
        while i < method_end:
            if '"""' in lines[i-1] and lines[i-1].count('"""') == 2:
                break
            if '"""' in lines[i]:
                docstring_lines.append(lines[i])
                break
            docstring_lines.append(lines[i])
            i += 1

    # 提取参数
    import re
    params_match = re.search(r'\(self(?:,\s*([^)]*))?\)', signature_line)
    if params_match and params_match.group(1):
        params = params_match.group(1).strip()
        delegation = f"        return self.evaluation_manager.{method_name}({params})\n"
    else:
        delegation = f"        return self.evaluation_manager.{method_name}()\n"

    # 构建新方法
    new_method = [signature_line] + docstring_lines + [delegation, '\n']

    # 替换
    removed = method_end - method_start
    new_lines = lines[:method_start] + new_method + lines[method_end:]

    print(f"  [OK] {method_name} ({removed} -> {len(new_method)} lines)")
    return new_lines, removed - len(new_method)

def main():
    print("Replacing evaluation methods...")

    with open('gui.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    original = len(lines)
    total_removed = 0

    # 从后往前替换
    for method_name, line_hint in reversed(METHODS):
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
