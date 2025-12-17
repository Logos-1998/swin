import os
import hashlib
from collections import defaultdict

def calculate_md5(file_path, chunk_size=8192):
    """计算文件的MD5哈希值，用于唯一标识文件内容"""
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        return None

def get_file_signatures(folder_path):
    """
    遍历源文件夹，生成 {md5: folder_path} 的映射
    返回: (signatures_dict, file_count)
    """
    signatures = {}
    count = 0

    # 转换为绝对路径
    abs_folder_path = os.path.abspath(folder_path)

    if not os.path.exists(abs_folder_path):
        return signatures, count

    for root, _, files in os.walk(abs_folder_path):
        for file in files:
            if file.startswith('.'): continue
            full_path = os.path.join(root, file)
            file_hash = calculate_md5(full_path)
            if file_hash:
                signatures[file_hash] = abs_folder_path
                count += 1
    return signatures, count

def analyze_destination(dst_path, source_signatures):
    """
    分析目标文件夹中的文件来源
    """
    abs_dst_path = os.path.abspath(dst_path)

    if not os.path.exists(abs_dst_path):
        print(f"  [错误] 目标文件夹不存在: {abs_dst_path}")
        return

    total_files = 0
    source_stats = defaultdict(int) # {source_path: count}
    unknown_count = 0

    files = [f for f in os.listdir(abs_dst_path) if os.path.isfile(os.path.join(abs_dst_path, f)) and not f.startswith('.')]
    total_files = len(files)

    if total_files == 0:
        print(f"  [空] 目标文件夹为空: {abs_dst_path}")
        print("-" * 60)
        return

    print(f"  正在分析 {len(files)} 个文件...")

    for f in files:
        full_path = os.path.join(abs_dst_path, f)
        f_hash = calculate_md5(full_path)

        if f_hash in source_signatures:
            origin_path = source_signatures[f_hash]
            source_stats[origin_path] += 1
        else:
            unknown_count += 1

    # === 输出报告 ===
    print("-" * 80) # 加长横线以适应长路径
    print(f"目标文件夹: {abs_dst_path}")
    print(f"文件总数: {total_files}")

    # 按数量降序排列输出
    sorted_stats = sorted(source_stats.items(), key=lambda item: item[1], reverse=True)

    for src_path, count in sorted_stats:
        percentage = (count / total_files) * 100
        # 【修改点】这里直接使用 src_path (绝对路径)，不再使用 os.path.basename
        print(f"  - 来自 [ {src_path} ]: {count} 张 ({percentage:.2f}%)")

    if unknown_count > 0:
        percentage = (unknown_count / total_files) * 100
        print(f"  - [未知来源 / 哈希未匹配]: {unknown_count} 张 ({percentage:.2f}%)")

    print("-" * 80)
    print()

def verify_group(group_name, src_list, dst_list):
    """
    处理一组 (A组, B组 或 C组)
    src_list: [path_center, path_p1, path_p2]
    dst_list: [path_dst_center, path_dst_p1, path_dst_p2]
    """
    print(f"====== 开始检查 {group_name} 组数据来源 ======")

    # 1. 建立源文件指纹库 (Hash Map)
    group_signatures = {}
    total_src_files = 0

    print("正在建立源文件索引 (计算MD5)...")
    for src_path in src_list:
        if os.path.exists(src_path):
            sigs, count = get_file_signatures(src_path)
            group_signatures.update(sigs)
            total_src_files += count
            print(f"  - 已索引源: {os.path.abspath(src_path)} ({count} 文件)")
        else:
            print(f"  ! [警告] 源路径不存在: {src_path}")

    print(f"索引完成，共计 {total_src_files} 个源文件指纹。\n")

    # 2. 逐个检查目标文件夹
    for dst_path in dst_list:
        analyze_destination(dst_path, group_signatures)

def main():
    # ================= 路径配置区域 =================
    # 请填入实际绝对路径，保持与生成脚本一致

    # NFH first,desktop second

    # --- A组 ---
    # src_A1 = r"E:\WM\DATA\dataset_original\train\Normal"
    # src_A2 = r"E:\WM\DATA\dataset_original\val\Normal"
    # src_A3 = r"E:\WM\DATA\dataset_original\test\Normal"

    # dst_A1 = r"E:\WM\Swin-Transformer\dataset\train\Normal"
    # dst_A2 = r"E:\WM\Swin-Transformer\dataset\val\Normal"
    # dst_A3 = r"E:\WM\Swin-Transformer\dataset\test\Normal"

    src_A1 = r"D:\Documents\Data\PNG512\dataset\train\Normal"
    src_A2 = r"D:\Documents\Data\PNG512\dataset\val\Normal"
    src_A3 = r"D:\Documents\Data\PNG512\dataset\test\Normal"

    dst_A1 = r"D:\Documents\Swin-Transformer\dataset\train\Normal"
    dst_A2 = r"D:\Documents\Swin-Transformer\dataset\val\Normal"
    dst_A3 = r"D:\Documents\Swin-Transformer\dataset\test\Normal"

    # --- B组 ---
    # src_B1 = r"E:\WM\DATA\dataset_original\train\OP"
    # src_B2 = r"E:\WM\DATA\dataset_original\val\OP"
    # src_B3 = r"E:\WM\DATA\dataset_original\test\OP"

    # dst_B1 = r"E:\WM\Swin-Transformer\dataset\train\OP"
    # dst_B2 = r"E:\WM\Swin-Transformer\dataset\val\OP"
    # dst_B3 = r"E:\WM\Swin-Transformer\dataset\test\OP"

    src_B1 = r"D:\Documents\Data\PNG512\dataset\train\OP"
    src_B2 = r"D:\Documents\Data\PNG512\dataset\val\OP"
    src_B3 = r"D:\Documents\Data\PNG512\dataset\test\OP"

    dst_B1 = r"D:\Documents\Swin-Transformer\dataset\train\OP"
    dst_B2 = r"D:\Documents\Swin-Transformer\dataset\val\OP"
    dst_B3 = r"D:\Documents\Swin-Transformer\dataset\test\OP"

    # --- C组 ---
    # src_C1 = r'E:\WM\DATA\dataset_original\train\OPA'
    # src_C2 = r'E:\WM\DATA\dataset_original\val\OPA'
    # src_C3 = r"E:\WM\DATA\dataset_original\test\OPA"
    #
    # dst_C1 = r"E:\WM\Swin-Transformer\dataset\train\OPA"
    # dst_C2 = r"E:\WM\Swin-Transformer\dataset\val\OPA"
    # dst_C3 = r"E:\WM\Swin-Transformer\dataset\test\OPA"

    src_C1 = r"D:\Documents\Data\PNG512\dataset\train\OPA"
    src_C2 = r"D:\Documents\Data\PNG512\dataset\val\OPA"
    src_C3 = r"D:\Documents\Data\PNG512\dataset\test\OPA"

    dst_C1 = r"D:\Documents\Swin-Transformer\dataset\train\OPA"
    dst_C2 = r"D:\Documents\Swin-Transformer\dataset\val\OPA"
    dst_C3 = r"D:\Documents\Swin-Transformer\dataset\test\OPA"

    # ===============================================

    # 执行 A 组检查
    verify_group("A",
                 [src_A1, src_A2, src_A3],
                 [dst_A1, dst_A2, dst_A3])

    # 执行 B 组检查
    verify_group("B",
                 [src_B1, src_B2, src_B3],
                 [dst_B1, dst_B2, dst_B3])

    # 执行 C 组检查
    verify_group("C",
                 [src_C1, src_C2, src_C3],
                 [dst_C1, dst_C2, dst_C3])

    print("所有检查任务完成。")

if __name__ == "__main__":
    main()