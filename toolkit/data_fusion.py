import os
import shutil
import random
import time
from datetime import datetime

def get_file_list(folder_path):
    """获取文件夹下所有文件列表"""
    if not os.path.exists(folder_path):
        return []
    # 获取绝对路径以确保后续输出准确
    folder_path = os.path.abspath(folder_path)
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]

def check_and_create_dir(folder_path):
    """
    检查文件夹是否存在，不存在则新建。
    并返回该文件夹的绝对路径。
    """
    abs_path = os.path.abspath(folder_path)
    if not os.path.exists(abs_path):
        try:
            os.makedirs(abs_path)
            print(f"  - [系统] 检测到目标不存在，已自动新建: {abs_path}")
        except OSError as e:
            print(f"  ! [错误] 无法创建文件夹 {abs_path}: {e}")
    return abs_path

def safe_copy(src_full_path, dst_folder, file_name):
    """
    安全复制：将文件从 src_full_path 复制到 dst_folder。
    若重名则自动重命名。
    """
    dst_path = os.path.join(dst_folder, file_name)

    # 冲突处理
    if os.path.exists(dst_path):
        name, ext = os.path.splitext(file_name)
        timestamp = str(int(time.time() * 1000000))[-6:]
        rand_suffix = random.randint(100, 999)
        new_name = f"{name}_copy_{timestamp}_{rand_suffix}{ext}"
        dst_path = os.path.join(dst_folder, new_name)
        # 递归检查
        if os.path.exists(dst_path):
            return safe_copy(src_full_path, dst_folder, new_name)

    try:
        shutil.copy2(src_full_path, dst_path)
    except Exception as e:
        print(f"  ! 复制失败: {src_full_path} -> {e}")

def process_group_safe(src_center, src_p1, src_p2,
                       dst_center, dst_p1, dst_p2,
                       proportion, seed_suffix):
    """
    非破坏性处理单组任务。
    src_xxx: 源文件夹路径 (只读)
    dst_xxx: 目标文件夹路径 (写入结果，若无则新建)
    """
    # 获取绝对路径用于显示
    src_center_abs = os.path.abspath(src_center)

    print(f"正在处理组: [{src_center_abs}] ...")

    # 0. 判定并创建目标文件夹 (Requirement #2)
    # 将路径转换为绝对路径方便后续处理
    dst_center = check_and_create_dir(dst_center)
    dst_p1 = check_and_create_dir(dst_p1)
    dst_p2 = check_and_create_dir(dst_p2)

    # 1. 设置随机种子
    base_time_str = datetime.now().strftime("%Y%m%d%H%M")
    full_seed_str = base_time_str + str(seed_suffix)
    seed_value = int(full_seed_str)
    random.seed(seed_value)
    print(f"  - 随机种子: {seed_value} | 比例: {proportion}")

    # 2. 读取源文件列表
    files_center = get_file_list(src_center) # A1
    files_p1 = get_file_list(src_p1)         # A2
    files_p2 = get_file_list(src_p2)         # A3

    # 验证非空
    if not all([files_center, files_p1, files_p2]):
        print("  ! 错误: 源文件夹为空或路径错误，跳过本组。")
        return

    # 3. 计算数量
    count_x = int(len(files_p1) * proportion) # 需从 A2 抽取的数量
    count_y = int(len(files_p2) * proportion) # 需从 A3 抽取的数量

    # 4. 模拟逻辑分裂 (内存操作)
    random.shuffle(files_center)
    mid_point = len(files_center) // 2

    # A1 前半区 (用于和 A2 交换)
    center_pool_1 = files_center[:mid_point]
    # A1 后半区 (用于和 A3 交换)
    center_pool_2 = files_center[mid_point:]

    # 5. 数量预检
    if len(center_pool_1) < count_x or len(center_pool_2) < count_y:
        print(f"  ! 终止: {os.path.basename(src_center)} 库存不足以进行交换。")
        return

    # 6. 确定交换名单 (List of filenames)
    swap_out_c1 = random.sample(center_pool_1, count_x) # A1 -> A2
    swap_out_p1 = random.sample(files_p1, count_x)      # A2 -> A1

    swap_out_c2 = random.sample(center_pool_2, count_y) # A1 -> A3
    swap_out_p2 = random.sample(files_p2, count_y)      # A3 -> A1

    # 7. 构建目标清单 (Origin File Path -> Target Folder)
    copy_tasks = []

    # --- 构建 Target A1 (中心) ---
    # A1 剩余 + A2 换入 + A3 换入
    remain_c1 = set(center_pool_1) - set(swap_out_c1)
    remain_c2 = set(center_pool_2) - set(swap_out_c2)

    for f in remain_c1: copy_tasks.append((os.path.join(src_center, f), dst_center))
    for f in remain_c2: copy_tasks.append((os.path.join(src_center, f), dst_center))
    for f in swap_out_p1: copy_tasks.append((os.path.join(src_p1, f), dst_center))
    for f in swap_out_p2: copy_tasks.append((os.path.join(src_p2, f), dst_center))

    # --- 构建 Target A2 (P1) ---
    # A2 剩余 + A1 换入
    remain_p1 = set(files_p1) - set(swap_out_p1)
    for f in remain_p1: copy_tasks.append((os.path.join(src_p1, f), dst_p1))
    for f in swap_out_c1: copy_tasks.append((os.path.join(src_center, f), dst_p1))

    # --- 构建 Target A3 (P2) ---
    # A3 剩余 + A1 换入
    remain_p2 = set(files_p2) - set(swap_out_p2)
    for f in remain_p2: copy_tasks.append((os.path.join(src_p2, f), dst_p2))
    for f in swap_out_c2: copy_tasks.append((os.path.join(src_center, f), dst_p2))

    # 8. 执行物理复制
    print(f"  - 逻辑构建完成，开始执行 {len(copy_tasks)} 个文件的复制操作...")
    for src_path, target_dir in copy_tasks:
        f_name = os.path.basename(src_path)
        safe_copy(src_path, target_dir, f_name)

    # 9. 输出具体变动数字 (Requirement #1: 完整绝对路径)
    print("-" * 30)

    # Target Center (A1) 报告
    print(f"  [统计报告] 目标: {dst_center}")
    print(f"    - 保留自 {os.path.abspath(src_center)}: {len(remain_c1) + len(remain_c2)}")
    print(f"    - 来自 {os.path.abspath(src_p1)}: {len(swap_out_p1)}")
    print(f"    - 来自 {os.path.abspath(src_p2)}: {len(swap_out_p2)}")
    print(f"    = 当前总计: {len(get_file_list(dst_center))}")

    # Target P1 (A2) 报告
    print(f"\n  [统计报告] 目标: {dst_p1}")
    print(f"    - 保留自 {os.path.abspath(src_p1)}: {len(remain_p1)}")
    print(f"    - 来自 {os.path.abspath(src_center)}: {len(swap_out_c1)}")
    print(f"    = 当前总计: {len(get_file_list(dst_p1))}")

    # Target P2 (A3) 报告
    print(f"\n  [统计报告] 目标: {dst_p2}")
    print(f"    - 保留自 {os.path.abspath(src_p2)}: {len(remain_p2)}")
    print(f"    - 来自 {os.path.abspath(src_center)}: {len(swap_out_c2)}")
    print(f"    = 当前总计: {len(get_file_list(dst_p2))}")

    print("=" * 50)

def main():
    # ================= 路径配置区域 =================
    # 请填入实际绝对路径 (Windows注意加 r)

    # NFH first,desktop second

    # --- A组 ---
    src_A1 = r"E:\WM\DATA\dataset_original\train\Normal"
    src_A2 = r"E:\WM\DATA\dataset_original\val\Normal"
    src_A3 = r"E:\WM\DATA\dataset_original\test\Normal"

    dst_A1 = r"E:\WM\Swin-Transformer\dataset\train\Normal"
    dst_A2 = r"E:\WM\Swin-Transformer\dataset\val\Normal"
    dst_A3 = r"E:\WM\Swin-Transformer\dataset\test\Normal"

    # src_A1 = r"D:\Documents\Data\PNG512\dataset\train\Normal"
    # src_A2 = r"D:\Documents\Data\PNG512\dataset\val\Normal"
    # src_A3 = r"D:\Documents\Data\PNG512\dataset\test\Normal"
    #
    # dst_A1 = r"D:\Documents\Swin-Transformer\dataset\train\Normal"
    # dst_A2 = r"D:\Documents\Swin-Transformer\dataset\val\Normal"
    # dst_A3 = r"D:\Documents\Swin-Transformer\dataset\test\Normal"

    # --- B组 ---
    src_B1 = r"E:\WM\DATA\dataset_original\train\OP"
    src_B2 = r"E:\WM\DATA\dataset_original\val\OP"
    src_B3 = r"E:\WM\DATA\dataset_original\test\OP"

    dst_B1 = r"E:\WM\Swin-Transformer\dataset\train\OP"
    dst_B2 = r"E:\WM\Swin-Transformer\dataset\val\OP"
    dst_B3 = r"E:\WM\Swin-Transformer\dataset\test\OP"

    # src_B1 = r"D:\Documents\Data\PNG512\dataset\train\OP"
    # src_B2 = r"D:\Documents\Data\PNG512\dataset\val\OP"
    # src_B3 = r"D:\Documents\Data\PNG512\dataset\test\OP"
    #
    # dst_B1 = r"D:\Documents\Swin-Transformer\dataset\train\OP"
    # dst_B2 = r"D:\Documents\Swin-Transformer\dataset\val\OP"
    # dst_B3 = r"D:\Documents\Swin-Transformer\dataset\test\OP"

    # --- C组 ---
    src_C1 = r'E:\WM\DATA\dataset_original\train\OPA'
    src_C2 = r'E:\WM\DATA\dataset_original\val\OPA'
    src_C3 = r"E:\WM\DATA\dataset_original\test\OPA"

    dst_C1 = r"E:\WM\Swin-Transformer\dataset\train\OPA"
    dst_C2 = r"E:\WM\Swin-Transformer\dataset\val\OPA"
    dst_C3 = r"E:\WM\Swin-Transformer\dataset\test\OPA"

    # src_C1 = r"D:\Documents\Data\PNG512\dataset\train\OPA"
    # src_C2 = r"D:\Documents\Data\PNG512\dataset\val\OPA"
    # src_C3 = r"D:\Documents\Data\PNG512\dataset\test\OPA"
    #
    # dst_C1 = r"D:\Documents\Swin-Transformer\dataset\train\OPA"
    # dst_C2 = r"D:\Documents\Swin-Transformer\dataset\val\OPA"
    # dst_C3 = r"D:\Documents\Swin-Transformer\dataset\test\OPA"

    # ================= 任务列表 =================
    # 格式: (源中, 源1, 源2, 目标中, 目标1, 目标2, 比例, 种子后缀)

    tasks = [
        # A组
        (src_A1, src_A2, src_A3, dst_A1, dst_A2, dst_A3, 1, "01"),
        # B组
        (src_B1, src_B2, src_B3, dst_B1, dst_B2, dst_B3, 1, "02"),
        # C组
        (src_C1, src_C2, src_C3, dst_C1, dst_C2, dst_C3, 1, "03"),
    ]

    # ===========================================

    print(f"=== 开始执行非破坏性混合复制 ===")
    print(f"时间基准: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    for args in tasks:
        process_group_safe(*args)

    print("所有任务执行完毕。")

if __name__ == "__main__":
    main()