import os
import glob
from collections import defaultdict

# ================= 配置区域 =================
# 搜索的根目录
SEARCH_ROOT = r'E:\WM\Swin-Transformer\dataset'

# 目标匹配规则 (必须同时满足以下所有条件)
TARGET_PROFILE = {
    "L1": -1.0,
    "L2": -1.4,
    "L3": -1.4,
    "L4": -1.7
}

# 浮点数比较的误差容忍度 (防止 -1.7000001 和 -1.7 不匹配的情况)
TOLERANCE = 0.001
# ===========================================

def is_float_match(val1, val2, tol=TOLERANCE):
    """判断两个浮点数是否足够接近"""
    return abs(val1 - val2) < tol

def parse_filename(filename):
    """
    解析文件名
    输入: A10014790_L2_-3.0_301.png
    输出: (id='A10014790', segment='L2', value=-3.0)
    """
    try:
        name_no_ext, _ = os.path.splitext(filename)
        parts = name_no_ext.split('_')

        if len(parts) < 3: return None

        pid = parts[0]      # A10014790
        segment = parts[1]  # L2
        val_str = parts[2]  # -3.0

        # 尝试将数值转换为 float
        value = float(val_str)

        return pid, segment, value
    except ValueError:
        return None
    except Exception:
        return None

def main():
    print(f"正在扫描目录: {SEARCH_ROOT} ...")
    print(f"寻找的目标特征 (必须全部符合): {TARGET_PROFILE}")
    print("-" * 50)

    # 1. 数据收集阶段
    # 结构: data_store['A1001'] = {'L1': -1.0, 'L2': -2.5, ...}
    patient_data = defaultdict(dict)

    # 遍历所有文件
    # 使用 glob 递归查找所有图片，效率也不错
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files_found = 0

    for root, dirs, files in os.walk(SEARCH_ROOT):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                res = parse_filename(f)
                if res:
                    pid, seg, val = res
                    # 只记录我们关心的节段 (L1-L4)
                    if seg in TARGET_PROFILE:
                        patient_data[pid][seg] = val
                        files_found += 1

    print(f"扫描结束。共解析了 {files_found} 个相关(L1-L4)文件，涉及 {len(patient_data)} 名患者。")
    print("正在筛选符合条件的患者...")
    print("-" * 50)

    # 2. 筛选阶段
    matched_patients = []

    for pid, records in patient_data.items():
        # 检查是否包含所有需要的节段
        # 比如 TARGET_PROFILE 需要 L1,L2,L3,L4，那么这个人的 records 里必须都有
        missing_segments = []
        is_full_match = True

        for target_seg, target_val in TARGET_PROFILE.items():
            # 检查该人是否有这个节段的数据
            if target_seg not in records:
                is_full_match = False
                break # 缺片子，直接淘汰

            # 检查数值是否匹配
            actual_val = records[target_seg]
            if not is_float_match(actual_val, target_val):
                is_full_match = False
                break # 数值对不上，淘汰

        if is_full_match:
            matched_patients.append(pid)

    # 3. 输出结果
    if not matched_patients:
        print("未找到任何同时满足这4个条件的患者。")
    else:
        print(f"🎉 找到 {len(matched_patients)} 名完全匹配的患者！")
        for pid in matched_patients:
            print(f"\n[目标锁定] 患者编号: {pid}")
            print(f"    检查数据详情:")
            # 把这个人的数据打印出来验证一下
            for seg in sorted(TARGET_PROFILE.keys()):
                val = patient_data[pid][seg]
                target = TARGET_PROFILE[seg]
                print(f"    - {seg}: {val} (目标: {target}) -> √")

if __name__ == "__main__":
    main()