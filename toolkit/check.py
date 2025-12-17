import pandas as pd
import os

# ================= 配置区域 =================
# 在这里填入你的 Excel 文件路径
EXCEL_PATH = r'D:\Documents\Swin-Transformer\original_usable_dta.xlsx'

# 要检查的列索引 (第6列对应索引5)
# Python索引从0开始：0=第1列, 1=第2列 ... 5=第6列
TARGET_COL_INDEX = 5
# ===========================================

def check_duplicates():
    print(f"正在读取文件: {EXCEL_PATH}")

    if not os.path.exists(EXCEL_PATH):
        print("错误: 文件不存在，请检查路径。")
        return

    try:
        # 读取 Excel，不设表头，方便按绝对位置索引
        # header=None 表示我们暂时不管表头名字，直接按位置取
        df = pd.read_excel(EXCEL_PATH, header=None)

        # 为了保险，跳过第一行（通常是表头标题），只检查数据行
        # 如果你的第6列连表头标题都重复了，可以把 [1:] 去掉
        data_col = df.iloc[1:, TARGET_COL_INDEX].astype(str).str.strip()

        # 查找重复项
        # keep=False 表示标记所有重复出现的项（比如A出现了两次，这两个都会被标记）
        duplicates = data_col[data_col.duplicated(keep=False)]

        print("-" * 30)
        print(f"正在检查第 {TARGET_COL_INDEX + 1} 列...")
        print(f"数据总行数: {len(data_col)}")

        if duplicates.empty:
            print("✅ 完美！没有发现重复项。")
        else:
            print(f"❌ 发现重复项！共涉及 {len(duplicates)} 行。")
            print("-" * 30)
            print("具体的重复内容及出现次数如下：")
            print(duplicates.value_counts())

            print("-" * 30)
            print("建议：请手动打开 Excel，使用 Ctrl+F 搜索上述 ID 进行修正。")

    except Exception as e:
        print(f"发生错误: {e}")
        # 如果列数不够，可能会报错，提示一下
        print("提示: 请确认 Excel 文件是否有第 6 列。")

if __name__ == "__main__":
    check_duplicates()