import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 您的数据集根目录 (包含 train/ 和 val/ 文件夹)
# DATA_ROOT = r'D:\Documents\Swin-Transformer\dataset'
DATA_ROOT = r'D:\Documents\Swin-Transformer\dataset'

# 您的临床信息 Excel 文件路径
EXCEL_PATH = r'D:\Documents\Swin-Transformer\original_usable_dta.xlsx'

# 输出的 CSV 文件路径
OUTPUT_CSV = './clinical_data.csv'

# Excel 列索引配置 (注意：Python索引从0开始，所以第6列是索引5)
# "Excel第六列是患者编号...第二列是患者年龄，第三列是患者性别...第四列是患者身高，第五列是患者体重"
COL_IDX = {
    'id': 5,      # 第6列
    'age': 1,     # 第2列
    'gender': 2,  # 第3列
    'height': 3,  # 第4列
    'weight': 4   # 第5列
}

# 性别映射字典
GENDER_MAP = {
    '男': 1,
    '女': 0,
    'Male': 1,
    'Female': 0,
    # 如果excel里有空格或大小写不一致，可以在代码里进一步处理
}
# ===========================================

def parse_patient_id(filename):
    """
    文件名解析函数
    输入: 'A10014790_L3_-2.8_275.png'
    输出: 'A10014790'
    """
    try:
        # 去掉路径，只留文件名
        basename = os.path.basename(filename)
        # 按下划线分割，取第一部分
        patient_id = basename.split('_')[0]
        return patient_id.strip()
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

def load_and_clean_excel(path):
    print(f"Loading clinical data from {path}...")
    try:
        # 读取Excel，header=0表示第一行是表头
        df_raw = pd.read_excel(path, header=0)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return None

    # 创建一个新的干净的 DataFrame
    clinical_df = pd.DataFrame()

    # 1. 提取 ID (强制转为字符串并去除空格，确保匹配准确)
    clinical_df['patient_id'] = df_raw.iloc[:, COL_IDX['id']].astype(str).str.strip()

    # 2. 提取并清洗 Age, Height, Weight (强制转为数字，非数字变为 NaN)
    for col_name in ['age', 'height', 'weight']:
        original_col_idx = COL_IDX[col_name]
        clinical_df[col_name] = pd.to_numeric(df_raw.iloc[:, original_col_idx], errors='coerce')

    # 3. 提取并清洗 Gender
    gender_raw = df_raw.iloc[:, COL_IDX['gender']].astype(str).str.strip()
    # 使用 map 进行转换，找不到的会变成 NaN
    clinical_df['gender'] = gender_raw.map(GENDER_MAP)

    # 4. 缺失值处理
    # 打印缺失情况
    print("\nMissing values before filling:")
    print(clinical_df.isnull().sum())

    # 对数值型特征用均值填充
    for col in ['age', 'height', 'weight']:
        mean_val = clinical_df[col].mean()
        clinical_df[col].fillna(mean_val, inplace=True)

    # 对性别用众数填充 (或者您可以选择删除)
    mode_gender = clinical_df['gender'].mode()[0]
    clinical_df['gender'].fillna(mode_gender, inplace=True)

    # 设 ID 为索引，方便后续查询
    clinical_df.set_index('patient_id', inplace=True)

    print(f"\nLoaded {len(clinical_df)} unique patients from Excel.")
    return clinical_df

def process_images(data_root, clinical_df):
    results = []
    # 支持的图像扩展名
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']

    # 遍历 train 和 val 文件夹
    # 假设结构是 data_root/train/class_x/xxx.png
    image_paths = []
    for ext in exts:
        # recursive=True 会搜索所有子文件夹
        image_paths.extend(glob.glob(os.path.join(data_root, '**', ext), recursive=True))

    print(f"\nFound {len(image_paths)} images. Processing...")

    missing_ids = set()

    for img_path in tqdm(image_paths):
        # 1. 解析文件名获取 ID
        pid = parse_patient_id(img_path)
        if pid is None:
            continue

        # 2. 在 Excel 表中查找
        if pid in clinical_df.index:
            info = clinical_df.loc[pid]

            # 3. 获取标签 (假设父文件夹名就是标签名，或者是 train/class_name 结构)
            # 这里我们简单记录父文件夹名字，后面Dataset类可以用 ImageFolder 的 class_to_idx 逻辑
            # 或者如果您的数据集已经是标准的 ImageFolder 结构，我们主要需要路径

            # 使用相对路径，方便移植
            rel_path = os.path.relpath(img_path, data_root)

            # 构建该样本的数据条目
            entry = {
                'image_path': rel_path,  # 关键索引
                'patient_id': pid,
                'age': info['age'],
                'gender': info['gender'],
                'height': info['height'],
                'weight': info['weight']
            }
            results.append(entry)
        else:
            missing_ids.add(pid)

    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} patient IDs from images were NOT found in Excel.")
        print(f"First 5 missing IDs: {list(missing_ids)[:5]}")
        print("These images will be skipped.")

    return pd.DataFrame(results)

def main():
    # 1. 加载并清洗 Excel
    clinical_df = load_and_clean_excel(EXCEL_PATH)
    if clinical_df is None:
        return

    # 2. 遍历图像并关联数据
    final_df = process_images(DATA_ROOT, clinical_df)

    # 3. 保存结果
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccess! CSV saved to {OUTPUT_CSV}")
    print(f"Total samples: {len(final_df)}")

    # 4. 计算并打印统计信息 (用于后续归一化)
    print("\n" + "="*40)
    print("STATISTICS FOR NORMALIZATION (Save these!)")
    print("="*40)

    stats = final_df[['age', 'height', 'weight']].agg(['mean', 'std'])
    print(stats)

    print("\nExample config usage:")
    print(f"_C.DATA.TABULAR_MEAN = {stats.loc['mean'].tolist()}")
    print(f"_C.DATA.TABULAR_STD  = {stats.loc['std'].tolist()}")

if __name__ == '__main__':
    main()