import os
import shutil
import logging
import pandas as pd
import sys

# ==================== 参数配置 ====================
DIR_REG = r"E:\regression\dicom"
DIR_INFO = r"E:\patients_information\dicom"
EXCEL_PATH = r"E:\regression\患者统计\患者信息.xlsx"
LOG_FILE = "process_error.log"

# ==================== 日志配置 ====================
# 配置日志，同时输出到控制台和日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

def flatten_regression_dicom_dir():
    """
    任务一：将 DIR_REG 下的文件去除患者文件夹这一层级
    """
    logging.info("--- 开始执行任务一：整理 regression 目录结构 ---")

    if not os.path.exists(DIR_REG):
        raise FileNotFoundError(f"目录不存在: {DIR_REG}")

    for item_name in os.listdir(DIR_REG):
        item_path = os.path.join(DIR_REG, item_name)

        # 仅处理文件夹
        if not os.path.isdir(item_path):
            continue

        # 获取该文件夹下的所有子文件夹
        sub_dirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]

        # 如果没有子文件夹，说明该文件夹可能已经是“序列文件夹”（或空文件夹），跳过
        if len(sub_dirs) == 0:
            continue

        # 异常情况 1：患者文件夹中有两个及以上的序列文件夹 -> 报错并暂停
        if len(sub_dirs) >= 2:
            msg = f"[错误] 任务一中断：患者文件夹 '{item_name}' 中包含 {len(sub_dirs)} 个序列文件夹（>=2）！"
            logging.error(msg)
            raise RuntimeError(msg)

        # 正常情况：只有一个序列文件夹
        if len(sub_dirs) == 1:
            seq_folder_name = sub_dirs[0]
            seq_folder_path = os.path.join(item_path, seq_folder_name)
            target_path = os.path.join(DIR_REG, seq_folder_name)

            # 异常情况 2：目标路径已存在同名序列文件夹（重名冲突） -> 报错并暂停，不覆盖
            if os.path.exists(target_path):
                msg = f"[错误] 任务一中断：提取序列时发生冲突，目标路径 '{target_path}' 已存在同名文件夹！"
                logging.error(msg)
                raise RuntimeError(msg)

            try:
                # 移动序列文件夹到外层
                shutil.move(seq_folder_path, target_path)
                # 删除原来变空的患者文件夹
                os.rmdir(item_path)
                logging.info(f"成功提取序列 '{seq_folder_name}'，已删除空患者层级 '{item_name}'。")
            except Exception as e:
                msg = f"[错误] 移动文件夹或删除空文件夹时发生系统异常: {str(e)}"
                logging.error(msg)
                raise RuntimeError(msg)

    logging.info("任务一执行完毕！当前 regression 目录下已全部为序列文件夹。\n")


def find_sequence_in_source(base_dir, target_seq_name):
    """
    辅助函数：在源目录及其子目录中寻找包含 .dcm 的同名序列文件夹
    """
    for root, dirs, files in os.walk(base_dir):
        if target_seq_name in dirs:
            candidate_path = os.path.join(root, target_seq_name)
            # 检查该同名文件夹内是否包含后缀为 .dcm 的文件（不区分大小写）
            contains_dcm = any(f.lower().endswith('.dcm') for f in os.listdir(candidate_path)
                               if os.path.isfile(os.path.join(candidate_path, f)))
            if contains_dcm:
                return candidate_path
    return None


def fill_missing_dicom_from_excel():
    """
    任务二：读取 Excel 表对比，补充缺失的序列文件夹
    """
    logging.info("--- 开始执行任务二：根据 Excel 校验并补充缺失的序列 ---")

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel文件不存在: {EXCEL_PATH}")

    try:
        # 读取 Excel，第一行默认作为表头 (header=0)
        df = pd.read_excel(EXCEL_PATH, header=0)
    except Exception as e:
        msg = f"[错误] 读取 Excel 文件失败: {str(e)}"
        logging.error(msg)
        raise RuntimeError(msg)

    # 提取第7列的数据 (索引为6)
    if df.shape[1] < 7:
        msg = f"[错误] Excel表格列数不足7列！当前列数：{df.shape[1]}"
        logging.error(msg)
        raise RuntimeError(msg)

    # 获取第7列数据，去除空值，转为字符串列表
    seq_names_in_excel = df.iloc[:, 6].dropna().astype(str).tolist()
    # 去除可能包含的头尾空格
    seq_names_in_excel = [name.strip() for name in seq_names_in_excel if name.strip()]

    for seq_name in seq_names_in_excel:
        target_path = os.path.join(DIR_REG, seq_name)

        # 情况 A：回归目录下已存在该序列文件夹 -> 跳过
        if os.path.exists(target_path):
            logging.info(f"序列 '{seq_name}' 已存在于 regression 目录，跳过。")
            continue

        # 情况 B：不存在，需要去 patients_information 下寻找
        logging.info(f"序列 '{seq_name}' 缺失，正在 patients_information 目录中搜索...")
        source_path = find_sequence_in_source(DIR_INFO, seq_name)

        # 异常情况 3：未能在源目录找到或未包含dcm文件 -> 报错并暂停
        if not source_path:
            msg = f"[错误] 任务二中断：在备用图库 '{DIR_INFO}' 中未找到名为 '{seq_name}' 且包含 dcm 文件的序列文件夹！"
            logging.error(msg)
            raise RuntimeError(msg)

        # 找到目标后进行复制（shutil.copytree 默认不会覆盖，如果目标存在会报错，做双重保险）
        try:
            shutil.copytree(source_path, target_path)
            logging.info(f"成功将缺失序列 '{seq_name}' 从源目录复制到 regression 目录。")
        except FileExistsError:
            msg = f"[错误] 复制时目标路径已存在，为了安全中止运行: {target_path}"
            logging.error(msg)
            raise RuntimeError(msg)
        except Exception as e:
            msg = f"[错误] 复制文件夹 '{seq_name}' 时发生异常: {str(e)}"
            logging.error(msg)
            raise RuntimeError(msg)

    logging.info("任务二执行完毕！所有 Excel 中的序列均已对齐。")


if __name__ == "__main__":
    try:
        logging.info("============= 批处理程序启动 =============")
        # 任务一
        flatten_regression_dicom_dir()
        # 任务二
        fill_missing_dicom_from_excel()
        logging.info("============= 全部操作成功完成 =============")

    except Exception as e:
        # 发生任何主动抛出的报错（异常情况1、2、3），程序即刻终止到此处
        logging.error("程序因严重错误已挂起终止。请查看上方日志或 log 文件排查问题。")