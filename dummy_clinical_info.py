import pandas as pd
import numpy as np
from timebase.data.static import DICT_STATE, DICT_TIME

# 这是 LABEL_COLS 的完整定义
LABEL_COLS = [
    "Sub_ID",
    "age",
    "sex",
    "status",
    "time",
    "Session_Code",
    "YMRS1",
    "YMRS2",
    "YMRS3",
    "YMRS4",
    "YMRS5",
    "YMRS6",
    "YMRS7",
    "YMRS8",
    "YMRS9",
    "YMRS10",
    "YMRS11",
    "YMRS_SUM",
    "HDRS1",
    "HDRS2",
    "HDRS3",
    "HDRS4",
    "HDRS5",
    "HDRS6",
    "HDRS7",
    "HDRS8",
    "HDRS9",
    "HDRS10",
    "HDRS11",
    "HDRS12",
    "HDRS13",
    "HDRS14",
    "HDRS15",
    "HDRS16",
    "HDRS17",
    "HDRS_SUM",
    "IPAQ_total",
    "YMRS_discretized",
    "HDRS_discretized",
]

def get_dummy_clinical_info():
    # 创建一个符合 LABEL_COLS 结构的字典来生成虚拟数据
    dummy_data = {
        # 基础信息 (这些是你在 Excel 中需要填写的)
        'Sub_ID': [f'{i:02d}' for i in range(1, 10+1)],
        'age': [28, 28, 28, 45, 45, 33, 51, 51, 29, 38],
        'sex': [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        'status': np.random.choice(list(DICT_STATE.keys()), 10), # 模拟被 DICT_STATE 替换后的数字
        'time': np.random.choice(list(DICT_TIME.keys()), 10),       # 模拟被 DICT_TIME 替换后的数字
        'Session_Code': [f'data/raw_data/WESAD/S{i}/S{i}_E4_Data' for i in range(2, 10+2)], # 模拟被脚本修改后的路径
        'IPAQ_total': np.random.choice([1500.0, 2100.0, -9.0], 10), # -9.0 代表缺失值
    }

    # 填充 YMRS 和 HDRS 的各项分数 (这些是你在 Excel 中需要填写的)
    for i in range(1, 12):
        dummy_data[f'YMRS{i}'] = np.random.randint(0, 5, 10)
    for i in range(1, 18):
        dummy_data[f'HDRS{i}'] = np.random.randint(0, 5, 10)

    # 填充脚本会覆盖的列 (在返回的 DataFrame 中，这些值是脚本计算后的)
    dummy_data['YMRS_SUM'] = np.random.randint(5, 40, 10).astype(float)
    dummy_data['HDRS_SUM'] = np.random.randint(3, 35, 10).astype(float)
    dummy_data['YMRS_discretized'] = np.random.randint(0, 5, 10).astype(float)
    dummy_data['HDRS_discretized'] = np.random.randint(0, 5, 10).astype(float)


    # 创建 DataFrame 并确保列的顺序与 LABEL_COLS 一致
    returned_dataframe_example = pd.DataFrame(dummy_data)
    returned_dataframe_example = returned_dataframe_example[LABEL_COLS]

    return returned_dataframe_example


# # 打印示例 DataFrame 的前几行，这就是 read 函数返回的最终样子
# print("`read` function return value example (first 5 rows):")
# print(returned_dataframe_example.head().to_string())