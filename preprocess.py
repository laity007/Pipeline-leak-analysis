import os
import pandas as pd
import numpy as np

def time_to_seconds(time_str):
    """
    将类似 '09:21.8' 的字符串转为秒数
    假设格式为 MM:SS.s 或 HH:MM:SS.s
    """
    parts = time_str.split(":")
    if len(parts) == 2:  # MM:SS.s
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS.s
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Unrecognized time format: {time_str}")

def preprocess_csv(input_path, output_path):
    # 读入原始文件
    df = pd.read_csv(input_path)

    # 提取三路时间列和三路幅值列
    time_cols = [c for c in df.columns if "时间" in c]
    value_cols = [c for c in df.columns if "幅值" in c]

    if len(time_cols) == 0 or len(value_cols) < 3:
        raise ValueError(f"{input_path} 文件格式不符合要求")

    # 取第一列时间作为统一时间
    t_str = df[time_cols[0]].astype(str).to_list()
    t_sec = np.array([time_to_seconds(ts) for ts in t_str])
    # 转换为相对秒（从0开始）
    t_sec = t_sec - t_sec[0]

    # 提取三路幅值
    channels = []
    for col in value_cols[:3]:
        channels.append(df[col].to_numpy(dtype=float))

    # 构建标准化DataFrame
    out_df = pd.DataFrame({
        "time_sec": t_sec,
        "channel_1": channels[0],
        "channel_2": channels[1],
        "channel_3": channels[2],
    })

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"已处理并保存: {output_path}")

def main():
    input_folder = os.path.join(os.path.dirname(__file__), "data")
    output_folder = os.path.join(os.path.dirname(__file__), "data2")
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".csv"):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            try:
                preprocess_csv(in_path, out_path)
            except Exception as e:
                print(f"处理 {fname} 时出错: {e}")

if __name__ == "__main__":
    main()
