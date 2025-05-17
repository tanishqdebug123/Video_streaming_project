import pandas as pd
from datetime import datetime

df = pd.read_csv('5G-4G-5G-1.csv')

time_stamp_col = df.columns[0] 
DL_bitrate_col = df.columns[12]

df['parsed_time'] = pd.to_datetime(df[time_stamp_col],format="%Y.%m.%d_%H.%M.%S")

start_time = df['parsed_time'][0]
df['timestamp'] = (df['parsed_time'] - start_time).dt.total_seconds()

out_df = df[['timestamp', DL_bitrate_col]]
out_df.to_csv('processed_DL_bitrate.csv', index=False)


