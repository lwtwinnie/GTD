import pandas as pd
import numpy as np

# 打开GTDdata.csv文件
df = pd.read_csv('GTDdata.csv', encoding='latin-1')

# 筛选出country取值为200的行
df_country_200 = df[df['country'] == 200]

# 将latitude变成[32:0.5:38]，即从32到38步长为0.5一组的数据
# longitude变成[35:0.5:43]，即从35到43步长为0.5一组的数据
lat_bins = np.arange(32, 38.5, 0.5)
lon_bins = np.arange(35, 43.5, 0.5)

# 计算每个格子的扩散系数大小
def cal_spread(grid1, grid2):
    ans = 0
    for index1, row1 in grid1.iterrows():
        for index2, row2 in grid2.iterrows():
            days_diff = abs((row1['imonth'] * 30 + row1['iday']) - (row2['imonth'] * 30 + row2['iday']))
            if days_diff <= 7:
                ans += 1
    return ans

# 创建一个二维列表来保存每个格子的DataFrame
grids = [[pd.DataFrame() for _ in range(len(lon_bins) - 1)] for _ in range(len(lat_bins) - 1)]

# 对于每个事件，将其添加到相应的格子DataFrame中
for index, row in df_country_200.iterrows():
    if not pd.isnull(row['latitude']) and not pd.isnull(row['longitude']):
        lat_index = int((row['latitude'] - 32) / 0.5)
        lon_index = int((row['longitude'] - 35) / 0.5)
        grids[lat_index][lon_index] = pd.concat([grids[lat_index][lon_index], row.to_frame().transpose()], ignore_index=True)

# 计算每个格子的扩散系数大小并打印输出
for i in range(len(lat_bins) - 1):
    for j in range(len(lon_bins) - 1):
        spread = 0
        if i > 0:
            spread += cal_spread(grids[i][j], grids[i - 1][j]) # 上方格子
            if j > 0:
                spread += cal_spread(grids[i][j], grids[i - 1][j - 1]) # 左上方格子
            if j < len(lon_bins) - 2:
                spread += cal_spread(grids[i][j], grids[i - 1][j + 1]) # 右上方格子
        if j > 0:
            spread += cal_spread(grids[i][j], grids[i][j - 1]) # 左方格子
        if j < len(lon_bins) - 2:
            spread += cal_spread(grids[i][j], grids[i][j + 1]) # 右方格子
        if i < len(lat_bins) - 2:
            spread += cal_spread(grids[i][j], grids[i + 1][j]) # 下方格子
            if j > 0:
                spread += cal_spread(grids[i][j], grids[i + 1][j - 1]) # 左下方格子
            if j < len(lon_bins) - 2:
                spread += cal_spread(grids[i][j], grids[i + 1][j + 1]) # 右下方格子
        print(f"Grid [{lat_bins[i]}:{lat_bins[i + 1]}, {lon_bins[j]}:{lon_bins[j + 1]}]: Spread coefficient = {spread}")
