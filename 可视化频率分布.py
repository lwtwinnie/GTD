import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 打开GTDdata.csv文件
df = pd.read_csv('GTDdata.csv', encoding='latin-1')


# 2. 筛选出country取值为200的行
df_country_200 = df[df['region'] == 10]

# 3. 将数据网格化
# latitude变成[32:0.5:38]
# longitude变成[35:0.5:43]
lat_bins = np.arange(10, 45.5, 0.5)
lon_bins = np.arange(0, 70, 0.5)

# 4. 创建二维直方图
hist, x_edges, y_edges = np.histogram2d(df_country_200['latitude'], df_country_200['longitude'], bins=[lat_bins, lon_bins])

# 5. 绘制热力图
plt.figure(figsize=(10, 8))
plt.pcolormesh(x_edges, y_edges, hist.T, cmap='hot')
plt.title('Heatmap of Incidents (Country Code 200)')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.colorbar(label='Incident Count')
plt.show()
