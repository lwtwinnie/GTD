# 数据准备
# 目的：提取和清理协变量和恐怖主义（GTD）数据，此外，该代码还生成了一个离散化的研究区域

# 将图保存到以下路径
figpath = "results/figs"


import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

# # #读取世界地理信息数据
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# 绘制没有grid的世界地图
# world.plot()

# 绘制有grid的世界地图
# 设置网格大小，这里使用经度和纬度的间隔来定义网格大小
lon_step = 2  # 经度间隔
lat_step = 2  # 纬度间隔

# 将世界地图网格化
# 将世界地图网格化
world_grid = gpd.overlay(world, world, how='intersection', make_valid=True, keep_geom_type=False)

# world_grid = gpd.overlay(world, world, how='intersection', make_valid=True)
# 绘制网格化后的世界地图
# world_grid.plot()
# 绘制网格化后的世界地图
ax = world_grid.plot()
# 添加经纬度点
longitude = [0, 30, 60, 90]  # 经度
latitude = [0, 30, 60, 90]   # 纬度
plt.scatter(longitude, latitude, color='red', marker='o', label='Points')  # 添加红色圆形点
plt.legend()  # 添加图例
ax.grid(True)
# 显示地图
plt.show()

# # 查看数据集的前几行
# print(world.head())
#
# # 查看数据集的基本信息，包括列名、数据类型等
# print(world.info())
#
# # 查看数据集的统计摘要，包括数值型特征的统计指标
# print(world.describe())


# # 选择特定区域的地理信息数据
# myregion = world[world['regnb'] == mm]
# # 创建区域的备份副本
# backcountry = myregion.copy()
#
# # 创建一个包含恐怖袭击事件地理位置的点数据集
# gtpt = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(gtdt['lon'], gtdt['lat'])])
#
# # 设置地理坐标系为与选择区域相同的坐标系
# gtpt.crs = backcountry.crs
#
# # 通过空间关联，从选择区域中提取与恐怖袭击事件相关的地理数据
# myregion = gpd.overlay(backcountry, gtpt, how='intersection')
#
# # 对选择的区域进行缓冲处理，确保边界完整性
# myregion['geometry'] = myregion.buffer(0)
#
# # 保存选择区域的数据备份
# regnodata = myregion.copy()
#
# # 清除不需要的区域数据列，仅保留关键信息
# regnodata = regnodata[['geometry']]
#
# # 通过与 PRIO 数据集进行空间交叉，提取选择区域内的 PRIO 数据
# PRIOreg = gpd.overlay(regnodata, PRIO, how='intersection')
#
# # 删除不需要的 PRIO 数据列
# PRIOreg.drop(columns=['iso_a3.1'], inplace=True)
#
# # 清理临时变量，释放内存空间
# del PRIO
