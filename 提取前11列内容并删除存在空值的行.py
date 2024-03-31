import pandas as pd

# 1. 打开GTDdata.csv文件
file_path = "GTDdata.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 2. 筛选出指定列
selected_columns = ['crit1', 'crit2', 'crit3', 'vicinity', 'extended', 'ishostkid',
                    'suicide', 'propextent', 'nkill', 'nwound', 'success']
filtered_df = df[selected_columns]

# 3. 删除存在空值的行
filtered_df = filtered_df.dropna()

# 4. 将筛选后的数据保存为pcadata_no_nan.csv文件
output_file = "pcadata_no_nan.csv"
filtered_df.to_csv(output_file, index=False)

print("数据已保存为pcadata_no_nan.csv文件。")

'''
用python实现：
1.打开GTDdata.csv文件
2.筛选出以下列：（列名是英文对应的部分）
入选标准1  crit1
入选标准2  crit2
入选标准3  crit3
是否发生在城市   vicinity
是否是持续事件  extended
是否绑架事件 ishostkid
是否是自杀式袭击  suicide
财产损失程度 propextent
死亡人数nkill
受伤人数  nwound
是否成功  success
3.删除存在空值的行，并保存为pcadata_no_nan.csv
'''
