import pandas as pd

# 1. 打开GTDdata.csv文件并指定编码格式
file_path = "GTDdata.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 2. 筛选出指定列
selected_columns = ['crit1', 'crit2', 'crit3', 'vicinity', 'extended', 'ishostkid',
                    'suicide', 'propextent', 'nkill', 'nwound', 'success']
filtered_df = df[selected_columns]

# 3. 将筛选后的数据保存为pcadata.csv文件
output_file = "pcadata.csv"
filtered_df.to_csv(output_file, index=False)

print("数据已保存为pcadata.csv文件。")
