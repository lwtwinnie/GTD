import pandas as pd
from sklearn.preprocessing import StandardScaler


def df2vector(df):
    # 1. 合并日期列
    df['datetime'] = df['iyear'] * 365 + df['imonth'] * 30 + df['iday']

    # 2. 选择需要保留的列
    columns_to_keep = ['eventid', 'datetime', 'extended', 'country', 'region', 'latitude', 'longitude',
                       'specificity', 'vicinity', 'crit1', 'crit2', 'crit3', 'success', 'suicide',
                       'attacktype1', 'targtype1', 'targsubtype1', 'nkill', 'nwound', 'property',
                       'propextent', 'ishostkid', 'ransom']
    df = df[columns_to_keep]

    # 3. 使用平均值填充空值
    df.fillna(df.mean(), inplace=True)

    # 4. 归一化（除了eventid和datetime之外的列）
    scaler = StandardScaler()
    columns_to_normalize = ['extended', 'latitude', 'longitude', 'specificity', 'vicinity', 'crit1', 'crit2',
                            'crit3', 'success', 'suicide', 'attacktype1', 'targtype1', 'targsubtype1',
                            'nkill', 'nwound', 'property', 'propextent', 'ishostkid', 'ransom']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # 5. 存储到本地
    df.to_csv('vector200.csv', index=False)


# 读入GTDdata.csv文件
df = pd.read_csv('GTDdata200.csv', encoding='ISO-8859-1')

# 调用df2vector函数
df2vector(df)

'''用python实现df2vector函数
它的输入是一个很大的df，列有eventid	iyear	imonth	iday	approxdate	extended	resolution	country	country_txt	region	region_txt	provstate	city	latitude	longitude	specificity	vicinity	location	summary	crit1	crit2	crit3	doubtterr	alternative	alternative_txt	multiple	success	suicide	attacktype1	attacktype1_txt	attacktype2	attacktype2_txt	attacktype3	attacktype3_txt	targtype1	targtype1_txt	targsubtype1	targsubtype1_txt	corp1	target1	natlty1	natlty1_txt	targtype2	targtype2_txt	targsubtype2	targsubtype2_txt	corp2	target2	natlty2	natlty2_txt	targtype3	targtype3_txt	targsubtype3	targsubtype3_txt	corp3	target3	natlty3	natlty3_txt	gname	gsubname	gname2	gsubname2	gname3	gsubname3	motive	guncertain1	guncertain2	guncertain3	individual	nperps	nperpcap	claimed	claimmode	claimmode_txt	claim2	claimmode2	claimmode2_txt	claim3	claimmode3	claimmode3_txt	compclaim	weaptype1	weaptype1_txt	weapsubtype1	weapsubtype1_txt	weaptype2	weaptype2_txt	weapsubtype2	weapsubtype2_txt	weaptype3	weaptype3_txt	weapsubtype3	weapsubtype3_txt	weaptype4	weaptype4_txt	weapsubtype4	weapsubtype4_txt	weapdetail	nkill	nkillus	nkillter	nwound	nwoundus	nwoundte	property	propextent	propextent_txt	propvalue	propcomment	ishostkid	nhostkid	nhostkidus	nhours	ndays	divert	kidhijcountry	ransom	ransomamt	ransomamtus	ransompaid	ransompaidus	ransomnote	hostkidoutcome	hostkidoutcome_txt	nreleased	addnotes	scite1	scite2	scite3	dbsource	INT_LOG	INT_IDEO	INT_MISC	INT_ANY	related

输出是一个矩阵
我希望进行的操作：
1.把iyear imonth iday通合并为列datetime=iyear*365+imonth*30+iday
2.保留列：eventid、datetime、extended、country、region、latitude、longitude、specificity、vicinity、crit1、crit2、crit3、
success、suicide、attacktype1、targtype1、targsubtype1、nkill、nwound、property、propextent、ishostkid、ransom
3.对除了eventid之外的列，用每一列的平均值填充空值
4.对除了eventid和datetime之外的列，每一列进行归一化（减去平均值、除以标准差）
5.存储df到本地，名为vector.csv
'''


import pandas as pd
from tqdm import tqdm

# 读取vector200.csv文件的前1000行
df = pd.read_csv('vector200.csv').head(1000)

# 生成事件对DataFrame
event_pairs = pd.DataFrame(columns=['datetime', 'eventid1', 'eventid2'])

# 计算总共需要迭代的次数
total_iterations = sum(range(1, len(df)))

# 使用tqdm显示进度条
with tqdm(total=total_iterations) as pbar:
    # 遍历每个事件，生成事件对
    rows = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            rows.append({'datetime': abs(df.iloc[i]['datetime'] - df.iloc[j]['datetime']),
                         'eventid1': df.iloc[i]['eventid'],
                         'eventid2': df.iloc[j]['eventid']})
            pbar.update(1)  # 更新进度条
    event_pairs = pd.concat([event_pairs, pd.DataFrame(rows)], ignore_index=True)

# 存储到本地
event_pairs.to_csv('event_pairs.csv', index=False)

'''截取前1000行，生成所有的事件对df，其列为时间差（datetime）、eventid1、eventid2，存到本地'''

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 读取event_pairs.csv文件
event_pairs_df = pd.read_csv('event_pairs.csv')

# 读取vector200.csv文件
vector_df = pd.read_csv('vector200.csv')

# 存储时间差和相似度对的列表
time_similarity_pairs = []

# 选择时间差为0、5、10、15、20、25……500的事件对
for time_diff in range(0, 501, 5):
    # 筛选时间差为time_diff的事件对
    filtered_pairs = event_pairs_df[event_pairs_df['datetime'] == time_diff]

    # 遍历每个事件对
    for _, pair in filtered_pairs.iterrows():
        # 找到事件对在vector200.csv中的对应行
        event1_row = vector_df[vector_df['eventid'] == pair['eventid1']].iloc[0]
        event2_row = vector_df[vector_df['eventid'] == pair['eventid2']].iloc[0]

        # 提取除了eventid和datetime之外的特征
        event1_features = event1_row.drop(['eventid', 'datetime'])
        event2_features = event2_row.drop(['eventid', 'datetime'])

        # 计算余弦相似度
        similarity = cosine_similarity([event1_features.values], [event2_features.values])[0][0]

        # 存储时间差和相似度对
        time_similarity_pairs.append([time_diff, similarity])

# 转换为DataFrame
time_similarity_df = pd.DataFrame(time_similarity_pairs, columns=['Time Difference', 'Similarity'])

# 存储为CSV文件
time_similarity_df.to_csv('time_similarity_pairs.csv', index=False)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(time_similarity_df['Time Difference'], time_similarity_df['Similarity'], marker='o', color='b', alpha=0.5)
plt.title('Time Difference vs Similarity')
plt.xlabel('Time Difference')
plt.ylabel('Similarity')
plt.grid(True)
plt.show()
