import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 总方差中的贡献率
# 生成示例数据
# np.random.seed(0)
# X = np.random.rand(100, 11)  # 生成一个100行11列的随机数据
#
# def randomize_data(X):
#     """
#     随机化数据集的特征列
#     """
#     X_randomized = np.copy(X)
#     for i in range(X.shape[1]):
#         np.random.shuffle(X_randomized[:, i])
#     return X_randomized

def get_pca_weights(X):
    """
    使用PCA获取每个维度的权重数值
    """
    # 首先，标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 创建PCA模型并拟合数据
    pca = PCA()
    pca.fit(X_scaled)

    # 获取主成分的权重数值
    weights = pca.explained_variance_ratio_

    return weights


# 读取 CSV 文件
df = pd.read_csv('output/pcadata_mean.csv')


# 提取特征列
X = df

# 对每一维度数据进行标准化，沿着轴 0 进行标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X.T).T

# 使用PCA获取每个维度的权重数值
weights = get_pca_weights(data_scaled)

# 打印每个维度的权重数值
for i, weight in enumerate(weights):
    print(f"Dimension {i + 1} weight: {weight}")
# 结果：
# Dimension 1 weight: 0.40786456067375637
# Dimension 2 weight: 0.1577808946477653
# Dimension 3 weight: 0.12326628498876713
# Dimension 4 weight: 0.08256812872246881
# Dimension 5 weight: 0.06637840451389275
# Dimension 6 weight: 0.05914609994809854
# Dimension 7 weight: 0.045018426494194304
# Dimension 8 weight: 0.03716354039794461
# Dimension 9 weight: 0.012539977232789113
# Dimension 10 weight: 0.00827368238032306
# Dimension 11 weight: 1.586312073522102e-28