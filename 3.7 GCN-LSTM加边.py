import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import layers

# 生成全1的邻接矩阵
def generate_adj_matrix(num_nodes):
    adj_matrix = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(adj_matrix, 0)  # 将对角线上的元素置为0，避免节点与自己连接
    return adj_matrix


# 构建GCN部分
# 构建整合模型
def build_model(adj_matrix, edge_weights):
    # 构建GCN模型
    def build_gcn(adj_matrix, edge_weights, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = inputs
        # GCN层
        x = gcn_layer(adj_matrix, edge_weights, x)
        # 添加激活函数
        x = layers.ReLU()(x)
        # 添加全连接层，将输出形状从 (None, 30, 86) 转换为 (None, 30, 64)
        x = layers.Dense(64)(x)
        return models.Model(inputs, x)

    # 构建GCN模型
    gcn_model = build_gcn(adj_matrix, edge_weights, input_shape=(30, 64))  # 修改这里的输入形状为 (30, 64)

    # 构建整合LSTM模型
    lstm_model = build_lstm(input_shape=(30, 64))

    # 组合GCN和LSTM模型
    combined_model_input = layers.Input(shape=(30, 86))  # 修改这里的输入形状为 (30, 86)
    gcn_output = gcn_model(combined_model_input)
    lstm_output = lstm_model(gcn_output)

    # 构建整合模型
    combined_model = models.Model(inputs=combined_model_input, outputs=lstm_output)

    return combined_model







def gcn_layer(adj_matrix, edge_weights, x):
    return CustomLayer()([adj_matrix, edge_weights, x])


# 构建整合LSTM部分
def build_lstm(input_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32),
        layers.Dense(86)
    ])
    return model


# 构建整合模型
def build_model(adj_matrix, edge_weights):
    # 构建GCN模型
    def build_gcn(adj_matrix, edge_weights, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = inputs
        # GCN层
        x = gcn_layer(adj_matrix, edge_weights, x)
        # 添加激活函数
        x = layers.ReLU()(x)
        # 返回模型
        return models.Model(inputs, x)

    # 构建GCN模型
    gcn_model = build_gcn(adj_matrix, edge_weights, input_shape=(30, 64))  # 修改这里的输入形状为 (30, 64)

    # 构建整合LSTM模型
    lstm_model = build_lstm(input_shape=(30, 64))

    # 组合GCN和LSTM模型
    combined_model_input = layers.Input(shape=(30, 86))
    gcn_output = gcn_model(combined_model_input)
    lstm_output = lstm_model(gcn_output)

    # 构建整合模型
    combined_model = models.Model(inputs=combined_model_input, outputs=lstm_output)

    return combined_model


# 导入数据
X = np.load('X_array.npy')
y = np.load('y_array.npy')
edge_weights = np.load('edge_weights.npy')

# 生成邻接矩阵
adj_matrix = generate_adj_matrix(num_nodes=86)

# 构建模型
model = build_model(adj_matrix, edge_weights)

# 设置Adam优化器的学习率为0.001
adam_optimizer = optimizers.Adam(learning_rate=0.001)

# 编译整合模型
model.compile(optimizer=adam_optimizer, loss='mse')

# 划分训练集和测试集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
