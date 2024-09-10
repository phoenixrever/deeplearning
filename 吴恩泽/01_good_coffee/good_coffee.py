import logging
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
from lab_utils_common import dlc
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
# 由于浮点数计算中的舍入误差，不同的计算顺序可能会导致略微不同的数值结果 在大多数情况下，OneDNN 提供的性能提升是值得保留的，除非你遇到了与数值精度相关的问题。
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
plt.style.use(os.path.join(os.path.dirname(__file__), 'deeplearning.mplstyle'))

# 设置TensorFlow日志级别
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# 加载咖啡数据
X, Y = load_coffee_data()
print(X.shape, Y.shape)  # (200, 2) (200, 1)

# 可视化烘焙数据
plt_roast(X, Y)


# 数据标准化
"""create a "Normalization Layer". Note, as applied here, this is not a layer in your model.
  'adapt' the data. This learns the mean and variance of the data set and saves the values internally.
  normalize the data. It is important to apply normalization to any future data that utilizes the learned model.
    """
print(f"Temperature Max, Min pre normalization: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}")  # 284.99, 151.32
print(f"Duration    Max, Min pre normalization: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}")  # 15.45, 11.51

'''
    tf.keras.layers.Normalization 是 TensorFlow 中用于对数据进行归一化处理的层  
    归一化是深度学习中常用的预处理步骤，用于将输入数据调整到某个范围内，通常是均值为 0，标准差为 1，以提高模型训练的稳定性和效率。
    axis=-1 通常表示最后一个轴，例如在图像数据中，最后一个轴通常对应于颜色通道（例如 RGB），在时间序列或其他高维数据中，最后一个轴可能对应于特征维度。
    对于 X (200, 2) 这个张量，axis=-1 指的是最后一个轴，即每行的那 2 个元素。
    
    axis=0: 可以看做 以 x 轴 垂直 求和 (垂直就是y方向)
    [1, 2]    # 第 1 列：1 + 3 + 5 = 9
    [3, 4] -> # 第 2 列：2 + 4 + 6 = 12
    [5, 6]

    结果: [9, 12]
    
    axis=1: 可以看做 以 Y 轴 垂直 求和 (垂直就是X方向,就是每行的2个元素相加)

    [1, 2]    # 第 1 行：1 + 2 = 3
    [3, 4] -> # 第 2 行：3 + 4 = 7
    [5, 6]    # 第 3 行：5 + 6 = 11

    结果: [3, 7, 11]
    
     
    例：
    data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
    
    均值是所有数据点的平均值
    1+2+3+4/10 = 2.5
    
    标准差
    计算每个数据点与均值的差
    1−2.5=−1.5
    2−2.5=−0.5
    3−2.5=0.5
    4−2.5=1.5
    
    计算这些差的平方：
    (−1.5) ^ 2 =2.25
    (−0.5) ^ 2 =0.25
    0.5 ^ 2 =0.25
    1.5 ^ 2 =2.25
    
    计算这些平方差的平均值（方差平方差总和除以）：
    (2.25 + 0.25 + 0.25 + 2.25)/4 =4/4=1.0  
    
    标准差是方差的平方根
    对1.0取平方根 = 1.0
    
    对于 [1, 2, 3, 4]，均值是 2.5，标准差是 1.0。
    
    对data 进行归一化 
    每个元素将根据其所在行的均值和标准差进行调整。例如，如果第 1 行的均值是 2.5，标准差是 1.118，那么元素 1 将被转换为 -1.265（即 (1-2.5)/1.118）。
    normalized_data_axis_minus1 = [
    [-1.265, -0.632, 0, 0.632],
    [-1.265, -0.632, 0, 0.632],
    [-1.265, -0.632, 0, 0.632]]

    总结 
        下面代码就是对数组中的每个元素 ，根据数据的均值和标准差将其转换为均值为 0、标准差为 1 的分布。不用深究，要研究的东西多着喃/(ㄒoㄒ)/~~
'''

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance 学习均值和方差
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")  # 1.66, -1.69
print(f"Duration    Max, Min post normalization: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")  # 1.79, -1.70


# 数据扩充: 将原始数据集复制1000次以扩大数据集 (1000, 1)：表示将 Xn 沿着第一个轴（样本轴x）重复 1000 次，第二个轴（特征轴y）不变。
Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000, 1))
print(Xt.shape, Yt.shape)


# 设置随机种子以获得一致的结果
tf.random.set_seed(1234)  # applied to achieve consistent results


# 构建神经网络模型

# Sequential 是 Keras 中的一种模型类型，用于逐层堆叠神经网络层。它表示一个线性的层堆叠模型，每一层的输入是上一层的输出。
model = Sequential(
    [
        # shape=(2,) 代表的是一维数组，每个样本具有 2 个特征(列)。
        # shape=(2, n) 代表的是二维数组，其中 2 是行数，n 是列数。
        tf.keras.Input(shape=(2,)),
        # 全连接层 该层有 3 个神经元 ,该层使用 sigmoid 激活函数。Sigmoid 函数将输出压缩到 0 到 1 的范围内。
        Dense(3, activation='sigmoid', name='layer1'),
        Dense(1, activation='sigmoid', name='layer2')
    ]
)

# 用于打印出模型的详细信息。具体来说，它提供了关于模型架构的概览，包括各层的名称、输出形状、参数数量等#
'''
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 2)]               0         
    _________________________________________________________________
    layer1 (Dense)               (None, 3)                 9         
    _________________________________________________________________
    layer2 (Dense)               (None, 1)                 4         
    =================================================================

    Output Shape：每一层的输出形状。
        [(None, 2)] None 表示样本数量是动态的，2 表示每个样本有 2 个特征。
        (None, 1)，表示该层的输出是单个值，通常用于二分类任务
    
    Param #：每层的参数数量。
        input_1 (InputLayer)：参数数量为 0，因为输入层没有可训练的参数。
        layer1 (Dense)：参数数量为 9。
            计算方式：输入 2 个特征，每个特征连接到 3 个神经元，因此有 2x3=6 个权重参数。
            还有 3 个偏置参数（每个神经元一个偏置）。
            总计 9 个参数。
        layer2 (Dense)：参数数量为 4。
            计算方式：输入 3 个神经元，每个神经元连接到 1 个神经元，因此有 3x1=3 个权重参数
            还有 1 个偏置参数（每个神经元一个偏置）。
            总计 4 个参数
            
    Total params：模型中所有层的总参数数量。这是所有可训练和不可训练参数的总和。总计为 13。
    Trainable params：在训练过程中会被更新的参数总数。这里是 13，因为所有参数都是可训练的。
    Non-trainable params：不可训练的参数数量。通常是 0，除非模型中包含冻结的层（如预训练模型）。
            
'''
model.summary()


# 计算模型参数数量 具体将上面的参数图分析
L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters  权重参数：2个特征连接到 3 个神经元 +   偏置参数： 3 个偏置
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters  权重参数：3个神经元连接到 1 个神经元 +  偏置参数： 1 个偏置
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params)  # 9 4


# 获取并打印初始权重
'''
W1(2, 3):
    [[ 0.86 -0.7   0.65]
    [ 0.08 -0.94  0.69]]
b1(3,): [0. 0. 0.]

W1(2, 3):
    形状：(2, 3)。这表示该层有 2 个输入特征和 3 个神经元（输出单元）。
    第一行 [0.86, -0.7, 0.65] 是第一个输入特征连接到 3 个神经元的权重。
    第二行 [0.08, -0.94, 0.69] 是第二个输入特征连接到 3 个神经元的权重。

b1(3,)
    形状：(3,)。表示该层有 3 个神经元，每个神经元有一个偏置。
    内容：[0. 0. 0.]。偏置值为零，可能是模型初始化时的默认值。

W2(3, 1):
    [[-0.37]
    [-1.13]
    [-0.43]]
b2(1,): [0.]



权重矩阵的来源
    初始化权重：
        在模型训练之前，神经网络的权重通常会被随机初始化。这是为了打破对称性，使得每个神经元可以学习到不同的特征。
        这些权重值最初是随机生成的。比如，0.86、-0.7 和 0.65 是在模型初始化时随机选择的值。这样做是为了让模型可以从多种可能的开始状态中选择最佳状态。

    模型训练：
        在训练过程中，模型会通过反向传播算法不断调整这些权重，使得模型的预测结果更加准确。这个过程会不断更新权重的值。
        训练过程中的更新：
        在训练过程中，通过反向传播和优化算法（如梯度下降），这些权重会不断调整。例如，模型在训练过程中发现 0.86 这个权重对最终预测结果的影响很大，就会根据训练数据调整它的值，以便模型更准确地预测结果。
        权重 0.86 表示第一个输入特征对第一个神经元的影响程度，-0.7 表示第一个输入特征对第二个神经元的影响，依此类推。

总结
    初始化阶段：权重值是随机生成的，目的是为了开始训练。
    训练阶段：通过优化算法调整这些权重值，使得模型在训练数据上表现得更好。
'''
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)


# 编译模型 编译模型时定义了损失函数、优化器以及其他一些配置。
model.compile(
    # 通过最小化损失函数的值来优化模型的参数。
    # BinaryCrossentropy 是用于二分类任务的损失函数。它计算预测概率与实际标签之间的交叉熵，适用于二分类问题，例如判断一张图片是否属于某个类别（例如是/否）。
    loss=tf.keras.losses.BinaryCrossentropy(),
    # 优化器用于更新模型的权重，以最小化损失函数的值。不同的优化器有不同的算法来调整权重。
    # Adam 是一种常用的优化算法，它结合了动量优化和自适应学习率的优点。它会根据每个参数的历史梯度自适应地调整学习率。
    # learning_rate=0.01：这是学习率，控制优化器更新模型权重的步长。学习率设置为 0.01，意味着每次权重更新时，步长为 0.01。合适的学习率对于模型的收敛速度和效果非常重要。
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

# 训练模型 训练模型是调整模型参数以最小化损失函数的过程。你会提供训练数据、目标标签以及训练的轮数（epochs），优化器会在每轮中更新模型的权重。
model.fit(
    # Xt 是训练数据，通常是输入特征 Yt 是训练数据的目标标签（或标签）
    Xt, Yt,
    # 轮数（epochs）表示模型将训练数据遍历的次数（梯度下降）。每次遍历整个训练数据集称为一个 epoch。
    epochs=10,
)
'''
总结
编译模型：配置了损失函数和优化器，定义了如何评估和优化模型。损失函数评估模型的表现，优化器通过调整权重来减少损失。
训练模型：使用指定的数据集和标签在多轮次中训练模型。模型通过优化算法更新权重，以更好地拟合数据和提高预测准确性。
'''


# 获取并打印训练后的权重
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
'''
我的训练结果,和下面的是不一样的
W1:
 [[ 14.82 -11.02   0.17]
 [ 12.37  -0.26  10.56]]
b1: [  2.08 -11.88  12.7 ]
W2:
 [[-47.51]
 [-55.94]
 [ 46.07]]
b2: [-14.34]

'''
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# 设置预定义的权重
# 由于神经网络训练的随机性质（初始化、数据批次顺序等），每次训练的结果可能会略有不同。
# 以下代码将权重设置为预定义的值，以确保每个人都能得到相同的结果。所有直接赋值了.
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]])
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("layer1").set_weights([W1, b1])
model.get_layer("layer2").set_weights([W2, b2])

# 使用模型进行预测
X_test = np.array([
    [200, 13.9],  # postive example
    [200, 17]])   # negative example

'''
归一化 使test的数据 在 0-1 之间
    X_testn =
    tf.Tensor(
    [[-0.47  0.42]
    [-0.47  3.16]], shape=(2, 2), dtype=float32)
'''
X_testn = norm_l(X_test)
print("X_testn = \n", X_testn)


predictions = model.predict(X_testn)

'''
    predictions =
    [[9.63e-01]
    [3.03e-08]]

'''
print("predictions = \n", predictions)


# 将预测结果转换为二元决策
# np.zeros_like() 是 NumPy 库中的一个函数。 创建一个新数组，其形状和数据类型与作为参数传入的数组相同。
# 新数组的所有元素都被初始化为 0。用来存储二元化（二分类）的决策结果，即 0 或 1。
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    # NumPy 的广播机制：虽然 predictions 是二维的，但 NumPy 的广播机制允许我们直接与标量（如 0.5）进行比较。0.963>0.5 true
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0

'''
    decisions =
    [[1.]
    [0.]]
'''
print(f"decisions = \n{yhat}")


# 另一种转换预测结果的方法
'''
这部分对 predictions 数组中的每个元素进行布尔比较。结果是一个布尔数组， [[True], [False]]。
.astype(int)将布尔数组转换为整数数组。[[1], [0]]。
速度快：这种向量化操作比我们之前使用循环实现的操作快得多，特别是对于大型数组。
'''
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")


# 可视化第一层( W1, b1)
'''
    plt_layer 函数画出的三个图表达了神经网络第一层中三个不同神经元的行为和决策边界。让我详细解释这些图的含义：
    1.每个图代表一个神经元：
        由于 W1.shape[1] 是 3，所以画出了三个子图，每个对应第一层的一个神经元。
    2.图的内容：
        x轴表示温度（摄氏度）
        y轴表示时间（分钟）
        散点：红色 'x' 表示好的烘焙，蓝色圆圈表示坏的烘焙
        颜色渐变：表示该神经元对输入的响应强度（概率）
    3.决策边界：
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
        颜色的变化表示神经元的激活程度，通常深蓝色区域表示高激活（接近1 大于0.5 代表坏的烘培），浅色区域表示低激活（接近0,小于0.5 代表好的烘培）
        颜色的变化模式展示了每个神经元如何划分输入空间

'''
plt_layer(X, Y.reshape(-1,), W1, b1, norm_l)

# 可视化输出单元
plt_output_unit(W2, b2)

# 定义网络预测函数


def netf(x): return model.predict(norm_l(x))


# 可视化整个网络
plt_network(X, Y, netf)
