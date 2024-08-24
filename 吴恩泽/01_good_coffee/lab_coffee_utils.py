from lab_utils_common import dlc
import matplotlib.colors as colors
from matplotlib import cm
from tensorflow.keras.activations import sigmoid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use(os.path.join(os.path.dirname(__file__), 'deeplearning.mplstyle'))


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """

    # 创建一个 NumPy 随机数生成器，种子设为 2，确保每次运行生成相同的随机数。
    rng = np.random.default_rng(2)

    # 生成 400 个随机数，并将其重塑为一个 200x2 的数组。每行代表一个样本，有两个特征。
    '''
        总共有 400 个元素
        我们指定了最后一个维度是 2
        所以 NumPy 会自动计算: 400 / 2 = 200
        结果就是一个 200 x 2 的数组


        等价写法：
        这行代码等同于 X = rng.random(400).reshape(200, 2)
        
        优势：
        使用 -1 的主要优势是灵活性。如果你改变了生成的随机数的数量，你不需要手动计算和更改 reshape 的参数。
        注意事项：

        你只能在一个维度上使用 -1
        总元素数必须能被其他指定维度整除
    '''
    X = rng.random(400).reshape(-1, 2)  # (200, 2)

    X[:, 1] = X[:, 1] * 4 + 11.5          # 12-15 min is best
    X[:, 0] = X[:, 0] * (285-150) + 150  # 350-500 F (175-260 C) is best

    # 在 NumPy 中，len(X) 返回的是数组 X 的第一个维度的长度。 它不是计算 row * column，而是仅仅返回行数（在这个例子中）。
    # 创建了一个长度为 200 的一维数组。Y 的形状是 (200,)，这是一个一维数组，不是 (200, 1) 的二维数组。
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        # 根据温度画一条时间边界线。
        y = -3/(260-175)*t + 21
        '''
            温度在 175°C 到 260°C 之间
            持续时间在 12 到 15 分钟之间
            并且点 (t, d) 在我们定义的线 y 下方或上面，就标记为 1
        '''
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    #  Y.reshape(-1, 1) 将 Y 从一维数组转换为列向量。 (200,1)
    return (X, Y.reshape(-1, 1))


def plt_roast(X, Y):
    # 确保Y是一维数组。
    Y = Y.reshape(-1,)
    colormap = np.array(['r', 'b'])
    fig, ax = plt.subplots(1, 1,)
    # 绘制"好"烘焙的散点图。使用红色'x'标记。
    # 这行代码使用 matplotlib 的 scatter 函数在图上绘制散点。
    # X[Y == 1, 0] NumPy 的高级索引语法, Y == 1 大概可以理解成Y (200,1)里面 吧 列值等与1 的row 取出来成为一个数组,然后 取出所有x 的 row和Y==1 row相等的 0列
    '''
        X = np.array([[100, 10],
                [200, 15],
                    [150, 12],
                    [250, 14]])
        Y = np.array([0, 1, 0, 1])

        result = X[Y == 1, 0]
        result 将是 [200, 250]
    '''
    # s=70：设置散点的大小为70。 marker='x'：使用 'x' 形状作为标记。 c='red':设置标记的颜色为红色。
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=70, marker='x', c='red', label="Good Roast")

    # 绘制"坏"烘焙的散点图。使用蓝色空心圆标记。
    '''
        X[Y == 0, 0] 选择 Y 值为 0（"坏"烘焙）的样本的第一个特征（温度）。 这将是散点的 x 坐标。
        X[Y == 0, 1] 选择 Y 值为 0 的样本的第二个特征（持续时间）。 这将是散点的 y 坐标。
        s=100 设置散点的大小为 100。 
        facecolors='none' 设置标记的填充颜色为透明
        edgecolors=dlc["dldarkblue"] 设置标记的边缘颜色。
        linewidth=1 设置标记边缘的线条宽度为 1
    '''
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o', facecolors='none',
               edgecolors=dlc["dldarkblue"], linewidth=1,  label="Bad Roast")

    '''
        想象你要画一条从 175°C 到 260°C 的温度线：

        起点是 175°C（最低温度）
        终点是 260°C（最高温度）
        你想在这条线上均匀地标记 50 个点

        它在 175 和 260 之间创建了 50 个均匀分布的数字。
        这就像在尺子上画等距离的刻度。
        
        np.linspace() 就是帮你算出这 50 个刻度应该在哪里，并把这些数值存在 tr 这个变量里。
        
        这样做的目的是：

        后面可以用这些点来画一条平滑的线，显示温度和时间的关系。
        有了这些均匀分布的点，画出来的线会很平滑，不会显得生硬或者断断续续。
    
    '''
    tr = np.linspace(175, 260, 50)

    # 绘制分隔"好"和"坏"烘焙的边界线。
    '''
        tr 是 x 坐标（温度）。
        (-3/85) * tr + 21 计算对应的 y 坐标（时间）。
        plot 函数会把这 50 个点按顺序连接起来。
    '''
    ax.plot(tr, (-3/85) * tr + 21, color=dlc["dlpurple"], linewidth=1)

    # 绘制水平线，表示最短烘焙时间。
    ax.axhline(y=12, color=dlc["dlpurple"], linewidth=1)

    #
    # 绘制垂直线，表示最低烘焙温度。
    ax.axvline(x=175, color=dlc["dlpurple"], linewidth=1)
    ax.set_title(f"Coffee Roasting", size=16)
    # 设置x轴标签。
    ax.set_xlabel("Temperature \n(Celsius)", size=12)
    # 设置y轴标签。
    ax.set_ylabel("Duration \n(minutes)", size=12)
    # 添加图例，放在右上角。
    ax.legend(loc='upper right')
    plt.show()


# 这个函数接受两个参数：ax（一个matplotlib轴对象）和fwb（一个函数，可能代表forward propagation）。
def plt_prob(ax, fwb):
    """ plots a decision boundary but include shading to indicate the probability """
    # setup useful ranges and common linspaces
    # 创建两个线性空间，分别代表两个特征（温度和时间）的范围。
    x0_space = np.linspace(150, 285, 40)
    x1_space = np.linspace(11.5, 15.5, 40)

    # get probability for x0,x1 ranges 使用meshgrid创建一个2D网格，用于后续计算。
    tmp_x0, tmp_x1 = np.meshgrid(x0_space, x1_space)

    # 创建一个与tmp_x0形状相同的零数组，用于存储概率值。
    z = np.zeros_like(tmp_x0)

    # 遍历网格的每个点，将坐标传入fwb函数（可能是神经网络的前向传播），计算概率值。
    for i in range(tmp_x0.shape[0]):
        for j in range(tmp_x1.shape[1]):
            x = np.array([[tmp_x0[i, j], tmp_x1[i, j]]])
            z[i, j] = fwb(x)

    # 获取'Blues'颜色映射，并使用truncate_colormap函数（未提供定义）截取其中一部分。
    cmap = plt.get_cmap('Blues')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    # TODO 待研究 pcolormesh 函数 /(ㄒoㄒ)/~~
    # 使用pcolormesh在给定的轴上绘制颜色网格。norm参数设置颜色范围，cmap使用自定义的颜色映射，shading设置为'nearest'以避免插值，alpha设置透明度。
    pcm = ax.pcolormesh(tmp_x0, tmp_x1, z,
                        norm=cm.colors.Normalize(vmin=0, vmax=1),
                        cmap=new_cmap, shading='nearest', alpha=0.9)

    # 为绘制的颜色网格添加一个颜色条。
    ax.figure.colorbar(pcm, ax=ax)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ truncates color map """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plt_layer(X, Y, W1, b1, norm_l):
    # 将Y重塑为一维数组，-1表示自动计算维度。 (200, 1) -> (200,)
    Y = Y.reshape(-1,)
    # 创建一个图形和子图数组。1行， 列数等于W1的列数（即神经元数量）
    '''
        W1.shape[1]=(2,)  np.array([1, 2]) 是一维的,矩阵运算中 (2,) 可以被视为行向量或列向量，取决于上下文
        (2, 1) np.array([[1], [2]]) 表示一个二维数组，有 2 行 1 列,总是被视为列向量
        可以这样理解 括号里面的每个数字代表一个维度,(2,) 这种就代表只有一个维度,就是普通的数组
        另外 arr_2d[0, 0] 和arr_2d[0:1] 
            [0, 0] 访问第0行第0列的元素
            0:1 是一个切片操作，表示"从索引0（包含）到索引1（不包含）"。
    '''
    fig, ax = plt.subplots(1, W1.shape[1], figsize=(16, 4))
    # 遍历第一层的每个神经元。
    for i in range(W1.shape[1]):
        # 定义一个内部函数，计算当前神经元的激活值。使用sigmoid激活函数。
        '''
            W1 = np.array([
                [-8.94,  0.29, 12.89],
                [-0.17, -7.34, 10.79]])
            b1 = np.array([-9.87, -9.28,  1.01])

            W1 是权重矩阵，[:, i] 表示取第 i 列的所有行。b1[i] 是第 i 个神经元的偏置项。
            归一化后的数据与权重进行点积运算。
            sigmoid 是一个激活函数，定义为 f(x) = 1 / (1 + e^(-x))。它将输入压缩到 (0, 1) 范围内，常用于二分类问题。它引入了非线性，允许网络学习更复杂的模式。

            经过归一化后，norm_l(x) 的形状仍然是 (2,)。
            W1[:, i] 取第 i 列，所以它的形状是 (2,)。
            当两个一维数组进行点积运算时，结果是一个标量（单个数字）。


            点积的计算过程： 
                第一行 [0.86, -0.7, 0.65] 是第一个输入特征连接到 3 个神经元的权重。
                第二行 [0.08, -0.94, 0.69] 是第二个输入特征连接到 3 个神经元的权重。
                假设 norm_l(x) = [a, b] 且 W1[:, i] = [w1, w2] 对应[0.86,0.08] 代表每个输入特征对应的权重,每一行代表输入特征,每一列代表了这个神经元的计算结果

                点积结果 = a * w1 + b * w2
            就是预测的模型 fx = w1 * x1 + w2 * x2 + b 多个特征与每个特征权重的乘机 直线图,代表了输入数据经过该神经元的线性变换后的值。

            所以，np.dot(norm_l(x), W1[:, i]) 返回的是一个单一的数值，代表了输入 x 经过第 i 个神经元的线性变换后的结果。
            这个数值随后会加上偏置 b1[i]，然后通过 sigmoid 函数进行非线性变换，得到神经元的最终输出

        '''
        def layerf(x): return sigmoid(np.dot(norm_l(x), W1[:, i]) + b1[i])
        '''
            在当前子图上绘制概率分布
        '''
        plt_prob(ax[i], layerf)

        # 绘制正样本（好的烘焙）为红色的 'x'。
        ax[i].scatter(X[Y == 1, 0], X[Y == 1, 1], s=70, marker='x', c='red', label="Good Roast")
        # 绘制负样本（坏的烘焙）为蓝色轮廓的圆圈。
        ax[i].scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o', facecolors='none',
                      edgecolors=dlc["dldarkblue"], linewidth=1,  label="Bad Roast")

        # 创建一个从175到260的等间隔数组，用于绘制线。
        tr = np.linspace(175, 260, 50)

        ax[i].plot(tr, (-3/85) * tr + 21, color=dlc["dlpurple"], linewidth=2)
        ax[i].axhline(y=12, color=dlc["dlpurple"], linewidth=2)
        ax[i].axvline(x=175, color=dlc["dlpurple"], linewidth=2)
        ax[i].set_title(f"Layer 1, unit {i}")
        ax[i].set_xlabel("Temperature \n(Celsius)", size=12)
    ax[0].set_ylabel("Duration \n(minutes)", size=12)
    plt.show()


'''
这段代码定义了一个名为`plt_network`的函数，用于可视化神经网络在咖啡烘焙问题上的性能。它生成两个子图来展示网络的概率输出和决策结果。让我为您详细解释：

1. 函数接受三个参数：`X`（输入数据），`Y`（实际标签），和`netf`（神经网络函数）。

2. 创建一个包含两个子图的图形。

3. 左侧子图（ax[0]）：
   - 调用`plt_prob`函数（未在此代码中定义）来绘制网络的概率输出。
   - 用红色'x'标记表示实际的好咖啡烘焙。
   - 用蓝色空心圆圈表示实际的坏咖啡烘焙。
   - 绘制三条紫色线，可能代表某些决策边界。
   - 设置轴标签和标题。

4. 右侧子图（ax[1]）：
   - 绘制与左图相同的三条紫色线。
   - 使用神经网络(`netf`)对输入数据进行预测。
   - 将概率大于0.5的预测分类为好烘焙（1），小于等于0.5的为坏烘焙（0）。
   - 用橙色'x'标记表示预测为好的咖啡烘焙。
   - 用蓝色空心圆圈表示预测为坏的咖啡烘焙。
   - 设置轴标签和标题。

5. 两个子图都设置了温度（摄氏度）作为x轴，持续时间（分钟）作为y轴。

这个函数的主要目的是比较神经网络的预测结果与实际数据，以及可视化网络的决策边界。左图显示实际数据和网络的概率输出，右图显示网络的最终决策结果。这有助于直观地评估模型在区分好坏咖啡烘焙方面的表现。

'''


def plt_network(X, Y, netf):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    Y = Y.reshape(-1,)
    plt_prob(ax[0], netf)
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1], s=70, marker='x', c='red', label="Good Roast")
    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o', facecolors='none',
                  edgecolors=dlc["dldarkblue"], linewidth=1,  label="Bad Roast")
    ax[0].plot(X[:, 0], (-3/85) * X[:, 0] + 21, color=dlc["dlpurple"], linewidth=1)
    ax[0].axhline(y=12, color=dlc["dlpurple"], linewidth=1)
    ax[0].axvline(x=175, color=dlc["dlpurple"], linewidth=1)
    ax[0].set_xlabel("Temperature \n(Celsius)", size=12)
    ax[0].set_ylabel("Duration \n(minutes)", size=12)
    ax[0].legend(loc='upper right')
    ax[0].set_title(f"network probability")

    ax[1].plot(X[:, 0], (-3/85) * X[:, 0] + 21, color=dlc["dlpurple"], linewidth=1)
    ax[1].axhline(y=12, color=dlc["dlpurple"], linewidth=1)
    ax[1].axvline(x=175, color=dlc["dlpurple"], linewidth=1)
    fwb = netf(X)
    yhat = (fwb > 0.5).astype(int)
    ax[1].scatter(X[yhat[:, 0] == 1, 0], X[yhat[:, 0] == 1, 1], s=70,
                  marker='x', c='orange', label="Predicted Good Roast")
    ax[1].scatter(X[yhat[:, 0] == 0, 0], X[yhat[:, 0] == 0, 1], s=100, marker='o', facecolors='none',
                  edgecolors=dlc["dldarkblue"], linewidth=1,  label="Bad Roast")
    ax[1].set_title(f"network decision")
    ax[1].set_xlabel("Temperature \n(Celsius)", size=12)
    ax[1].set_ylabel("Duration \n(minutes)", size=12)
    ax[1].legend(loc='upper right')


def plt_output_unit(W, b):
    """ plots a single unit function with 3 inputs """
    # 定义步数为10,这将决定后面生成数据点的精细程度。
    steps = 10
    fig = plt.figure()
    # 添加一个3D子图。
    ax = fig.add_subplot(projection='3d')
    # 在0到1之间生成10个均匀分布的点,分别用于x、y、z轴。
    x_ = np.linspace(0., 1., steps)
    y_ = np.linspace(0., 1., steps)
    z_ = np.linspace(0., 1., steps)
    # 使用meshgrid创建3D网格点。
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    # 创建一个10x10x10的零矩阵,用于存储后面计算的值。
    d = np.zeros((steps, steps, steps))
    # 选择'Blues'颜色映射。
    cmap = plt.get_cmap('Blues')
    # 这是一个三重循环,遍历所有网格点。对每个点:创建一个包含x、y、z坐标的向量v,计算v与权重W的点积,加上偏置b,将结果传入sigmoid函数,将最终值存储在d数组中
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                v = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                d[i, j, k] = tf.keras.activations.sigmoid(np.dot(v, W[:, 0])+b).numpy()
    # 在3D图中绘制所有点,颜色由d中的值决定。
    pcm = ax.scatter(x, y, z, c=d, cmap=cmap, alpha=1)
    ax.set_xlabel("unit 0")
    ax.set_ylabel("unit 1")
    ax.set_zlabel("unit 2")
    # 设置图形的初始视角。
    ax.view_init(30, -120)
    # 添加颜色条。
    ax.figure.colorbar(pcm, ax=ax)
    ax.set_title(f"Layer 2, output unit")

    plt.show()
