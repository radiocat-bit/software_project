import numpy as np 
import matplotlib.pyplot as plt 
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # 设置随机数种子

# 加载和查看数据集
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c = np.squeeze(Y), s = 40, cmap = plt.cm.Spectral)
plt.show()

m = Y.shape[1] # 训练集的数量

print("X的维度： " + str(X.shape))
print("Y的维度： " + str(Y.shape))
print("数据集里的数据有: " + str(m) + " 个")


# 定义神经网络结构
def layer_sizes(X, Y):
	"""
	参数：
		X - 输入数据集，维度为（输入的数量，训练/测试的数量)
		Y - 标签，维度为（输出的数量，训练/测试数量）
	返回：
		n_x - 输入层的数量
		n_h - 隐藏层的数量
		n_y - 输出层的数量
	"""
	n_x = X.shape[0] # 输入层
	n_h = 4 # 隐藏层，硬编码为4
	n_y = Y.shape[0] # 输出层

	return (n_x, n_h, n_y)


# 初始化模型的随机参数
def initialize_parameters(n_x, n_h, n_y):
	"""
	返回：
		parameters - 包含参数的字典
			W1 - 权重矩阵，维度(n_h, n_x）
			b1 - 偏向量，维度(n_h, 1)
			W2 - 权重矩阵，维度(n_y, n_h)
			b2 - 偏向量，维度为(n_y, 1)
	"""
	np.random.seed(2)
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros(shape = (n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros(shape = (n_y, 1))

	parameters = {
			"W1" : W1,
			"b1" : b1,
			"W2" : W2,
			"b2" : b2}

	return parameters


# 前向传播
def forward_propagation(X, parameters):
	"""
	参数：
		X - 维度为(n_x, m)的输入数据
		parameters - 初始化函数(initialize_parameters)的输出
	返回：
		A2 - 使用sigmoid()函数计数的第二次激活后的数值
		cache - 包含"Z1", "A1", "Z2", "A2"的字典类型变量
	"""
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	# 前向传播计算A2
	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)

	cache = {
		"Z1" : Z1,
		"A1" : A1,
		"Z2" : Z2,
		"A2" : A2 
	}

	return (A2, cache)


# 计算损失
def compute_cost(A2, Y, parameters):
	"""
	返回：
		cost - 交叉熵给出成本方程
	"""

	m = Y.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]

	# 计算成本
	logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
	cost = -np.sum(logprobs) / m 
	cost = float(np.squeeze(cost))

	return cost 


# 反向传播
def backward_propagation(parameters, cache, X, Y):
	"""
	返回：
		grads - 包含W和b的导数一个字典类型的变量
	"""
	m = X.shape[1]

	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y 
	dW2 = (1 / m) * np.dot(dZ2, A1.T)
	db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)
	dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
	dW1 = (1 / m) * np.dot(dZ1, X.T)
	db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)

	grads = {
		"dW1" : dW1,
		"db1" : db1,
		"dW2" : dW2,
		"db2" : db2}

	return grads


# 利用梯度更新规则更新参数
def update_parameters(parameters, grads, learning_rate = 1.2):
	"""
	参数：
		learning_rate - 学习速率
	返回：
		parameters - 包含更新参数的字典类型变量
	"""
	W1, W2 = parameters["W1"], parameters["W2"]
	b1, b2 = parameters["b1"], parameters["b2"]

	dW1, dW2 = grads["dW1"], grads["dW2"]
	db1, db2 = grads["db1"], grads["db2"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {
		"W1" : W1,
		"b1" : b1,
		"W2" : W2,
		"b2" : b2
	}

	return parameters


# 建立含有一个隐藏层的模型
def nn_model(X, Y, n_h, num_iterations, print_cost = False):
	"""
	参数：
		num_iterations - 梯度下降循环中的迭代次数
		print_cost - 每1000次迭代打印一次成本数值
	返回：
		parameters - 模型学习的参数，用于进行预测
	"""

	np.random.seed(3)
	n_x = layer_sizes(X, Y)[0]
	n_y = layer_sizes(X, Y)[2]

	parameters = initialize_parameters(n_x, n_h, n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(num_iterations):
		A2, cache = forward_propagation(X, parameters)
		cost = compute_cost(A2, Y, parameters)
		grads = backward_propagation(parameters, cache, X, Y)
		parameters  =update_parameters(parameters, grads, learning_rate = 0.5)

		if print_cost:
			if i%1000 == 0:
				print("第", i, "次循环，成本为： " + str(cost))

	return parameters


# 使用学习的参数，为X的每个实例预测一个类
def predict(parameters, X):
	"""
	返回：
		predictions - 预测的向量(红色：0 / 蓝色：1)
	"""
	A2, cache = forward_propagation(X, parameters)
	predictions = np.round(A2)

	return predictions

# 正式运行
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True)

# 绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print("准确率： %d" % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

plt.show()


# 更改隐藏节点的数目来改善准确率
print("========================更改隐藏节点的数目======================")
plt.figure(figsize = (16,32))
hidden_layer_sizes = [1, 2, 3, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
	plt.subplot(5, 2, i + 1)
	plt.title("Hidden Layer of size %d" % n_h)
	parameters = nn_model(X, Y, n_h, num_iterations = 5000)
	plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
	predictions = predict(parameters, X)
	accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
	print("隐藏节点数 ： {} ， 准确率： {} %".format(n_h, accuracy))

plt.show()