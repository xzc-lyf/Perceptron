import math
import matplotlib.pyplot as plt
import numpy as np

'''
:param
    file_path: The dataset file path.
:returns
    features: A list used to store the coordinates in the dataset.
        ['-8.459748692855033', '5.621530829461882'],
        ['4.237208820754372', '-6.462792989330957'],
        ['11.644215057276138', '9.451504823451543'],
        ['12.66037789730385', '-3.6062685957325904'], ...
    labels: A list stores the label (1 or -1) for each data point.
        [-1, 1, -1, 1, ...]
    dimension: Number of features (2, 4, 8)
    radius: radius (16, 24, 12).
'''
def load_dataset(file_path):
    features, labels = [], []
    with open(file_path, 'r') as dataset:
        lines = dataset.readlines()
        metadata = lines[0].strip().split(',')
        dimension, radius = int(metadata[0]), int(metadata[-1])
        for line in lines[1:]:
            values = line.strip().split(',')
            features.append(values[:dimension])
            labels.append(int(values[-1]))
    return features, labels, dimension, radius

'''
:param
    vec1: The vector1 to be computed.
    vec2: The vector2 to be computed.
:returns
    res: The result of dot product.
'''
def compute_dot_product(vec1, vec2):
    return sum(float(a) * float(b) for a, b in zip(vec1, vec2))

"""
:param
    radius: The points of S fall in a ball that centers at the origin and has radius R.
    gamma: Denote by γ the largest margin of all the separation planes.
:returns
    res: Maximum number of iterations.
"""
def calculate_max_iterations(radius, gamma_opt):
    return math.ceil(12 * (radius ** 2) / (gamma_opt ** 2))

"""
:param
    weights: The weight vector to be adjusted.
    features: A list used to store the coordinates in the dataset.
    labels: A list stores the label (1 or -1) for each data point.
:returns
    res: Maximum number of iterations.
"""
def update_weights(weights, point, label):
    return [w + label * float(x) for w, x in zip(weights, point)]

"""
:param
    features: A list used to store the coordinates in the dataset.
    labels: A list stores the label (1 or -1) for each data point.
    weights: The weight vector to be adjusted.
    gamma_guess: An arbitrary value (Set to radius in this case).
:returns
    res1: Index
    res2: Distance between the violation point and the plane.
"""
def check_violation(features, labels, weights, gamma_guess):
    len_w = math.sqrt(sum(w ** 2 for w in weights)) if any(weights) else 0
    for index, (point, label) in enumerate(zip(features, labels)):
        dot_val = compute_dot_product(weights, point)
        curr_margin = math.fabs(dot_val / len_w) if len_w else 0
        if (label == 1 and dot_val <= 0) or (label == -1 and dot_val >= 0) or curr_margin < gamma_guess / 2:
            return index, curr_margin
    return -1, curr_margin


def train_model(features, labels, max_iterations, radius, dimension):
    weights = [0] * dimension
    gamma_threshold = radius

    for iteration in range(max_iterations):
        violation_index, margin = check_violation(features, labels, weights, gamma_threshold)

        if violation_index == -1:
            print("Training complete - no violations.")
            print(f"Final Weights after {iteration} iterations:", weights)
            return weights, True  # 返回权重和训练成功标志

        # 打印信息
        print(f"\nIteration {iteration + 1}")
        print("Current Weights:", weights)
        print("Margin of violating sample:", margin)
        print("Violating sample:", features[violation_index])
        print("Label of violating sample:", labels[violation_index])

        # 更新权重
        weights = update_weights(weights, features[violation_index], labels[violation_index])

        # 输出更新后的权重
        print("Weights after update:", weights)
        print('-' * 40)

    print("Training stopped - maximum iterations reached.")
    return weights, False  # 如果最大迭代数达到，则返回False


def train_margin_perceptron(file_list):
    for file in file_list:
        features, labels, dimension, radius = load_dataset(file)
        gamma = radius
        max_iterations = calculate_max_iterations(radius, gamma)

        while True:
            weights, training_complete = train_model(features, labels, max_iterations, radius, dimension)

            if training_complete or gamma <= 1e-8:
                if gamma <= 1e-8:
                    print("Converged to approximate requirement; stopping margin perceptron.")
                break

            gamma /= 2
            max_iterations = calculate_max_iterations(radius, gamma)

        print(f"Final gamma: {gamma}")
        print("Final weights:", weights)
        break  # 假设仅训练第一个文件


def plot_decision_boundary(features, labels, weights, title="Margin Perceptron Training"):
    """
    可视化数据点和当前决策边界.
    :param features: 数据点的坐标列表
    :param labels: 数据点的标签列表 (1 或 -1)
    :param weights: 当前权重向量
    :param title: 图的标题
    """
    # 转换features为浮点型矩阵
    features = np.array(features, dtype=float)

    # 创建一个二维平面
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # 计算决策边界
    Z = np.sign(weights[0] * xx + weights[1] * yy)
    plt.contourf(xx, yy, Z, alpha=0.1, cmap=plt.cm.coolwarm)

    # 绘制数据点
    positive_samples = features[np.array(labels) == 1]
    negative_samples = features[np.array(labels) == -1]
    plt.scatter(positive_samples[:, 0], positive_samples[:, 1], c='b', label="Positive (1)", edgecolor='k')
    plt.scatter(negative_samples[:, 0], negative_samples[:, 1], c='r', label="Negative (-1)", edgecolor='k')

    # 绘制权重向量 (权重的方向即为分隔线的方向)
    if any(weights):  # 避免初始0向量情况
        slope = -weights[0] / (weights[1] if weights[1] != 0 else 1e-10)
        intercept = 0
        x_vals = np.array([x_min, x_max])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, 'k--', linewidth=1, label="Decision Boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(loc="best")
    plt.show()


# 更新 train_model 函数，加入可视化
def train_model_with_visualization(features, labels, max_iterations, radius, dimension):
    weights = [0] * dimension
    gamma_threshold = radius

    for iteration in range(max_iterations):
        violation_index, margin = check_violation(features, labels, weights, gamma_threshold)

        # 可视化当前决策边界
        plot_decision_boundary(features, labels, weights, title=f"Iteration {iteration + 1}")

        if violation_index == -1:
            print("Training complete - no violations.")
            print(f"Final Weights after {iteration} iterations:", weights)
            return weights, True

        print(f"\nIteration {iteration + 1}")
        print("Current Weights:", weights)
        print("Margin of violating sample:", margin)
        print("Violating sample:", features[violation_index])
        print("Label of violating sample:", labels[violation_index])

        # 更新权重
        weights = update_weights(weights, features[violation_index], labels[violation_index])

    print("Training stopped - maximum iterations reached.")
    return weights, False


# 在 train_margin_perceptron 中调用新函数
def train_margin_perceptron_with_visualization(file_list):
    for file in file_list:
        features, labels, dimension, radius = load_dataset(file)
        gamma = radius
        max_iterations = calculate_max_iterations(radius, gamma)

        while True:
            weights, training_complete = train_model_with_visualization(features, labels, max_iterations, radius,
                                                                        dimension)

            if training_complete or gamma <= 1e-8:
                if gamma <= 1e-8:
                    print("Converged to approximate requirement; stopping margin perceptron.")
                break

            gamma /= 2
            max_iterations = calculate_max_iterations(radius, gamma)

        print(f"Final gamma: {gamma}")
        print("Final weights:", weights)

# 调用训练和可视化函数
train_margin_perceptron_with_visualization(["./Dataset1.txt", "./Dataset2.txt", "./Dataset3.txt"])
