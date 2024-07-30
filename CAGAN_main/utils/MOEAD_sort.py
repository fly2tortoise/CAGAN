import numpy as np

def Prefer_MOEAD(random_F,  N, weight):
    # 归一化函数
    def normalize_data(F):
        F_min = np.min(F, axis=0)
        F_max = np.max(F, axis=0)
        return (F - F_min) / (F_max - F_min)
    # 对数据进行归一化
    random_F = normalize_data(random_F)
    # 定义权重向量
    # weights = np.array([1/3, 1/3, 1/3])
    # weights = np.array([3/10, 3/10, 3/10])
    weights = weight
    # 确定理想点
    ideal_point = np.min(random_F, axis=0)
    # 计算切比雪夫距离
    def calculate_tchebycheff_distance(F, weights, ideal_point):
        tchebycheff_distances = np.max(weights * np.abs(F - ideal_point), axis=1)
        return tchebycheff_distances
    # 计算所有个体的切比雪夫距离
    tchebycheff_distances = calculate_tchebycheff_distance(random_F, weights, ideal_point)
    # 根据切比雪夫距离进行排序
    sorted_indices_tchebycheff = np.argsort(tchebycheff_distances)

    return sorted_indices_tchebycheff[:N]

