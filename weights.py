import numpy as np
from scipy.linalg import eig

def check_matrix(matrix):
    """  
    核查判断矩阵是否满足以下条件：
        条件1: 判断矩阵是方阵;
        条件2: 判断矩阵的对角线元素全为1;
        条件3: 判断矩阵的对角线元素互为倒数。
    
    Args:
        matrix (numpy.ndarray): 判断矩阵, 对于 n 个因素，其形状为 (n, n)。
    
    return:
        判断矩阵的维度 (integer)
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("判断矩阵不是方阵!")
    
    diag_elements = np.diag(matrix)
    
    if not np.allclose(diag_elements, 1):
        raise ValueError("判断矩阵对角线元素不全为1!")
    
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            if not abs(matrix[i][j] * matrix[j][i] - 1) < 1e-7:
                raise ValueError("判断矩阵对角线元素不全互为倒数!")
    
    return matrix.shape[0]

def ahp_weight_calculation(matrix, index):
    """
    使用层次分析法 (AHP) 计算权重并检查一致性。
    
    Args:
        matrix (numpy.ndarray): 判断矩阵, 对于 n 个因素，其形状为 (n, n);
        index (list): 判断矩阵的行或列索引。
    
    return:
        包含权重向量和一致性比率等信息的字典 (dict)
    """

    n = check_matrix(matrix)

    # step 1: 计算特征向量和特征值
    eigenvalues, eigenvectors = eig(matrix)

    # 输出判断矩阵的最大特征值
    print("最大特征值：")
    print(np.round(np.max(eigenvalues).real, 4))

    # 取最大特征值对应的特征向量，并归一化
    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvector = np.real(eigenvectors[:, max_eigenvalue_index])
    weights = max_eigenvector / np.sum(max_eigenvector)
    weights = np.round(weights, 4)

    # 输出权重
    print("各因素的权重：")
    for i, j in zip(index, weights):
        print(f'{i}: {j}')

    # 步骤 2: 检查一致性
    # 计算一致性指标CI
    n = matrix.shape[0]
    CI = (np.real(eigenvalues[max_eigenvalue_index]) - n) / (n - 1)

    # 随机一致性指标RI
    RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.89, 5: 1.12, 6: 1.26, 7: 1.36, 8: 1.41, 9: 1.46, 10: 1.49}[n]

    # 计算一致性比率CR
    CR = CI / RI
    CR = np.round(CR, 4)

    print("\n一致性比率 CR:")
    print(CR)

    # 判断是否一致性合理
    if CR < 0.1:
        print("\n矩阵的一致性良好!")
        return {"index": index, "max_eigenvalue": np.round(np.max(eigenvalues).real, 4), "CR ": CR, "pass": True, "weights": weights}
    else:
        print("\n矩阵的一致性较差, 需要重新评估!")
        return {"index": index, "max_eigenvalue": np.round(np.max(eigenvalues).real, 4), "CR ": CR, "pass": False, "weights": weights}

def matrix_combine(arr):
    """
    对多位专家给出的打分矩阵进行合并, 得到最终的判断矩阵。

    Args:
        arr (numpy.ndarray): 多位专家给出的打分矩阵, 形状为 (n, m, m), 其中 n 为专家数量, m 为矩阵阶数(指标数量)。

    return:
        合并后的判断矩阵, 形状为 (m, m)。 (numpy.ndarray)
    """

    # 确保输入矩阵是三维的 (n, m, m)
    assert arr.ndim == 3, "输入矩阵必须是三维的 (n, m, m)"
    
    # 计算几何平均值: 对应值相乘然后取1/m次方
    product_arr = np.prod(arr, axis = 0)
    result = np.power(product_arr, 1 / arr.shape[0])
    
    return result

# 函数测试示例
if __name__ == "__main__":
    matrix_1 = np.array([
        [1  , 4  , 5  , 5  ],
        [1/4, 1  , 2  , 4  ],
        [1/5, 1/2, 1  , 2  ],
        [1/5, 1/4, 1/2, 1  ]
    ])

    matrix_2 = np.array([
            [1  , 3  , 2  , 5  ],
            [1/3, 1  , 2  , 4  ],
            [1/2, 1/2, 1  , 2  ],
            [1/5, 1/4, 1/2, 1  ]
        ])

    matrix_3 = np.array([
            [1  , 3  , 2  , 5  ],
            [1/3, 1  , 2  , 2  ],
            [1/2, 1/2, 1  , 4  ],
            [1/5, 1/2, 1/4, 1  ]
        ])

    arr = matrix_combine(np.array([matrix_1, matrix_2, matrix_3]))
    result = ahp_weight_calculation(arr, ["因素1", "因素2", "因素3", "因素4"])
    print(result)

# 实验测试
if __name__ == "__main__":
    import pandas as pd

    ## indicators (pandas.DataFrame): 企业税务风险指标数据
    indicators = pd.read_csv("output/indicators.csv")
    indicators['证券代码'] = indicators['证券代码'].apply(lambda x: f'{x:06d}')

    # ## 这里为测试方便，只选择了部分指标，实际中下面一行代码需注释掉
    # indicators = indicators[[
    #     '证券代码',
    #     '证券简称',
    #     '营业收入变动率',
    #     '营业成本变动率',
    #     '营业费用变动率',
    #     '销售费用变动率',
    #     '管理费用变动率',
    #     '财务费用变动率',
    #     '研发费用变动率',
    #     '成本费用利润率',
    #     '营业利润变动率',
    #     '营业外收入变动率',
    #     '营业收入变动率与营业利润变动率配比',
    #     '营业收入变动率与营业成本变动率配比',
    #     '营业收入变动率与营业费用变动率配比',
    #     '营业成本变动率与营业利润变动率配比',
    #     '增值税税收负担率',
    #     '应纳税额与工业增加值弹性系数',
    #     '工业增加值税负差异率'
    # ]]

    import json

    ## 载入对所选指标的分组信息（增值税、企业所得税和其他税种）
    ## 这部分数据需前端向后端发送
    ## 注意：这里只是示例，'data/二级指标划分.json'中的数据不完全正确，具体数据需根据学生实际填写情况来定
    with open('data/二级指标划分.json', 'r', encoding='utf-8') as file:
        three_group_infos = json.load(file)

    indicators_1 = indicators[three_group_infos['增值税风险识别指标']]
    indicators_2 = indicators[three_group_infos['企业所得税风险识别指标']]
    indicators_3 = indicators[three_group_infos['其他税种税务风险识别指标']]

    ## 假设pairwise_matrix_1是五位专家打分矩阵合并后的判断矩阵(第一类指标：假设为"增值税风险识别指标")
    pairwise_matrix_1 = np.array([
        [1  , 2  , 2  ],
        [1/2, 1  , 2  ],
        [1/2, 1/2, 1  ]
    ])

    index_1 = list(indicators_1.columns)

    result_AHP_1 = ahp_weight_calculation(pairwise_matrix_1, index_1)
    result_AHP_1

    ## 假设pairwise_matrix_2是五位专家打分矩阵合并后的判断矩阵(第二类指标：假设为"企业所得税风险识别指标")
    pairwise_matrix_2 = np.array([
        [1  , 1  , 1  , 4  , 3  , 2  , 2  , 2  , 2  , 1  ],
        [1  , 1  , 1  , 3  , 1  , 3  , 2  , 4  , 2  , 2  ],
        [1  , 1  , 1  , 3  , 5  , 2  , 1  , 2  , 4  , 2  ],
        [1/4, 1/3, 1/3, 1  , 3  , 2  , 2  , 3  , 2  , 4  ],
        [1/3, 1  , 1/5, 1/3, 1  , 3  , 2  , 2  , 2  , 2  ],
        [1/2, 1/3, 1/2, 1/2, 1/3, 1  , 3  , 2  , 2  , 3  ],
        [1/2, 1/2, 1  , 1/2, 1/2, 1/3, 1  , 3  , 1  , 2  ],
        [1/2, 1/4, 1/2, 1/3, 1/2, 1/2, 1/3, 1  , 1  , 1  ],
        [1/2, 1/2, 1/4, 1/2, 1/2, 1/2, 1  , 1  , 1  , 1  ],
        [1  , 1/2, 1/2, 1/4, 1/2, 1/3, 1/2, 1  , 1  , 1  ]
    ])

    index_2 = list(indicators_2.columns)

    result_AHP_2 = ahp_weight_calculation(pairwise_matrix_2, index_2)

    ## 假设pairwise_matrix_3是五位专家打分矩阵合并后的判断矩阵(第三类指标：假设为"其他税种税务风险识别指标")
    pairwise_matrix_3 = np.array([
        [1  , 4  , 5  , 5  ],
        [1/4, 1  , 2  , 4  ],
        [1/5, 1/2, 1  , 2  ],
        [1/5, 1/4, 1/2, 1  ]
    ])

    index_3 = list(indicators_3.columns)

    result_AHP_3 = ahp_weight_calculation(pairwise_matrix_3, index_3)

    ## 假设pairwise_matrix_0是对'增值税风险识别指标', '企业所得税风险识别指标', '其他税种税务风险识别指标'三类总指标的判断矩阵
    pairwise_matrix_0 = np.array([
        [1  , 1/4, 2  ],
        [4  , 1  , 5  ],
        [1/2, 1/5, 1  ]
    ])

    index_0 = ['增值税风险识别指标', '企业所得税风险识别指标', '其他税种税务风险识别指标']

    result_AHP_0 = ahp_weight_calculation(pairwise_matrix_0, index_0)

    ## 将三类指标的权重合并
    result_AHP = {
        'index': result_AHP_1['index'] + result_AHP_2['index'] + result_AHP_3['index'],
        'total_weights': np.array([result_AHP_0['weights'][0]] * len(result_AHP_1['weights']) + [result_AHP_0['weights'][1]] * len(result_AHP_2['weights']) + [result_AHP_0['weights'][2]] * len(result_AHP_3['weights'])),
        'weights': np.array(list(result_AHP_1['weights']) + list(result_AHP_2['weights']) + list(result_AHP_3['weights']))
    }

    ## weights.csv: 存放行业税务风险指标权重
    weights = pd.DataFrame(np.concatenate((result_AHP['total_weights'].reshape(1, -1), result_AHP['weights'].reshape(1, -1)), axis = 0), columns = result_AHP['index'])
    weights.index = ['二级指标权重', '三级指标权重']
    weights.to_csv('output/weights.csv', encoding = 'utf-8-sig')
    