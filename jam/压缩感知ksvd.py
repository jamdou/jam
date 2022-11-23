import numpy as np
import pandas as pd
import segyio
from scipy.io import loadmat
'''
train_data_mat = loadmat("../data/train_data2.mat")
 
train_data = train_data_mat["Data"]
train_label = train_data_mat["Label"]
 
print(train_data.shape, train_label.shape)
'''

# --------------------读sgy为矩阵------------------------#
# 初始化一个列表存数据，最后再转为矩阵
data_list = []
# sgy文件路径
pp_path = r"D:\matlabm\stk_2000.sgy"
with segyio.open(pp_path, mode="r+", iline=17,
                 xline=13,
                 strict=True,
                 ignore_geometry=False) as f:
    for t in range(len(f.trace)):
        data_list.append(f.trace[t])
# 转为矩阵
data_numpy = np.array(data_list)

#train_data = data_numpy
train_data = np.transpose(data_numpy)


u, s, v = np.linalg.svd(train_data)
n_comp = 20
dict_data = u[:, :n_comp]


def dict_update(y, d, x, n_components):
    """
    使用KSVD更新字典的过程
    """
    for i in range(n_components):
        index = np.nonzero(x[i, :])[0]
        if len(index) == 0:
            continue
        # 更新第i列
        d[:, i] = 0
        # 计算误差矩阵
        r = (y - np.dot(d, x))[:, index]
        # 利用svd的方法，来求解更新字典和稀疏系数矩阵
        u, s, v = np.linalg.svd(r, full_matrices=False)
        # 使用左奇异矩阵的第0列更新字典
        d[:, i] = u[:, 0]
        # 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
        for j,k in enumerate(index):
            x[i, k] = s[0] * v[0, j]
    return d, x

from sklearn import linear_model
 
max_iter = 20
dictionary = dict_data
 
y = train_data
tolerance = 1e-6
 
for ii in range(max_iter):
    # 稀疏编码
    x = linear_model.orthogonal_mp(dictionary, y)
    #e = np.linalg.norm(y - np.dot(dictionary, x))
    e = np.linalg.norm(y - np.dot(dictionary, x))/np.linalg.norm(y)
    if e < tolerance:
        break
    dict_update(y, dictionary, x, n_comp)
 
sparsecode = linear_model.orthogonal_mp(dictionary, y)
 
train_restruct = dictionary.dot(sparsecode)


#name22=r"d:\jam\python\stk_raw.csv"
#np.savetxt(name22,data_numpy,delimiter = ',')
