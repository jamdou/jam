import numpy as np
import pandas as pd
import segyio
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog

'''
train_data_mat = loadmat("../data/train_data2.mat")
 
train_data = train_data_mat["Data"]
train_label = train_data_mat["Label"]
 
print(train_data.shape, train_label.shape)
'''

# --------------------读sgy为矩阵------------------------#
# 初始化一个列表存数据，最后再转为矩阵




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


def normalizejam(data):

    mins = data.min(0) #返回data矩阵中每一列中最小的元素，返回一个列表

    maxs = data.max(0) #返回data矩阵中每一列中最大的元素，返回一个列表

    ranges = maxs - mins #最大值列表 - 最小值列表 = 差值列表

    normData = np.zeros(np.shape(data)) #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据

    row = data.shape[0] #返回 data矩阵的行数

    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值

    normData = normData / np.tile(ranges,(row,1)) #data矩阵每一列数据都除去每一列的差值(差值 = 某列的最大值- 某列最小值)

    return normData





#for tt in range(5998140,5999098):
#name1=r"D:\line2000\cmp5998471.segy"
#name1=r"D:\line2000\cmp_962_860.sgy"
#name1=r"D:\line2000\cip_1078.sgy"
root = tk.Tk()
root.withdraw()
name1 = filedialog.askopenfilename()  

   
#    name1=r"C:\Users\wts_duxin\Desktop\line2000\cmp"+str(tt)+".segy"
data_list = []
    # sgy文件路径
pp_path = name1
with segyio.open(pp_path, mode="r+", iline=17,
                 xline=13,
                 strict=False,
                 ignore_geometry=False) as f:
        for i in range(len(f.trace)):
            data_list.append(f.trace[i])
            # 转为矩阵
data_numpy = np.array(data_list)
   
    #exec(f'cmp{tt}=data_numpy')
#train_data=data_numpy
train_data=np.transpose(data_numpy)
train_data2=train_data[680:790,0:224]






u, s, v = np.linalg.svd(train_data2)


n_comp =30
    #n_comp = 50
for tt in range(2,n_comp):
#    if tt==2:
#      train_restruct=train_data*0
      
#    train_old=train_restruct    
#    err1=train_data-train_restruct

    dirct_data = u[:, :tt]



  
    from sklearn import linear_model

    max_iter = 40
    dictionary = dirct_data
 
    y = train_data2
    tolerance = 1e-6
 
    for i in range(max_iter):
               # 稀疏编码
               x = linear_model.orthogonal_mp(dictionary, y)
               e = np.linalg.norm(y - np.dot(dictionary, x))/np.linalg.norm(y)
               if e < tolerance:
                   break
               dict_update(y, dictionary, x, tt)
 
    sparsecode = linear_model.orthogonal_mp(dictionary, y)
 
    train_restruct = dictionary.dot(sparsecode)