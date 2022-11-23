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
name1=r"D:\line2000\cmp5998471.segy"
    
#    name1=r"C:\Users\wts_duxin\Desktop\line2000\cmp"+str(tt)+".segy"
data_list = []
    # sgy文件路径
pp_path = name1
with segyio.open(pp_path, mode="r+", iline=17,
                 xline=13,
                 strict=True,
                 ignore_geometry=False) as f:
        for i in range(len(f.trace)):
            data_list.append(f.trace[i])
            # 转为矩阵
data_numpy = np.array(data_list)
   
    #exec(f'cmp{tt}=data_numpy')
#train_data=data_numpy
train_data=np.transpose(data_numpy)
train_data1=np.transpose(train_data[550:750,:])





u, s, v = np.linalg.svd(train_data1)


n_comp = 30
    #n_comp = 50
for tt in range(2,n_comp):
#    if tt==2:
#      train_restruct=train_data*0
      
#    train_old=train_restruct    
#    err1=train_data-train_restruct

    dirct_data = u[:, :tt]



  
    from sklearn import linear_model

    max_iter = 30
    dictionary = dirct_data
 
    y = train_data1
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
'''     
    exec(f'train_restruct_{tt}=train_restruct')
    if tt==2:
        train_restruct_1=train_data-train_data
        errr_1=1
    exec(f'train_only_{tt}=train_restruct_{tt}-train_restruct_{tt-1}')
    exec(f'train_er_{tt}=train_data-train_restruct_{tt}')
    exec(f'err_{tt}=np.sum(train_data**2)')
    exec(f'err2_{tt}=np.sum(train_er_{tt})**2')
    exec(f'errr_{tt}=err_{tt}/err2_{tt}')
    
    bb=eval(f'errr_{tt-1}')  
    aa=eval(f'errr_{tt}')
    name2=r"d:\jam_python\ksvdout\ksvdcmp5998471_m30_"+str(tt)+".csv"
    np.savetxt(name2,train_restruct,delimiter = ',')
    noise=train_data-train_restruct
    name3=r"d:\jam_python\ksvdout\ksvdcmp5998471_m30_noise_"+str(tt)+".csv"
    np.savetxt(name3,noise,delimiter = ',')
    
#    if aa<bb:
#       break 




    err2=train_old-train_restruct
    
    err1no=normalizejam(err1)
    err2no=normalizejam(err2)
    train_no=normalizejam(train_old)
  
  

name2=r"d:\jam_python\ksvdout\test\ksvdcmp5998471_t.csv"
np.savetxt(name2,train_restruct,delimiter = ',')

name3=r"d:\jam_python\ksvdout\test\cmp5998471.csv"
np.savetxt(name3,train_data,delimiter = ',')

ccc=train_data-train_restruct
name4=r"d:\jam_python\ksvdout\test\cmp5998471noise.csv"
np.savetxt(name4,ccc,delimiter = ',')
#for tt in range(5998043,5999195) 8140
'''