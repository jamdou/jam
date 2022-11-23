import segyio
import numpy as np


# --------------------读sgy为矩阵------------------------#
# 初始化一个列表存数据，最后再转为矩阵
data_list = []
# sgy文件路径
pp_path = r"D:\matlabm\stk_2000.sgy"
with segyio.open(pp_path, mode="r+", iline=17,
                 xline=13,
                 strict=True,
                 ignore_geometry=False) as f:
    for i in range(len(f.trace)):
        data_list.append(f.trace[i])
# 转为矩阵
data_numpy = np.array(data_list)
'''
# --------------------矩阵写回sgy------------------------#
with segyio.open(pp_path, mode="r+", iline=17,
                 xline=13,
                 strict=True,
                 ignore_geometry=False) as f:
    for i in range(len(f.trace)):
        f.trace[i]=data_numpy[i]
'''