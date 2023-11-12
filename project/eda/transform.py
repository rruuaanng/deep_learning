import tensorflow as tf
import pandas as pd
import numpy as np
import os

# 数据目录
happy = 'E:/dataset/EDA/happy'
normal = 'E:/dataset/EDA/normal'
sad = 'E:/dataset/EDA/sad'

# # MP150
# data = np.array([])

# # 遍历文件夹中的数据并保存到data中
# for fn in os.listdir(sad):
#     tmp = pd.read_csv(f'{sad}/{fn}') \
#                  .to_numpy() \
#                  .T[0]
#     data = np.append(data, tmp)

# np.save('./sad.npy',data)

