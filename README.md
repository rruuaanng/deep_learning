# deep_learning
2023 Autumn Course Neural Networks and Deep Learning


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Accuracy


# 整理数据
# 将三个文件的数据整合到data中

# 导入三个数据集
happy = np.load(f'./process/happy.npy')[:8000]
normal = np.load(f'./process/normal.npy')[:8000]
sad = np.load(f'./process/sad.npy')[:8000]
happy_label = np.full_like(happy, 0)
normal_label = np.full_like(normal, 1)
sad_label = np.full_like(sad, 2)

happy = happy.reshape(-1,8000,1)
happy_label = happy_label.reshape(-1,8000,1)
happy.shape

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,8000,1)),
    tf.keras.layers.Conv1D(3,3),
    tf.keras.layers.Dense(1, activation='softmax')  # 输出层为 3 个单元，使用 softmax 激活函数
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy')

model.fit(happy, happy_label, epochs=5)

输出：
...
    File "e:\anaconda3\Lib\site-packages\keras\src\engine\input_spec.py", line 298, in assert_input_compatibility
        raise ValueError(

    ValueError: Input 0 of layer "sequential_75" is incompatible with the layer: expected shape=(None, 1, 8000, 1), found shape=(None, 8000, 1)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
