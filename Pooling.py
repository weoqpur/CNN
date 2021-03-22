# Max Pooling : 필터안에 최대 값을 구함
# Average Pooling : 필터안에 평균 값을 구함

# data_format: (batch, channels, height, width)

import tensorflow as tf
import numpy as np
from tensorflow import keras

image = tf.constant([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
                    # pool_size: 정수형이나 튜플(2개의 정수)로 값을 줌
                    # strides: 정수형, 튜플(2개의 정수)로 값을 줌
                    # VALID = 패딩을 하지 않음 
                    # SAME = 패딩을 함
pool = keras.layers.MaxPool2D(pool_size = (2, 2), strides = 1, padding = 'SAME')(image)

# shape: data_format을 출력
print(pool.shape)
print(pool.numpy())