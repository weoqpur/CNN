#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# mnist data를 불러옴
mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# train이미지와 test이미지로 분리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 픽셀 값을 0과 1로 나누기 위해 255로 나눔
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# train 이미지의 첫번째 이미지를 가져 옴
img = train_images[0]

# image를 4차원으로 나눔
img = img.reshape(-1, 28, 28, 1)
# convert_to_tensor() : 텐서로 변경
img = tf.convert_to_tensor(img)
# 정규 분포에 따라 텐서를 생성 (stddev: 생성하는 텐서의 표준 편차)
weight_init = keras.initializers.RandomNormal(stddev=0.01)
# kernel_initializer 초기값 설정기를 전달하는 인수
conv2d = keras.layers.Conv2D(filters=5, kernel_size=3, strides=(2, 2),
        padding='SAME', kernel_initializer=weight_init)(img)

print(conv2d.shape)

# np.swapaxes(array, 바꿀 축1, 바꿀 축2) 0이면
# 가장 바깥 축, 숫자가 커질 수록 차원이 증가
feature_maps = np.swapaxes(conv2d, 0, 3)
# enumerate(순서가 있는 자료형) 인덱스를 포함한 enumerate 객체로 반환
for i, feature_map in enumerate(feature_maps):
    # subplot(행, 열, 인덱스)
    plt.subplot(1,5,i+1), plt.imshow(feature_map.reshape(14, 14),
    cmap = 'gray')

pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
        padding='SAME')(conv2d)
print(pool.shape)

feature_maps = np.swapaxes(pool, 0, 3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1, 5, i+1), plt.imshow(feature_map.reshape(7, 7),
    cmap = 'gray')
plt.show() 
# %%