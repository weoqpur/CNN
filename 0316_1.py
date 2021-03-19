import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

image = tf.constant([[[[1], [2], [3]],
                    [[4], [5], [6]],
                    [[7], [8], [9]]]], dtype = np.float32)
print(image.shape)
plt.imshow(image.numpy().reshape(3, 3), cmap='Greys')
plt.show()

weight = np.array([[[[1.]],[[1.]]],
                   [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID', 
                             kernel_initializer=weight_init)(image)
print("conv2d.shape", conv2d.shape)
print(conv2d.numpy().reshape(2,2))
plt.imshow(conv2d.numpy().reshape(2,2), cmap='gray')
plt.show()



print("image.shape", image.shape)

weight = np.array([[[[1., 10., -1.]], [[1., 10., -1.]]],
                    [[[1., 10., -1.]],[[1., 10., -1.]]]])
print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding='SAME',
                            kernel_initializer=weight_init)(image)
print("con2d.shape", conv2d.shape)
feature_maps= np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    print(feature_map.reshape(3,3))
    plt.subplot(1, 3, i+1), plt.imshow(feature_map.reshape(3,3), cmap='gray')
plt.show()