import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Set Hyper Parameters

learning_rate = 0.001
training_epochs = 15
batch_size = 100

cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'minst_cnn_seq'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

# 2. Make a Data Pipelining
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터를 0과 1로 구분하기 위해 255로 나눔
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# axis -1은 마지막 인덱스로 채널을 추가해 줌
train_images = np.expand_dims(train_images, axis = -1)
test_images = np.expand_dims(test_images, axis = -1)

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size = 100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

def create_model():
    model = keras.Sequential()
    # convolution 3x3x32 relu를 써서 음수를 없앴음
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu,padding='SAME',input_shape=(28, 28, 1)))
    # MaxPooling 2x2(default)
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    # convolution 3x3x64 relu를 써서 음수를 없앴음
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    # MaxPooling 2x2(default)
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    # convolution 3x3x128 relu를 써서 음수를 없앴음
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    # MaxPooling 2x2(default)
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    # 앞에서 나왔던 모든 피처맵을 백터로 펴 줌
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    # 과적합 : 이미 주어진 데이터를 분석하는 능력은 뛰어나나 새로운 데이터를 분석하는 능력은 떨어지는 현상
    # 과적합 현상을 방지하기 위해 드롭아웃 방법으로 정규화 시켜 줌
    model.add(keras.layers.Dropout(0.4))
    # 레이어를 한 번 더 생성
    model.add(keras.layers.Dense(10))
    return model
model = create_model()
model.summary()

def loss_fn(model, images, labels):
    # 이미지의 드랍아웃을 적용
    logits = model(images, training = True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels = labels))
    return loss

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
        # 테이프를 거꾸로 감아서 gradient를 계산해 줌
    return tape.gradient(loss, model.variables)

# 아담옵티마이저로 가중치를 조절
optimizer = tf.optimizers.Adam(learning_rate = learning_rate)

def evaluate(model, images, labels):
    # 트레이닝을 하는 단계가 아님
    logits = model(images, training = False)
    #                               결과와                      정답 둘이 비교 같으면 맞고 다르면 틀림
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # 위 연산의 평균
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

checkpoint = tf.train.Checkpoint(cnn=model)
#13:54