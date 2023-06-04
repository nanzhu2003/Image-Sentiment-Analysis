from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import keras.backend as K

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os

from tensorflow.python.keras.engine.input_layer import InputLayer

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

img_rows, img_cols = 300, 300  # 指定图像的高度和宽度
img_channels = 1  # 指定图像的通道数

path1 = "images"  # 指定原始图像的路径
path2 = "images_resized"  # 指定调整尺寸后的图像保存路径（处理成灰度图像）

listing = os.listdir(path1)
num_samples = np.size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '//' + file)
    img = im.resize((img_rows, img_cols))
    gray = img.convert('L')
    # need to do some more processing here
    gray.save(path2 + '//' + file, "JPEG")

imlist = os.listdir(path2)

im1 = np.array(Image.open('images_resized' + '//' + imlist[0]))
m, n = im1.shape[0:2]
imnbr = len(imlist)

immatrix = np.array([np.array(Image.open('images_resized' + '//' + im2)).flatten()
                     for im2 in imlist], 'f')

label = np.ones((num_samples,), dtype=int)
label[0:1260] = 0
label[1260:2716] = 1
label[2716:] = 2
data, Label = shuffle(immatrix, label, random_state=2)
train_data = [data, Label]  # 将数据和标签打包成数据集

img = immatrix[167].reshape(img_rows, img_cols)
plt.imshow(img)
plt.imshow(img, cmap='gray')
print(train_data[0].shape)
print(train_data[1].shape)

batch_size = 16  # 32
nb_classes = 3
nb_epoch = 20

nb_filters = 32
nb_pool = 2
nb_conv = 3

(X, y) = (train_data[0], train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 256
plt.figure()
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i, :])
plt.show()
model = Sequential()
model.add(InputLayer(input_shape=(img_rows, img_cols, img_channels)))

# 添加卷积层，指定卷积核的数量、卷积核的大小和边界模式
model.add(Convolution2D(nb_filters, kernel_size=(3, 3), strides=(1, 1), padding='valid'))

'''
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        padding='valid',
                        input_shape=(1, img_rows, img_cols)))
'''
convout1 = Activation('relu')  # 定义第一个激活函数
model.add(convout1)  # 将第一个激活函数添加到模型中

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout3 = Activation('relu')
model.add(convout3)

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))  # 添加池化层，指定池化窗口的大小
model.add(Dropout(0.5))  # 添加Dropout层，防止过拟合

model.add(Flatten())
model.add(Dense(512))  # 添加全连接层，指定神经元的数量
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_data=(X_test, Y_test))
K.clear_session()
#  模型训练
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)
K.clear_session()
# visualizing losses and accuracy

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(nb_epoch)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
print(plt.style.available)  # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
y_pred = model.predict(X_test[1:5])
predicted_classes = np.argmax(y_pred, axis=1)
print(predicted_classes)

print(Y_test[1:5])

# visualizing intermediate layers

output_layer = model.layers[1].output
output_fn = K.function([model.layers[0].input], [output_layer])

# the input image

input_image = X_train[0:1, :, :, :]
print(input_image.shape)

plt.imshow(input_image[0, 0, :, :], cmap='gray')
plt.imshow(input_image[0, 0, :, :])
plt.show()
output_image = output_fn(input_image)
output_image = np.array(output_image[0])

# Rearrange dimension so we can plot the result
# output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)  # numpy.rollaxis(arr, axis, start)

fig = plt.figure(figsize=(8, 8))
for i in range(32):
    ax = fig.add_subplot(6, 6, i + 1)
    ax.imshow(output_image[0, :, :, i], interpolation='nearest')  # to see the first filter
    ax.imshow(output_image[0, :, :, i], cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt.show()

# Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(X_test)
print("Y_predition:\n", Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print("y_predition:\n", y_pred)

p = model.predict(X_test)  # 获取类别的概率预测

target_names = ['Negative', 'Neutral', 'Positive']
print(classification_report(np.argmax(Y_test, axis=1), y_pred, target_names=target_names))
print(confusion_matrix(np.argmax(Y_test, axis=1), y_pred))
