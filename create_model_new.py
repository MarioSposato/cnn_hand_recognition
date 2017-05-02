import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import adam
from keras.utils import np_utils
import sys
from keras.applications import VGG16
from keras.layers.core import Dense,Activation,Dropout,Reshape,Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from generators import my_generator
sys.setrecursionlimit(4000)


base_folder = "/home/lapis-14/Desktop/new_augmented_dataset/"


# ###Creo una lista ordinata con i nomi dei file di immagine e dei file di labels del train set
train_data_list = np.asarray(sorted([i for i in os.listdir(base_folder+"train/") if i.endswith(".jpg")]))
train_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder+"train/") if i.endswith(".npy")]))
print "sorted"
perm = np.random.permutation(len(train_data_list))
train_data_list = train_data_list[perm]
train_labels_list = train_labels_list[perm]
print "permuted"
###Qui creo una lista in cui carico le immagini e i file delle labels del train_set
NUM_EL_TRAIN = 150000
data_train = [cv2.imread(base_folder+ "train/" + i, cv2.IMREAD_GRAYSCALE) for i in train_data_list[0:NUM_EL_TRAIN]]
data_train = np.array(data_train).astype("uint8")
train_labels = [np.load(base_folder+ "train/" + i) for i in train_labels_list[0:NUM_EL_TRAIN]]
train_labels = np.array(train_labels).astype("float16").reshape(-1, 12)
print "loaded"
###Creo una lista ordinata con i nomi dei file di immagine e dei file di labels del test set
# batch_size = 64
# g = my_generator(base_folder+"train/",batch_size)
# train_len = g.next()
# print "generator with {} images".format(train_len)
# #TEST
test_data_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".jpg")]))
test_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".npy")]))
perm = np.random.permutation(len(test_data_list))
test_data_list = test_data_list[perm]
test_labels_list = test_labels_list[perm]
###Qui creo una lista in cui carico le immagini e i file delle labels del test_set
NUM_EL_TEST = 500
data_test = [cv2.imread(base_folder+"test/" + i, cv2.IMREAD_GRAYSCALE) for i in test_data_list[0:NUM_EL_TEST]]
test_labels = [np.load(base_folder+"test/"+i) for i in test_labels_list[0:NUM_EL_TEST]]
data_test = np.array(data_test).astype("uint8")
test_labels = np.array(test_labels).astype("float16").reshape(-1, 12)

data_train = data_train[:, :, :,  np.newaxis]
data_test = data_test[:, :, :, np.newaxis]

print "maracaibo"
# exit()
# build the CNN
model = Sequential()
# first set of CONV => RELU => POOL
# erano 20 filtri 9x9
model.add(Conv2D(20, (9,9), padding="same",
                        input_shape=(120, 160, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Conv2D(48, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(48, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# # set of FC => RELU layers
model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(500))
model.add(Activation("relu"))

# softmax classifier
model.add(Dense(12))
model.add(Activation("relu"))

for m in model.layers:
    print m.output_shape
# train the model using adam
print("[INFO] compiling model...")
model.compile(loss="mean_squared_error", optimizer=adam(),
              metrics=["accuracy"])
while True:
    epochs = raw_input("epoch ")
    if epochs == "":
        break
    epochs = int(epochs)
    model.fit(data_train, train_labels, nb_epoch=epochs, batch_size=128,
              verbose=1,validation_data=(data_test,test_labels))
    # print model.predict(data_train[0:10])
    # while True:
    #     index = raw_input("index 0-499")
    #     if index == "":
    #         break
    #     index = int(index)
    #     image = data_test[index:index+1].copy()
    #     centers = model.predict(image)[0]
    #     image = np.squeeze(image)
    #
    #     for i in range(0,len(train_labels[0]),2):
    #         cv2.circle(image, tuple(centers[i:i+2]), 2, (255,255,255), 2)
    #
    #     cv2.imshow("circle", image)
    #     cv2.waitKey(0)
# #saving
# model.save("/home/lapis-14/Desktop/cnn_hand_recognition/model_CONVNET4_2.h5")
# exit()
# vgg = VGG16(include_top=False,weights="imagenet",input_shape=(120,160,3))
# for layer in vgg.layers:
#     layer.trainable = False
#
# x = Flatten()(vgg.output)
# x = Dense(4096, name="a")(x)
# x = Activation("relu", name="b")(x)
# x = Dense(1024, name = "c")(x)
# # x = Dropout(0.25)(x)
# x = Activation("relu", name="d")(x)
# x = Dense(128, name="e")(x)
# # x = Dropout(0.15)(x)
# x = Activation("relu", name="f")(x)
# x = Dense(32, name = "g")(x)
# x = Activation("relu", name="h")(x)
# x = Dense(12, name ="i")(x)
# # x = Activation("relu", name="l")(x)
#
# m = Model(inputs=vgg.input,outputs=x)
#
# for layer in m.layers:
#     print "{} {}".format(layer.output_shape, layer.trainable)


m.compile(optimizer=Adam(), loss="mse")
#batch da 100,ne tengo in ram 250
# m.fit_generator(g,steps_per_epoch=train_len/batch_size,epochs=40,verbose=1,callbacks=[ModelCheckpoint("/home/lapis-14/Desktop/cnn_hand_recognition/model_CONVNET_VGG.h5", period=1)],validation_data=(data_test,test_labels),max_q_size=25,workers=1)
m.fit(data_train, train_labels, epochs=40, batch_size=128, verbose=1, shuffle=True, validation_data=(data_test, test_labels), callbacks=[ModelCheckpoint("/home/lapis-14/Desktop/cnn_hand_recognition/model_CONVNET_CNN.h5", period=5)])
# while True:
#     index = raw_input("index 0-499")
#     if index == "":
#         break
#     index = int(index)
#     image = data_test[index:index + 1].copy()
#     centers = m.predict(image)[0]
#     image = np.squeeze(image)
#     centers = centers.reshape(6,2)
#     for center in centers:
#         cv2.circle(image, tuple(center.astype("int")), 2, (255, 255, 255), 2)
#
#     cv2.imshow("circle", image)
#     cv2.waitKey(0)