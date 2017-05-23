import cv2
import os
import numpy as np
import time
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input,Concatenate
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import adam
from keras.utils import np_utils
import sys
from keras.applications import VGG16, VGG19
from keras.layers.core import Dense,Activation,Dropout,Reshape,Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, History
from generators import my_generator, train_generator
from keras.applications.imagenet_utils import preprocess_input
from keras.layers.normalization import BatchNormalization
from losses import weighted_loss
# sys.setrecursionlimit(4000)
from matplotlib import pyplot as plt
loss = []
val_loss = []
class MyCallbacks(Callback):

    def __init__(self):
        super(MyCallbacks, self).__init__()
        self.i = 0
        self.tic = time.clock()

    def on_batch_end(self, batch, logs=None):
        self.i += 1
        if self.i == 100:
            tac = time.clock()
            print "loss: {}".format(logs['loss'])
            # print "time: {}".format(tac-self.tic)
            loss.append(logs['loss'])
            self.tic = tac
            self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss.append(logs['val_loss'])

# base_folder = "/home/lapis-15/Desktop/cnn_hand_recognition/processed/"
### Dataset aumentato delle mie mani annotate
base_folder = "/home/lapis-15/Desktop/my_processed_rc/"

# ###Creo una lista ordinata con i nomi dei file di immagine e dei file di labels del train set
# train_data_list = np.asarray(sorted([i for i in os.listdir(base_folder+"train/") if i.endswith(".jpg")]))
# train_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder+"train/") if i.endswith(".npy")]))
# print "sorted"
# perm = np.random.permutation(len(train_data_list))
# train_data_list = train_data_list[perm]
# train_labels_list = train_labels_list[perm]
# print "permuted"
# ###Qui creo una lista in cui carico le immagini e i file delle labels del train_set
# NUM_EL_TRAIN = 150000
# data_train = [cv2.imread(base_folder+ "train/" + i, cv2.IMREAD_GRAYSCALE) for i in train_data_list[0:NUM_EL_TRAIN]]
# data_train = np.array(data_train).astype("uint8")
# train_labels = [np.load(base_folder+ "train/" + i) for i in train_labels_list[0:NUM_EL_TRAIN]]
# train_labels = np.array(train_labels).astype("float16").reshape(-1, 12)
# print "loaded"
###Creo una lista ordinata con i nomi dei file di immagine e dei file di labels del test set
batch_size = 128
g = train_generator(base_folder+"train/", batch_size)
train_len = g.get_len()
print "generator with {} images".format(train_len)
# #TEST
test_data_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".jpg")]))
test_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".npy")]))
perm = np.random.permutation(len(test_data_list))
test_data_list = test_data_list[perm]
test_labels_list = test_labels_list[perm]
###Qui creo una lista in cui carico le immagini e i file delle labels del test_set
NUM_EL_TEST = 2500
data_test = [cv2.imread(base_folder+"test/" + i, cv2.IMREAD_GRAYSCALE) for i in test_data_list[0:NUM_EL_TEST]]
test_labels = [np.load(base_folder+"test/"+i) for i in test_labels_list[0:NUM_EL_TEST]]
data_test = np.array(data_test).astype("float16")
# data_test = preprocess_input(data_test[...,::-1], data_format="channels_last")
# print len(test_labels[0])
# print test_labels[0]
test_labels = np.array(test_labels).astype("float16").reshape(-1, 13)

#data_train = data_train[:, :, :,  np.newaxis]
data_test = data_test[:, :, :, np.newaxis]

print "maracaibo"
# exit()
# build the CNN
in_conv = Input(shape=(120, 160, 1))
conv = Conv2D(64, (3, 3), padding="same")(in_conv)
#conv = BatchNormalization(axis=-1)(conv)
conv = Activation("relu")(conv)
conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

conv = Conv2D(128, (3, 3), padding="same")(conv)
#x = BatchNormalization(axis=-1)(x)
conv = Activation("relu")(conv)
conv = Conv2D(128, (3, 3), padding="same")(conv)
conv = Activation("relu")(conv)

conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

conv = Conv2D(256, (3, 3), padding="same")(conv)
conv = Activation("relu")(conv)
conv = Conv2D(256, (3, 3), padding="same")(conv)
conv = Activation("relu")(conv)
conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

conv = Conv2D(512, (3, 3), padding="same")(conv)
conv = Activation("relu")(conv)
conv = MaxPooling2D(pool_size=(2, 2))(conv)
conv = GlobalAveragePooling2D()(conv)
#COMUNE

conv = Dense(64)(conv)
conv = Activation("relu")(conv)

#REGRESSIONE
reg = Dense(12)(conv)
#CATEGORIZZAZIONE
cat = Dense(11)(conv)
#cat = Activation("softmax")(conv)

#RIUNISCO
out = Concatenate(axis=-1)([reg,cat])
model = Model(inputs=[in_conv],outputs=[out])

# for m in model.layers:
#     print m.output_shape
# train the model using adam

model.summary()
print("[INFO] compiling model...")

model.compile(optimizer=Adam(), loss=weighted_loss)
#batch da 100,ne tengo in ram 250
while True:
    epochs = raw_input("Epochs: ")

    if epochs == '':
        break
    else:
        epochs = int(epochs)
        model.fit_generator(g,steps_per_epoch=train_len/batch_size,epochs=epochs,verbose=1,callbacks=[ModelCheckpoint("/home/lapis-15/Desktop/cnn_hand_recognition/models/my_model_5cnnbatchnorm_c_1.h5", period=1), MyCallbacks()],validation_data=(data_test,test_labels),max_q_size=50, workers=6)

# epochs = raw_input("# of epochs trained: ")
conv = np.arange(0, len(loss), 1)
plt.plot(conv, np.asarray(loss))
plt.show()

