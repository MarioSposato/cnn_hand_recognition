import cv2
import os
import numpy as np
import time
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,Callback
from generators import my_generator, train_generator
from generators_for_cineca import my_generator
from keras.applications import VGG16
from keras.layers.core import Dense,Activation,Dropout,Reshape,Flatten
from keras.optimizers import Adam
from keras.models import Model

class MyCallbacks(Callback):
    def __init__(self):
        super(MyCallbacks, self).__init__()
        self.i = 0
        self.tic = time.clock()
    def on_batch_end(self, batch, logs=None):
        self.i += 1

        if self.i == 1000:
            tac = time.clock()
            print logs['loss'], logs['batch']
            print tac-self.tic
            self.tic = tac
            self.i = 0


base_folder = "/home/lapis-14/Desktop/new_augmented_dataset/"


g = my_generator(base_folder+"train/",num_el=5100)
# print "generator with {} images".format(len(data_train))

# #TEST
test_data_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".jpg")]))
test_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".npy")]))
perm = np.random.permutation(len(test_data_list))
test_data_list = test_data_list[perm]
test_labels_list = test_labels_list[perm]
###Qui creo una lista in cui carico le immagini e i file delle labels del test_set
NUM_EL_TEST = 250
data_test = np.asarray([(cv2.imread(base_folder + "test/"+ i).astype("float16")-(103.939,116.779,123.68))[...,::-1] for i in test_data_list[0:NUM_EL_TEST]])
test_labels = [np.load(base_folder+"test/"+i) for i in test_labels_list[0:NUM_EL_TEST]]
test_labels = np.array(test_labels).astype("uint8").reshape(-1, 12)

vgg = VGG16(include_top=False,weights="imagenet",input_shape=(120,160,3))
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(4096, name="a")(x)
x = Activation("relu", name="b")(x)
x = Dense(1024, name = "c")(x)
x = Dropout(0.05)(x)
x = Activation("relu", name="d")(x)
x = Dense(128, name="e")(x)
#x = Dropout(0.15)(x)
x = Activation("relu", name="f")(x)
x = Dense(32, name = "g")(x)
x = Activation("relu", name="h")(x)
x = Dense(12, name ="i")(x)
# x = Activation("relu", name="l")(x)

model = Model(inputs=vgg.input,outputs=x)
model.compile(optimizer=Adam(), loss="mse")
print "maracaibo at CINECA"

for giga_batch in g:
    model.fit(giga_batch[0],giga_batch[1],batch_size=128,epochs=3,verbose=1,shuffle=True,validation_data=(data_test,test_labels),callbacks=[ModelCheckpoint("models/aere.h5", period=1), MyCallbacks()])

