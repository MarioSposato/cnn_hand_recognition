import cv2
import os
import numpy as np
import time
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,Callback
from generators import my_generator, train_generator


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
model = load_model("./models/model_CONVNET_VGG.h5")
batch_size = 100
g = train_generator(base_folder=base_folder+"train/",generator_batch=batch_size)
# g = my_generator(base_folder+"train/",batch_size)
train_len = g.next()
# print "generator with {} images".format(train_len)
print "generator with {} images".format(g.get_len())

# #TEST
test_data_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".jpg")]))
test_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".npy")]))
perm = np.random.permutation(len(test_data_list))
test_data_list = test_data_list[perm]
test_labels_list = test_labels_list[perm]
###Qui creo una lista in cui carico le immagini e i file delle labels del test_set
NUM_EL_TEST = 2500
data_test = [cv2.imread(base_folder+"test/" + i) for i in test_data_list[0:NUM_EL_TEST]]
test_labels = [np.load(base_folder+"test/"+i) for i in test_labels_list[0:NUM_EL_TEST]]
data_test = np.array(data_test).astype("uint8")
test_labels = np.array(test_labels).astype("float16").reshape(-1, 12)



print "maracaibo"
model.fit_generator(g, steps_per_epoch=g.get_len()/batch_size, epochs=40, verbose=2, callbacks=[ModelCheckpoint(".models/model_CONVNET_VGG.h5", period=1), MyCallbacks()], validation_data=(data_test, test_labels), max_q_size=25, workers=1)
