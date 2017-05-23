import cv2
import os
import numpy as np
import time
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,Callback
from keras.applications.imagenet_utils import preprocess_input
from generators import my_generator, train_generator
from matplotlib import pyplot as plt
from losses import weighted_loss
loss = []
val_loss = []
class MyCallbacks(Callback):

    def __init__(self):
        super(MyCallbacks, self).__init__()
        self.i = 0
        self.tic = time.clock()

    def on_batch_end(self, batch, logs=None):
        self.i += 1
        if self.i == 1000:
            tac = time.clock()
            print "loss: {}".format(logs['loss'])
            # print "time: {}".format(tac-self.tic)
            loss.append(logs['loss'])
            self.tic = tac
            self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss.append(logs['val_loss'])


base_folder = "/home/lapis-15/Desktop/my_processed_rc/"
model = load_model("./models/CNN_RC_L100_2e.h5", custom_objects={"weighted_loss":weighted_loss})
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

data_test = data_test[:, :, :, np.newaxis]



print "maracaibo"
while True:
    epochs = raw_input("Epochs: ")

    if epochs == '':
        break
    else:
        epochs = int(epochs)
        model.fit_generator(g,steps_per_epoch=train_len/batch_size,epochs=epochs,verbose=2,callbacks=[ModelCheckpoint("/home/lapis-15/Desktop/cnn_hand_recognition/models/CNN_RC_L100_2e.h5", period=1), MyCallbacks()],validation_data=(data_test,test_labels),max_q_size=50, workers=6)

# epochs = raw_input("# of epochs trained: ")
x = np.arange(0, len(loss), 1)
y = np.arange(0,len(val_loss), 1)
plt.plot(x, np.asarray(loss))
plt.title("Training loss")
plt.plot(y, np.asarray(val_loss))
plt.title("Validation loss")
plt.show()