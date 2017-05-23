import numpy as np
import os
import cv2
from keras.applications.imagenet_utils import preprocess_input

def my_generator(base_folder, batch_size=100):
    """

    :param base_folder: path base
    :param batch_size: dimensione estratta ogni volta
    :return: tupla batch dati e labels
    """
    data_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".jpg")]))
    labels_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".npy")]))
    assert len(data_list) == len(labels_list)
    yield len(data_list)
    i = 0
    while True:
        if i == 0 or len(data_list[i * batch_size:(i + 1) * batch_size]) == 0:
            perm = np.random.permutation(len(data_list))
            data_list = data_list[perm]
            labels_list = labels_list[perm]
            i = 0

        data_batch = np.asarray(
            [cv2.imread(base_folder + j) for j in data_list[i * batch_size:(i + 1) * batch_size]]).astype("float16")
        labels_batch = np.asarray([np.load(base_folder+j) for j in labels_list[i * batch_size:(i + 1) * batch_size]]).astype(
            "float16")

        yield (data_batch, labels_batch.reshape(-1,12))
        i += 1
import threading

# generatore per estrarre il train
class train_generator(object):
    def __init__(self,base_folder,generator_batch):
        self.train_data_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".jpg")]))
        self.train_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".npy")]))
        self.generator_batch = generator_batch
        self.base_folder = base_folder
        assert len(self.train_data_list) == len(self.train_labels_list)
        self.lock = threading.Lock()
        self.i = 0
    def __iter__(self):
        return self
    def get_len(self):
        return len(self.train_data_list)

    def next(self):
        #LOCK
        with self.lock:
            if self.i == 0 or len(self.train_data_list[self.i * self.generator_batch:(self.i + 1) * self.generator_batch]) == 0:
                #inizio batch o fine
                self.i = 0
                #shuffle
                perm = np.random.permutation(self.get_len())
                self.train_data_list = self.train_data_list[perm]
                self.train_labels_list = self.train_labels_list[perm]

            train_data_list_t = self.train_data_list[self.i * self.generator_batch:(self.i + 1) * self.generator_batch]
            train_labels_list_t = self.train_labels_list[self.i * self.generator_batch:(self.i + 1) * self.generator_batch]
            self.i += 1

        #NO LOCK
        # se sono qui ho qualcosa, devo caricarlo
        # train data come float32 e tolgo la media(Baraldi)

        train_data = np.zeros((len(train_data_list_t), 120, 160), dtype="float16")
        train_labels = np.zeros((len(train_labels_list_t), 13), dtype="float16")

        for j in xrange(len(train_data_list_t)):
            train_data[j] = cv2.imread(self.base_folder + train_data_list_t[j], cv2.IMREAD_GRAYSCALE).astype("float16")
            # t[...,0] -= 103.939
            # t[...,1] -= 116.779
            # t[...,2] -= 123.68
            # t = t[...,::-1]
            # print train_data_list_t[j]

            # train_data[j] = t[..., ::-1]
            train_labels[j] = np.load(self.base_folder + train_labels_list_t[j])
        # print train_data.shape
        return (train_data[:, :, :,  np.newaxis], train_labels)
        # return (preprocess_input(train_data, data_format="channels_last"), train_labels.reshape(-1,12))
