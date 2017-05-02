import numpy as np
import os
import cv2


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
