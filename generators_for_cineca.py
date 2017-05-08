import numpy as np
import os
import cv2
import gc


def my_generator(base_folder, num_el=25000):
    """

    :param base_folder: path base
    :param batch_size: dimensione estratta ogni volta
    :return: tupla batch dati e labels
    """
    data_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".jpg")]))
    labels_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".npy")]))

    perm = np.random.permutation(len(data_list))
    data_list = data_list[perm]
    labels_list = labels_list[perm]

    assert len(data_list) == len(labels_list)

    i = 0
    while True:
        for j in xrange(20):
            gc.collect()
        if len(data_list[i * num_el:(i + 1) * num_el]) == 0:
            raise StopIteration

        data_batch = np.asarray(
            [(cv2.imread(base_folder + j).astype("float16")-(103.939,116.779,123.68))[...,::-1] for j in data_list[i * num_el:(i + 1) * num_el]])
        labels_batch = np.asarray([np.load(base_folder+j) for j in labels_list[i * num_el:(i + 1) * num_el]]).astype(
            "uint8")

        yield (data_batch, labels_batch.reshape(-1,12))
        del data_batch
        i += 1