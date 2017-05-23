import cv2
import os
import numpy as np
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from losses import weighted_loss

import sys
import time

# path = "/home/lapis-14/Desktop/cnn_hand_recognition/Training/Depth/201406191044/image_"

# path = "/home/lapis-14/Desktop"
model = load_model("/home/lapis-15/Desktop/cnn_hand_recognition/models/CNN_RC_L100_2e.h5",custom_objects={"weighted_loss":weighted_loss})


def test_on_dataset(net="mine"):

    base_folder = "/home/lapis-15/Desktop/my_processed_rc/test/"

    NUM_EL_TEST = 500
    test_data_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".jpg")]))
    # test_labels_list = np.asarray(sorted([i for i in os.listdir(base_folder+"test") if i.endswith(".npy")]))
    data_test = [cv2.imread(base_folder + i, cv2.IMREAD_GRAYSCALE) for i in test_data_list[0:NUM_EL_TEST]]
    # test_labels = [np.load(base_folder+"test/"+i) for i in test_labels_list[0:NUM_EL_TEST]]
    data_test = np.array(data_test).astype("uint8")
    # test_labels = np.array(test_labels).astype("float16").reshape(-1, 12)
    j = 0
    while True:
        # index = raw_input("index 0000-1724 ")
        # if index == "":
        #     break
        #index = str(index)
        ##Carico immagini in grayscale se uso la rete custom
        #image = cv2.imread(path+index+".png", cv2.IMREAD_GRAYSCALE)
        ##Carico immagini a 3 canali se uso la VGG
        # image = cv2.imread(base_folder + test_data_list)
        image = data_test[j]
        img = image.copy()
        # mask = image.copy().astype("float16")
        if image is None:
            print "Image not found.\n"
            x = raw_input("w for next, s for previous")
            if x == 'w':
                j += 1
            else:
                j -= 1
            continue
        if net == "VGG":
            mask1 = image.copy().astype("float16")
            image = preprocess_input(mask1[np.newaxis, ..., ::-1], data_format="channels_last")
        else:
            # mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask1 = image.copy().astype("float16")
            print mask1.shape
            image = mask1[np.newaxis, :, :, np.newaxis]
            print image.shape
        tic = time.clock()
        centers = model.predict(image)[0][:12].reshape(6, 2)
        print time.clock()-tic
        # image = np.squeeze(image)
        # for i in range(0, len(centers), 2):
        #     print str(tuple(centers[i:i+2]))
        #     cv2.circle(image, tuple(centers[i:i+2]), 1, (255, 255, 255), 2)
        for center in centers:
            cv2.circle(img, tuple(center), 1, (255, 255, 255), 2)

        print model.predict(image)[0]
        print "Classe: " +str(np.argmax(model.predict(image)[0][12:]))
        cv2.imshow("circle", img)
        if cv2.waitKey(0) & 0xFF == ord('w'):
            j += 1
            cv2.destroyAllWindows()
        elif cv2.waitKey(0) & 0xFF == ord('s'):
            j -= 1
            cv2.destroyAllWindows()
        elif cv2.waitKey(0) & 0xFF == ord('d'):
            j = int(raw_input("index :"))
        elif cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def bgr_norm(frame):
    frame = frame.astype("float32")
    norm_bgr = np.zeros_like(frame, dtype="float32")

    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    s = b + g + r
    # print s.shape
    norm_bgr[:, :, 0] = b / s * 255
    norm_bgr[:, :, 1] = g / s * 255
    norm_bgr[:, :, 2] = r / s * 255
    # norm_bgr=cv2.convertScaleAbs(norm_bgr)

    return norm_bgr.astype("uint8")


def test_my_hand(name, ind, net):
    print "ciao"
    path = "/home/lapis-15/Desktop/cnn_hand_recognition/"+name+".jpg"
    ###model = load_model("/home/lapis-14/Desktop/cnn_hand_recognition/model_CONVNET4_2.h5")
    image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = bgr_norm(image)
    lower_skin = np.array([0, 95, 101])
    upper_skin = np.array([19, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert from bgr 2 hsv format
    mask = cv2.inRange(hsv, lower_skin, upper_skin)  # make a mask with the image, selecting colors in defined range
    mask = cv2.erode(mask, None, iterations=4)  # old value = 2
    mask = cv2.dilate(mask, None, iterations=7)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones([5,5]), iterations =2)
    res1 = cv2.bitwise_and(image, image, mask=mask)
    #mask = np.where(mask == 255, 0, 125).astype("uint8")
    mask = np.where(res1 == 0, 125, 0).astype("uint8")
    mask = cv2.flip(mask, 1)
    # mask1 = mask[45:, 80:440]
    # img = cv2.resize(mask1, (int(mask1.shape[1] * 0.75), int(mask1.shape[0] * 0.75)))
    # mask[45:, 80:440] = 125
    # mask[image.shape[0] / 2 - img.shape[0] / 2:image.shape[0] / 2 + img.shape[0] / 2+ind,
    # image.shape[1] / 2 - img.shape[1] / 2: image.shape[1] / 2 + img.shape[1] / 2+ind] = img
    # mask1 = mask[45:, 80:440]
    mask1 = mask[25:, 80:540]
    img = cv2.resize(mask1, (int(mask1.shape[1] * 0.65), int(mask1.shape[0] * 0.65)))
    mask[25:, 80:540] = 125
    mask[image.shape[0] / 2 - img.shape[0] / 2:image.shape[0] / 2 + img.shape[0] / 2+ind,
    image.shape[1] / 2 - img.shape[1] / 2: image.shape[1] / 2 + img.shape[1] / 2+ind] = img

    mask = cv2.resize(mask, (160, 120))
    if net == "VGG":
        mask1 = mask.copy().astype("float16")
        image = preprocess_input(mask1[np.newaxis, ..., ::-1], data_format="channels_last")
    else:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask1 = mask.copy().astype("float16")
        image = mask1[np.newaxis, :, :, np.newaxis]


    ##Commento la seguente riga se uso VGG
    #image = image[np.newaxis, np.newaxis, :, :]
    # image = image[np.newaxis, :, :]
    tic = time.clock()

    centers = model.predict(image)[0][:12].reshape(6,2)
    print time.clock() - tic
    # image = np.squeeze(image)
    for center in centers:
        cv2.circle(mask, tuple(center), 1, (255, 255, 255), 2)

    cv2.imshow(name, mask)
    # cv2.imwrite("image_test_norelu.jpg", image)
    # cv2.putText(mask,"{}".format(model.predict(image)[0][12]), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    print "Classe: " + str(np.argmax(model.predict(image)[0][12:]))
    cv2.imwrite(name+str("1e.jpg"),mask)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def test_real_time(net):
    cap = cv2.VideoCapture(0)
    while True:

        r, frame = cap.read()
        image = bgr_norm(frame)
        lower_skin = np.array([0, 0, 92])
        upper_skin = np.array([26, 255, 255])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert from bgr 2 hsv format
        mask = cv2.inRange(hsv, lower_skin, upper_skin)  # make a mask with the image, selecting colors in defined range
        mask = cv2.erode(mask, None, iterations=4)  # old value = 2
        mask = cv2.dilate(mask, None, iterations=7)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones([5,5]), iterations =2)
        res1 = cv2.bitwise_and(image, image, mask=mask)
        # mask = np.where(mask == 255, 0, 125).astype("uint8")
        mask = np.where(res1 == 0, 125, 0).astype("uint8")
        mask = cv2.flip(mask, 1)
        # print mask.shape
        # print mask
        mask = cv2.resize(mask, (160, 120))

        if net == "VGG":
            mask1 = mask.copy().astype("float16")
            image = preprocess_input(mask1[np.newaxis, ..., ::-1], data_format="channels_last")
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask1 = mask.copy().astype("float16")
            image = mask1[np.newaxis, :, :, np.newaxis]

        ##Commento la seguente riga se uso VGG

        ##Tsetti
        # image = image[np.newaxis, :, :]
        tic = time.clock()

        centers, classe = model.predict(image)[0][:12].reshape(6, 2), int(np.argmax(model.predict(image)[0][12:]))
        # print time.clock() - tic
        # image = np.squeeze(image)
        for center in centers:
            # print centers[i]
            cv2.circle(mask, tuple(center), 1, (255, 255, 255), 2)
        cv2.imshow("input", mask)
        print "Classe: " + str(classe)
        # print model.predict(image)[0]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():

    # test_my_hand("open_hand", ind=0, net="mine")
    # test_my_hand("open_hand_new", ind=1, net="mine")
    test_real_time(net="mine")
    # test_on_dataset("mine")

if __name__ == '__main__':
    main()
