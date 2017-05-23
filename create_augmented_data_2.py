import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

def print_label(img,label):
    img = img.copy()
    for l in label:
        cv2.circle(img,tuple(l.astype("int")),2,(0,255,0),-1)
    return img


#FUNZIONA TUTTO(PERCHe SONO UN GENIO)
W = 160
H = 120
TRAIN_SIZE = 0.95

center = (80, 60)
### Cartella con dataset originale
# base_folder = "/home/lapis-15/Desktop/cnn_hand_recognition/new_data/"
### Cartella da cui prendo il dataset con le mie mani
base_folder = "/home/lapis-15/Desktop/cnn_hand_recognition/my_hand_rc/"

data_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".jpg")]))
labels_list = np.asarray(sorted([i for i in os.listdir(base_folder) if i.endswith(".npy")]))
#numero elementi
#permuto
perm = np.random.permutation(len(data_list))
data_list = data_list[perm]
labels_list = labels_list[perm]
#divido
train_data_list, test_data_list, train_labels_list, test_labels_list  = train_test_split(data_list, labels_list, train_size=TRAIN_SIZE)

# dest_path = "/home/lapis-15/Desktop/cnn_hand_recognition/processed/"
###Salvo il dataset aumentato con le mie mani sul desktop
dest_path = "/home/lapis-15/Desktop/my_processed_rc/"
os.mkdir(dest_path)
os.mkdir(dest_path+"train/")
os.mkdir(dest_path+"test/")

train_n = 0
test_n = 0
img_n = 0
for img_name, label_name in zip(data_list, labels_list):
    print "START WORKING ON IMG {} of {}".format(img_n,len(data_list))
    img = cv2.imread(base_folder + img_name)
    label = np.load(base_folder+label_name)
    class_img = label[-1]
    label = label[:-1].reshape(-1,2)

    # TRASLAZIONE INIZIALE AL CENTRO
    img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = np.where(img_bin < 115, 255, 0).astype("uint8")
    contour = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = max(contour, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    T = np.array([[1, 0, center[0]*2-cx],[0, 1, center[1]*2-cy]], dtype="float32")
    img = cv2.warpAffine(img, T, (W*2,H*2), borderValue=(125, 125, 125))
    label = T.dot(np.insert(label,2,1,axis=1).T).T
    #da qui in poi cambiano center W e h
    img = cv2.resize(img,(160,120),cv2.INTER_CUBIC)
    label = label/2
    #FLIP
    flips = [None,0,1]
    for flip in flips:
        if flip is None:
            img_flip = img.copy()
            label_flip = label.copy()
        elif flip == 0:
            img_flip = cv2.flip(img,0)
            label_flip = label.copy()
            label_flip[:,1] = H-label_flip[:,1]-1


        elif flip == 1:
            img_flip = cv2.flip(img, 1)
            label_flip = label.copy()
            label_flip[:, 0] = W - label_flip[:, 0] -1


        #usa sempre img diverse per le transformazioni perche altrimenti propaghi errori
        #SCALING
        #TUNABLE
        scales = [1.25,1.0,0.75]
        for scale in scales:
            T = cv2.getRotationMatrix2D(center, 0, scale)
            img_s = cv2.warpAffine(img_flip, T, (W, H), borderValue=(125, 125, 125))
            label_s = T.dot(np.insert(label_flip, 2, 1, axis=1).T).T

            # ROTAZIONE
            # TUNABLE
            angle_inc = 10
            for angle in xrange(angle_inc,360,angle_inc):
                T = cv2.getRotationMatrix2D(center,angle,1)
                img_r = cv2.warpAffine(img_s, T, (W, H), borderValue=(125, 125, 125))
                label_r = T.dot(np.insert(label_s, 2, 1, axis=1).T).T

                # TRASLAZIONE GRIGLIA
                # TUNABLE
                grid_inc = 20
                grid = np.mgrid[-center[0]+grid_inc:center[0]-grid_inc:grid_inc,-center[1]+grid_inc:center[1]-grid_inc:grid_inc]
                grid = np.concatenate((grid[0][...,np.newaxis],grid[1][...,np.newaxis]),axis=2).reshape(-1,2)
                for tran in grid:
                    T = np.array([[1, 0, tran[0]], [0, 1, tran[1]]], dtype="float32")
                    img_t = cv2.warpAffine(img_r, T, (W, H), borderValue=(125, 125, 125))
                    label_t = T.dot(np.insert(label_r, 2, 1, axis=1).T).T
                    #IMMAGINE FINALE
                    # background enhancement?
                    img_final = img_t.copy()
                    label_final = label_t.copy().flatten()
                    label_final = np.append(label_final,[class_img])
                    # SAVE
                    if img_name in train_data_list:
                        cv2.imwrite(dest_path+"train/{}.jpg".format(train_n),img_final)
                        np.save(dest_path+"train/{}.npy".format(train_n),label_final)
                        train_n+=1

                    else:
                        cv2.imwrite(dest_path+"test/{}.jpg".format(test_n),img_final)
                        np.save(dest_path+"test/{}.npy".format(test_n),label_final)
                        test_n+=1
    img_n +=1

print "generate {} for train and {} for test".format(train_n,test_n)






