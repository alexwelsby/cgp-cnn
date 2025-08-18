import numpy as np
from chainer import dataset
import cv2
import chainer
import os
import glob
import numpy as np
from chainer.datasets import TupleDataset

#basing it off this https://github.com/chainer/chainer/blob/master/chainer/datasets/cifar.py
def build_labels_dict(dir):
        subdirs = sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])
        labels_dict = {name: idx for idx, name in enumerate(subdirs)}
        print("label dictionary created", labels_dict)
        return labels_dict #will return a dict of ints:strings
    
def get_kaakaa_dataset(data_dir):
    #we assume we're passed the base directory
    train_dir = os.path.join(data_dir, "train")
    print("train_dir", train_dir)
    #chainer/cgp-cnn doesn't use a val directory ;_; 
    test_dir = os.path.join(data_dir, "val")
    print("test_dir", test_dir)

    labels_dict = build_labels_dict(train_dir) #each folder should have all birds, so this is fine

    train = _get_kaakaa_dataset(train_dir, labels_dict, size=(128, 128))
    test = _get_kaakaa_dataset(test_dir, labels_dict, size=(128, 128))
        
    return train, test

def _get_kaakaa_dataset(data_dir, labels_dict, size=(224, 224)):
    data = sorted(glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True))
    x_list = []
    y_list = []
    for path in data:
        img = cv2.imread(path)[:, :, ::-1]  # BGR -> RGB

        img = cv2.resize(img, size) #resize to our new size

        img = img.astype(chainer.config.dtype) #what cifar does and cgp-cnn expects
            
        img *= (1 / 255.0) #normalizing

        img = img.transpose(2, 0, 1) # (H,W,C) -> (C,H,W) for chainer
            
        fname = os.path.basename(os.path.dirname(path)) #getting the folder name
        label = labels_dict[fname] #getting the int label associated w/ it

        x_list.append(img)
        y_list.append(label) #adding label

    images = np.stack(x_list, axis=0)
    labels = np.array(y_list, dtype=np.int32)

    return TupleDataset(images, labels)
    