import numpy as np
from chainer import dataset
import cv2
import os
import glob

class fetch_kaakaa:
    def build_labels_dict(self, dir):
        subdirs = sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])
        labels_dict = {name: idx for idx, name in enumerate(subdirs)}
        print("label dictionary created", labels_dict)
        return labels_dict #will return a dict of ints:strings
    
    def get_kaakaa_Dataset(self, data_dir):
        #we assume we're passed the base directory
        train_dir = os.path.join(data_dir, "train")
        print("train_dir", train_dir)
        #chainer/cgp-cnn doesn't use a val directory ;_; 
        test_dir = os.path.join(data_dir, "val")
        print("test_dir", test_dir)

        labels_dict = self.build_labels_dict(train_dir) #each folder should have all birds, so this is fine

        train = kaakaa_dataset(train_dir, labels_dict, size=(224, 224))
        test = kaakaa_dataset(test_dir, labels_dict, size=(224, 224))
        
        return train, test

    


class kaakaa_dataset(dataset.DatasetMixin):
    def __init__(self, data_dir, labels_dict, size=(224, 224)):
        """
        data_dir: root directory with images
        labels_dict: mapping from filename -> label (int)
        size: resize images to this shape (H, W)
        """
        self.data = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.labels_dict = labels_dict
        self.size = size
        print(f"size of {data_dir}: {size} ")
    
    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        # Load image
        path = self.data[i]
        img = cv2.imread(path)[:, :, ::-1]  # BGR -> RGB
        img = cv2.resize(img, self.size)

        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W) for chainer

        # the folder label
        fname = os.path.basename(os.path.dirname(path))
        label = self.labels_dict[fname]

        return img, np.int32(label)