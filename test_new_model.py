import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F
import pickle
from cgp import *
from chainer.backends import cuda
from cnn_model import CGP2CNN
import argparse
import pandas as pd
from cnn_train import CNN_train  # Ensure this matches the class name in cnn_model.py
from kaakaa_dataset import build_labels_dict, _get_kaakaa_dataset


def load_model():
    parser = argparse.ArgumentParser(description='Arguments for evaluating a model after its generation.')
    parser.add_argument('--eval_directory', '-eval_dir', default='./', help='The directory the set of images you\'re testing on can be found.')
    parser.add_argument('--log_directory', '-log_dir', default='./', help='The directory the pickle, log, and .model can be found in.')
    args = parser.parse_args()

    
    network_info_pickle = os.path.join(args.log_directory, "network_info.pickle")
    log_file = os.path.join(args.log_directory, "log_cgp.txt")
    model_file = os.path.join(args.log_directory, "retrained_net.model")

    with open(network_info_pickle, mode='rb') as f:
        network_info = pickle.load(f)

    #loading the network architecture from the pickle
    cgp = CGP(network_info, None)
    data = pd.read_csv(log_file, header=None)  #loading the log file
    cgp.load_log(list(data.tail(1).values.flatten().astype(int)))  # Read the log at final generation
    cgp_new = cgp.pop[0].active_net_list()

    #provides us with our base model w/o weights
    model = CGP2CNN(cgp_new, 17) 
    serializers.load_npz(model_file, model) #loads our weights into model

    labels_dict = build_labels_dict(args.eval_directory)
    reversed_labels = {v: k for k, v in labels_dict.items()}
    dataset = _get_kaakaa_dataset(args.eval_directory, labels_dict, (224,224))

    x_test = np.stack([image[0] for image in dataset])  
    y_test = np.array([label[1] for label in dataset], dtype=np.int32)

    #sorting things just in case they're not in class order
    sort_idx = np.argsort(y_test)

    x_test = x_test[sort_idx]
    y_test = y_test[sort_idx]

    num_classes = np.unique(y_test)
    batchsize = 16
    y_pred = []
    per_class_stats = {
        cls: {"accuracy": None, "precision": None, "recall": None, "f1": None}
        for cls in range(len(num_classes))
    }
    for cls in range(len(num_classes)):
        #since 
        idx = np.where(y_test == cls)[0]
        x_cls = x_test[idx]
        y_cls = y_test[idx]

        correct = 0
        total = len(y_cls)

        
        for i in range(0, total, batchsize):
            x_batch = x_cls[i:i+batchsize]
            y_batch = y_cls[i:i+batchsize]

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                model(x_batch, y_batch)
            
            logits = model.outputs[-1]
            y_pred_batch = F.argmax(logits, axis=1).array
            y_pred.extend(y_pred_batch)
                
            batch_acc = float(model.accuracy.data)
                
            correct += (batch_acc * len(y_batch))
            
        per_class_stats[cls]["accuracy"] = correct / total if total > 0 else None
    
    
    #now that we've gotten all predictions, we go through the classes again to calculate f1, precision, recall
    for cls in range(len(num_classes)):
        per_class_stats[cls]["precision"], per_class_stats[cls]["recall"], per_class_stats[cls]["f1"] = compute_stats(cls, y_test, y_pred)

    for cls, stats in per_class_stats.items():
        print(f"  Class {cls}: {stats}")

def compute_stats(cls, y_test, y_pred):
    y_pred = np.array(y_pred, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    true_pos = np.sum((y_pred == cls) & (y_test == cls))

    false_pos = np.sum((y_pred == cls) & (y_test != cls))

    false_neg = np.sum((y_pred != cls) & (y_test == cls))

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0

    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

load_model()