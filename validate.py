''' Script for task 1 validate '''
import argparse
import os

import numpy as np
import joblib
from tqdm import tqdm

from utils import *
from vad_utils import *

"""----------------------------- Validation options -----------------------------"""
parser = argparse.ArgumentParser(description='VAD Validation')
parser.add_argument('--model', type=str, default='svm_not_normalized_0.064_0.032',
                    help='model name')
parser.add_argument('--save_name', type=str, default='train',
                    help='file name while saving data for lazing loading')
parser.add_argument('--task', type=int, default=1,
                    help='task id')

args = parser.parse_args()

model_path = './models/' + args.model + '.pkl'

frame_size = float(args.model.split('_')[-2])
frame_shift = float(args.model.split('_')[-1])

prefix = './task' + str(args.task) + '/' + args.save_name + '_' + str(frame_size) + '_' + str(frame_shift)
features_path = prefix + '_features.npy'
target_path = prefix + '_target.npy'

label_path = './task' + str(args.task) + '/val_label' + '_' + str(frame_size) + '_' + str(frame_shift) + '.json'

def validate(m, X, y):
    y_pred = m.predict(X).tolist()
    y_true = y.tolist()
    auc, eer = get_metrics(y_pred, y_true)
    return auc, eer

def main():
    if not os.path.exists(label_path):
        label_file = "data/dev_label.txt"
        label = read_label_from_file(label_file, frame_size=frame_size, frame_shift=frame_shift)
        save_json(label, label_path)
    else:
        label = read_json(label_path)

    features, target = sklearn_dataset(
                        label, task=args.task,
                        frame_size=frame_size, frame_shift=frame_shift,
                        features_path=features_path, target_path=target_path
                        )
    
    m = joblib.load(model_path)

    auc, eer = validate(m, features, target)

if __name__ == '__main__':
    main()