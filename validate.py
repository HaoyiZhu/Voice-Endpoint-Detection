''' Script for task 1 validate '''
import argparse
import os

import numpy as np
import joblib
from tqdm import tqdm

from utils import *
from vad_utils import *

parser = argparse.ArgumentParser(description='VAD Validation')
parser.add_argument('--model', type=str, default='svm_not_normalized_0.064_0.032',
                    help='model name')

args = parser.parse_args()

model_path = './models/' + args.model + '.pkl'

frame_size = float(args.model.split('_')[-2])
frame_shift = float(args.model.split('_')[-1])

prefix = './task1/train_' + str(frame_size) + '_' + str(frame_shift)
features_path = prefix + '_features.npy'
target_path = prefix + '_target.npy'

def validate(m, X, y):
    y_pred = m.predict(X).tolist()
    y_true = y.tolist()
    auc, eer = get_metrics(y_pred, y_true)
    return auc, eer

def main():
    if not os.path.exists('./task1/label.json'):
        label_file = "data/dev_label.txt"
        label = read_label_from_file(label_file)
        save_json(label, './task1/label.json')
    else:
        label = read_json('./task1/label.json')

    features, target = sklearn_dataset_for_task_1(
                        label, frame_size=frame_size, frame_shift=frame_shift,
                        features_path=features_path, target_path=target_path
                        )
    target = target.astype(np.float32)
    features = features.astype(np.float32)
    assert features.shape[0] == target.shape[0], \
          ('The shape of features', features.shape, \
           'must match that of target', target.shape)
    
    m = joblib.load(model_path)

    auc, eer = validate(m, features, target)

if __name__ == '__main__':
    main()