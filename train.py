''' Script for task 1 training'''

import argparse
import os
import time

import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from vad_utils import read_label_from_file
from evaluate import get_metrics
from utils import sklearn_dataset, read_json, save_json, build_model

"""----------------------------- Training options -----------------------------"""
parser = argparse.ArgumentParser(description='VAD Training')
parser.add_argument('--f_size', type=float, default=0.032,
                    help='frame size')
parser.add_argument('--f_shift', type=float, default=0.008,
                    help='frame shift')
parser.add_argument('--task', type=int, default=1,
                    help='task id')
parser.add_argument('--exp', type=str, default='svm_not_normalized',
                    help='Experiment ID')
parser.add_argument('--save_name', type=str, default='train',
                    help='file name while saving data for lazing loading')
"""----------------------------- Only meaningful in task1 -----------------------------"""
parser.add_argument('--model', type=str, default='svm',
                    help='what kind of model to use, supported: svm, linear, ridge, logistic, lasso')
parser.add_argument('--scaler', default=False, action='store_true',
                    help='whether to normalize the input data')
parser.add_argument('--cv', type=int, default=None,
                    help='cv value, only supported while using ridge, logistic or lasso')
parser.add_argument('--tol', type=float, default=0.001,
                    help='Tolerance for stopping criterion when using SVM.')
parser.add_argument('--c', type=float, default=1.0,
                    help='Regularization parameter when using SVM.')
"""----------------------------- Only meaningful in task2 -----------------------------"""
parser.add_argument('--n_cpnt', type=int, default=2,
                    help='The number of mixture components.')

args = parser.parse_args()

frame_size = args.f_size
frame_shift = args.f_shift

prefix = './task' + str(args.task) + '/' + args.save_name + '_' + str(frame_size) + '_' + str(frame_shift)
features_path = prefix + '_features.npy'
target_path = prefix + '_target.npy'

exp_id = args.exp + '_' + str(frame_size) + '_' + str(frame_shift)

label_path = './task' + str(args.task) + '/train_label' + '_' + str(frame_size) + '_' + str(frame_shift) + '.json'

def train(m, X, y):
    m.fit(X, y)
    score = m.score(X, y)
    return score

def validate(m, X, y):
    y_pred = m.predict(X).tolist()
    y_true = y.tolist()
    auc, eer = get_metrics(y_pred, y_true)
    return auc, eer

def exp(m, features, target, name):
    features_positive = features[target > 0.5, :]
    features_negative = features[target < 0.5, :]
    target_positive = target[target > 0.5]
    target_negative = target[target < 0.5]

    print('Experiment Name: ', name)
    start_time = time.time()
    score = train(m, features, target)
    print('Training score :', score)
    print('Training time :', time.time() - start_time)
    print('-----------------------------------------')

    joblib.dump(m, './models/' + name + '.pkl')

    auc, err = validate(m, features, target)
    print('AUC :', auc)
    print('ERR :', err)

    pred_pos = m.predict(features_positive)
    pred_neg = m.predict(features_negative)
    acc_pos = np.sum(pred_pos) / np.sum(target_positive)
    acc_neg = np.sum(pred_neg) / np.sum(target_negative)
    print('Positive Acc :', acc_pos)
    print('Negative Acc :', acc_pos)

def main():
    if not os.path.exists(label_path):
    	print('loading labels...')
        label_file = "data/dev_label.txt" if args.task == 1 else "data/train_label.txt"
        label = read_label_from_file(label_file, frame_size=frame_size, frame_shift=frame_shift)
        save_json(label, label_path)
    else:
    	print('lazy loading labels...')
        label = read_json(label_path)

    features, target = sklearn_dataset(
                        label, task=args.task,
                        frame_size=frame_size, frame_shift=frame_shift,
                        features_path=features_path, target_path=target_path
                        )

    m = build_model(args)

    exp(m, features, target, exp_id)

if __name__ == '__main__':
    main()