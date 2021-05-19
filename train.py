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

train_label_path = './task' + str(args.task) + '/train_label' + '_' + str(frame_size) + '_' + str(frame_shift) + '.json'
val_label_path = './task' + str(args.task) + '/val_label' + '_' + str(frame_size) + '_' + str(frame_shift) + '.json'

def train(m, X, y):
    m.fit(X, y)
    score = m.score(X, y)
    return score

def validate(m, X, y):
    y_pred = m.predict(X).tolist()
    y_true = y.tolist()
    auc, eer = get_metrics(y_pred, y_true)
    return auc, eer

def exp(m, features_train, target_train, features_val, target_val, name):
    print('Experiment Name: ', name)
    start_time = time.time()
    score = train(m, features_train, target_train)
    print('Training score :', score)
    print('Training time :', time.time() - start_time)
    print('-----------------------------------------')

    joblib.dump(m, './models/' + name + '.pkl')

    train_auc, train_err = validate(m, features_train, target_train)
    print('Taining AUC :', train_auc)
    print('Taining ERR :', train_err)

    if target_val:
        print('-----------------------------------------')
        train_auc, train_err = validate(m, features_val, target_val)
        print('Val AUC :', val_auc)
        print('Val ERR :', val_err)

def main():
    if not os.path.exists(train_label_path):
        print('loading training labels...')
        train_label_file = "data/dev_label.txt" if args.task == 1 else "data/train_label.txt"
        train_label = read_label_from_file(train_label_file, frame_size=frame_size, frame_shift=frame_shift)
        save_json(train_label, train_label_path)
    else:
        print('lazy loading training labels...')
        train_label = read_json(train_label_path)

    features_train, target_train = sklearn_dataset(
                        train_label, task=args.task, mode='train',
                        frame_size=frame_size, frame_shift=frame_shift,
                        features_path=features_path, target_path=target_path
                        )

    if args.task == 2:
        if not os.path.exists(val_label_path):
            print('loading validation labels...')
            val_label_file = "data/dev_label.txt"
            val_label = read_label_from_file(val_label_file, frame_size=frame_size, frame_shift=frame_shift)
            save_json(val_label, val_label_path)
        else:
            print('lazy loading validation labels...')
            val_label = read_json(val_label_path)
        features_val, target_val = sklearn_dataset(
                            val_label, mode='val',
                            frame_size=frame_size, frame_shift=frame_shift,
                            features_path=features_path, target_path=target_path
                            )
    else:
        features_val, target_val = None, None


    m = build_model(args)

    exp(m, features_train, target_train, features_val, target_val, exp_id)

if __name__ == '__main__':
    main()