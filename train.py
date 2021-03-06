''' Script for task 1 training'''

import argparse
import os
import time

import numpy as np
import joblib

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
# parser.add_argument('--save_name', type=str, default='train',
#                     help='file name while saving data for lazing loading')
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

prefix_train = './task' + str(args.task) + '/' + 'train' + '_' + str(frame_size) + '_' + str(frame_shift)
prefix_val = './task' + str(args.task) + '/' + 'val' + '_' + str(frame_size) + '_' + str(frame_shift)
train_features_path = prefix_train + '_features.npy'
train_target_path = prefix_train + '_target.npy'
val_features_path = prefix_val + '_features.npy'
val_target_path = prefix_val + '_target.npy'

exp_id = args.exp + '_' + str(frame_size) + '_' + str(frame_shift)

train_label_path = './task' + str(args.task) + '/train_label' + '_' + str(frame_size) + '_' + str(frame_shift) + '.json'
val_label_path = './task' + str(args.task) + '/val_label' + '_' + str(frame_size) + '_' + str(frame_shift) + '.json'

def train(m, X, y):
    if args.task == 1:
        m.fit(X, y)
        score = m.score(X, y)
    else:
        score1 = m[0].fit(X[y >= 0.5])
        score2 = m[1].fit(X[y < 0.5])
        score = [score1, score2]

    return score

def validate(m, X, y):
    if args.task == 1:
        y_pred = m.predict(X).tolist()
    else:
        y_pred1 = m[0].score_samples(X)
        y_pred2 = m[1].score_samples(X)
        y_pred = (y_pred1 > y_pred2).tolist()
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

    if target_val is not None:
        print('-----------------------------------------')
        val_auc, val_eer = validate(m, features_val, target_val)
        print('Val AUC :', val_auc)
        print('Val ERR :', val_eer)

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
                        features_path=train_features_path, target_path=train_target_path
                        )

    '''optional'''
    # from sklearn.manifold import TSNE
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # X_embedded = TSNE(n_components=2).fit_transform(features_train[0::500,:])
    # plt.scatter(X_embedded[:,0], X_embedded[:,1],c=target_train[0::500])
    # plt.savefig('vis.png')
    '''optional'''

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
                            val_label, task=args.task, mode='val',
                            frame_size=frame_size, frame_shift=frame_shift,
                            features_path=val_features_path, target_path=val_target_path
                            )
    else:
        features_val, target_val = None, None

    m = build_model(args)

    exp(m, features_train, target_train, features_val, target_val, exp_id)

if __name__ == '__main__':
    main()