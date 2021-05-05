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
from utils import sklearn_dataset_for_task_1, read_json, save_json

parser = argparse.ArgumentParser(description='VAD Training')
parser.add_argument('--f_size', type=float, default=0.032,
                    help='frame size')
parser.add_argument('--f_shift', type=float, default=0.008,
                    help='frame shift')
parser.add_argument('--exp', type=str, default='train',
                    help='Experiment ID')
parser.add_argument('--save_name', type=str, default='task1_train',
                    help='file name while saving data for lazing loading')
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

args = parser.parse_args()

frame_size = args.f_size
frame_shift = args.f_shift

prefix = './task1/' + args.save_name + '_' + str(frame_size) + '_' + str(frame_shift)
features_path = prefix + '_features.npy'
target_path = prefix + '_target.npy'

exp_id = args.exp + '_' + str(frame_size) + '_' + str(frame_shift)

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
    
    model_type = args.model
    if model_type == 'svm':
        from sklearn import svm
        clf = svm.SVC(C=args.c, tol=args.tol)
    elif model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()
    elif model_type == 'ridge':
        from sklearn.linear_model import RidgeCV
        clf = RidgeCV(cv=args.cv)
    elif model_type == 'logistic':
        from sklearn.linear_model import LogisticRegressionCV
        clf = LogisticRegressionCV(cv=args.cv)
    elif model_type == 'lasso':
        from sklearn.linear_model import LassoCV
        clf = LassoCV(cv=args.cv)
    else:
        raise NotImplementedError

    if args.scaler:
        m = Pipeline([('scaler', StandardScaler()),
                    ('clf', clf)])
    else:
        m = clf

    exp(m, features, target, exp_id)

if __name__ == '__main__':
    main()