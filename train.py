'''task 1'''

import os
import numpy as np

import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                 LogisticRegressionCV, RidgeCV, Lasso, LassoCV)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from vad_utils import read_label_from_file
from evaluate import get_metrics
from utils import sklearn_dataset_for_task_1, read_json, save_json

frame_size = 0.032
frame_shift = 0.008
c = 1.0
tol = 1e-5

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
    features_test = features[-50000:, :]
    target_test = target[-50000:]
    features = features[:-50000, :]
    target = target[:-50000]

    print('Experiment Name: ', name)
    score = train(m, features, target)
    print('Training score :', score)
    joblib.dump(m, './task1/' + name + '.pkl')
    auc, err = validate(m, features_test, target_test)
    print('AUC :', auc)
    print('ERR :', err)
    print()


def main():
    if not os.path.exists('./task1/label.json'):
        label_file = "data/dev_label.txt"
        label = read_label_from_file(label_file)
        save_json(label, './task1/label.json')
    else:
        label = read_json('./task1/label.json')

    features, target = sklearn_dataset_for_task_1(label)
    target = target.astype(np.float32)
    features = features.astype(np.float32)
    assert features.shape[0] == target.shape[0], \
          ('The shape of features', features.shape, \
           'must match that of target', target.shape)
    
    # features = features[:, -2].reshape(-1,1)

    # m = LinearSVC(max_iter=1000)
    # exp(m, features, target, 'linear_svc_2')

    # m = Pipeline([('scaler', StandardScaler()),
    #               ('linear_regression', LinearRegression())])
    # exp(m, features, target, 'linear_regression_2')


    # m = Pipeline([('scaler', StandardScaler()),
    #               ('logistic_regression', LogisticRegression(max_iter=200))])
    # exp(m, features, target, 'logistic_regression_2')


    # m = Pipeline([('scaler', StandardScaler()),
    #               ('ridge_regression', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]))])
    # exp(m, features, target, 'ridge_regression_cv_2')

    m = Pipeline([('scaler', StandardScaler()),
                  ('lasso_regression', LassoCV(cv=7))])
    exp(m, features, target, 'lasso_regression_cv_7')

    # m = Pipeline([('scaler', StandardScaler()),
    #               ('logistic_regression', LogisticRegressionCV(cv=5, max_iter=1000))])
    # exp(m, features, target, 'logistic_regression_cv_5')


    # m = Pipeline([('scaler', StandardScaler()),
    #               ('logistic_regression', LogisticRegressionCV(cv=7, max_iter=1000))])
    # exp(m, features, target, 'logistic_regression_cv_7')


if __name__ == '__main__':
    main()