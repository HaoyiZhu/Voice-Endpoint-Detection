''' Script for task 1 inference '''

import argparse
import os

import numpy as np
import joblib
from tqdm import tqdm

from utils import read_wav, extract_features, save_prediction_labels
from vad_utils import prediction_to_vad_label

"""----------------------------- Inference options -----------------------------"""
parser = argparse.ArgumentParser(description='VAD Inference')
parser.add_argument('--model', type=str, default='svm_not_normalized_0.064_0.032',
                    help='model name')
parser.add_argument('--f_size', type=float, default=0.032,
                    help='frame size')
parser.add_argument('--f_shift', type=float, default=0.008,
                    help='frame shift')
parser.add_argument('--task', type=int, default=1,
                    help='task id')
parser.add_argument('--data_dir', type=str, default='./wavs/test',
                    help='directory contains test data')
parser.add_argument('--save_dir', type=str, default='./task1/',
                    help='directory to save the results')
parser.add_argument('--save_name', type=str, default='task1_prediction_on_test',
                    help='file to save prediction labels')

args = parser.parse_args()

model_path = './models/' + args.model + '.pkl'

frame_size = args.f_size
frame_shift = args.f_shift

def main():
    file_names = os.listdir(args.data_dir)
    result = dict()

    print('Loading model...')
    m = joblib.load(model_path)

    for i in tqdm(range(len(file_names))):
        _, wav_data = read_wav(file_names[i], 'test', silence=True)
        feature = extract_features(wav_data, task=args.task, frame_size=frame_size, frame_shift=frame_shift)

        if args.task == 1:
        	pred = m.predict(feature).tolist()
        else:
        	pred1 = m[0].score_samples(feature)
        	pred2 = m[1].score_samples(feature)
        	pred = (pred1 > pred2).astype(np.int64).tolist()
        pred = prediction_to_vad_label(pred, frame_size=frame_size, frame_shift=frame_shift)

        file_name = file_names[i].split('.')[0]
        result[file_name] = pred

    print('Saving results...')
    save_prediction_labels(result, save_dir=args.save_dir, file_name=args.save_name, save_format='txt')

if __name__ == '__main__':
    main()

