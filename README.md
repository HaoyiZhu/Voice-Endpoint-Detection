#  Voice-Endpoint-Detection

This repository contains codes of Project 1 for Intelligent speech recognition. This project
aims to detect the voice endpoints in audio, which consists of two tasks: the first one based 
on simple linear classifier and short-term signal features, while the second one
is based on statistical model classifier and frequency domain features.

## Requirements

- python 3.6+
- scikit-learn == 0.24.2
- tqdm

## Training
- Note that sklearn only supports cpu, so it may takes a long time for training. You can reduce the number of features
or samples to make it faster.
```bash
# Start training with: 
python train.py

# Some flags are supported for convinient customizing your strategy, you can check them in train.py.
# For example: 
python train.py --f_size 0.064 --f_shift 0.032 --exp svm_not_normalized --save_name train --model svm
```

## Inference
- To be continued.


