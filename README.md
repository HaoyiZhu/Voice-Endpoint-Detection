#  Voice-Endpoint-Detection

This repository contains codes of Project 1 for Intelligent speech recognition. This project
aims to detect the voice endpoints in audio, which consists of two tasks: the first one based 
on simple linear classifier and short-term signal features, while the second one
is based on statistical model classifier and frequency domain features.

# Requirements
- python 3.6+
- scikit-learn == 0.24.2
- tqdm

# Task 1

## Training
- Note that sklearn only supports cpu, so it may takes a long time for training. You can reduce the number of features
or samples to make it faster.
```bash
# Start training with: 
python train.py --task 1

# Some flags are supported for convinient customizing your strategy, you can check them in train.py.
# For example: 
python train.py --task 1 --f_size 0.064 --f_shift 0.032 --exp svm_not_normalized --model svm
```

## Inference
```bash
# By default, the results will be saved in './task1/task1_prediction_on_test'. Please refer to the code for all flags.
# Example:
python inference.py --model {model name} --f_size 0.064 --f_shift 0.032
```

## Validation
- To use existed model to validate, you can run:
```bash
python validate.py --model {model name} --f_size 0.064 --f_shift 0.032
```
# Task 2

- Features visualization:

<div align="center">
    <img src="./pictures/vis1.png", width="400" alt><br>
    Feature Distribution (from partial samples)
</div>

<div align="center">
    <img src="./pictures/vis2.png", width="400" alt><br>
    Feature Distribution (from all samples)
</div>

## Training
- Just add the task 2 options.
```bash
# Start training with: 
python train.py --task 2

# Task 2 only supports GMM currently. You can customize the number of components.
# For example: 
python train.py --task 2 --f_size 0.032 --f_shift 0.008 --exp gmm --n_cpnt 10
```

## Inference
```bash
# By default, the results will be saved in './task1/task1_prediction_on_test'. Please refer to the code for all flags.
# Example:
python inference.py --task 2 --model {model name} --f_size 0.064 --f_shift 0.032
```

## Validation
- To use existed model to validate, you can run:
```bash
python validate.py --task 2 --model {model name} --f_size 0.064 --f_shift 0.032
```

