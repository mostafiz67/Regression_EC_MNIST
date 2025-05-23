# MNIST Digits Regression EC with K-fold validation

## Goal
Treat MNIST Digits Classification dataset as a Regression dataset using a Convolutional Neural Network (CNN).

## Implementation
Desige and developed a CNN model to solve the MNIST digits classification problem as a regression problem and checked the performance of the model using some Goodness-of-Fit test metrics.

## CNN Model Architecture

1) Two fully convolutional layers, 
2) Relu activation function and MaxPooling,
3) Mean Squared Error (MSELoss) as loss function, 
4) Stochastic Gradient Descent (SGD),
5) Learning Rate 0.01,
6) Number of Epochs 50,
7) K-fold (k=5) external holdout cross-validation, and
8) Number of repeat is 15.


## Regression ECs and Goodness of fit for the MNIST Digits using CNN

Model | EC Method: Value | MAE (Std) | MAPE (Std) | MSqE (Std) | R2 (Std)
---------- | ---------- | ---------- | ---------- | ---------- | ---------- | 
CNN | Ratio: 0.48 (0.01) <br /> Ratio-diff: 0.40 (0.01) <br /> Ratio-sign: 0.21 (0.05) <br /> Ratio-sign-diff: 0.09 (0.02) <br /> Intersection-union-sample: 0.34 (0.03) <br /> Intersection-union-all: 0.40 (0.04) <br /> Intersection-union-distance: 0.49 (0.07) | 0.57 (0.06) | 0.19 (0.20) | 0.68 (0.12) | 0.92 (0.01)


## Model Prediction VS Actual Number

Actual| 7| 2| 1| 0| 4| 1| 4| 9| 5| 9|
-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
Prediction| 7.1|  1.8| 1.6| 0.1| 4.4| 0.7| 4.5| 7.3| 4.9| 8.9 




## Model Train using K-fold and Collect Regression EC (Test)

```
python3 run.py --choices=Train-Test
```

## Model Compare (with actual vs prediction)

```
python3 run.py --choices=Compare
```

