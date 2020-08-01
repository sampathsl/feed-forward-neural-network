import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

numberOfSeeds = 100
cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
ds = pd.read_table('data/crx.data', sep=",", names=cols, header=None)
dsModified = ds.replace('?', np.nan)

dsModified.columns

for col in ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13', 'A14', 'A16']:
    dsModified[col] = pd.Categorical(dsModified[col])
    dsModified[col] = dsModified[col].cat.codes

dsModified.head()

minMaxScaler = MinMaxScaler()
dsFinal = pd.DataFrame(minMaxScaler.fit_transform(dsModified))

dsFinal.describe()

X = dsFinal.iloc[:, 0:15].values
Y = dsFinal.iloc[:, 15:16].values

np.random.seed(50)


def getModel(verbose=False):
    model = tf.keras.models.Sequential()

    if verbose:
        print('Network configuration ', neuronCountForLayer)
    model.add(tf.keras.layers.Dense(neuronCountForLayer[0], input_dim=15, activation=activationFuncsForLayer[0],
                                    kernel_regularizer=regFunction))

    for x in range(1, depthOfNetwork - 1):
        model.add(tf.keras.layers.Dense(neuronCountForLayer[x], activation=activationFuncsForLayer[x],
                                        kernel_regularizer=regFunction))

    model.add(tf.keras.layers.Dense(neuronCountForLayer[depthOfNetwork - 1],
                                    activation=activationFuncsForLayer[depthOfNetwork - 1]))

    model.compile(loss=lossFunction, optimizer='adam', metrics=['accuracy'])

    return model


def evaluateModel(verbose=False):
    numberOfSplits = 5
    fOneScores = []

    for trainIndex, testIndex in StratifiedKFold(numberOfSplits).split(X, Y):
        xTrain, xTest = X[trainIndex], X[testIndex]
        yTrain, yTest = Y[trainIndex], Y[testIndex]

        model = getModel(verbose)
        model.fit(xTrain, yTrain, epochs=100, verbose=0)
        evaluationMetrics = model.evaluate(xTest, yTest, verbose=0)

        if verbose:
            print('Model evaluation ', evaluationMetrics)

        yPredict = np.where(model.predict(xTest) > 0.5, 1, 0)
        fOneScore = f1_score(yTest, yPredict, average="macro")

        if verbose:
            print('F1 score ', fOneScore)

        fOneScores.append(fOneScore)

    return np.mean(fOneScores)


print("-------------------------------------------------------------------------------")
print("----------------------------------START----------------------------------------")

# print("F1 Score For 5 K-Fold Cross Validation")
# depthOfNetwork = 2
# neuronCountForLayer = [2, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
#
# print("\n Final Mean F1 Score: ", evaluateModel(True))
#
# print("----------------------------------END----------------------------------------\n")
# print("----------------------------------START----------------------------------------")
#
# print("Changing neuron count of first hidden layer")
# depthOfNetwork = 2
# activationFuncsForLayer = ['sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
#
# for i in range(5, 105, 5):
#     neuronCountForLayer = [i, 1]
#     print("'Node count : % 3d, Mean F1 score : % 10.5f" % (i, evaluateModel()))
#
# print("----------------------------------END----------------------------------------\n")
# print("----------------------------------START----------------------------------------")
#
# print("Changing no. of hidden layers with constant Neurons(15)")
# print("1 Hidden layers F1 score")
# depthOfNetwork = 2
# neuronCountForLayer = [15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
#
# print("'Neurons [% 3d], Mean F1 score : % 10.5f" % (15, evaluateModel()))
#
# print("\n--------------------------------------------------------------------------")
# print("Changing no. of hidden layers with constant Neurons(15)")
# print("2 Hidden layers F1 score")
# depthOfNetwork = 3
# neuronCountForLayer = [15, 15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
# print("'Neurons [% 3d, % 3d], Mean F1 score : % 10.5f" % (15, 15, evaluateModel()))
#
# print("\n--------------------------------------------------------------------------")
# print("Changing no. of hidden layers with constant Neurons(15)")
# print("3 Hidden layers F1 score")
# depthOfNetwork = 4
# neuronCountForLayer = [15, 15, 15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
# print("'Neurons [% 3d, % 3d,% 3d], Mean F1 score : % 10.5f" % (15, 15, 15, evaluateModel()))
#
# print("\n--------------------------------------------------------------------------")
# print("Changing no. of hidden layers with constant Neurons(15)")
# print("4 Hidden layers F1 score")
# depthOfNetwork = 5
# neuronCountForLayer = [15, 15, 15, 15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
# print("'Neurons [% 3d, % 3d,% 3d,% 3d], Mean F1 score : % 10.5f" % (15, 15, 15, 15, evaluateModel()))
#
# print("\n--------------------------------------------------------------------------")
# print("Changing no. of hidden layers with constant Neurons(15)")
# print("5 Hidden layers F1 score")
# depthOfNetwork = 6
# neuronCountForLayer = [15, 15, 15, 15, 15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
# print("'Neurons [% 3d, % 3d,% 3d,% 3d,% 3d], Mean F1 score : % 10.5f" % (15, 15, 15, 15, 15, evaluateModel()))
#
# print("\n--------------------------------------------------------------------------")
# print("Changing no. of hidden layers with constant Neurons(15)")
# print("6 Hidden layers F1 score")
# depthOfNetwork = 7
# neuronCountForLayer = [15, 15, 15, 15, 15, 15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
# print("'Neurons [% 3d, % 3d,% 3d,% 3d,% 3d,% 3d], Mean F1 score : % 10.5f" % (15, 15, 15, 15, 15, 15, evaluateModel()))
#
# print("\n--------------------------------------------------------------------------")
# print("Changing no. of hidden layers with constant Neurons(15)")
# print("7 Hidden layers F1 score")
# depthOfNetwork = 8
# neuronCountForLayer = [15, 15, 15, 15, 15, 15, 15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
# lossFunction = 'binary_crossentropy'
# regFunction = tf.keras.regularizers.l2(0)
# print("'Neurons [% 3d, % 3d,% 3d,% 3d,% 3d,% 3d,% 3d], Mean F1 score : % 10.5f" % (
#     15, 15, 15, 15, 15, 15, 15, evaluateModel()))
#
# print("----------------------------------END----------------------------------------\n")
# print("----------------------------------START----------------------------------------")
#
# print("Changing different Error functions with 1 hidden layer with constant Neurons(15)")
# depthOfNetwork = 2
# neuronCountForLayer = [15, 1]
# activationFuncsForLayer = ['sigmoid', 'sigmoid']
# lossFunction_list = ['binary_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error']
# regFunction = tf.keras.regularizers.l2(0)
# for each in lossFunction_list:
#     lossFunction = each
#     print("'Neurons [% 3d], Mean F1 score : % 10.5f" % (15, evaluateModel()))
#
# print("----------------------------------END----------------------------------------\n")
# print("----------------------------------START----------------------------------------")

print("Changing different activation functions with 1 hidden layer with constant Neurons(15)")
depthOfNetwork = 2
neuronCountForLayer = [15, 1]
activationFuncList = ['relu', 'tanh', 'sigmoid']
lossFunction = 'binary_crossentropy'
regFunction = tf.keras.regularizers.l2(0)
for each in activationFuncList:
    activationFuncsForLayer = [each, 'sigmoid']
    print("'Neurons [% 3d], Mean F1 score : % 10.5f" % (15, evaluateModel()))

print("----------------------------------END----------------------------------------\n")
print("----------------------------------START----------------------------------------")

print("L1 regularization with different lambda values")
depthOfNetwork = 2
neuronCountForLayer = [15, 1]
activationFuncsForLayer = ['tanh', 'sigmoid']
lossFunction = 'binary_crossentropy'
regFunction = tf.keras.regularizers.l1(0)

for i in range(-5, 5):
    regFunction = tf.keras.regularizers.l1(10 ** i)
    print("'Regularizor : l1 with lambda : % 10.5f , Mean F1 score : % 10.5f" % (10 ** i, evaluateModel()))

print("----------------------------------END----------------------------------------\n")
print("----------------------------------START----------------------------------------")

print("L2 regularization with different lambda values")
depthOfNetwork = 2
neuronCountForLayer = [15, 1]
activationFuncsForLayer = ['tanh', 'sigmoid']
lossFunction = 'binary_crossentropy'
regFunction = tf.keras.regularizers.l2(0)

for i in range(-5, 5):
    regFunction = tf.keras.regularizers.l2(10 ** i)
    print("Regularizer : l2 with lambda : % 10.5f , Mean F1 score : % 10.5f" % (10 ** i, evaluateModel()))

print("----------------------------------END----------------------------------------\n")
