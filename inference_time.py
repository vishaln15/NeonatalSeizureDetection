import helpermethods as helper
import numpy as np
import sys
import edgeml_pytorch.utils as utils
from edgeml_pytorch.graph.protoNN import ProtoNN
import torch
import time
import scipy
from antropy.antropy import entropy
from scipy.signal import periodogram, welch
import pandas as pd


# HyperParams

hyperParam = {}
fs = 32
window = np.load('1window.npy')
window_length = fs * 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataDimension = window.shape[0]
dataDimension = 20
PROJECTION_DIM = 10
NUM_PROTOTYPES = 20
numClasses = 2
times = list()

# Feature Methods

def hMob(x):
    row = np.array(x)
    return (np.sqrt(np.var(np.gradient(x)) / np.var(x)))

def feature_mean(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(np.mean(x, axis = 0) for x in row)

def feature_stddev(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(np.std(x, axis = 0) for x in row)

def kurtosis(row):
    row = np.array(row)
    # annotation = row[-1]
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(scipy.stats.kurtosis(x, axis = 0, bias = False) for x in row)

def skewness(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(scipy.stats.skew(x, axis = 0, bias = False) for x in row)
    
def spectral_entropy(row, sf = 32, nperseg = window_length, axis = 1):
    row = np.array(row)
    # annotation = row[-1]
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    _, psd = welch(row, sf, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = - np.where(psd_norm == 0, 0, psd_norm * np.log(psd_norm) / np.log(2)).sum(axis=axis)
    return pd.Series(se)

def hjorthActivity(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(np.var(x, axis = 0) for x in row)

def hjorthMobility(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(np.sqrt(np.var(np.gradient(x)) / np.var(x)) for x in row)

def hjorthComplexity(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series((hMob(np.gradient(x)) / hMob(x)) for x in row)

def permutation_entropy(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(entropy.perm_entropy(x) for x in row)

def sample_entropy(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.array(entropy.sample_entropy(x) for x in row)

def approximate_entropy(row):
    row = np.array(row)
    # row = row[:-1]
    row = np.reshape(row, (21, window_length))
    return pd.Series(entropy.app_entropy(x) for x in row)

def getFeatures(window):
    list_of_feature_methods = [feature_mean, feature_stddev, kurtosis, skewness, spectral_entropy, hjorthActivity, hjorthMobility, 
                          hjorthComplexity, permutation_entropy, sample_entropy, approximate_entropy]
    return np.array([i(window) for i in list_of_feature_methods])

def loadModel():
    hyperParam['B'] = np.load('WEIGHTS/1sec/output_pca_20_test_20_lr_0.0001_e_1000/B.npy')
    hyperParam['gamma'] = np.load('WEIGHTS/1sec/output_pca_20_test_20_lr_0.0001_e_1000/gamma.npy')
    hyperParam['W'] = np.load('WEIGHTS/1sec/output_pca_20_test_20_lr_0.0001_e_1000/W.npy')
    hyperParam['Z'] = np.load('WEIGHTS/1sec/output_pca_20_test_20_lr_0.0001_e_1000/Z.npy')
    return

loadModel()
# print(hyperParam['W'].shape, dataDimension, PROJECTION_DIM)
protoNN = ProtoNN(dataDimension, PROJECTION_DIM, NUM_PROTOTYPES, numClasses, hyperParam['gamma'], W = hyperParam['W'], B = hyperParam['B']).to(device)

for i in range(int(sys.argv[1])):
    start = time.time()
    features = getFeatures(window)
    features = features.flatten()[:200].reshape((10, 20))
    logits = protoNN.forward(torch.Tensor(features).to(device))
    _, _ = torch.max(logits, dim=1)
    end = time.time()
    times.append(end - start)

# print(times)
print("%.5f milli seconds!" % (np.mean(times) * 1000))