import os, sys, shutil
import warnings
warnings.filterwarnings(action='ignore')

import multiprocessing
import random
import time

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.metrics

from glob import glob
import tqdm
from natsort import natsorted

import scipy
import scipy.io as sio
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import kaiserord, firwin, filtfilt, butter
from scipy.ndimage import label, binary_closing
from skimage import morphology
from scipy import ndimage

# import kornia
import neurokit2 as nk
import librosa as lb
import soundfile as sf

# import cv2
import monai
from monai.inferers import sliding_window_inference
from monai.config import print_config

import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import *

import wandb
from pytorch_lightning import loggers as pl_loggers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *

import scipy
import scipy.io as sio
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import kaiserord, firwin, filtfilt, butter
from scipy.ndimage import label, binary_closing
from skimage import morphology
from scipy import ndimage

import sklearn
from sklearn.metrics import *

from neurokit2.misc import NeuroKitWarning, listify
from neurokit2.signal.signal_resample import signal_resample
from neurokit2.signal.signal_simulate import signal_simulate

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    monai.utils.misc.set_determinism(seed=seed)
    pl.seed_everything(seed,True)    
    
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_command(string):
    return os.system(string)

def find_maxF1(y_true, y_score, show=False):

    prec, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2*recall*prec/(recall+prec)
    threshold = thresholds[np.argmax(f1_scores)]
    
    if show:
        plt.figure(figsize=(16,4))
        plt.subplot(121)
        plt.title('PR curve')
        plt.plot(recall,prec)
        plt.scatter(recall[idx], prec[idx], color='r')
        plt.subplot(122)
        plt.title('Threshold curve')
        plt.plot(thresholds)
        plt.scatter(idx,threshold,color='r')
        plt.show()
    return threshold    

def Youden_index(y_true, y_score):
    '''Find data-driven cut-off for classification    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).

    === Example ===
    y = [0,0,0,1,1,1]
    yhat = [0.3,0.6,0.4,.7,.9,.8]
    Youden_index(y, yhat)

    References
    ----------    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def get_Binaryindex(arr):
    """
    from QRS, 
    """
    idxs = []
    arr_ = arr.copy()
    arr_ = arr_.round()

    label_result, count = scipy.ndimage.label(arr_)
    for i in range(1,count+1):
        index = np.where(label_result == i)[0]
        start = index[0]
        end = index[-1]
        idxs.append(int(np.mean([start,end])))
    return idxs

def remove_baseline_wander(signal, fs):    
    order = 4
    nyq = 0.5 * fs
    lowcut = 0.67 #0.5
    highcut = 50
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    res = filtfilt(b, a, signal)
    # res = lfilter(b, a, signal)
    return res

def EDA_zscore(arr_dataset):
    all = []
    for a in arr_dataset:
        if len(a['signal'])>1800:
            all.append(remove_baseline_wander(a['signal'], 360))
            # all.append(a['signal'])
    
    all = np.array(all)#.flatten().all()
    # all01 = np.percentile(all,1)
    # all[all<all01] = all01
    # all99 = np.percentile(all,99)
    # all[all>all99] = all99
    mean = np.mean(all)
    std = np.std(all)
    print(f'mean{mean},std{std}')
    return mean, std
