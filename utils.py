import os, sys, shutil
import warnings
warnings.filterwarnings(action='ignore')

import multiprocessing
import random
import time

import pickle
import pylab as plt
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

import kornia
import neurokit2 as nk
import librosa as lb
import soundfile as sf

import cv2
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


def EDA_zscore(arr_dataset):
    all = []
    for a in arr_dataset:
        all.append(a['signal'])
    all = np.array(all)#.flatten().all()
    print(all.shape)
    # all01 = np.percentile(all,1)
    # all[all<all01] = all01
    # all99 = np.percentile(all,99)
    # all[all>all99] = all99
    mean = np.mean(all)
    std = np.std(all)
    return mean, std

def EDA():
    set_seed()
    hyperparameters = dict(config_defaults)
    model = PVC_NET(hyperparameters)

    classes = model.hyperparameters['outChannels']
    srTarget = model.hyperparameters['srTarget']
    featureLength = model.hyperparameters['featureLength']       
    dataNorm = model.hyperparameters['dataNorm']

    train_files = glob('dataset/MIT-BIH_NPY/train/*.npy')
    # train_data, valid_data = seed_MITBIH(train_files, model.hyperparameters['dataSeed'])
    train_data, valid_data = FOLD5_MITBIH(train_files, model.hyperparameters['dataSeed'])
    
    train_dataset = MIT_DATASET(train_data,featureLength,srTarget, classes, dataNorm, model.hyperparameters['trainaug'], True)
    valid_dataset = MIT_DATASET(valid_data,featureLength,srTarget, classes, dataNorm, False)
    test_dataset = MIT_DATASET(test_data,featureLength,srTarget, classes, dataNorm, False)
    AMC_dataset = MIT_DATASET(AMC_data,featureLength,srTarget, classes, dataNorm, False)
    CPSC2020_dataset = MIT_DATASET(CPSC2020_data,featureLength, srTarget, classes, dataNorm,False)
    # CU_dataset = MIT_DATASET(CU_data,featureLength, srTarget, classes, False)
    ESC_dataset = MIT_DATASET(ESC_data,featureLength, srTarget, classes, False)
    # FANTASIA_dataset = MIT_DATASET(FANTASIA_data,featureLength, srTarget, classes, False)
    INCART_dataset = MIT_DATASET(INCART_data,featureLength, srTarget, classes, dataNorm, False)
    NS_dataset = MIT_DATASET(NS_data,featureLength, srTarget, classes, dataNorm, False)
    STDB_dataset = MIT_DATASET(STDB_data,featureLength, srTarget, classes, dataNorm, False)
    SVDB_dataset = MIT_DATASET(SVDB_data,featureLength, srTarget, classes, dataNorm, False)
    AMCREAL_dataset = MIT_DATASET(AMCREAL_data,featureLength, srTarget, classes, dataNorm, False)

    if model.hyperparameters['sampler']:
        train_loader = DataLoader(train_dataset, batch_size = model.hyperparameters['batch_size'], shuffle = False, num_workers=4, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset))
    else:
        train_loader = DataLoader(train_dataset, batch_size = model.hyperparameters['batch_size'], shuffle = True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size = 64, shuffle = False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size = 64, num_workers=2, shuffle = False)
    AMC_loader = DataLoader(AMC_dataset,batch_size = 64, num_workers=2, shuffle = False)
    CPSC2020_loader = DataLoader(CPSC2020_dataset,batch_size = 64, num_workers=2, shuffle = False)
    # CU_loader = DataLoader(CU_dataset,batch_size = 64, num_workers=2, shuffle = False)
    ESC_loader = DataLoader(ESC_dataset,batch_size = 64, num_workers=2, shuffle = False)
    # FANTASIA_loader = DataLoader(FANTASIA_dataset,batch_size = 64, num_workers=2, shuffle = False)
    INCART_loader = DataLoader(INCART_dataset, batch_size = 64, num_workers=2, shuffle = False)
    NS_loader = DataLoader(NS_dataset, batch_size = 64, num_workers=2, shuffle = False)
    # STDB_loader = DataLoader(STDB_dataset, batch_size = 64, num_workers=2, shuffle = False)
    SVDB_loader = DataLoader(SVDB_dataset, batch_size = 64, num_workers=2, shuffle = False)
    
    loaders = [train_loader, valid_loader, test_loader, AMC_loader, CPSC2020_loader, ESC_loader, INCART_loader, NS_loader, SVDB_loader]
    
    for l in loaders:
        batch = next(iter(l))
        signal_original = batch['signal_original']
        signal = batch['signal']
        y_seg = batch['y_seg']
        print(f"dataSource:{batch['dataSource'][0]} fname:{batch['pid'][0]} shape:{signal.shape} unique:{torch.unique(signal)}")
        
        i = 0
        idx_QRS = get_Binaryindex(y_seg[i,0].numpy())
        idx_PVC = get_Binaryindex(y_seg[i,1].numpy())
        signal_min = torch.min(signal[i,0]) - .2
        signal_max = torch.max(signal[i,0]) + .2
        
        plt.figure(figsize=(16,4))
        # plt.plot(signal_original[i,0],label='Orignal ECG')
        plt.plot(signal[i,0],label='Preprocessed ECG',color='black')
        plt.scatter(idx_QRS,[signal_min]*len(idx_QRS),label='R-peak',alpha=0.8,marker="o")
        plt.scatter(idx_PVC,[signal_max]*len(idx_PVC),label='PVC',alpha=0.8,marker="v")
        
        plt.legend()
        plt.show()