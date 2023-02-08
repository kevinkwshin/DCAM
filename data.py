from data import *
from utils import *

# train_data = np.load('dataset/mit-bih-arrhythmia-database-1.0.0_trainSeg_seed4.npy',allow_pickle=True) # B x (C) x Signal
# valid_data = np.load('dataset/mit-bih-arrhythmia-database-1.0.0_validSeg_seed4.npy',allow_pickle=True) # B x (C) x Signal
test_data = np.load('dataset/mit-bih-arrhythmia-database-1.0.0_testSeg.npy',allow_pickle=True) # B x (C) x Signal

AMC_data  = np.load('dataset/AMC_PeakLabel_3rd_125Hz.npy',allow_pickle=True) # 497 samples
CPSC2020_data  = np.load('dataset/CPSC2020_testSeg_ver2.npy',allow_pickle=True)
CU_data  = np.load('dataset/cu-ventricular-tachyarrhythmia-database-1.0.0_testSeg.npy',allow_pickle=True)
ESC_data  = np.load('dataset/european-st-t-database-1.0.0_testSeg.npy',allow_pickle=True)
FANTASIA_data = np.load('dataset/fantasia-database-1.0.0_testSeg.npy', allow_pickle=True) # B x (C) x Signal
INCART_data  = np.load('dataset/INCART_testSeg.npy',allow_pickle=True)
NS_data = np.load('dataset/mit-bih-noise-stress-test-database-1.0.0_testSeg.npy',allow_pickle=True)
STDB_data = np.load('dataset/mit-bih-st-change-database-1.0.0_testSeg.npy',allow_pickle=True)
SVDB_data = np.load('dataset/mit-bih-supraventricular-arrhythmia-database-1.0.0_testSeg.npy',allow_pickle=True)
# AMCREAL_data = np.load('dataset/AMCREAL_testSeg.npy',allow_pickle=True)

def add_datainfo(data, info_string):
    new_data = []
    for d in data:
        d['dataSource'] = info_string
        new_data.append(d)
    return np.array(new_data)

def seed_MITBIH(files, seed):
    train_files, valid_files = sklearn.model_selection.train_test_split(files, test_size=.2, random_state= seed)

    train_seg = []
    for f in train_files:
        data = np.load(f,allow_pickle=True)
        train_seg.extend(data)

    valid_seg = []
    for f in valid_files:
        data = np.load(f,allow_pickle=True)
        valid_seg.extend(data)

    print('seed:',seed)
    add_datainfo(train_seg,1)
    add_datainfo(valid_seg,2)
    return train_seg, valid_seg

def FOLD5_MITBIH(files, seed):
    files = np.array(files)
    kf = sklearn.model_selection.KFold(5, shuffle=True, random_state=0)
    FOLD = []
    
    for train_index, test_index in kf.split(files):
        train_files = files[train_index]
        test_files = files[test_index]
        FOLD.append([train_files,test_files])

    train_files, valid_files=FOLD[seed]

    train_seg = []
    for f in train_files:
        data = np.load(f,allow_pickle=True)
        train_seg.extend(data)

    valid_seg = []
    for f in valid_files:
        data = np.load(f,allow_pickle=True)
        valid_seg.extend(data)

    # print(f'seed:{seed}, train_files:{train_files}, valid_files:{valid_files}')
    add_datainfo(train_seg,1)
    add_datainfo(valid_seg,2)
    return train_seg, valid_seg

# add_datainfo(train_data,'train')
# add_datainfo(valid_data,'val')
# add_datainfo(test_data,'test')

# add_datainfo(AMC_data,'AMC')
# add_datainfo(CPSC2020_data,'CPSC2020')
# add_datainfo(INCART_data,'INCART')
# add_datainfo(FANTASIA_data,'FANTASIA')
# add_datainfo(AMCREAL_data,'AMCREAL')

# add_datainfo(train_data,1)
# add_datainfo(valid_data,2)
add_datainfo(test_data,3)

add_datainfo(AMC_data,11)
add_datainfo(CPSC2020_data,12)
add_datainfo(CU_data,13)
add_datainfo(ESC_data,14)
add_datainfo(FANTASIA_data,15)
add_datainfo(INCART_data,16)
add_datainfo(NS_data,17)
add_datainfo(STDB_data,18)
add_datainfo(SVDB_data,19)
# add_datainfo(AMCREAL_data,21)
print()

def signal_distort(
    signal,
    sampling_rate=1000,
    noise_shape="laplace",
    noise_amplitude=0,
    noise_frequency=100,
    powerline_amplitude=0,
    powerline_frequency=50,
    artifacts_amplitude=0,
    artifacts_frequency=100,
    artifacts_number=5,
    linear_drift=False,
    random_state=None,
    silent=False,
):
    """**Signal distortion**

    Add noise of a given frequency, amplitude and shape to a signal.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    noise_shape : str
        The shape of the noise. Can be one of ``"laplace"`` (default) or
        ``"gaussian"``.
    noise_amplitude : float
        The amplitude of the noise (the scale of the random function, relative
        to the standard deviation of the signal).
    noise_frequency : float
        The frequency of the noise (in Hz, i.e., samples/second).
    powerline_amplitude : float
        The amplitude of the powerline noise (relative to the standard
        deviation of the signal).
    powerline_frequency : float
        The frequency of the powerline noise (in Hz, i.e., samples/second).
    artifacts_amplitude : float
        The amplitude of the artifacts (relative to the standard deviation of
        the signal).
    artifacts_frequency : int
        The frequency of the artifacts (in Hz, i.e., samples/second).
    artifacts_number : int
        The number of artifact bursts. The bursts have a random duration
        between 1 and 10% of the signal duration.
    linear_drift : bool
        Whether or not to add linear drift to the signal.
    random_state : int
        Seed for the random number generator. Keep it fixed for reproducible
        results.
    silent : bool
        Whether or not to display warning messages.

    Returns
    -------
    array
        Vector containing the distorted signal.

    Examples
    --------
    .. ipython:: python

      import numpy as np
      import pandas as pd
      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, frequency=0.5)

      # Noise
      @savefig p_signal_distort1.png scale=100%
      noise = pd.DataFrame({"Freq100": nk.signal_distort(signal, noise_frequency=200),
                           "Freq50": nk.signal_distort(signal, noise_frequency=50),
                           "Freq10": nk.signal_distort(signal, noise_frequency=10),
                           "Freq5": nk.signal_distort(signal, noise_frequency=5),
                           "Raw": signal}).plot()
      @suppress
      plt.close()

    .. ipython:: python

      # Artifacts
      @savefig p_signal_distort2.png scale=100%
      artifacts = pd.DataFrame({"1Hz": nk.signal_distort(signal, noise_amplitude=0,
                                                        artifacts_frequency=1,
                                                        artifacts_amplitude=0.5),
                               "5Hz": nk.signal_distort(signal, noise_amplitude=0,
                                                        artifacts_frequency=5,
                                                        artifacts_amplitude=0.2),
                               "Raw": signal}).plot()
      @suppress
      plt.close()

    """
    # Seed the random generator for reproducible results.
    # np.random.seed(random_state)

    # Make sure that noise_amplitude is a list.
    if isinstance(noise_amplitude, (int, float)):
        noise_amplitude = [noise_amplitude]

    signal_sd = np.std(signal, ddof=1)
    if signal_sd == 0:
        signal_sd = None

    noise = 0

    # Basic noise.
    if min(noise_amplitude) > 0:
        noise += _signal_distort_noise_multifrequency(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            noise_amplitude=noise_amplitude,
            noise_frequency=noise_frequency,
            noise_shape=noise_shape,
            silent=silent,
        )

    # Powerline noise.
    if powerline_amplitude > 0:
        noise += _signal_distort_powerline(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            powerline_frequency=powerline_frequency,
            powerline_amplitude=powerline_amplitude,
            silent=silent,
        )

    # Artifacts.
    if artifacts_amplitude > 0:
        noise += _signal_distort_artifacts(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            artifacts_frequency=artifacts_frequency,
            artifacts_amplitude=artifacts_amplitude,
            artifacts_number=artifacts_number,
            silent=silent,
        )

    if linear_drift:
        noise += _signal_linear_drift(signal)

    distorted = signal + noise

    # Reset random seed (so it doesn't affect global)
    # np.random.seed(None)

    return distorted

def _signal_distort_artifacts(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    artifacts_frequency=0,
    artifacts_amplitude=0.1,
    artifacts_number=5,
    artifacts_shape="laplace",
    silent=False,
):

    # Generate artifact burst with random onset and random duration.
    artifacts = _signal_distort_noise(
        len(signal),
        sampling_rate=sampling_rate,
        noise_frequency=artifacts_frequency,
        noise_amplitude=artifacts_amplitude,
        noise_shape=artifacts_shape,
        silent=silent,
    )
    if artifacts.sum() == 0:
        return artifacts

    min_duration = int(np.rint(len(artifacts) * 0.001))
    max_duration = int(np.rint(len(artifacts) * 0.01))
    artifact_durations = np.random.randint(min_duration, max_duration, artifacts_number)

    artifact_onsets = np.random.randint(0, len(artifacts) - max_duration, artifacts_number)
    artifact_offsets = artifact_onsets + artifact_durations

    artifact_idcs = np.array([False] * len(artifacts))
    for i in range(artifacts_number):
        artifact_idcs[artifact_onsets[i] : artifact_offsets[i]] = True

    artifacts[~artifact_idcs] = 0

    # Scale amplitude by the signal's standard deviation.
    if signal_sd is not None:
        artifacts_amplitude *= signal_sd
    artifacts *= artifacts_amplitude

    return artifacts

def _signal_distort_noise_multifrequency(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    noise_amplitude=0.1,
    noise_frequency=100,
    noise_shape="laplace",
    silent=False,
):
    base_noise = np.zeros(len(signal))
    params = listify(
        noise_amplitude=noise_amplitude, noise_frequency=noise_frequency, noise_shape=noise_shape
    )

    for i in range(len(params["noise_amplitude"])):

        freq = params["noise_frequency"][i]
        amp = params["noise_amplitude"][i]
        shape = params["noise_shape"][i]

        if signal_sd is not None:
            amp *= signal_sd

        # Make some noise!
        _base_noise = _signal_distort_noise(
            len(signal),
            sampling_rate=sampling_rate,
            noise_frequency=freq,
            noise_amplitude=amp,
            noise_shape=shape,
            silent=silent,
        )
        base_noise += _base_noise

    return base_noise


def _signal_distort_noise(
    n_samples,
    sampling_rate=1000,
    noise_frequency=100,
    noise_amplitude=0.1,
    noise_shape="laplace",
    silent=False,
):

    _noise = np.zeros(n_samples)
    # Apply a very conservative Nyquist criterion in order to ensure
    # sufficiently sampled signals.
    nyquist = sampling_rate * 0.1
    if noise_frequency > nyquist:
        if not silent:
            warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since it cannot be resolved at "
                f" the sampling rate of {sampling_rate} Hz. Please increase "
                f" sampling rate to {noise_frequency * 10} Hz or choose "
                f" frequencies smaller than or equal to {nyquist} Hz.",
                category=NeuroKitWarning,
            )
        return _noise
    # Also make sure that at least one period of the frequency can be
    # captured over the duration of the signal.
    duration = n_samples / sampling_rate
    if (1 / noise_frequency) > duration:
        if not silent:
            warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since its period of {1 / noise_frequency} "
                f" seconds exceeds the signal duration of {duration} seconds. "
                f" Please choose noise frequencies larger than "
                f" {1 / duration} Hz or increase the duration of the "
                f" signal above {1 / noise_frequency} seconds.",
                category=NeuroKitWarning,
            )
        return _noise

    noise_duration = int(duration * noise_frequency)

    if noise_shape in ["normal", "gaussian"]:
        _noise = np.random.normal(0, noise_amplitude, noise_duration)
    elif noise_shape == "laplace":
        _noise = np.random.laplace(0, noise_amplitude, noise_duration)
    else:
        raise ValueError(
            "NeuroKit error: signal_distort(): 'noise_shape' should be one of 'gaussian' or 'laplace'."
        )

    if len(_noise) != n_samples:
        _noise = signal_resample(_noise, desired_length=n_samples, method="interpolation")
    return _noise


def _signal_distort_powerline(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    powerline_frequency=50,
    powerline_amplitude=0.1,
    silent=False,
):

    duration = len(signal) / sampling_rate
    powerline_noise = signal_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        frequency=powerline_frequency,
        amplitude=1,
        silent=silent,
    )

    if signal_sd is not None:
        powerline_amplitude *= signal_sd
    powerline_noise *= powerline_amplitude

    return powerline_noise

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices = None, num_samples = None, callback_get_label = None):
        self.indices = list(range(len(dataset))) if indices is None else indices        # if indices is not provided, all elements in the dataset will be considered
        self.callback_get_label = callback_get_label                                    # define custom callback
        self.num_samples = len(self.indices) if num_samples is None else num_samples    # if num_samples is not provided, draw `len(indices)` samples in each iteration
        set_seed()
        df = pd.DataFrame()                                                             # distribution of classes in the dataset
        
        label = []
        for idx in tqdm.trange(len(dataset), desc="Sampling"):
            ########## customize here ###############
            l = dataset[idx]['y_PVC_seg'] # <- 수정
            if 1 in l:
                label.append(1)
            else:
                label.append(0)
            # 총 5개    
            ########## customize here ###############
        label = torch.tensor(label)
        
        df["label"] = label
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        
        # weights = 1.0 / np.sqrt(np.sqrt(label_to_count[df["label"]]))
        # weights = 1.0 / np.sqrt(label_to_count[df["label"]]) 
        weights = 1.0 / label_to_count[df["label"]] # almost equally
        # weights = 1.0 / (label_to_count[df["label"]])**2 # slightly weighted to 1
        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

# old preprocessing
def remove_baseline_wander(signal, fs):    
    order = 4
    nyq = 0.5 * fs
    lowcut = 0.67 #0.5
    highcut = 40
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    res = filtfilt(b, a, signal)
    # res = lfilter(b, a, signal)
    return res

from audiomentations import *
p=.2
augment_audiomentation = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=p),
    AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=40.0, p=p),
    Gain(min_gain_in_db=-12, max_gain_in_db=12, p=p),
    FrequencyMask(min_frequency_band=0.0, max_frequency_band=.5, p=p),
    TanhDistortion(min_distortion= 0.01, max_distortion = 0.4, p=p),
    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=p),
])

import neurokit2 as nk
def augment_neurokit(ecg_signal, sr):
    noise_shape = ['gaussian', 'laplace']
    n_noise_shape = np.random.randint(0,2)

    powerline_frequency = np.random.randint(50,60)
    noise_frequency = np.random.randint(2,30)
    artifacts_frequency= np.random.randint(2,20)
    # artifacts_number = np.random.randint(2,20)
    artifacts_number = 1

    powerline_amplitude = np.random.rand(1)*.2 #/ powerline_frequency
    noise_amplitude = np.random.rand(1)*.2 #/ noise_frequency
    artifacts_amplitude = np.random.rand(1)*1 #/ artifacts_frequency
    
    ecg_signal = signal_distort(ecg_signal,
                                sampling_rate=sr,
                                noise_shape=noise_shape[n_noise_shape],
                                noise_amplitude=noise_amplitude,
                                noise_frequency=noise_frequency,
                                powerline_amplitude=powerline_amplitude,
                                powerline_frequency=powerline_frequency,
                                artifacts_amplitude=artifacts_amplitude,
                                artifacts_frequency=artifacts_frequency,
                                artifacts_number=artifacts_number,
                                linear_drift=False,
                                random_state=None,#42,
                                silent=True)
    return ecg_signal

def minmax(arr):
    """
    numpy
    """
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def zscore(arr, mean=None, std=None):
    """
    numpy
    """
    if mean == None or std == None:
        mean = np.mean(arr)
        std = np.std(arr)
    return (arr-mean)/std

def augment_neurokit2(sig, sr, p=0.3):
    
    if np.random.rand(1) <= p:
        beta = (np.random.rand(1)-.5)*4
        amp = np.random.rand(1)

        noise = nk.signal.signal_noise(duration=len(sig)/sr, sampling_rate=sr, beta=beta) * amp
        temp1 = np.random.rand(1)
        temp2 = np.random.rand(1)
        start = int(np.min([temp1,temp2])*len(sig))
        end = int(np.max([temp1,temp2])*len(sig))
        filt = np.zeros_like(sig)
        filt[start:end]= 1
        filt = scipy.ndimage.gaussian_filter1d(filt,11,order=0,mode='nearest')
        
        result = np.zeros(len(sig))
        aug = augment_neurokit(noise, sr=sr)
        result[:len(aug)] = aug
        result = result * filt
        result = sig + result
        return result

    else:
        return sig

class MIT_DATASET():
    def __init__(self, data, featureLength, srTarget, classes=4, normalize='instance', augmentation="NONE", random_crop=False):
        self.data = data
        self.classes = classes
        self.augmentation = augmentation
        if augmentation == "NONE":
            self.augmentation = False
        elif augmentation =='NEUROKIT':
            self.augmentation=augment_neurokit
        elif augmentation =='NEUROKIT2':
            self.augmentation=augment_neurokit2
        elif augmentation =='AUDIOMENTATION':
            self.augmentation=augment_audiomentation

        self.random_crop = random_crop
        self.srTarget = srTarget
        self.featureLength = featureLength
        self.normalize = normalize
        self.mean, self.std = EDA_zscore(data)
        # print('mean', self.mean, 'std', self.std)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pid = self.data[idx]['pid']
        signal = self.data[idx]['signal']
        srOriginal = self.data[idx]['sr']
        time = self.data[idx]['time']
        idx_Normal = self.data[idx]['idx_Normal']
        idx_PVC = self.data[idx]['idx_PVC']
        idx_AFIB = self.data[idx]['idx_Afib']
        idx_Others = self.data[idx]['idx_Others']
        idx_Artifact = self.data[idx]['idx_Artifact']
        dataSource = self.data[idx]['dataSource']
        
        y_Normal_seg = np.zeros_like(signal)
        y_PVC_seg = np.zeros_like(signal)
        y_AFIB_seg = np.zeros_like(signal)
        y_Others_seg = np.zeros_like(signal)
        
        interval = int(srOriginal * 0.1) # this is to set peak interval
        
        # grab annotations
        for idx_ in idx_Normal:  
            y_Normal_seg[idx_-interval:idx_+interval] = 1
        for idx_ in idx_PVC:
            y_PVC_seg[idx_-interval:idx_+interval] = 1
        for idx_ in idx_AFIB:
            y_AFIB_seg[idx_-interval:idx_+interval] = 1
        for idx_ in idx_Others:
            y_Others_seg[idx_-interval:idx_+interval] = 1
    
        # resampling
        if self.augmentation:
            srTarget = np.random.randint(int(self.srTarget*0.97),int(self.srTarget*1.03)) # time stretching, you need to carefully check here
        else: 
            srTarget = self.srTarget
            
        signal = lb.resample(signal, orig_sr=srOriginal, target_sr=srTarget) if srTarget != srOriginal else signal # resample
        y_Normal_seg = scipy.ndimage.zoom(y_Normal_seg, srTarget/srOriginal, order=0, mode='nearest',) if srTarget != srOriginal else y_Normal_seg # resample
        y_PVC_seg = scipy.ndimage.zoom(y_PVC_seg, srTarget/srOriginal, order=0, mode='nearest',) if srTarget != srOriginal else y_PVC_seg # resample
        y_AFIB_seg = scipy.ndimage.zoom(y_AFIB_seg, srTarget/srOriginal, order=0, mode='nearest',) if srTarget != srOriginal else y_AFIB_seg # resample
        y_Others_seg = scipy.ndimage.zoom(y_Others_seg, srTarget/srOriginal, order=0, mode='nearest',) if srTarget != srOriginal else y_Others_seg # resample
        
        if self.random_crop:
            if int(len(signal)) > self.featureLength:  # randomly crop 
                randnum = np.random.randint(0,len(signal)-self.featureLength)
                start = randnum if self.random_crop else 0
                end = start+self.featureLength
            elif int(len(signal)) == self.featureLength:
                start = 0
                end = 0 + self.featureLength
            else:
                print('too short data:: need check sampling rate or featureLength', int(len(signal)),self.featureLength)
                
            signal = signal[start:end]
            y_Normal_seg = y_Normal_seg[start:end]
            y_PVC_seg = y_PVC_seg[start:end]
            y_AFIB_seg = y_AFIB_seg[start:end]
            y_Others_seg = y_Others_seg[start:end]
        # print('after crop',signal.shape)

        y_peak_seg = y_Normal_seg + y_PVC_seg + y_Others_seg + y_AFIB_seg # R-peak
        y_peak_seg[y_peak_seg!=0] =1

        if self.classes == 1:
            y_seg = np.expand_dims(y_PVC_seg,0) # 1 class
        elif self.classes == 2:
            y_seg = np.stack((y_peak_seg, y_PVC_seg), axis=0).astype(float) # 2 multi class
        elif self.classes == 3:
            # y_Others_seg = y_Others_seg + y_AFIB_seg # non PVC
            # y_Others_seg[y_Others_seg!=0] =1
            # y_seg = np.stack((y_peak_seg, y_PVC_seg, y_Others_seg), axis=0).astype(float) # 3 multi class    
            y_seg = np.stack((y_peak_seg, y_PVC_seg, y_AFIB_seg), axis=0).astype(float) # 3 multi class    
        elif self.classes == 4:
            y_seg = np.stack((y_peak_seg, y_PVC_seg, y_AFIB_seg, y_Others_seg), axis=0).astype(float) # 4 multi class
        
        y_Normal = np.array([0]) if 1 in y_Normal_seg else np.array([1]) # classification task
        y_Others = np.array([0]) if 1 in y_Others_seg else np.array([1]) # classification task
        y_PVC    = np.array([0]) if 1 in y_PVC_seg else np.array([1]) # classification task
        y_AFIB   = np.array([0]) if 1 in y_AFIB_seg else np.array([1]) # classification task
        
        signal_original = signal.copy()
        signal_original = np.expand_dims(signal_original,0)
        
        # augmentation
        signal = signal if not self.augmentation else self.augmentation(signal, srTarget)
        
        signal = remove_baseline_wander(signal,srTarget)
        signal = np.expand_dims(signal,0)
        
        if self.normalize =='minmaxI':
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) # normalize  
        elif self.normalize =='zscoreI':
            signal = zscore(signal)
        elif self.normalize =='zscoreO':
            signal = zscore(signal, self.mean, self.std)
            
        # signal = torch.tensor(signal).float() # shape should be Channel X Signal
        signal = torch.from_numpy(signal.copy()).float()
        
        return {'dataSource':dataSource,
                'pid':pid,
                'srOriginal': srOriginal,
                'srTarget':srTarget,
                'time':time,
                'fname':f'{pid}_time{time}',
                'signal':signal,
                'signal_original':signal_original,
                'y_AFIB':y_AFIB, 
                'y_PVC':y_PVC,
                'y_AFIB_seg':y_AFIB_seg,
                'y_PVC_seg':y_PVC_seg, 
                'y_Normal_seg':y_Normal_seg,
                'y_Others_seg':y_Others_seg, 
                'y_seg':y_seg,}