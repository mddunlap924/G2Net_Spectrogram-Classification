import numpy as np
import pandas as pd
import pickle
import os
import yaml
from scipy import signal
from nnAudio.Spectrogram import CQT
import itertools
import wandb
from pytorch_lightning import loggers as pl_loggers
from gwpy.timeseries import TimeSeries


np.random.seed(42)

# Measure the average power spectral (asd) density of the noise targets
# The asd will be used for data whitening
def avg_psd_calculate(X_train, y_train, tukey_alpha):
    N_SAMPLES = 1_000
    idxs = {'noise': np.random.permutation(np.where(y_train == 0)[0])[0:N_SAMPLES],
            'sig': np.random.permutation(np.where(y_train == 1)[0])[0:N_SAMPLES],
            }

    # High-pass Filter each signal
    Nt = len(X_train[0][0])
    freqs = np.fft.rfftfreq(Nt, 1 / 2048)
    psds = np.empty((N_SAMPLES, 3, len(freqs)))

    # Tukey Window
    leakage_window = signal.tukey(4096, tukey_alpha)

    # Noise Q-Transform
    # https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.highpass
    # https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html#gwpy.timeseries.TimeSeries.asd
    for i, idx in enumerate(idxs['noise']):
        sigs = X_train[idx]
        for j in range(3):
            sig = TimeSeries(sigs[j], sample_rate=2048)
            sig = sig * leakage_window
            psds[i, j] = sig.asd().value
    psd_avg = psds.mean(axis=0)
    Pxx = {'noise': dict(pxx=psds, psd_avg=psd_avg)}
    del X_train, y_train
    return freqs, Pxx['noise']['psd_avg']


# Return the list of configuration (cfg) files for bash execution
def list_of_cfgs():
    list_files = []
    for (dirpath, dirnames, filenames) in os.walk(f'./Model_Cfgs'):
        list_files += [os.path.join(dirpath, file) for file in filenames]

    list_files.sort()
    cfgs = []
    for file_name in list_files:
        # Read YAML file
        with open(file_name, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
            cfgs.append(data_loaded)

    return cfgs


# Get the path of a checkpoint's weight for loading a model
def get_ckpt_path(name, id, ckpt_type, *, existing=False):
    """ Lookup Checkpoint Path """
    if existing:
        log_path = os.path.join('gwave', id, 'checkpoints')
    else:
        log_path = os.path.join(name, id, 'checkpoints')

    if ckpt_type == 'last':
        ckpt_file_name = [file for file in os.listdir(log_path) if 'last' in file]
    else:
        ckpt_file_name = [file for file in os.listdir(log_path) if ('last' not in file) and ('step' in file)]
    ckpt_file_name = ckpt_file_name[0]

    if existing:
        ckpt_path = {'dir': log_path,
                     'file_name': ckpt_file_name,
                     'path': os.path.join(log_path, ckpt_file_name)}
        print(f'Retrieved Existing *.ckpt: {os.path.join(log_path, ckpt_file_name)}')
    else:
        ckpt_path = {'dir': os.path.join(name, id, 'checkpoints'),
                     'file_name': ckpt_file_name,
                     'path': os.path.join(log_path, ckpt_file_name)}

    return ckpt_path


# Load the parameters from a configuration file
def load_cfg(file_name):
    file_path = os.path.join(f'./Model_Cfgs', file_name)
    # Read YAML file
    with open(file_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


# Measure various descriptive statistics about the data for EDA and normalizations
def sig_stats(x, y, hpf):
    x_filt = hpf.highpass_filter(x[0])

    sig_mean = np.empty((x.shape[0], x_filt.shape[0]))
    sig_std = np.empty((x.shape[0], x_filt.shape[0]))
    sig_min = np.empty((x.shape[0], x_filt.shape[0]))
    sig_max = np.empty((x.shape[0], x_filt.shape[0]))
    for i in range(x.shape[0]):
        sig_filt = hpf.highpass_filter(x[i])
        sig_mean[i] = sig_filt.mean(axis=1)
        sig_std[i] = sig_filt.std(axis=1)
        sig_min[i] = sig_filt.min(axis=1)
        sig_max[i] = sig_filt.max(axis=1)

    hpf.mean = sig_mean.mean(axis=0)
    hpf.std = sig_std.mean(axis=0)
    hpf.min = sig_min.mean(axis=0)
    hpf.max = sig_max.mean(axis=0)

    return hpf


# Load the raw data from disk into RAM
class LoadData:
    def __init__(self, dataset_name, data_type, *, partial=False):
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.partial = partial

        if partial:
            data_path = f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{dataset_name}/{data_type}_signals_partial.npy'
        else:
            data_path = f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{dataset_name}/{data_type}_signals.npy'
        self.data_path = data_path
        self.data_list_path = f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{data_type}_files.pkl'

        stats = {
            'train': {
                'mean': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/mean_train_signals.npy'),
                'std': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/std_train_signals.npy')},
            'test': {
                'mean': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/mean_test_signals.npy'),
                'std': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/std_test_signals.npy')},
        }
        stats['norm'] = {'mean': np.array([stats['train']['mean'].mean(axis=0),
                                           stats['test']['mean'].mean(axis=0)]).mean(axis=0),
                         'std': np.array([stats['train']['std'].mean(axis=0),
                                          stats['test']['std'].mean(axis=0)]).mean(axis=0),
                         }
        self.stats = stats

    def load_data(self):
        # Load Numpy Data
        self.X = np.load(self.data_path)
        self.N = self.X.shape[0]

        # Load list of data directories from data
        with open(self.data_list_path, 'rb') as f:
            list_files = pickle.load(f)
        list_files = [i for i in list_files[0:self.N]]

        # Split data directory to get data ids
        ids = [i.split('/')[-1].split('.')[0] for i in list_files]

        # Place file paths and ids into dictionary
        self.data_info = {'paths': list_files,
                          'id': ids}

        # Return train or test data
        if self.data_type == 'train':
            train_df = pd.read_csv('./Data/g2net-gravitational-wave-detection/training_labels.csv')
            y = train_df.target.to_numpy()
            self.y = y[0:self.N]


# Data preprocessing with various signal processing filters, scaling, cross-correlation, etc.
class SigFilter:
    def __init__(self, *, btype='highpass', cutoff=18, order=8, tukey_alpha=0.2, leakage_window_type='tukey'):
        self.btype = btype
        self.cutoff = cutoff
        self.order = order
        self.fs = 2048
        self.b, self.a = signal.butter(N=self.order,
                                       Wn=self.cutoff,
                                       fs=self.fs,
                                       btype=self.btype)
        self.leakage_window_type = leakage_window_type
        if leakage_window_type == 'tukey':
            self.leakage_window = signal.tukey(4096, tukey_alpha)
        elif leakage_window_type == 'boxcar':
            self.leakage_window = signal.boxcar(4096)

    def highpass_filter(self, sigs):
        sigs_filt = np.empty(sigs.shape)
        for i in range(3):
            sig = TimeSeries(sigs[i], sample_rate=self.fs)
            sig = sig * self.leakage_window
            sig = sig.whiten(asd=self.psd[i], window='boxcar')
            sig = sig.highpass(self.cutoff, filtfilt=True)
            sigs_filt[i] = sig.value
        return sigs_filt


class QTransform:
    def __init__(self, spec, spec_mix, *, TTA=False):
        self.TTA = TTA
        self.spec = spec
        self.spec_mix = spec_mix

    def __create_q_transform(self, q_spec):
        q_transform = CQT(sr=2048,
                          fmin=q_spec[0],
                          fmax=q_spec[1],
                          hop_length=q_spec[2],
                          filter_scale=q_spec[3],
                          bins_per_octave=q_spec[4],
                          n_bins=q_spec[5],
                          )
        q_transform.fmin = q_spec[0]
        q_transform.fmax = q_spec[1]
        q_transform.filter_scale = q_spec[3]
        q_transform.bins_per_octave = q_spec[4]
        q_transform.n_bins = q_spec[5]
        return q_transform

    def __q_transforms_list(self, specs):
        vars = [specs['fmin'],
                specs['fmax'],
                specs['hop'],
                specs['filter_scale'],
                specs['bins_per_octave'],
                specs['n_bins']]
        q_specs = list(itertools.product(*vars))
        q_transforms = []
        for q_spec in q_specs:
            q_transform = self.__create_q_transform(q_spec)
            q_transforms.append(q_transform)
        return q_transforms

    def create_q_transforms(self):
        # Q-Transform from SPEC
        q_transform0 = self.__create_q_transform(q_spec=[self.spec['fmin'],
                                                         self.spec['fmax'],
                                                         self.spec['hop'],
                                                         self.spec['filter_scale'],
                                                         self.spec['bins_per_octave'],
                                                         self.spec['n_bins'],
                                                         ])
        # This is for the scenario where TTA
        if self.TTA:
            q_transform_ = self.__q_transforms_list(specs=self.spec_mix)
            q_transform = [q_transform0] + q_transform_
        # This is for the scenario where TTA=False
        else:
            # If spec_mix=True
            if self.spec_mix['apply']:
                q_transform_ = self.__q_transforms_list(specs=self.spec_mix)
                q_transform = [q_transform0] + q_transform_
                del q_transform_
            # If spec_mix=False
            else:
                q_transform = [q_transform0]
        return q_transform


# Log results for Weights & Biases
class WandB:
    def __init__(self, cfg):
        self.key = ''   # Enter your wandb key here
        self.id = wandb.util.generate_id()  # Generate version name for tracking in wandb
        self.wb_logger = pl_loggers.WandbLogger(project='gwave',
                                                config=cfg,
                                                name=self.id,
                                                version=self.id)
        wandb.login(key=self.key)
