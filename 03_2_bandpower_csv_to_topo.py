#%%
import mne
import os
import pickle
from glob import glob
from scipy.stats import sem
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing

# Paths
raw_dir = '/Users/arthurlecoz/Desktop/boki'
preprocessed_dir = os.path.join(raw_dir, 'Preproc')
power_dir = os.path.join(raw_dir, 'Power')

# Create necessary directories if they don't exist
# os.makedirs(raw_dir, exist_ok=True)
# os.makedirs(preprocessed_dir, exist_ok=True)
# os.makedirs(results_dir, exist_ok=True)

sfreq = 256

freqs = np.linspace(1, 40, 157)

method = "welch"
fmin = 1
fmax = 40
n_fft = 4*sfreq
n_per_seg = n_fft
n_overlap = int(n_per_seg/2)
window = "hamming"

threshold = dict(eeg = 300e-6)

freq_bands = {
     'delta': (1, 4),
     'theta': (4, 8),
     'alpha': (8, 12),
     'beta': (12, 30),
     'gamma': (30, 40)
     }

bp_files = glob(os.path.join(power_dir, "*bandpower.csv"))

stages = ["W", "N1", "N2", "N3", "R"]
channels = np.array(
    ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
       'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'F9', 'F10', 'T9',
       'T10', 'P9', 'P10', 'AF7', 'AF3', 'F11', 'F5', 'F1', 'FT11', 'FT9',
       'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'C5', 'C1', 'TP11', 'TP9',
       'TP7', 'CP3', 'CP1', 'P11', 'P5', 'P1', 'PO7', 'PO3', 'POZ', 'OZ',
       'FPZ', 'AFZ', 'AF4', 'AF8', 'F2', 'F6', 'F12', 'FC2', 'FC4', 'FC6',
       'FT8', 'FT10', 'FT12', 'C6', 'C2', 'CPZ', 'CP2', 'CP4', 'CP6',
       'TP8', 'TP10', 'TP12', 'P2', 'P6', 'P12', 'PO4', 'PO8', 'CP5']
    )

midline = ["FZ", "CZ", "PZ", "OZ"]

#%% Gather all of the dataframe and concat it

all_subject_bp_savename = os.path.join(
    power_dir, "all_subject_bandpower.csv"
    )

if not os.path.exists(all_subject_bp_savename) 
    all_csv = [pd.read_csv(csv) for csv in bp_files]
    df = pd.concat(all_csv)
    del df['Unnamed: 0']
else : df = pd.read_csv(all_subject_bp_savename)

# %% Group it by subject however you want 

"""
You obtained a dataframe for each subject for each epochs.
You want to average per subject to have a value of absolute and relative power
per frequency band per sleep stage.

But depending on how you want to plot it afterward, you can adjust it.

Here I'll code it so you can compare everything to wake, but you could play
with it.
"""

mean_df = df.groupby(['sub_id', 'channel', 'stage'], as_index = False).mean()

# %% Topographies all sleep stages : Eye Check





# %% Topographies : Statistical comparisons




