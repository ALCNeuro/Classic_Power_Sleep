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

files = glob(os.path.join(preprocessed_dir, "*.fif"))

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

coi = ['sub_id', 'channel', 'stage', 
   'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
   'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma']

cols_power = [
    'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
    'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma'
    ]

#%% Loop over all subjects in the data directory

file = files[0]

for i, file in enumerate(files) :
    
    sub_id = file.split('Preproc/')[-1].split('_epo')[0]
    this_subject_savepath = os.path.join(
        power_dir, f"{sub_id}_bandpower.csv"
        )
    this_dic = {f : [] for f in coi}
    
    print(f"Processing : {sub_id}... [{i+1} / {len(files)}]")
        
    if not os.path.exists(this_subject_savepath) : 
    
        temp_dic = {stage : {chan : [] for chan in channels}
                              for stage in stages}

        print(f"...processing {sub_id}")
        
        epochs = mne.read_epochs(file, preload = True)
        epochs.drop_bad(threshold)
        
        for st in stages:
            print(f'processing {st}')
            if st not in epochs.event_id.keys() : continue
        
            temp_power = epochs[st].compute_psd(
                    method = method,
                    fmin = fmin, 
                    fmax = fmax,
                    n_fft = n_fft,
                    n_overlap = n_overlap,
                    n_per_seg = n_per_seg,
                    window = window,
                    picks = channels
                    )
            
            for i_epoch in range(len(epochs[st])):
                this_power = np.squeeze(temp_power[i_epoch])
                
                abs_bandpower_ch = {
                     f"abs_{band}" : np.nanmean(this_power[:,
                             np.logical_and(
                                 freqs >= borders[0], freqs <= borders[1]
                                 )], axis = 1)
                     for band, borders in freq_bands.items()}
                
                total_power = np.sum(
                    [abs_bandpower_ch[f"abs_{band}"] 
                     for band, borders in freq_bands.items()], axis = 0
                    )
                
                rel_bandpower_ch = {
                    f"rel_{band}" : abs_bandpower_ch[f"abs_{band}"] / total_power
                    for band in freq_bands.keys()
                    }
                
                abs_bandpower_ch_log = {
                    f"abs_{band}":10*np.log(abs_bandpower_ch[f"abs_{band}"]) 
                    for band in freq_bands.keys()
                    }
                
                for i_ch, channel in enumerate(channels) :
                    this_dic['sub_id'].append(sub_id)
                    this_dic['stage'].append(st)
                    this_dic['channel'].append(channel)
                    for col in cols_power :
                        if col.startswith('abs'):
                            this_dic[col].append(abs_bandpower_ch_log[col][i_ch])
                        if col.startswith('rel'):
                            this_dic[col].append(rel_bandpower_ch[col][i_ch])
                               
    subject_df = pd.DataFrame.from_dict(this_dic)
    subject_df.to_csv(this_subject_savepath)
