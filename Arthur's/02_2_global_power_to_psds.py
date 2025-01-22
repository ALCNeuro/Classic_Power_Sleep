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

gp_files = glob(os.path.join(power_dir, "*.pickle"))
midline = ["FZ", "CZ", "PZ", "OZ"]

#%% Loop over all subjects in the data directory

big_dic = {st : {channel : [] for channel in channels} for st in stages}
file = files[0]

for i, file in enumerate(gp_files) :
    
    this_dic = pd.read_pickle(file)
    # if subtype.startswith('N') : continue
    sub_id = file.split('Power/')[-1].split('_global')[0]
    # session = sub_id[-2:]
    
    print(f"Processing : {sub_id}... [{i+1} / {len(gp_files)}]")
        
    for st in stages:
        for channel in channels:
            if len(this_dic[st][channel]) < 1 :
                big_dic[st][channel].append(
                    np.nan * np.empty(freqs.shape[0]))
            else : 
                big_dic[st][channel].append(
                    10 * np.log10(this_dic[st][channel][0]))

# %% converting for plotting

dic_psd = {st : {chan : [] for chan in channels}
                      for st in stages}
dic_sem = {st :  {chan : [] for chan in channels}
                      for st in stages}

for stage in big_dic.keys() :
    for channel in big_dic[stage].keys() :
        dic_psd[stage][channel] = np.nanmean(big_dic[stage][channel], axis = 0)
        dic_sem[stage][channel] = sem(big_dic[stage][channel], nan_policy = 'omit')

# %% PSD plots per stages at midline (inspection)

fig, axs = plt.subplots(
    nrows=1, 
    ncols=len(midline), 
    figsize=(20, 6), 
    sharey=True, 
    layout = "constrained"
    )
for st in stages : 
    for i, channel in enumerate(midline):
        ax = axs[i]
    
        # Convert power to dB
        psd_db = dic_psd[st][channel]
    
        # Calculate the SEM
        sem_db = dic_sem[st][channel]
    
        # Plot the PSD and SEM
        ax.plot(
            freqs, 
            psd_db, 
            label = st, 
            # color = palette[j],
            alpha = .7,
            linewidth = 2
            )
        ax.fill_between(
            freqs, 
            psd_db - sem_db, 
            psd_db + sem_db, 
            alpha= 0.2, 
            # color = palette[j]
            )
    
        # Set the title and labels
        ax.set_title('Channel: ' + channel)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim([1, 40])
        # ax.set_ylim([-30, 60])
        ax.legend()
    
    # Add the condition name as a title for the entire figure
    fig.suptitle('Averaged Stage in Midline')
    
    # Add a y-axis label to the first subplot
    axs[0].set_ylabel('Power (dB)')
    for i in range(len(midline)) :
        if i < len(midline)-1:
            axs[i].get_legend().set_visible(False)
    
    # Adjust the layout of the subplots
    # plt.constrained_layout()
    
    # Show the plot
    plt.show()
    
# %% 
