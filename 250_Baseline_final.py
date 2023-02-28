import pandas as pd
import numpy as np
import glob
import scipy.signal as signal
from scipy import sparse
from scipy.signal import medfilt
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def baseline_als(file, lam, p, niter=10):
    L = len(file)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*file)
        w = p * (file > z) + (1-p) * (file < z)
    return z

def lowpass(file):
    '''
    Low Pass Filter

    Parameters
    -------------------------------------------------------------
    file: input data on which low pass filter needs to be applied
    -------------------------------------------------------------
    '''
    b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, file)
    return low_passed

def baseline_reconstruction_250(file, kernel_size):
    als_baseline = baseline_als(file, 16**5, 0.01) 
    s_als = df1 - als_baseline
    s_corrected = signal.detrend(s_als)
    corrected_baseline = s_corrected - medfilt(s_corrected, kernel_size)
    return corrected_baseline

header_list = ['MLII', 'Value', 'VALUE', 'ECG', '0', 'val','V1', 'II', 'noise1', 'noise2','ii']

for i in glob.glob('C:\\Users\\Hops\\Downloads\\all_files\\137.csv'):
    name = i.split("\\")[-1].split(".csv")[0] 
    # print(i)
    print(name)
    # print(i)
    # count = 0
    # k = 0
    # l= 3750
    
    for header in header_list:
        try:
            df1 = pd.read_csv(i)[header]
        except:
            pass
    
    # initial_file = df1.iloc[k:l]
    # kernel_size = 51; for heavy noise or for any activity tag
    # kernel_size = 101; for 200 and 250 Hz
    # kernel_size = 150; for 360 Hz
    baseline = baseline_reconstruction_250(file = df1, kernel_size = 101)
    low_pass = lowpass(baseline)

    ax0 = plt.subplot(311)
    ax0.set_title("Raw Signal")
    ax0.plot(df1)
    # ax1 = plt.subplot(412, sharex = ax0, sharey =ax0)
    # ax1.set_title("Base Line Shift")
    # ax1.plot(s_corrected)
    ax2 = plt.subplot(312, sharex = ax0, sharey =ax0)
    ax2.set_title("Baseline")
    ax2.plot(baseline)
    ax3 = plt.subplot(313, sharex = ax0, sharey =ax0)
    ax3.set_title("Low Pass")
    ax3.plot(low_pass)
    plt.show()

    # k=+3750
    # l+=3750
    # count=+1
    # if l > len(df1):
    #     break