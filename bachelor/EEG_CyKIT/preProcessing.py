# from scipy.fftpack import fft
from scipy import signal
from scipy.integrate import simps
from numpy.fft import fft, fftfreq, ifft
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pyedflib
import os
import csv
from sklearn.decomposition import FastICA


def plot(subject, stage='W', fs=256):
    if stage == 'W':
        y = np.loadtxt(
            './Dataset/data_cooked/%s/%s' % (subject, subject + "-Sleep Stage W.csv"),
            unpack=False,
            delimiter=',',
            skiprows=0,
            usecols=range(1, 15))
    else:
        y = np.loadtxt(
            './Dataset/data_cooked/%s/%s' % (subject, subject + "-Sleep Stage 1.csv"),
            unpack=False,
            delimiter=',',
            skiprows=0,
            usecols=range(1, 15))
    # Original Signal
    y = notch_filter(60, y)
    y = notch_filter(120, y)
    y = butter_highpass(1, fs, y)
    y = y[:, 1]
    mean = np.mean(y)
    std = np.std(y)
    var = std ** 2
    print(mean, std, var)
    n = np.size(y)
    t = np.arange(0, len(y) / fs, 1 / fs)

    # FFT
    freq_fft = (fs / 2) * np.linspace(0, 1, int(n / 2))
    y_fft = fft(y)
    y_fft_theo = (2 / n) * np.abs(y_fft[0:np.size(freq_fft)])

    # Original Signal Plot
    plt.subplot(2, 1, 1)
    plt.title("Original Signal Plot")
    plt.ylabel("Amplitude (uV)")
    plt.xlabel("Time (sec)")
    plt.plot(t, y)
    # FFT Plot
    plt.subplot(2, 1, 2)
    plt.title("Fast Fourier Transform (FFT) Plot")
    plt.ylabel("Amplitude (uV)")
    plt.xlabel("Freq (hz)")
    plt.plot(freq_fft, y_fft_theo)

    plt.tight_layout()
    plt.show()


def fft_psd_estimation(y_arr, Fs=256):
    values = {}
    for idx, y in enumerate(y_arr):
        n = np.size(y)
        # FFT
        freq_fft = (Fs / 2) * np.linspace(0, 1, int(n / 2))
        y_fft = fft(y)
        y_fft_theo = (2 / n) * np.abs(y_fft[0:np.size(freq_fft)])
        values["freq_fft"] = freq_fft
        values["y" + str(idx + 1) + "_fft_theo"] = y_fft_theo
        # PSD
        # FFT Classic
        y_psd_fft = 2 * (np.abs(y_fft[0:np.size(freq_fft)] / n) ** 2)
        values["y" + str(idx + 1) + "_psd_classic"] = y_psd_fft
        # Welch's Method
        window = 4 * Fs  # Define window length (4 seconds) -> 2 cycles/lowest freq we have -> 2/0.5 = 4
        freq_psd, y_psd_welch = signal.welch(y, fs=Fs, nperseg=window)
        values["freq_psd"] = freq_psd
        values["y" + str(idx + 1) + "_psd_welch"] = y_psd_welch
    return values


def average_channels(arr):
    mean = np.mean(arr, axis=1, keepdims=True)
    arr -= mean
    return arr


def moving_average(arr, moving_average=13):
    for sample in range(arr.shape[0] - moving_average):
        arr[sample, :] = np.mean(arr[sample: sample + moving_average, :], axis=0, keepdims=True)
    return arr


def z_score(arr):
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.mean(arr, axis=0, keepdims=True)
    arr = (arr - mean) / std
    return arr


def notch_filter(freqToRemove, arr, q, fs=256):
    w0 = freqToRemove/(fs/2)
    Q = w0/q
    b, a = signal.iirnotch(w0, Q, fs)

    for channel in range(arr.shape[1]):
        arr[:, channel] = signal.lfilter(b, a, arr[:, channel])

    return arr


def butter_highpass(cutoff, fs, arr, order=6):
    nyq = 0.5 * fs
    cut = cutoff / nyq
    b, a = signal.butter(order, cut, btype='high')
    for channel in range(arr.shape[1]):
        arr[:, channel] = signal.lfilter(b, a, arr[:, channel])

    return arr


def butter_bandpass(lowcut, highcut, fs, arr, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    for channel in range(arr.shape[1]):
        arr[:, channel] = signal.lfilter(b, a, arr[:, channel])

    return arr


def apply_ICA(arr):
    ica = FastICA()
    for channel in range(arr.shape[1]):
        arr[:, channel] = ica.fit_transform(arr[:, channel])

    return arr


def feature_extraction(epoch_length=1., duration=300, arr=None, method=0, fs=256):
    if arr is None:
        if not os.path.isdir('Dataset/data_cooked/features'):
            os.makedirs('Dataset/data_cooked/features')

        with open('Dataset/data_cooked/features/W_fft.csv', 'w', newline='') as w_fft_obj, \
                open('Dataset/data_cooked/features/N1_fft.csv', 'w', newline='') as n1_fft_obj, \
                open('Dataset/data_cooked/features/W_psd_classic.csv', 'w', newline='') as w_psd_classic_obj, \
                open('Dataset/data_cooked/features/N1_psd_classic.csv', 'w', newline='') as n1_psd_classic_obj, \
                open('Dataset/data_cooked/features/W_psd_welch.csv', 'w', newline='') as w_psd_welch_obj, \
                open('Dataset/data_cooked/features/N1_psd_welch.csv', 'w', newline='') as n1_psd_welch_obj:
            w_fft_writer = csv.writer(w_fft_obj)
            n1_fft_writer = csv.writer(n1_fft_obj)
            w_psd_classic_writer = csv.writer(w_psd_classic_obj)
            n1_psd_classic_writer = csv.writer(n1_psd_classic_obj)
            w_psd_welch_writer = csv.writer(w_psd_welch_obj)
            n1_psd_welch_writer = csv.writer(n1_psd_welch_obj)
            header = ["File Name",
                      "F3 Delta",
                      "F3 Theta",
                      "F3 Alpha",
                      "F3 Beta",
                      "F3 Gamma",
                      "FC5 Delta",
                      "FC5 Theta",
                      "FC5 Alpha",
                      "FC5 Beta",
                      "FC5 Gamma",
                      "AF3 Delta",
                      "AF3 Theta",
                      "AF3 Alpha",
                      "AF3 Beta",
                      "AF3 Gamma",
                      "F7 Delta",
                      "F7 Theta",
                      "F7 Alpha",
                      "F7 Beta",
                      "F7 Gamma",
                      "T7 Delta",
                      "T7 Theta",
                      "T7 Alpha",
                      "T7 Beta",
                      "T7 Gamma",
                      "P7 Delta",
                      "P7 Theta",
                      "P7 Alpha",
                      "P7 Beta",
                      "P7 Gamma",
                      "O1 Delta",
                      "O1 Theta",
                      "O1 Alpha",
                      "O1 Beta",
                      "O1 Gamma",
                      "O2 Delta",
                      "O2 Theta",
                      "O2 Alpha",
                      "O2 Beta",
                      "O2 Gamma",
                      "P8 Delta",
                      "P8 Theta",
                      "P8 Alpha",
                      "P8 Beta",
                      "P8 Gamma",
                      "T8 Delta",
                      "T8 Theta",
                      "T8 Alpha",
                      "T8 Beta",
                      "T8 Gamma",
                      "F8 Delta",
                      "F8 Theta",
                      "F8 Alpha",
                      "F8 Beta",
                      "F8 Gamma",
                      "AF4 Delta",
                      "AF4 Theta",
                      "AF4 Alpha",
                      "AF4 Beta",
                      "AF4 Gamma",
                      "FC6 Delta",
                      "FC6 Theta",
                      "FC6 Alpha",
                      "FC6 Beta",
                      "FC6 Gamma",
                      "F4 Delta",
                      "F4 Theta",
                      "F4 Alpha",
                      "F4 Beta",
                      "F4 Gamma"]
            w_fft_writer.writerow(header)
            n1_fft_writer.writerow(header)
            w_psd_classic_writer.writerow(header)
            n1_psd_classic_writer.writerow(header)
            w_psd_welch_writer.writerow(header)
            n1_psd_welch_writer.writerow(header)

            for patientFolder in os.listdir('Dataset/data_cooked'):
                if patientFolder.startswith("features") or patientFolder.startswith("parameters") or patientFolder.startswith("subjects"):
                    continue
                print("**************   %s  **************" % patientFolder)

                epoch_size = int(epoch_length * fs)
                epochs_per_subject = int(duration / epoch_length)
                w_arr = np.loadtxt(
                    './Dataset/data_cooked/%s/%s' % (patientFolder, patientFolder + "-Sleep Stage W.csv"),
                    unpack=False,
                    delimiter=',',
                    skiprows=0,
                    usecols=range(1, 15))
                n1_arr = np.loadtxt(
                    './Dataset/data_cooked/%s/%s' % (patientFolder, patientFolder + "-Sleep Stage 1.csv"),
                    unpack=False,
                    delimiter=',',
                    skiprows=0,
                    usecols=range(1, 15))
                # w_arr = notch_filter(50, w_arr, 35, fs)
                # n1_arr = notch_filter(50, n1_arr, 35, fs)
                # w_arr = notch_filter(120, w_arr, 0.0333, fs)
                # n1_arr = notch_filter(120, n1_arr, 0.0333, fs)
                # w_arr = butter_highpass(1, fs, w_arr)
                # n1_arr = butter_highpass(1, fs, n1_arr)
                # w_arr = apply_ICA(w_arr)
                # n1_arr = apply_ICA(n1_arr)
                # w_arr = butter_bandpass(0.5, 50, 256, w_arr)
                # n1_arr = butter_bandpass(0.5, 50, 256, n1_arr)
                # w_arr = average_channels(w_arr)
                # n1_arr = average_channels(n1_arr)
                # w_arr = moving_average(w_arr, 13)
                # n1_arr = moving_average(n1_arr, 13)
                # w_arr = z_score(w_arr)
                # n1_arr = z_score(n1_arr)

                for i in range(epochs_per_subject):
                    file_name_w = patientFolder + "-Sleep stage W-" + str(i).zfill(len(str(epochs_per_subject)))
                    file_name_n1 = patientFolder + "-Sleep stage 1-" + str(i).zfill(len(str(epochs_per_subject)))
                    w_arr_subset = []
                    n1_arr_subset = []
                    for j in range(w_arr.shape[1]):
                        w_arr_subset.append(w_arr[i * epoch_size: i * epoch_size + epoch_size, j])
                        n1_arr_subset.append(n1_arr[i * epoch_size: i * epoch_size + epoch_size, j])
                    values_w = fft_psd_estimation(w_arr_subset, fs)
                    values_n1 = fft_psd_estimation(n1_arr_subset, fs)

                    row_fft_w = [file_name_w]
                    row_fft_n1 = [file_name_n1]
                    row_psd_classic_w = [file_name_w]
                    row_psd_classic_n1 = [file_name_n1]
                    row_psd_welch_w = [file_name_w]
                    row_psd_welch_n1 = [file_name_n1]
                    N = epoch_size
                    for j in range(w_arr.shape[1]):
                        # Extraction from FFT
                        # To get a range of frequencies (f1,f2) from a fft list, N*(f1/fs) ... N*(f2/fs)
                        # W
                        Y_m = values_w["y" + str(j + 1) + "_fft_theo"]
                        # Delta band 0-4 Hz
                        deltaBand = sum(Y_m[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                        row_fft_w.append(format(deltaBand, '.3f'))
                        # Theta band 4-8 Hz
                        thetaBand = sum(Y_m[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                        row_fft_w.append(format(thetaBand, '.3f'))
                        # Alpha band 8-13 Hz
                        alphaBand = sum(Y_m[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                        row_fft_w.append(format(alphaBand, '.3f'))
                        # Beta band 13-30 Hz
                        betaBand = sum(Y_m[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                        row_fft_w.append(format(betaBand, '.3f'))
                        # Gamma Band >30 Hz
                        gammaBand = sum(Y_m[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                        row_fft_w.append(format(gammaBand, '.3f'))
                        # N1
                        Y_m = values_n1["y" + str(j + 1) + "_fft_theo"]
                        # Delta band 0-4 Hz
                        deltaBand = sum(Y_m[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                        row_fft_n1.append(format(deltaBand, '.3f'))
                        # Theta band 4-8 Hz
                        thetaBand = sum(Y_m[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                        row_fft_n1.append(format(thetaBand, '.3f'))
                        # Alpha band 8-13 Hz
                        alphaBand = sum(Y_m[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                        row_fft_n1.append(format(alphaBand, '.3f'))
                        # Beta band 13-30 Hz
                        betaBand = sum(Y_m[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                        row_fft_n1.append(format(betaBand, '.3f'))
                        # Gamma Band >30 Hz
                        gammaBand = sum(Y_m[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                        row_fft_n1.append(format(gammaBand, '.3f'))
                        # Extraction from PSD Classic
                        # W
                        Y_m = values_w["y" + str(j + 1) + "_psd_classic"]
                        # Delta band 0-4 Hz
                        deltaBand = sum(Y_m[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                        row_psd_classic_w.append(format(deltaBand, '.3f'))
                        # Theta band 4-8 Hz
                        thetaBand = sum(Y_m[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                        row_psd_classic_w.append(format(thetaBand, '.3f'))
                        # Alpha band 8-13 Hz
                        alphaBand = sum(Y_m[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                        row_psd_classic_w.append(format(alphaBand, '.3f'))
                        # Beta band 13-30 Hz
                        betaBand = sum(Y_m[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                        row_psd_classic_w.append(format(betaBand, '.3f'))
                        # Gamma Band >30 Hz
                        gammaBand = sum(Y_m[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                        row_psd_classic_w.append(format(gammaBand, '.3f'))
                        # N1
                        Y_m1 = values_n1["y" + str(j + 1) + "_psd_classic"]
                        # Delta band 0-4 Hz
                        deltaBand = sum(Y_m1[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                        row_psd_classic_n1.append(format(deltaBand, '.3f'))
                        # Theta band 4-8 Hz
                        thetaBand = sum(Y_m1[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                        row_psd_classic_n1.append(format(thetaBand, '.3f'))
                        # Alpha band 8-13 Hz
                        alphaBand = sum(Y_m1[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                        row_psd_classic_n1.append(format(alphaBand, '.3f'))
                        # Beta band 13-30 Hz
                        betaBand = sum(Y_m1[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                        row_psd_classic_n1.append(format(betaBand, '.3f'))
                        # Gamma Band >30 Hz
                        gammaBand = sum(Y_m1[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                        row_psd_classic_n1.append(format(gammaBand, '.3f'))
                        # Extraction from PSD Welch
                        # W
                        psd = values_w["y" + str(j + 1) + "_psd_welch"]
                        freq_psd = values_w["freq_psd"]
                        freq_res = freq_psd[1] - freq_psd[0]  # Frequency resolution = 1 / 4 = 0.25
                        # Compute the absolute power by approximating the area under the curve
                        # Delta band 0.5-4 Hz
                        deltaBand = simps(psd[np.logical_and(freq_psd >= 0.5, freq_psd <= 4)], dx=freq_res)
                        row_psd_welch_w.append(format(deltaBand, '.3f'))
                        # Theta band 4-8 Hz
                        thetaBand = simps(psd[np.logical_and(freq_psd >= 4, freq_psd <= 8)], dx=freq_res)
                        row_psd_welch_w.append(format(thetaBand, '.3f'))
                        # Alpha band 8-13 Hz
                        alphaBand = simps(psd[np.logical_and(freq_psd >= 8, freq_psd <= 13)], dx=freq_res)
                        row_psd_welch_w.append(format(alphaBand, '.3f'))
                        # Beta band 13-30 Hz
                        betaBand = simps(psd[np.logical_and(freq_psd >= 13, freq_psd <= 30)], dx=freq_res)
                        row_psd_welch_w.append(format(betaBand, '.3f'))
                        # Gamma Band >30 Hz
                        gammaBand = simps(psd[np.logical_and(freq_psd >= 30, freq_psd <= 50)], dx=freq_res)
                        row_psd_welch_w.append(format(gammaBand, '.3f'))
                        # N1
                        psd = values_n1["y" + str(j + 1) + "_psd_welch"]
                        freq_psd = values_n1["freq_psd"]
                        freq_res = freq_psd[1] - freq_psd[0]  # Frequency resolution = 1 / 4 = 0.25
                        # Compute the absolute power by approximating the area under the curve
                        # Delta band 0.5-4 Hz
                        deltaBand = simps(psd[np.logical_and(freq_psd >= 0.5, freq_psd <= 4)], dx=freq_res)
                        row_psd_welch_n1.append(format(deltaBand, '.3f'))
                        # Theta band 4-8 Hz
                        thetaBand = simps(psd[np.logical_and(freq_psd >= 4, freq_psd <= 8)], dx=freq_res)
                        row_psd_welch_n1.append(format(thetaBand, '.3f'))
                        # Alpha band 8-13 Hz
                        alphaBand = simps(psd[np.logical_and(freq_psd >= 8, freq_psd <= 13)], dx=freq_res)
                        row_psd_welch_n1.append(format(alphaBand, '.3f'))
                        # Beta band 13-30 Hz
                        betaBand = simps(psd[np.logical_and(freq_psd >= 13, freq_psd <= 30)], dx=freq_res)
                        row_psd_welch_n1.append(format(betaBand, '.3f'))
                        # Gamma Band >30 Hz
                        gammaBand = simps(psd[np.logical_and(freq_psd >= 30, freq_psd <= 50)], dx=freq_res)
                        row_psd_welch_n1.append(format(gammaBand, '.3f'))

                    w_fft_writer.writerow(row_fft_w)
                    n1_fft_writer.writerow(row_fft_n1)
                    w_psd_classic_writer.writerow(row_psd_classic_w)
                    n1_psd_classic_writer.writerow(row_psd_classic_n1)
                    w_psd_welch_writer.writerow(row_psd_welch_w)
                    n1_psd_welch_writer.writerow(row_psd_welch_n1)
        return None
    else:
        # arr = average_channels(arr)
        # arr = moving_average(arr, 13)
        # arr = z_score(arr)
        # arr = notch_filter(60, arr, fs)
        # arr = notch_filter(120, arr, fs)
        # arr = butter_highpass(0.5, fs, arr)
        epoch_size = int(epoch_length * fs)
        epochs_per_subject = int(duration / epoch_length)
        rows = []
        for i in range(epochs_per_subject):
            row = []
            arr_subset = []
            for j in range(arr.shape[1]):
                arr_subset.append(arr[i * epoch_size: i * epoch_size + epoch_size, j])
            values = fft_psd_estimation(arr_subset, fs)
            N = epoch_size

            for j in range(arr.shape[1]):
                if method == 0:
                    # Extraction from FFT
                    Y_m = values["y" + str(j + 1) + "_fft_theo"]
                    # Delta band 0-4 Hz
                    deltaBand = sum(Y_m[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                    row.append(format(deltaBand, '.3f'))
                    # Theta band 4-8 Hz
                    thetaBand = sum(Y_m[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                    row.append(format(thetaBand, '.3f'))
                    # Alpha band 8-13 Hz
                    alphaBand = sum(Y_m[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                    row.append(format(alphaBand, '.3f'))
                    # Beta band 13-30 Hz
                    betaBand = sum(Y_m[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                    row.append(format(betaBand, '.3f'))
                    # Gamma Band >30 Hz
                    gammaBand = sum(Y_m[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                    row.append(format(gammaBand, '.3f'))
                elif method == 1:
                    # Extraction from PSD Classic
                    Y_m = values["y" + str(j + 1) + "_psd_classic"]
                    # Delta band 0-4 Hz
                    deltaBand = sum(Y_m[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                    row.append(format(deltaBand, '.3f'))
                    # Theta band 4-8 Hz
                    thetaBand = sum(Y_m[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                    row.append(format(thetaBand, '.3f'))
                    # Alpha band 8-13 Hz
                    alphaBand = sum(Y_m[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                    row.append(format(alphaBand, '.3f'))
                    # Beta band 13-30 Hz
                    betaBand = sum(Y_m[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                    row.append(format(betaBand, '.3f'))
                    # Gamma Band >30 Hz
                    gammaBand = sum(Y_m[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                    row.append(format(gammaBand, '.3f'))
                elif method == 2:
                    # Extraction from PSD Welch
                    psd = values["y" + str(j + 1) + "_psd_welch"]
                    freq_psd = values["freq_psd"]
                    freq_res = freq_psd[1] - freq_psd[0]  # Frequency resolution = 1 / 4 = 0.25
                    # Compute the absolute power by approximating the area under the curve
                    # Delta band 0.5-4 Hz
                    deltaBand = simps(psd[np.logical_and(freq_psd >= 0.5, freq_psd <= 4)], dx=freq_res)
                    row.append(format(deltaBand, '.3f'))
                    # Theta band 4-8 Hz
                    thetaBand = simps(psd[np.logical_and(freq_psd >= 4, freq_psd <= 8)], dx=freq_res)
                    row.append(format(thetaBand, '.3f'))
                    # Alpha band 8-13 Hz
                    alphaBand = simps(psd[np.logical_and(freq_psd >= 8, freq_psd <= 13)], dx=freq_res)
                    row.append(format(alphaBand, '.3f'))
                    # Beta band 13-30 Hz
                    betaBand = simps(psd[np.logical_and(freq_psd >= 13, freq_psd <= 30)], dx=freq_res)
                    row.append(format(betaBand, '.3f'))
                    # Gamma Band >30 Hz
                    gammaBand = simps(psd[np.logical_and(freq_psd >= 30, freq_psd <= 50)], dx=freq_res)
                    row.append(format(gammaBand, '.3f'))
            rows.append(row)
        return rows


# feature_extraction(2., 90)
# plot('OmarN0', '1')
