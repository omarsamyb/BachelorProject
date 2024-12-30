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

# style.use('ggplot')
file_duration = 0.25
total_individuals = 75
files_count = 570


def min_duration(edf2csv):
    min_duration = 99999999999
    min_name = None
    min_stage = None
    for filename in os.listdir('Dataset_EEG/data_raw/sleep/sleep-cassette'):
        if filename.endswith("-Hypnogram.edf"):
            w_duration = 0
            n1_duration = 0
            Hypnogram = pyedflib.EdfReader('./Dataset_EEG/data_raw/sleep/sleep-cassette/' + filename)
            annotations = Hypnogram.readAnnotations()
            for annotation in np.arange(Hypnogram.annotations_in_file):
                if annotations[2][annotation] == "Sleep stage W":
                    w_duration += annotations[1][annotation]
                if annotations[2][annotation] == "Sleep stage 1":
                    n1_duration += annotations[1][annotation]
            # print(filename, w_duration, n1_duration)
            if w_duration < min_duration:
                min_duration = w_duration
                min_name = filename
                min_stage = "W"
            if n1_duration < min_duration:
                min_duration = n1_duration
                min_name = filename
                min_stage = "N1"
            Hypnogram.close()
    # print(min_name, min_duration, min_stage)
    if edf2csv:
        return min_duration
    else:
        return files_count


def edf2csv():
    epoch = file_duration * 100  # 1 second
    wLimit = min_duration(True) / file_duration
    s1Limit = min_duration(True) / file_duration
    # cassette & telemetry
    for filename in os.listdir('Dataset_EEG/data_raw/sleep/sleep-cassette'):
        if filename.endswith("-PSG.edf"):
            codeName = filename[0:8]
            psgName = codeName + '-PSG'
            continue
        print(codeName)
        hypnogramName = filename[0:8] + '-Hypnogram'
        PSG = pyedflib.EdfReader('./Dataset_EEG/data_raw/sleep/sleep-cassette/' + psgName + '.edf')
        Hypnogram = pyedflib.EdfReader('./Dataset_EEG/data_raw/sleep/sleep-cassette/' + hypnogramName + '.edf')
        n = PSG.signals_in_file
        Fpz_Cz = PSG.readSignal(0)
        Pz_Oz = PSG.readSignal(1)

        if os.path.isdir('Dataset_EEG/data_cooked/%s' % codeName):
            continue
        else:
            os.makedirs('Dataset_EEG/data_cooked/%s' % codeName)

        sW = -1
        s1 = -1
        s2 = -1
        s3 = -1
        s4 = -1
        sR = -1
        mT = -1
        counter = -1
        annotations = Hypnogram.readAnnotations()
        for annotation in np.arange(Hypnogram.annotations_in_file):
            if s1 >= s1Limit and sW >= wLimit:
                break
            if annotations[2][annotation] == "Sleep stage 1":
                s1 += 1
                counter = s1
                if s1 >= s1Limit:
                    continue
            elif annotations[2][annotation] == "Sleep stage 2":
                s2 += 1
                counter = s2
                continue
            elif annotations[2][annotation] == "Sleep stage 3":
                s3 += 1
                counter = s3
                continue
            elif annotations[2][annotation] == "Sleep stage 4":
                s4 += 1
                counter = s4
                continue
            elif annotations[2][annotation] == "Sleep stage W":
                sW += 1
                counter = sW
                if sW >= wLimit:
                    continue
            elif annotations[2][annotation] == "Sleep stage R":
                sR += 1
                counter = sR
                continue
            elif annotations[2][annotation] == "Movement time":
                mT += 1
                counter = mT
                continue
            elif annotations[2][annotation] == "Sleep stage ?":
                continue
            else:
                continue

            print("annotation: onset is %f    duration is %s    description is %s" % (
                annotations[0][annotation], annotations[1][annotation], annotations[2][annotation]))

            doubleEpochCounter = 0
            for signal in np.arange(int(annotations[0][annotation] * 100),
                                    int(annotations[0][annotation] * 100) + int(annotations[1][annotation] * 100)):
                if doubleEpochCounter % epoch == 0:
                    if doubleEpochCounter > 0:
                        counter += 1
                        if annotations[2][annotation] == "Sleep stage 1" and counter >= s1Limit:
                            break
                        if annotations[2][annotation] == "Sleep stage W" and counter >= wLimit:
                            break
                        csvFile.close()
                    if counter < 10:
                        csvFile = open(
                            './Dataset_EEG/data_cooked/%s/%s.csv' % (
                                codeName, codeName + "-" + annotations[2][annotation] + "-000" + str(counter)),
                            'w')
                    elif 10 <= counter < 100:
                        csvFile = open(
                            './Dataset_EEG/data_cooked/%s/%s.csv' % (
                                codeName, codeName + "-" + annotations[2][annotation] + "-00" + str(counter)),
                            'w')
                    elif 100 <= counter < 1000:
                        csvFile = open(
                            './Dataset_EEG/data_cooked/%s/%s.csv' % (
                                codeName, codeName + "-" + annotations[2][annotation] + "-0" + str(counter)),
                            'w')
                    else:
                        csvFile = open(
                            './Dataset_EEG/data_cooked/%s/%s.csv' % (
                                codeName, codeName + "-" + annotations[2][annotation] + "-" + str(counter)),
                            'w')

                    csvFile.write(PSG.getSignalLabels()[0] + "," + PSG.getSignalLabels()[1] + "\n")
                doubleEpochCounter += 1
                csvFile.write(format(Fpz_Cz[int(signal)], '.3f') + "," + format(Pz_Oz[int(signal)], '.3f') + "\n")
            csvFile.close()
            if annotations[2][annotation] == "Sleep stage 1":
                s1 = counter
            elif annotations[2][annotation] == "Sleep stage 2":
                s2 = counter
            elif annotations[2][annotation] == "Sleep stage 3":
                s3 = counter
            elif annotations[2][annotation] == "Sleep stage 4":
                s4 = counter
            elif annotations[2][annotation] == "Sleep stage W":
                sW = counter
            elif annotations[2][annotation] == "Sleep stage R":
                sR = counter
            elif annotations[2][annotation] == "Movement time":
                mT = counter
            elif annotations[2][annotation] == "Sleep stage ?":
                continue
            else:
                continue


def plot(patient_id, file_name, channel, seconds, method):
    # channel = 0 ->Fpz_Cz, channel = 1 ->Pz_Oz
    plt.close('all')
    y1, y2 = np.loadtxt('./Dataset_EEG/data_cooked/%s/%s.csv' % (patient_id, file_name),
                        unpack=True,
                        delimiter=',',
                        skiprows=1,
                        usecols=range(0, 2),
                        max_rows=int(seconds * 100))
    values = fft_psd_estimation(y1, y2)
    fs = 100
    if channel == 0:
        y = y1
    else:
        y = y2
    t = np.arange(0, len(y) / 100, 1 / fs)
    mean_y = np.mean(y)
    std_y = np.std(y)
    var_y = std_y ** 2
    print("Mean", mean_y)
    print("Standard Deviation", std_y)
    print("Variance", var_y)
    # Original Signal Plot
    plt.subplot(2, 1, 1)
    plt.title("Original Signal Plot")
    plt.ylabel("Amplitude (uV)")
    plt.xlabel("Time (sec)")
    plt.plot(t, y)
    # FFT Plot
    if method == 0:
        plt.subplot(2, 1, 2)
        plt.title("Fast Fourier Transform (FFT) Plot")
        plt.ylabel("Amplitude (uV)")
        plt.xlabel("Freq (hz)")
        if channel == 0:
            plt.plot(values["freq_fft"], values["y1_fft_theo"])
        else:
            plt.plot(values["freq_fft"], values["y2_fft_theo"])
    # PSD Classic Plot
    elif method == 1:
        plt.subplot(2, 1, 2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (uV^2 / Hz)')
        plt.title("Power Spectrum Density (PSD) Plot - Classic")
        if channel == 0:
            plt.plot(values["freq_fft"], values["y1_psd_classic"])
        else:
            plt.plot(values["freq_fft"], values["y2_psd_classic"])
    # PSD Welch Plot
    elif method == 2:
        plt.subplot(2, 1, 2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (uV^2 / Hz)')
        plt.title("Power Spectrum Density (PSD) Plot - Welch")
        if channel == 0:
            plt.plot(values["freq_psd"], values["y1_psd_welch"])
        else:
            plt.plot(values["freq_psd"], values["y2_psd_welch"])
    plt.tight_layout()
    plt.show()


def plot_bands(method, channel, n, band1, band2, show_w, show_n1):
    # epoch = file_duration * 100
    # Cassette Only
    if n == 0:
        max_rows_n1 = total_individuals * 2 * int(min_duration(False) / file_duration)  # Last SC row number - 1
        max_rows_w = total_individuals * 2 * int(min_duration(False) / file_duration)
        skip_rows_n1 = 1
        skip_rows_w = 1
    # Telemetry Only
    elif n == 1:
        max_rows_n1 = None
        max_rows_w = None
        skip_rows_n1 = (total_individuals * 2 * int(min_duration(False) / file_duration)) + 1  # Last SC row number
        skip_rows_w = (total_individuals * 2 * int(min_duration(False) / file_duration)) + 1
    # All Data
    elif n == 2:
        max_rows_n1 = None
        max_rows_w = None
        skip_rows_n1 = 1
        skip_rows_w = 1
    # First n data
    else:
        max_rows_n1 = n
        max_rows_w = n
        skip_rows_n1 = 1
        skip_rows_w = 1

    # FFT Values
    if method == 0:
        delta1_W, theta1_W, alpha1_W, beta1_W, gamma1_W = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y1_W_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 6),
            max_rows=max_rows_w)
        delta2_W, theta2_W, alpha2_W, beta2_W, gamma2_W = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y2_W_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 6),
            max_rows=max_rows_w)
        delta1_1, theta1_1, alpha1_1, beta1_1, gamma1_1 = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y1_N1_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 6),
            max_rows=max_rows_n1)
        delta2_1, theta2_1, alpha2_1, beta2_1, gamma2_1 = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y2_N1_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 6),
            max_rows=max_rows_n1)
    # PSD Classic
    elif method == 1:
        delta1_W, theta1_W, alpha1_W, beta1_W, gamma1_W = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y1_W_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 6),
            max_rows=max_rows_w)
        delta2_W, theta2_W, alpha2_W, beta2_W, gamma2_W = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y2_W_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 6),
            max_rows=max_rows_w)
        delta1_1, theta1_1, alpha1_1, beta1_1, gamma1_1 = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y1_N1_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 6),
            max_rows=max_rows_n1)
        delta2_1, theta2_1, alpha2_1, beta2_1, gamma2_1 = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y2_N1_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 6),
            max_rows=max_rows_n1)
    # PSD Welch
    elif method == 2:
        delta1_W, theta1_W, alpha1_W, beta1_W, gamma1_W = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y1_W_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 6),
            max_rows=max_rows_w)
        delta2_W, theta2_W, alpha2_W, beta2_W, gamma2_W = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y2_W_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 6),
            max_rows=max_rows_w)
        delta1_1, theta1_1, alpha1_1, beta1_1, gamma1_1 = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y1_N1_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 6),
            max_rows=max_rows_n1)
        delta2_1, theta2_1, alpha2_1, beta2_1, gamma2_1 = np.loadtxt(
            './Dataset_EEG/data_cooked/features/y2_N1_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 6),
            max_rows=max_rows_n1)

    if band1 == 'delta':
        plt.xlabel("Delta")
        if channel == 0:
            x1 = delta1_W
            x2 = delta1_1
        elif channel == 1:
            x1 = delta2_W
            x2 = delta2_1
    if band1 == 'theta':
        plt.xlabel("Theta")
        if channel == 0:
            x1 = theta1_W
            x2 = theta1_1
        elif channel == 1:
            x1 = theta2_W
            x2 = theta2_1
    if band1 == 'alpha':
        plt.xlabel("Alpha")
        if channel == 0:
            x1 = alpha1_W
            x2 = alpha1_1
        elif channel == 1:
            x1 = alpha2_W
            x2 = alpha2_1
    if band1 == 'beta':
        plt.xlabel("Beta")
        if channel == 0:
            x1 = beta1_W
            x2 = beta1_1
        elif channel == 1:
            x1 = beta2_W
            x2 = beta2_1
    if band1 == 'gamma':
        plt.xlabel("Gamma")
        if channel == 0:
            x1 = gamma1_W
            x2 = gamma1_1
        elif channel == 1:
            x1 = gamma2_W
            x2 = gamma2_1
    if band2 == 'delta':
        plt.ylabel("Delta")
        if channel == 0:
            y1 = delta1_W
            y2 = delta1_1
        elif channel == 1:
            y1 = delta2_W
            y2 = delta2_1
    if band2 == 'theta':
        plt.ylabel("Theta")
        if channel == 0:
            y1 = theta1_W
            y2 = theta1_1
        elif channel == 1:
            y1 = theta2_W
            y2 = theta2_1
    if band2 == 'alpha':
        plt.ylabel("Alpha")
        if channel == 0:
            y1 = alpha1_W
            y2 = alpha1_1
        elif channel == 1:
            y1 = alpha2_W
            y2 = alpha2_1
    if band2 == 'beta':
        plt.ylabel("Beta")
        if channel == 0:
            y1 = beta1_W
            y2 = beta1_1
        elif channel == 1:
            y1 = beta2_W
            y2 = beta2_1
    if band2 == 'gamma':
        plt.ylabel("Gamma")
        if channel == 0:
            y1 = gamma1_W
            y2 = gamma1_1
        elif channel == 1:
            y1 = gamma2_W
            y2 = gamma2_1

    plt.title("Red = Awake & Blue = Sleep Stage N1")
    if show_w:
        plt.plot(x1, y1, 'r+')
    if show_n1:
        plt.plot(x2, y2, 'b+')
    plt.show()


def feature_extraction():
    if not os.path.isdir('Dataset_EEG/data_cooked/features'):
        os.makedirs('Dataset_EEG/data_cooked/features')
    with open('Dataset_EEG/data_cooked/features/y1_W_fft.csv', 'w', newline='') as y1_w_fft_obj, \
            open('Dataset_EEG/data_cooked/features/y1_W_psd_classic.csv', 'w', newline='') as y1_w_psd_classic_obj, \
            open('Dataset_EEG/data_cooked/features/y1_W_psd_welch.csv', 'w', newline='') as y1_w_psd_welch_obj, \
            open('Dataset_EEG/data_cooked/features/y1_N1_fft.csv', 'w', newline='') as y1_n1_fft_obj, \
            open('Dataset_EEG/data_cooked/features/y1_N1_psd_classic.csv', 'w', newline='') as y1_n1_psd_classic_obj, \
            open('Dataset_EEG/data_cooked/features/y1_N1_psd_welch.csv', 'w', newline='') as y1_n1_psd_welch_obj, \
            open('Dataset_EEG/data_cooked/features/y2_W_fft.csv', 'w', newline='') as y2_w_fft_obj, \
            open('Dataset_EEG/data_cooked/features/y2_W_psd_classic.csv', 'w', newline='') as y2_w_psd_classic_obj, \
            open('Dataset_EEG/data_cooked/features/y2_W_psd_welch.csv', 'w', newline='') as y2_w_psd_welch_obj, \
            open('Dataset_EEG/data_cooked/features/y2_N1_fft.csv', 'w', newline='') as y2_n1_fft_obj, \
            open('Dataset_EEG/data_cooked/features/y2_N1_psd_classic.csv', 'w', newline='') as y2_n1_psd_classic_obj, \
            open('Dataset_EEG/data_cooked/features/y2_N1_psd_welch.csv', 'w', newline='') as y2_n1_psd_welch_obj:
        y1_w_fft_writer = csv.writer(y1_w_fft_obj)
        y1_w_psd_classic_writer = csv.writer(y1_w_psd_classic_obj)
        y1_w_psd_welch_writer = csv.writer(y1_w_psd_welch_obj)
        y1_n1_fft_writer = csv.writer(y1_n1_fft_obj)
        y1_n1_psd_classic_writer = csv.writer(y1_n1_psd_classic_obj)
        y1_n1_psd_welch_writer = csv.writer(y1_n1_psd_welch_obj)
        y2_w_fft_writer = csv.writer(y2_w_fft_obj)
        y2_w_psd_classic_writer = csv.writer(y2_w_psd_classic_obj)
        y2_w_psd_welch_writer = csv.writer(y2_w_psd_welch_obj)
        y2_n1_fft_writer = csv.writer(y2_n1_fft_obj)
        y2_n1_psd_classic_writer = csv.writer(y2_n1_psd_classic_obj)
        y2_n1_psd_welch_writer = csv.writer(y2_n1_psd_welch_obj)
        header_y1 = ["File Name",
                     "Fpz-Cz Delta",
                     "Fpz-Cz Theta",
                     "Fpz-Cz Alpha",
                     "Fpz-Cz Beta",
                     "Fpz-Cz Gamma"]
        header_y2 = ["File Name",
                     "Pz_Oz Delta",
                     "Pz_Oz Theta",
                     "Pz_Oz Alpha",
                     "Pz_Oz Beta",
                     "Pz_Oz Gamma"]
        y1_w_fft_writer.writerow(header_y1)
        y1_w_psd_classic_writer.writerow(header_y1)
        y1_w_psd_welch_writer.writerow(header_y1)
        y1_n1_fft_writer.writerow(header_y1)
        y1_n1_psd_classic_writer.writerow(header_y1)
        y1_n1_psd_welch_writer.writerow(header_y1)
        y2_w_fft_writer.writerow(header_y2)
        y2_w_psd_classic_writer.writerow(header_y2)
        y2_w_psd_welch_writer.writerow(header_y2)
        y2_n1_fft_writer.writerow(header_y2)
        y2_n1_psd_classic_writer.writerow(header_y2)
        y2_n1_psd_welch_writer.writerow(header_y2)
        wLimit = int(min_duration(False))
        n1Limit = int(min_duration(False))
        for patientFolder in os.listdir('Dataset_EEG/data_cooked'):
            if patientFolder.startswith("features") or patientFolder.startswith("parameters"):
                continue
            print("**************   %s  **************" % patientFolder)
            wCounter = 0
            n1Counter = 0
            for file in os.listdir('./Dataset_EEG/data_cooked/%s' % patientFolder):
                if file == 'features_W.csv' or file == 'features_1.csv':
                    continue
                score = file[21]
                if wCounter >= wLimit and n1Counter >= n1Limit:
                    break
                if wCounter >= wLimit and score == "W":
                    continue
                if n1Counter >= n1Limit and score == "1":
                    continue

                y1, y2 = np.loadtxt('./Dataset_EEG/data_cooked/%s/%s' % (patientFolder, file),
                                    unpack=True,
                                    delimiter=',',
                                    skiprows=1,
                                    usecols=range(0, 2))
                values = fft_psd_estimation(y1, y2)
                N = np.size(y1)
                fs = 100
                # Extraction from FFT
                # To get a range of frequencies (f1,f2) from a fft list, N*(f1/fs) ... N*(f2/fs)
                Y_m1 = values["y1_fft_theo"]
                Y_m2 = values["y2_fft_theo"]
                # Delta band 0-4 Hz
                deltaBand1 = sum(Y_m1[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                deltaBand2 = sum(Y_m2[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                # Theta band 4-8 Hz
                thetaBand1 = sum(Y_m1[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                thetaBand2 = sum(Y_m2[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                # Alpha band 8-13 Hz
                alphaBand1 = sum(Y_m1[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                alphaBand2 = sum(Y_m2[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                # Beta band 13-30 Hz
                betaBand1 = sum(Y_m1[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                betaBand2 = sum(Y_m2[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                # Gamma Band >30 Hz
                gammaBand1 = sum(Y_m1[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                gammaBand2 = sum(Y_m2[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                row_y1_fft = [file,
                              format(deltaBand1, '.3f'),
                              format(thetaBand1, '.3f'),
                              format(alphaBand1, '.3f'),
                              format(betaBand1, '.3f'),
                              format(gammaBand1, '.3f')]
                row_y2_fft = [file,
                              format(deltaBand2, '.3f'),
                              format(thetaBand2, '.3f'),
                              format(alphaBand2, '.3f'),
                              format(betaBand2, '.3f'),
                              format(gammaBand2, '.3f')]
                # Extraction from PSD Classic
                Y_m1 = values["y1_psd_classic"]
                Y_m2 = values["y2_psd_classic"]
                # Delta band 0-4 Hz
                deltaBand1 = sum(Y_m1[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                deltaBand2 = sum(Y_m2[int(N * (0 / fs)):int(N * (4 / fs)) + 1])
                # Theta band 4-8 Hz
                thetaBand1 = sum(Y_m1[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                thetaBand2 = sum(Y_m2[int(N * (4 / fs)):int(N * (8 / fs)) + 1])
                # Alpha band 8-13 Hz
                alphaBand1 = sum(Y_m1[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                alphaBand2 = sum(Y_m2[int(N * (8 / fs)):int(N * (13 / fs)) + 1])
                # Beta band 13-30 Hz
                betaBand1 = sum(Y_m1[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                betaBand2 = sum(Y_m2[int(N * (13 / fs)):int(N * (30 / fs)) + 1])
                # Gamma Band >30 Hz
                gammaBand1 = sum(Y_m1[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                gammaBand2 = sum(Y_m2[int(N * (30 / fs)):int(N * (50 / fs)) + 1])
                row_y1_psd_classic = [file,
                                      format(deltaBand1, '.3f'),
                                      format(thetaBand1, '.3f'),
                                      format(alphaBand1, '.3f'),
                                      format(betaBand1, '.3f'),
                                      format(gammaBand1, '.3f')]
                row_y2_psd_classic = [file,
                                      format(deltaBand2, '.3f'),
                                      format(thetaBand2, '.3f'),
                                      format(alphaBand2, '.3f'),
                                      format(betaBand2, '.3f'),
                                      format(gammaBand2, '.3f')]
                # Extraction from PSD Welch
                psd1 = values["y1_psd_welch"]
                psd2 = values["y2_psd_welch"]
                freq_psd = values["freq_psd"]
                freq_res = freq_psd[1] - freq_psd[0]  # Frequency resolution = 1 / 4 = 0.25
                # Compute the absolute power by approximating the area under the curve
                # Delta band 0.5-4 Hz
                deltaBand1 = simps(psd1[np.logical_and(freq_psd >= 0.5, freq_psd <= 4)], dx=freq_res)
                deltaBand2 = simps(psd2[np.logical_and(freq_psd >= 0.5, freq_psd <= 4)], dx=freq_res)
                # Theta band 4-8 Hz
                thetaBand1 = simps(psd1[np.logical_and(freq_psd >= 4, freq_psd <= 8)], dx=freq_res)
                thetaBand2 = simps(psd2[np.logical_and(freq_psd >= 4, freq_psd <= 8)], dx=freq_res)
                # Alpha band 8-13 Hz
                alphaBand1 = simps(psd1[np.logical_and(freq_psd >= 8, freq_psd <= 13)], dx=freq_res)
                alphaBand2 = simps(psd2[np.logical_and(freq_psd >= 8, freq_psd <= 13)], dx=freq_res)
                # Beta band 13-30 Hz
                betaBand1 = simps(psd1[np.logical_and(freq_psd >= 13, freq_psd <= 30)], dx=freq_res)
                betaBand2 = simps(psd2[np.logical_and(freq_psd >= 13, freq_psd <= 30)], dx=freq_res)
                # Gamma Band >30 Hz
                gammaBand1 = simps(psd1[np.logical_and(freq_psd >= 30, freq_psd <= 50)], dx=freq_res)
                gammaBand2 = simps(psd2[np.logical_and(freq_psd >= 30, freq_psd <= 50)], dx=freq_res)
                row_y1_psd_welch = [file,
                                    format(deltaBand1, '.3f'),
                                    format(thetaBand1, '.3f'),
                                    format(alphaBand1, '.3f'),
                                    format(betaBand1, '.3f'),
                                    format(gammaBand1, '.3f')]
                row_y2_psd_welch = [file,
                                    format(deltaBand2, '.3f'),
                                    format(thetaBand2, '.3f'),
                                    format(alphaBand2, '.3f'),
                                    format(betaBand2, '.3f'),
                                    format(gammaBand2, '.3f')]
                if score == 'W':
                    wCounter += 1
                    y1_w_fft_writer.writerow(row_y1_fft)
                    y1_w_psd_classic_writer.writerow(row_y1_psd_classic)
                    y1_w_psd_welch_writer.writerow(row_y1_psd_welch)
                    y2_w_fft_writer.writerow(row_y2_fft)
                    y2_w_psd_classic_writer.writerow(row_y2_psd_classic)
                    y2_w_psd_welch_writer.writerow(row_y2_psd_welch)
                if score == '1':
                    n1Counter += 1
                    y1_n1_fft_writer.writerow(row_y1_fft)
                    y1_n1_psd_classic_writer.writerow(row_y1_psd_classic)
                    y1_n1_psd_welch_writer.writerow(row_y1_psd_welch)
                    y2_n1_fft_writer.writerow(row_y2_fft)
                    y2_n1_psd_classic_writer.writerow(row_y2_psd_classic)
                    y2_n1_psd_welch_writer.writerow(row_y2_psd_welch)
                # if same_file == 1:
                #     with open('./data_cooked/%s/%s' % (patientFolder, file), 'r') as read_obj:
                #         csv_reader = csv.reader(read_obj)
                #         with open('./data_cooked/%s/%sM' % (patientFolder, file), 'w', newline='') as write_obj:
                #             csv_writer = csv.writer(write_obj)
                #             valuesAdded = False
                #             for row in csv_reader:
                #                 if not valuesAdded:
                #                     if np.size(row) > 2:
                #                         if row[0] == "EEG Fpz-Cz":
                #                             row[2] = "Fpz-Cz Delta"
                #                             row[3] = "Pz_Oz Delta"
                #                             row[4] = "Fpz-Cz Theta"
                #                             row[5] = "Pz_Oz Theta"
                #                             row[6] = "Fpz-Cz Alpha"
                #                             row[7] = "Pz_Oz Alpha"
                #                             row[8] = "Fpz-Cz Beta"
                #                             row[9] = "Pz_Oz Beta"
                #                             row[10] = "Fpz-Cz Gamma"
                #                             row[11] = "Pz_Oz Gamma"
                #                             row[12] = "Fpz-Cz SS"
                #                             row[13] = "Pz_Oz SS"
                #                             row[14] = "Fpz-Cz K-Complex"
                #                             row[15] = "Pz_Oz K-Complex"
                #                         else:
                #                             row[2] = deltaBand1
                #                             row[3] = deltaBand2
                #                             row[4] = thetaBand1
                #                             row[5] = thetaBand2
                #                             row[6] = alphaBand1
                #                             row[7] = alphaBand2
                #                             row[8] = betaBand1
                #                             row[9] = betaBand2
                #                             row[10] = gammaBand1
                #                             row[11] = gammaBand2
                #                             row[12] = sleepSpindlesBand1
                #                             row[13] = sleepSpindlesBand2
                #                             row[14] = kComplexBand1
                #                             row[15] = kComplexBand2
                #                             valuesAdded = True
                #                     else:
                #                         if row[0] == "EEG Fpz-Cz":
                #                             row.append("Fpz-Cz Delta")
                #                             row.append("Pz_Oz Delta")
                #                             row.append("Fpz-Cz Theta")
                #                             row.append("Pz_Oz Theta")
                #                             row.append("Fpz-Cz Alpha")
                #                             row.append("Pz_Oz Alpha")
                #                             row.append("Fpz-Cz Beta")
                #                             row.append("Pz_Oz Beta")
                #                             row.append("Fpz-Cz Gamma")
                #                             row.append("Pz_Oz Gamma")
                #                             row.append("Fpz-Cz SS")
                #                             row.append("Pz_Oz SS")
                #                             row.append("Fpz-Cz K-Complex")
                #                             row.append("Pz_Oz K-Complex")
                #                         else:
                #                             row.append(deltaBand1)
                #                             row.append(deltaBand2)
                #                             row.append(thetaBand1)
                #                             row.append(thetaBand2)
                #                             row.append(alphaBand1)
                #                             row.append(alphaBand2)
                #                             row.append(betaBand1)
                #                             row.append(betaBand2)
                #                             row.append(gammaBand1)
                #                             row.append(gammaBand2)
                #                             row.append(sleepSpindlesBand1)
                #                             row.append(sleepSpindlesBand2)
                #                             row.append(kComplexBand1)
                #                             row.append(kComplexBand2)
                #                             valuesAdded = True
                #                 csv_writer.writerow(row)
                #     os.remove('./data_cooked/%s/%s' % (patientFolder, file))
                #     os.rename('./data_cooked/%s/%sM' % (patientFolder, file), './data_cooked/%s/%s' % (patientFolder, file))


def fft_psd_estimation(y1, y2):
    # Removing Artifacts
    # for i in np.arange(len(y1)):
    #     if y1[i] > 100 or y1[i] < -100:
    #         y1[i] = 0
    #     if y2[i] > 100 or y2[i] < -100:
    #         y2[i] = 0
    # Normalization
    # Average of 2 channels
    # sum = y1 + y2
    # average = sum / 2
    # y1 = average
    # y2 = average
    # Subtract Average of 2 channels
    # sum = y1 + y2
    # average = sum / 2
    # y1 -= average
    # y2 -= average
    # Subtract Average of whole epoch
    mean_y1 = np.mean(y1)
    mean_y2 = np.mean(y2)
    y1 -= mean_y1
    y2 -= mean_y2

    Fs = 100
    n = np.size(y1)
    # FFT
    freq_fft = (Fs / 2) * np.linspace(0, 1, int(n / 2))
    # freq_fft = fftfreq(n)
    # mask = freq_fft >= 0
    y1_fft = fft(y1)
    y2_fft = fft(y2)
    y1_fft_theo = (2 / n) * np.abs(y1_fft[0:np.size(freq_fft)])
    y2_fft_theo = (2 / n) * np.abs(y2_fft[0:np.size(freq_fft)])
    values = {"freq_fft": freq_fft,
              "y1_fft_theo": y1_fft_theo,
              "y2_fft_theo": y2_fft_theo}
    # PSD
    # FFT Classic
    y1_psd_fft = 2 * (np.abs(y1_fft[0:np.size(freq_fft)] / n) ** 2)
    y2_psd_fft = 2 * (np.abs(y1_fft[0:np.size(freq_fft)] / n) ** 2)
    values["y1_psd_classic"] = y1_psd_fft
    values["y2_psd_classic"] = y2_psd_fft
    # Welch's Method
    window = 4 * Fs  # Define window length (4 seconds) -> 2 cycles/lowest freq we have -> 2/0.5 = 4
    freq_psd, y1_psd_welch = signal.welch(y1, fs=Fs, window='hamming', nperseg=window, scaling='density')
    y2_psd_welch = signal.welch(y2, fs=Fs, window='hamming', nperseg=window, scaling='density')[1]
    values["freq_psd"] = freq_psd
    values["y1_psd_welch"] = y1_psd_welch
    values["y2_psd_welch"] = y2_psd_welch
    # for i in np.arange(0,200,4):
    #     norm = np.sum(psd[i:i+5])/5
    #     psd[i:i+5] /= [norm] * 5
    return values


def revise():
    # Original Signal
    Fs = 100
    # t = np.arange(0, 60, 1/Fs)
    # y = 2 * np.sin(2*pi*5*t) + 2*np.sin(2*pi*20*t) + 6*np.sin(2*pi*35*t)
    y1, y2 = np.loadtxt('./Dataset_EEG/data_cooked/SC4001E0/SC4001E0-Sleep stage W-0.csv',
                        unpack=True,
                        delimiter=',',
                        skiprows=1,
                        usecols=range(0, 2))
    y = y1
    t = np.arange(0, len(y) / 100, 1 / Fs)
    mean_y = np.mean(y)
    std_y = np.std(y)
    var_y = std_y ** 2
    print(mean_y, std_y, var_y)
    # FFT
    n = np.size(y)
    y_fft = fft(y)
    freq_fft = (Fs / 2) * np.linspace(0, 1, int(n / 2))
    y_fft_theo = (2 / n) * np.abs(y_fft[0:np.size(freq_fft)])
    # PSD
    y_psd_fft = 2 * (np.abs(y_fft[0:np.size(freq_fft)] / n) ** 2)
    window = (2 / 0.5) * Fs
    freq_psd, y_psd_welch = signal.welch(y, fs=Fs, window='hamming', nperseg=window, scaling='density')
    freq_res = freq_psd[1] - freq_psd[0]  # = 1 / 4 = 0.25
    total_power = simps(y_psd_welch, dx=freq_res)
    fft5 = simps(y_psd_fft[int(n * (0 / Fs)):int(n * (10 / Fs)) + 1], dx=freq_res)
    power5 = simps(y_psd_welch[np.logical_and(freq_psd >= 0, freq_psd <= 10)], dx=freq_res)

    # Original Signal Plot
    plt.subplot(3, 1, 1)
    plt.title("Original Signal Plot")
    plt.ylabel("Amplitude (uV)")
    plt.xlabel("Time (sec)")
    plt.plot(t, y)
    # FFT Plot
    plt.subplot(3, 1, 2)
    plt.title("Fast Fourier Transform (FFT) Plot")
    plt.ylabel("Amplitude (uV)")
    plt.xlabel("Freq (hz)")
    plt.plot(freq_fft, y_fft_theo)
    # PSD Plot
    plt.subplot(3, 1, 3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (uV^2 / Hz)')
    # plt.title("Power Spectrum Density (PSD) Plot - FFT")
    # plt.plot(freq_fft, y_psd_fft)
    plt.title("Power Spectrum Density (PSD) Plot - Welch")
    plt.plot(freq_psd, y_psd_welch)

    plt.tight_layout()
    plt.show()


def count_files():
    min_count = 9999
    min_name = None
    for patientFolder in os.listdir('Dataset_EEG/data_cooked'):
        if patientFolder.startswith("features") or patientFolder.startswith("parameters"):
            continue
        counter_n1 = 0
        counter_w = 0
        for file in os.listdir('./Dataset_EEG/data_cooked/%s' % patientFolder):
            if file.startswith("%s-Sleep stage 1" % patientFolder):
                counter_n1 += 1
            if file.startswith("%s-Sleep stage W" % patientFolder):
                counter_w += 1
        if counter_n1 < min_count:
            min_count = counter_n1
            min_name = patientFolder
        if counter_w < min_count:
            min_count = counter_w
            min_name = patientFolder
        print(patientFolder, counter_w, counter_n1)
    print(min_name, min_count)


def checkAmps():
    for patientFolder in os.listdir('Dataset_EEG/data_cooked'):
        min_y1 = 9999
        min_y2 = 9999
        max_y1 = -1
        max_y2 = -1
        if patientFolder.startswith("features") or patientFolder.startswith("parameters"):
            continue
        print("**************   %s  **************" % patientFolder)
        for file in os.listdir('./Dataset_EEG/data_cooked/%s' % patientFolder):
            y1, y2 = np.loadtxt('./Dataset_EEG/data_cooked/%s/%s' % (patientFolder, file),
                                unpack=True,
                                delimiter=',',
                                skiprows=1,
                                usecols=range(0, 2))
            for i in np.arange(len(y1)):
                if y1[i] > 100 or y1[i] < -100:
                    y1[i] = 0
                if y2[i] > 100 or y2[i] < -100:
                    y2[i] = 0
            if np.max(y1) > max_y1:
                max_y1 = np.max(y1)
            if np.max(y2) > max_y2:
                max_y2 = np.max(y2)
            if np.min(y1) < min_y1:
                min_y1 = np.min(y1)
            if np.min(y2) < min_y2:
                min_y2 = np.min(y2)

        print(patientFolder)
        print(min_y1, max_y1)
        print(min_y2, max_y2)


# revise()
# min_duration(True)
# edf2csv()
# feature_extraction()
# plot('SC4021E0', 'SC4021E0-Sleep stage W-0002', 0, 0.25, 0)
# plot_bands(0, 0, 0, 'alpha', 'beta', True, True)
# count_files()
# checkAmps()
