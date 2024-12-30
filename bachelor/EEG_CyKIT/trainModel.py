import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from bachelor.EEG.model_tensorflow import model, predict
from bachelor.EEG.model import save_obj, load_obj
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

total_individuals = 2


def load_data(method, test_size, test_sepIndividuals, epoch_length=0.25, duration=300):
    epochs_per_subject = int(duration / epoch_length)

    # FFT Values
    if method == 0:
        w_data = np.loadtxt(
            './Dataset/data_cooked/features/W_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=1,
            usecols=range(1, 71))
        n1_data = np.loadtxt(
            './Dataset/data_cooked/features/N1_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=1,
            usecols=range(1, 71))
    # Classic
    elif method == 1:
        w_data = np.loadtxt(
            './Dataset/data_cooked/features/W_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=1,
            usecols=range(1, 71))
        n1_data = np.loadtxt(
            './Dataset/data_cooked/features/N1_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=1,
            usecols=range(1, 71))
    # PSD Welch
    elif method == 2:
        w_data = np.loadtxt(
            './Dataset/data_cooked/features/W_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=1,
            usecols=range(1, 71))
        n1_data = np.loadtxt(
            './Dataset/data_cooked/features/N1_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=1,
            usecols=range(1, 71))

    if test_sepIndividuals:
        test_individuals = int(test_size * total_individuals)  # 22
        train_individuals = total_individuals - test_individuals  # 53
        print("Train Subjects Count", train_individuals)
        print("Test Subjects Count", test_individuals)
        split_index = train_individuals * epochs_per_subject
        w_split = np.split(w_data, indices_or_sections=[int(split_index)], axis=1)  # [0] = train, [1] = test
        n1_split = np.split(n1_data, indices_or_sections=[int(split_index)], axis=1)
        y_w_train = np.zeros((1, w_split[0].shape[1]))
        y_w_test = np.zeros((1, w_split[1].shape[1]))
        y_n1_train = np.ones((1, n1_split[0].shape[1]))
        y_n1_test = np.ones((1, n1_split[1].shape[1]))
        train_x_orig = np.concatenate((w_split[0], n1_split[0]), axis=1)
        test_x_orig = np.concatenate((w_split[1], n1_split[1]), axis=1)
        train_y_orig = np.concatenate((y_w_train, y_n1_train), axis=1)
        test_y_orig = np.concatenate((y_w_test, y_n1_test), axis=1)
        indices_train = np.arange(train_x_orig.shape[1])
        indices_test = np.arange(test_x_orig.shape[1])
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_test)
        train_x = train_x_orig.T[indices_train]
        train_x = train_x.T
        train_y = train_y_orig.T[indices_train]
        train_y = train_y.T
        test_x = test_x_orig.T[indices_test]
        test_x = test_x.T
        test_y = test_y_orig.T[indices_test]
        test_y = test_y.T
        mean = np.mean(train_x, axis=1, keepdims=True)
        std = np.std(train_x, axis=1, keepdims=True)
        train_x -= mean
        train_x /= std
        test_x -= mean
        test_x /= std

    else:
        y_w = np.zeros((1, w_data.shape[1]))
        y_n1 = np.ones((1, n1_data.shape[1]))
        data_x = np.concatenate((w_data, n1_data), axis=1)
        data_y = np.concatenate((y_w, y_n1), axis=1)

        indices = np.arange(data_x.shape[1])
        np.random.shuffle(indices)
        data_x_shuffled = data_x.T[indices]
        data_y_shuffled = data_y.T[indices]
        train_x, test_x, train_y, test_y = train_test_split(data_x_shuffled, data_y_shuffled, test_size=test_size,
                                                            random_state=42)
        train_x = train_x.T
        test_x = test_x.T
        train_y = train_y.T
        test_y = test_y.T
        mean = np.mean(train_x, axis=1, keepdims=True)
        std = np.std(train_x, axis=1, keepdims=True)
        train_x -= mean
        train_x /= std
        test_x -= mean
        test_x /= std
    return train_x, test_x, train_y, test_y, mean, std


def model_train(classifier='NN'):
    # Load Data
    method = 0
    test_size = 0.50
    test_sepIndividuals = True
    epoch_length = 2.
    duration = 90
    train_x, test_x, train_y, test_y, mean, std = load_data(method, test_size, test_sepIndividuals, epoch_length, duration)
    # tempx = train_x
    # tempy = train_y
    # train_x = test_x
    # train_y = test_y
    # test_x = tempx
    # test_y = tempy
    #  5   10  15  20   25  30  35  40  45  50  55   60  65     70
    # AF3, F7, F3, FC5, T7, P7, 01, 02, P8, T8, FC6, F4, F8 and AF4
    # F7 F8 F3 F4 FC5 FC6 P8  +2 -1
    train_x = np.concatenate((train_x[7:9, :], train_x[62:64, :],
                              train_x[12:14, :], train_x[57:59, :],
                              train_x[17:19, :], train_x[52:54, :],
                              train_x[42:44, :]), axis=0)
    test_x = np.concatenate((test_x[7:9, :], test_x[62:64, :],
                              test_x[12:14, :], test_x[57:59, :],
                              test_x[17:19, :], test_x[52:54, :],
                              test_x[42:44, :]), axis=0)
    # train_x = train_x[67:69, :]
    # test_x = test_x[67:69, :]
    # mean = mean[50:65]
    # std = std[50:65]
    # alpha beta only
    # train_x = np.concatenate((train_x[7:9, :], train_x[42:44, :], train_x[52:54, :], train_x[57:59, :], train_x[62:64, :], train_x[67:69, :]), axis=0)
    # test_x = np.concatenate((test_x[7:9, :], test_x[42:44, :], test_x[52:54, :], test_x[57:59, :], test_x[62:64, :], test_x[67:69, :]), axis=0)

    # train_x = train_x[65:70, :]
    # test_x = test_x[65:70, :]
    print(train_x.shape, test_x.shape)
    # NN Model
    if classifier == "NN" or classifier == "all":
        layers_dims = [14, 2000, 1]
        # layers_dims = [30, 3, 1]
        parameters, train_acc, test_acc = model(layers_dims, train_x, train_y, test_x, test_y, minibatch_size=32,
                                                num_epochs=60, learning_rate=0.001)
        save_obj(parameters, "NN_train_%.2f_test_%.2f" % (train_acc * 100, test_acc * 100), './Dataset/data_cooked/parameters/')
        save_obj(mean, "NN_train_%.2f_test_%.2f_MEAN" % (train_acc * 100, test_acc * 100), './Dataset/data_cooked/parameters/')
        save_obj(std, "NN_train_%.2f_test_%.2f_STD" % (train_acc * 100, test_acc * 100),
                 './Dataset/data_cooked/parameters/')

        # predictions = predict(train_x, parameters, None)
        # print(predictions)
        # print(train_y)
    # RF
    if classifier == "RF" or classifier == "all":
        clf = RandomForestClassifier(n_estimators=200)
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        RF_predictions_train = clf.predict(train_x.T)
        RF_predictions_test = clf.predict(test_x.T)
        print('Accuracy of RF: %s ' % str(np.sum(
            (np.array(RF_predictions_train).reshape(1, len(RF_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of RF: %s ' % str(
            np.sum((np.array(RF_predictions_test).reshape(1, len(RF_predictions_test)) == test_y) / test_x.shape[1])))
        train_acc = clf.score(train_x.T, train_y.T)
        test_acc = clf.score(test_x.T, test_y.T)
        print(train_acc)
        print(test_acc)
    # SVM
    if classifier == "SVM" or classifier == "all":
        # sc = StandardScaler()
        # train_x = sc.fit_transform(train_x.T)
        # test_x = sc.transform(test_x.T)
        # train_x = train_x.T
        # test_x = test_x.T
        # ica = FastICA(n_components=20)
        # train_x = ica.fit_transform(train_x.T)
        # test_x = ica.transform(test_x.T)
        # train_x = train_x.T
        # test_x = test_x.T
        # pca = PCA(n_components=16)
        # train_x = pca.fit_transform(train_x.T)
        # test_x = pca.transform(test_x.T)
        # train_x = train_x.T
        # test_x = test_x.T
        clf = svm.SVC()
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        SVM_predictions_train = clf.predict(train_x.T)
        SVM_predictions_test = clf.predict(test_x.T)
        print('Accuracy of SVM: %s ' % str(np.sum(
            (np.array(SVM_predictions_train).reshape(1, len(SVM_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of SVM: %s ' % str(
            np.sum((np.array(SVM_predictions_test).reshape(1, len(SVM_predictions_test)) == test_y) / test_x.shape[1])))
        train_acc = clf.score(train_x.T, train_y.T)
        test_acc = clf.score(test_x.T, test_y.T)
        print(train_acc)
        print(test_acc)
        # save_obj(clf, "subjects_%s_SVM_testAcc_%.2f" % ('Omar0Omar1', test_acc * 100),
        #          './Dataset/data_cooked/parameters/')
        # save_obj(mean, "subjects_%s_SVM_testAcc_%.2f_mean" % ('Omar0Omar1', test_acc * 100),
        #          './Dataset/data_cooked/parameters/')
        # save_obj(std, "subjects_%s_SVM_testAcc_%.2f_std" % ('Omar0Omar1', test_acc * 100),
        #          './Dataset/data_cooked/parameters/')


def model_test():
    model_parameters = load_obj('testSize_OmarSamy_layers_[70, 20, 10, 1]_train_99.75',
                                './Dataset/data_cooked/parameters/')
    model_mean = load_obj('testSize_OmarSamy_layers_[70, 20, 10, 1]_train_99.75_MEAN',
                          './Dataset/data_cooked/parameters/')
    # Load Data
    method = 0
    test_size = 0.20
    test_sepIndividuals = True
    epoch_length = 0.25
    duration = 300
    train_x, test_x, train_y, test_y, mean = load_data(method, test_size, test_sepIndividuals, epoch_length, duration)
    print(train_x.shape, test_x.shape)
    predictions = predict(train_x, model_parameters)
    print("Predictions")
    print(predictions[0, 10:20])
    print("True Values")
    print(train_y[0, 10:20])

    # x = np.asarray(
    #     [[138, 466],
    #      [90, 152],
    #      [52, 71],
    #      [75, 181],
    #      [42, 205]])
    # y = np.asarray([[1, 0]])
    # pred_m, acc_m = predict(x, y, parameters)
    # print(pred_m, acc_m)


model_train('NN')
# model_test()
