import numpy as np
import matplotlib.pyplot as plt
from bachelor.EEG.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from bachelor.EEG.main import min_duration

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
# %load_ext autoreload
# %autoreload 2

total_individuals = 75
epoch = 1


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros(shape=(layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                             activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from AL and Y.
    # logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))
    # cost = -1 / m * np.sum(logprobs)
    # cost = - np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))/m
    cost = (-1. / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation="sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (
            np.multiply(learning_rate, grads["dW" + str(l + 1)]))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (
            np.multiply(learning_rate, grads["db" + str(l + 1)]))
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []  # keep track of cost

    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)

    counter = 0
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            if len(costs) > 0 and costs[-1] == cost:
                counter += 1
            costs.append(cost)
            if counter == 5:
                break

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, Y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(100 * np.sum((p == Y) / m)))

    return p, np.sum((p == Y) / m)


def load_data(method, n, test_size, test_sepIndividuals, epoch_length=1., duration=570):
    epochs_per_subject = int(duration / epoch_length)
    # Cassette Only
    if n == 0:
        max_rows_n1 = total_individuals * 2 * (min_duration(False) / epoch)  # Last SC row number - 1
        max_rows_w = total_individuals * 2 * (min_duration(False) / epoch)
        skip_rows_n1 = 1
        skip_rows_w = 1
    # Telemetry Only
    elif n == 1:
        max_rows_n1 = None
        max_rows_w = None
        skip_rows_n1 = (total_individuals * 2 * (min_duration(False) / epoch)) + 1  # Last SC row number
        skip_rows_w = (total_individuals * 2 * (min_duration(False) / epoch)) + 1
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
        w_data = np.loadtxt(
            './Dataset_EEG/data_cooked/features/W_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 11),
            max_rows=max_rows_w)
        n1_data = np.loadtxt(
            './Dataset_EEG/data_cooked/features/N1_fft.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 11),
            max_rows=max_rows_n1)
    # Classic
    elif method == 1:
        w_data = np.loadtxt(
            './Dataset_EEG/data_cooked/features/W_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 11),
            max_rows=max_rows_w)
        n1_data = np.loadtxt(
            './Dataset_EEG/data_cooked/features/N1_psd_classic.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 11),
            max_rows=max_rows_n1)
    # PSD Welch
    elif method == 2:
        w_data = np.loadtxt(
            './Dataset_EEG/data_cooked/features/W_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_w,
            usecols=range(1, 11),
            max_rows=max_rows_w)
        n1_data = np.loadtxt(
            './Dataset_EEG/data_cooked/features/N1_psd_welch.csv',
            unpack=True,
            delimiter=',',
            skiprows=skip_rows_n1,
            usecols=range(1, 11),
            max_rows=max_rows_n1)

    if test_sepIndividuals:
        test_individuals = int(test_size * total_individuals)  # 22
        train_individuals = total_individuals - test_individuals  # 53
        print("Train Subjects Count", train_individuals)
        print("Test Subjects Count", test_individuals)
        split_index = (
                    train_individuals * 2 * epochs_per_subject)  # Target SC row number 30% test or train_ind * file count per ind * 2 for 2 nights +1 to compenstate header
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
        # train_x /= std
        test_x -= mean
        # test_x /= std

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
    return train_x, test_x, train_y, test_y


def save_obj(obj, name, path):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def model_train(model):
    # Load Data
    method = 0
    n = 2
    test_size = 0.20
    test_sepIndividuals = True
    epoch_length = 0.25
    duration = 60
    train_x, test_x, train_y, test_y = load_data(method, n, test_size, test_sepIndividuals, epoch_length, duration)
    print(train_x.shape, test_x.shape)
    # NN Model
    if model == "NN" or model == "all":
        # layers_dims = [10, 20, 10, 1]  # 2-layer model 74.9% 0.50 avg 2 channels 5 10 1
        layers_dims = [10, 20, 1]
        parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.01, num_iterations=4000,
                                   print_cost=True)
        pred_train, acc_train = predict(train_x, train_y, parameters)
        pred_test, acc_test = predict(test_x, test_y, parameters)
        # save_obj(parameters, "testSize_%s_data_%s%s%s_layers_%s_train_%.2f_test_%.2f" % (test_size, method, channel, n, str(layers_dims), acc_train * 100, acc_test * 100), './Dataset_EEG/data_cooked/parameters/')
    # Logistic regression
    if model == "LR" or model == "all":
        clf = sklearn.linear_model.LogisticRegressionCV()
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        LR_predictions_train = clf.predict(train_x.T)
        LR_predictions_test = clf.predict(test_x.T)
        print(LR_predictions_train)
        print('Accuracy of LR: %s ' % str(np.sum(
            (np.array(LR_predictions_train).reshape(1, len(LR_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of LR: %s ' % str(
            np.sum((np.array(LR_predictions_test).reshape(1, len(LR_predictions_test)) == test_y) / test_x.shape[1])))
    # SVM
    if model == "SVM" or model == "all":
        clf = svm.SVC()
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        SVM_predictions_train = clf.predict(train_x.T)
        SVM_predictions_test = clf.predict(test_x.T)
        print(SVM_predictions_train)
        print('Accuracy of SVM: %s ' % str(np.sum(
            (np.array(SVM_predictions_train).reshape(1, len(SVM_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of SVM: %s ' % str(
            np.sum((np.array(SVM_predictions_test).reshape(1, len(SVM_predictions_test)) == test_y) / test_x.shape[1])))
    # SGD
    if model == "SGD" or model == "all":
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5000)
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        SGD_predictions_train = clf.predict(train_x.T)
        SGD_predictions_test = clf.predict(test_x.T)
        print(SGD_predictions_train)
        print('Accuracy of SGD: %s ' % str(np.sum(
            (np.array(SGD_predictions_train).reshape(1, len(SGD_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of SGD: %s ' % str(
            np.sum((np.array(SGD_predictions_test).reshape(1, len(SGD_predictions_test)) == test_y) / test_x.shape[1])))
    # DT
    if model == "DT" or model == "all":
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        DT_predictions_train = clf.predict(train_x.T)
        DT_predictions_test = clf.predict(test_x.T)
        print(DT_predictions_train)
        print('Accuracy of DT: %s ' % str(np.sum(
            (np.array(DT_predictions_train).reshape(1, len(DT_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of DT: %s ' % str(
            np.sum((np.array(DT_predictions_test).reshape(1, len(DT_predictions_test)) == test_y) / test_x.shape[1])))
    # RF
    if model == "RF" or model == "all":
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        RF_predictions_train = clf.predict(train_x.T)
        RF_predictions_test = clf.predict(test_x.T)
        print(RF_predictions_train)
        print('Accuracy of RF: %s ' % str(np.sum(
            (np.array(RF_predictions_train).reshape(1, len(RF_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of RF: %s ' % str(
            np.sum((np.array(RF_predictions_test).reshape(1, len(RF_predictions_test)) == test_y) / test_x.shape[1])))
    # NN_scikit
    if model == "NN_scikit" or model == "all":
        clf = MLPClassifier(solver='lbfgs', alpha=0.0075, hidden_layer_sizes=(100, 50), random_state=1, max_iter=200)
        clf.fit(train_x.T, np.array(train_y[0]).tolist())
        # Print accuracy
        NN_predictions_train = clf.predict(train_x.T)
        NN_predictions_test = clf.predict(test_x.T)
        print(NN_predictions_train)
        print('Accuracy of NN: %s ' % str(np.sum(
            (np.array(NN_predictions_train).reshape(1, len(NN_predictions_train)) == train_y) / train_x.shape[1])))
        print('Accuracy of NN: %s ' % str(
            np.sum((np.array(NN_predictions_test).reshape(1, len(NN_predictions_test)) == test_y) / test_x.shape[1])))


def manual_test(parameters_obj):
    parameters = load_obj(parameters_obj)
    x = np.asarray(
        [[138, 466],
         [90, 152],
         [52, 71],
         [75, 181],
         [42, 205]])
    y = np.asarray([[1, 0]])
    pred_m, acc_m = predict(x, y, parameters)
    print(pred_m, acc_m)


# model_train("SVM")
