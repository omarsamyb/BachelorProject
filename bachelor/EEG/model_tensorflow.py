import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
from bachelor.EEG.model import load_data, save_obj

tf.disable_v2_behavior()


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Description:
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector, of shape (n_y, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of examples
    mini_batches = []
    # np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_x, n_y):
    """
    Description:
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of input features
    n_y -- scalar, number of classes (Ex: from 0 to 5, so -> 6 classes)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None], name="Y")

    return X, Y


def initialize_parameters(layer_dims):
    """
    Description:
    Initializes parameters to build a neural network with tensorflow

    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters tensors "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)

    Tips:
    tf.get_variable() creates a new variable that can be shared and referenced while tf.Variable can't be shared.
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network + 1

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(name='W' + str(l),
                                                   shape=[layer_dims[l], layer_dims[l - 1]],
                                                   initializer=tf.initializers.glorot_uniform())
        parameters['b' + str(l)] = tf.get_variable(name='b' + str(l),
                                                   shape=[layer_dims[l], 1],
                                                   initializer=tf.zeros_initializer())

    return parameters


def forward_propagation(X, parameters):
    """
    Description:
    Implements the forward propagation for the model

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", .., "WL", "bL"

    Returns:
    ZL -- the output of the last LINEAR unit
    """

    L = len(parameters) // 2  # number of layers in the neural network
    A = X

    # Implement [LINEAR -> RELU]*(L-1).
    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters["W" + str(l)], A_prev), parameters["b" + str(l)])
        A = tf.nn.relu(Z)
    # Implement last LINEAR -> -
    ZL = tf.add(tf.matmul(parameters["W" + str(L)], A), parameters["b" + str(L)])

    return ZL


def compute_cost(ZL, Y):
    """
    Description:
    Computes the cost

    Arguments:
    ZL -- output of forward propagation (output of the last LINEAR unit), of shape (n_y, number of examples)
    Y -- "true" labels vector placeholder, same shape as ZL

    Returns:
    cost - Tensor of the cost function

    Tips:
    -"logits" and "labels" inputs are expected to be of shape (number of examples, num_classes) so we need to Transpose.
    -tf.reduce_mean basically does the summation over the examples.
    -What we've been calling "ZL" and "Y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation.
    """

    # to fit the tensorflow requirement we need to Transpose.
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(layer_dims, X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Description:
    Implements a L-layer tensorflow neural network.

    Arguments:
    X_train -- training set, of shape (input size = n_x, number of training examples = m_train)
    Y_train -- training set labels, of shape (output size = n_y, number of training examples = m_train)
    X_test -- test set, of shape (input size = n_x, number of training examples = m_test)
    Y_test -- test set labels, of shape (output size = 6, number of test examples = m_test)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    # tf.set_random_seed(1)   # to keep consistent results
    # seed = 3
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ZL = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(ZL, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            # seed = seed + 1
            if minibatch_size == -1:
                minibatches = random_mini_batches(X_train, Y_train, m, 0)
            else:
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, 0)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost is True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        # Softmax
        # correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Sigmoid
        AL = tf.nn.sigmoid(ZL)
        prediction = tf.round(AL)
        correct_prediction = tf.equal(prediction, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_acc = accuracy.eval({X: X_train, Y: Y_train})
        test_acc = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_acc)
        print("Test Accuracy:", test_acc)

        return parameters, train_acc, test_acc


def predict(X_input, parameters, Y_input=None):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    (n_x, m) = X_input.shape
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, m], name="X")
    if Y_input is not None:
        n_y = Y_input.shape[0]
        Y = tf.placeholder(dtype=tf.float32, shape=[n_y, m], name="Y")
    ZL = forward_propagation(X, parameters)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        AL = tf.nn.sigmoid(ZL)
        prediction = tf.round(AL)
        if Y_input is not None:
            correct_prediction = tf.equal(prediction, Y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            pred_acc = accuracy.eval({X: X_input, Y: Y_input})
            print("Prediction Accuracy:", pred_acc)
        pred_values = prediction.eval({X: X_input})
        return pred_values


def model_train(model_id):
    # Load Data
    method = 0
    n = 2
    test_size = 0.20
    test_sepIndividuals = True
    epoch_length = 0.25
    duration = 570
    train_x, test_x, train_y, test_y = load_data(method, n, test_size, test_sepIndividuals, epoch_length, duration)
    print(train_x.shape, test_x.shape)
    # NN Model
    if model_id == "NN" or model_id == "all":
        layers_dims = [10, 20, 10, 1]
        parameters, train_acc, test_acc = model(layers_dims, train_x, train_y, test_x, test_y, minibatch_size=64,
                                                num_epochs=60, learning_rate=0.001)
        save_obj(parameters, "testSize_%s_layers_%s_train_%.2f_test_%.2f" % (test_size, str(layers_dims), train_acc * 100, test_acc * 100))

# model_train("NN")
