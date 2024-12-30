import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Model
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, concatenate
from keras.preprocessing import image
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from keras_tqdm import TQDMNotebookCallback
import keras.backend as k
# from Generator import DriveDataGenerator
from PIL import ImageDraw
import matplotlib
import h5py
import math
import os


print(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())
print(device_lib.list_local_devices())
