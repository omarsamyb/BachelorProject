from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras_tqdm import TQDMNotebookCallback
import json
import os
import numpy as np
import pandas as pd
import h5py
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
from bachelor.Simulator_E2EDL import Generator, Cooking


def draw_image_with_label(imgArr, label, y_labels, prediction=None, flipped=None, brighten_value=None):
    if flipped is not None:
        print('Is Horizontally flipped: {0}'.format(flipped))
    if brighten_value is not None:
        print('Brighten Value: {0}'.format(brighten_value))
    if 'Throttle' in y_labels:
        print('Actual Throttle = {0}'.format(label[y_labels.index('Throttle')]))
        if prediction is not None:
            print('Predicted Throttle = {0}'.format(prediction[y_labels.index('Throttle')]))
            print('Throttle L1 Error: {0}'.format(
                abs(prediction[y_labels.index('Throttle')] - label[y_labels.index('Throttle')])))
    if 'Brake' in y_labels:
        print('Actual Brake = {0}'.format(label[y_labels.index('Brake')]))
        if prediction is not None:
            print('Predicted Brake = {0}'.format(prediction[y_labels.index('Brake')]))
            print(
                'Brake L1 Error: {0}'.format(abs(prediction[y_labels.index('Brake')] - label[y_labels.index('Brake')])))
    if 'Steering' in y_labels:
        print('Actual Steering Angle = {0}'.format(label[y_labels.index('Steering')]))
        # Steering range for the car is +- 40 degrees -> radians = 40*pi/180 = 0.69 radians
        theta = label[y_labels.index('Steering')] * 0.69
        line_length = 50
        line_thickness = 3
        label_line_color = (255, 0, 0)
        prediction_line_color = (0, 0, 255)
        pil_image = image.array_to_img(imgArr, K.image_data_format(), scale=True)
        draw_image = pil_image.copy()
        image_draw = ImageDraw.Draw(draw_image)
        first_point = (int(imgArr.shape[1] / 2), imgArr.shape[0])
        second_point = (int((imgArr.shape[1] / 2) + (line_length * math.sin(theta))),
                        int(imgArr.shape[0] - (line_length * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=label_line_color, width=line_thickness)
        if prediction is not None:
            print('Predicted Steering Angle = {0}'.format(prediction[y_labels.index('Steering')]))
            print('Steering L1 Error: {0}'.format(
                abs(prediction[y_labels.index('Steering')] - label[y_labels.index('Steering')])))
            theta = prediction[y_labels.index('Steering')] * 0.69
            second_point = (int((imgArr.shape[1] / 2) + (line_length * math.sin(theta))),
                            int(imgArr.shape[0] - (line_length * math.cos(theta))))
            image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)
        del image_draw
        plt.imshow(draw_image)
        plt.show()


# Data & Model directories
COOKED_DATA_DIR = 'Dataset-E2EDL/data_cooked'
MODEL_OUTPUT_DIR = 'model'

# Read Datasets
batch_size = 32
train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5').replace("\\", "/"), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5').replace("\\", "/"), 'r')
test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5').replace("\\", "/"), 'r')
num_train_examples = train_dataset['img_0_0'].shape[0]
num_eval_examples = eval_dataset['img_0_0'].shape[0]
num_test_examples = test_dataset['img_0_0'].shape[0]
print(num_train_examples, num_eval_examples, num_test_examples)

# Modify images to cut useful areas, drop 90% of 0 valued labels, randomly flip vertically & adjust brightness
data_generator = Generator.DriveDataGenerator(rescale=1. / 255., horizontal_flip=True, brightness_range=[0.6, 1.4])
train_generator = data_generator.flow \
    (x_img_0_0=train_dataset['img_0_0'],
     x_img_0_3=train_dataset['img_0_3'],
     x_img_0_5=train_dataset['img_0_5'],
     # x_img_4_3=train_dataset['img_4_3'],
     x_previous_state=train_dataset['previous_state'], y=train_dataset['label'],
     x_previous_state_labels=['Steering', 'Throttle', 'Brake', 'Speed'], y_labels=['Steering', 'Throttle', 'Brake'],
     batch_size=batch_size, zero_drop_percentage_steer=0.95, roi=[76, 135, 0, 255], shuffle=True)
eval_generator = data_generator.flow \
    (x_img_0_0=eval_dataset['img_0_0'],
     x_img_0_3=eval_dataset['img_0_3'],
     x_img_0_5=eval_dataset['img_0_5'],
     # x_img_4_3=eval_dataset['img_4_3'],
     x_previous_state=eval_dataset['previous_state'], y=eval_dataset['label'],
     x_previous_state_labels=['Steering', 'Throttle', 'Brake', 'Speed'], y_labels=['Steering', 'Throttle', 'Brake'],
     batch_size=batch_size, zero_drop_percentage_steer=0.95, roi=[76, 135, 0, 255], shuffle=True)
test_generator = data_generator.flow \
    (x_img_0_0=test_dataset['img_0_0'],
     x_img_0_3=test_dataset['img_0_3'],
     x_img_0_5=test_dataset['img_0_5'],
     # x_img_4_3=test_dataset['img_4_3'],
     x_previous_state=test_dataset['previous_state'], y=test_dataset['label'],
     x_previous_state_labels=['Steering', 'Throttle', 'Brake', 'Speed'], y_labels=['Steering', 'Throttle', 'Brake'],
     batch_size=batch_size, zero_drop_percentage_steer=0.95, roi=[76, 135, 0, 255], shuffle=True)

# # Explore a sample batch
[sample_batch_x_data, sample_batch_y_data] = next(train_generator)
# for i in range(0, 3, 1):
#     draw_image_with_label(imgArr=sample_batch_x_data[0][i], label=sample_batch_y_data[i],
#                           y_labels=['Steering', 'Throttle', 'Brake'],
#                           flipped=is_horizontally_flipped[i], brighten_value=brighten_values[i])

# Model Architecture
activation = 'relu'

# Create the convolutional stacks
# front_center_scene
png_image_input_0 = Input(shape=sample_batch_x_data[0].shape[1:])
png_stack_0 = Conv2D(16, (3, 3), activation=activation, padding='same', name='convolution0')(png_image_input_0)
png_stack_0 = MaxPooling2D(pool_size=(2, 2))(png_stack_0)
png_stack_0 = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(png_stack_0)
png_stack_0 = MaxPooling2D(pool_size=(2, 2))(png_stack_0)
png_stack_0 = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(png_stack_0)
png_stack_0 = MaxPooling2D(pool_size=(2, 2))(png_stack_0)
png_stack_0 = Flatten()(png_stack_0)
png_stack_0 = Dropout(0.2)(png_stack_0)
# # front_center_segmentation
# png_image_input_1 = Input(shape=sample_batch_x_data[2].shape[1:])
# png_stack_1 = Conv2D(16, (3, 3), activation=activation, padding='same', name='convolution3')(png_image_input_1)
# png_stack_1 = MaxPooling2D(pool_size=(2, 2))(png_stack_1)
# png_stack_1 = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution4')(png_stack_1)
# png_stack_1 = MaxPooling2D(pool_size=(2, 2))(png_stack_1)
# png_stack_1 = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution5')(png_stack_1)
# png_stack_1 = MaxPooling2D(pool_size=(2, 2))(png_stack_1)
# png_stack_1 = Flatten()(png_stack_1)
# png_stack_1 = Dropout(0.2)(png_stack_1)
# # Inject pfm pic input & state input
# front_center_DepthVis
# pfm_image_input_0 = Input(shape=sample_batch_x_data[1].shape[1:])
# pfm_stack_0 = Flatten()(pfm_image_input_0)
# pfm_stack_0 = Dropout(0.2)(pfm_stack_0)
# # back_center_DepthVis
# # pfm_pic_input_1 = Input(shape=pfm_image_input_shape)
# # pfm_stack_1 = Flatten()(pfm_pic_input_1)
# # pfm_stack_1 = Dropout(0.2)(pfm_stack_1)
# previous_state
state_input = Input(shape=sample_batch_x_data[3].shape[1:])
# # # merged = concatenate([png_stack_0, png_stack_1, pfm_stack_0, pfm_stack_1, state_input])
# merged = concatenate([png_stack_0, png_stack_1, pfm_stack_0, state_input])
merged = concatenate([png_stack_0, state_input])
# Add a few dense layers to finish the model
merged = Dense(64, activation=activation, name='dense0')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(10, activation=activation, name='dense1')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(3, name='output')(merged)

adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model = Model(inputs=[png_image_input_0, pfm_image_input_0, png_image_input_1, state_input], outputs=merged)
model = Model(inputs=[png_image_input_0, state_input], outputs=merged)
model.compile(optimizer=adam, loss='mse')

# # Model Summary
# model.summary()

# # Callbacks
# plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
# checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'models',
#                                    '{0}_model.{1}-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))
# Cooking.checkAndCreateDir(checkpoint_filepath)
# checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)
# csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
# callbacks = [plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback, TQDMNotebookCallback()]
#
# # Train
# history = model.fit_generator(train_generator, steps_per_epoch=num_train_examples // batch_size, epochs=500,
#                               callbacks=callbacks, validation_data=eval_generator,
#                               validation_steps=num_eval_examples // batch_size,
#                               verbose=2)
#
# Sanity check
[sample_batch_x_data, sample_batch_y_data] = next(test_generator)
STEER_THROTTLE_BRAKE_MODEL_PATH = 'model/models/steer_throttle_brake_00_model.17-0.0192344.h5'
steer_throttle_brake_model = load_model(STEER_THROTTLE_BRAKE_MODEL_PATH)

# predictions = model.predict([sample_batch_x_data[0], sample_batch_x_data[1], sample_batch_x_data[2], sample_batch_x_data[3]])
predictions = steer_throttle_brake_model.predict([sample_batch_x_data[0], sample_batch_x_data[3]])
for i in range(0, 3, 1):
    draw_image_with_label(imgArr=sample_batch_x_data[0][i], label=sample_batch_y_data[i], prediction=predictions[i],
                          y_labels=['Steering', 'Throttle', 'Brake'])

[sample_batch_x_data, sample_batch_y_data] = next(test_generator)
predictions = model.predict([sample_batch_x_data[0], sample_batch_x_data[1]])
for i in range(0, 3, 1):
    draw_image_with_label(imgArr=sample_batch_x_data[0][i], label=sample_batch_y_data[i], prediction=predictions[i],
                          y_labels=['Steering', 'Throttle', 'Brake'])
