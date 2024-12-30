import os
import sys
import time
import queue
import signal
import bachelor.EEG_CyKIT.CyKit.Py3.cyPyWinUSB as hid
from bachelor.EEG_CyKIT.CyKit.Py3.cyCrypto.Cipher import AES
from bachelor.EEG_CyKIT.CyKit.Py3.cyCrypto import Random
from bachelor.EEG.model_tensorflow import predict
from bachelor.EEG.model import load_obj
from bachelor.EEG_CyKIT.preProcessing import feature_extraction
import tkinter as tk

from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import airsim

tasks = []
Fs = 256
epoch_length = 2.
duration = 2
number_of_epochs = duration / epoch_length
required_samples = duration * Fs
model_parameters = load_obj('NN_train_100.00_test_90.00', './Dataset/data_cooked/parameters/')
model_mean = load_obj('NN_train_100.00_test_90.00_MEAN', './Dataset/data_cooked/parameters/')
model_std = load_obj('NN_train_100.00_test_90.00_STD', './Dataset/data_cooked/parameters/')

STEER_THROTTLE_BRAKE_MODEL_PATH = './Dataset/data_cooked/parameters/steer_throttle_brake_00_model.17-0.0192344.h5'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


class EEG(object):

    def __init__(self):
        self.hid = None
        self.delimiter = ", "

        devicesUsed = 0

        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devicesUsed += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                device.set_raw_data_handler(self.dataHandler)
        if devicesUsed == 0:
            os._exit(0)
        sn = self.serial_number

        # EPOC+ in 16-bit Mode.
        k = ['\0'] * 16
        k = [sn[-1], sn[-2], sn[-2], sn[-3], sn[-3], sn[-3], sn[-2], sn[-4], sn[-1], sn[-4], sn[-2], sn[-2], sn[-4],
             sn[-4], sn[-2], sn[-1]]

        # EPOC+ in 14-bit Mode.
        # k = [sn[-1],00,sn[-2],21,sn[-3],00,sn[-4],12,sn[-3],00,sn[-2],68,sn[-1],00,sn[-2],88]

        self.key = str(''.join(k))
        self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)

    def dataHandler(self, data):
        try:
            tasks.append((data[1:], time.time_ns()))
        except Exception as exception2:
            print(str(exception2))

    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (
                ((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) - 128) * 32.82051289))
        return edk_value

    def get_data(self, required_samples):
        try:
            desired_data = tasks[-required_samples:]
            return_arr = []
            for (raw_data, timestamp) in desired_data:
                join_data = ''.join(map(chr, raw_data))
                data = self.cipher.decrypt(bytes(join_data, 'latin-1')[0:32])
                if str(data[1]) == "32":  # No Gyro Data.
                    print("gyro data")
                    return

                # counter
                packet_data = []
                for i in range(2, 16, 2):
                    packet_data.append(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1])))

                for i in range(18, len(data), 2):
                    packet_data.append(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1])))

                return_arr.append(packet_data)
            return np.asarray(return_arr)

        except Exception as exception2:
            print(str(exception2))


def getEpoch(required_samples):
    return cyHeadset.get_data(required_samples)


def get_images():
    images_list = []
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, pixels_as_float=False, compress=False)])
    # airsim.ImageRequest("front_center", airsim.ImageType.DepthVis, pixels_as_float=True, compress=False),
    # airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False),
    # airsim.ImageRequest("back_center", airsim.ImageType.DepthVis, pixels_as_float=True, compress=False)])
    for response in responses:
        if response.pixels_as_float:
            images_list.append(np.asarray(airsim.get_pfm_array(response)[76:135, 0:255]) / 65504)
        else:
            image1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            image_rgb = image1d.reshape(responses[0].height, responses[0].width, 3)
            image_rgb = image.image.flip_axis(image_rgb, 2)
            images_list.append(np.asarray(image_rgb[76:135, 0:255, 0:3]) / 255)
    return images_list

steer_throttle_brake_model = load_model(STEER_THROTTLE_BRAKE_MODEL_PATH)
print('Using model {0} for steer & throttle testing.'.format(STEER_THROTTLE_BRAKE_MODEL_PATH))
# Connect to AirSim
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
print('Connection established!')
png_image_buf_0 = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1, 4))


cyHeadset = EEG()


while True:
    time.sleep(2.5)
    data = getEpoch(required_samples)
    data = data.astype(float)
    if data.shape[0] != required_samples:
        print("insufficient samples")
        break
    X_orig = feature_extraction(epoch_length=epoch_length, duration=duration, arr=data, method=0, fs=256)
    X = np.asarray(X_orig, dtype=float).T  # Expected shape (70, 8)
    # X = np.concatenate((X[50:65, :]), axis=0)
    # X = X[50:65, :]
    X = (X - model_mean) / model_std
    X = np.concatenate((X[7:9, :], X[62:64, :],
                        X[12:14, :], X[57:59, :],
                        X[17:19, :], X[52:54, :],
                        X[42:44, :]), axis=0)
    # X = model_ss.transform(X.T)
    # X = X.T

    predictions = predict(X, model_parameters)
    # predictions = model_parameters.predict(X.T)
    if np.sum(predictions) == 1:
        print(predictions)
        print("Detected: Sleeping")

        # Controlling the Vehicle
        client.enableApiControl(True)
        while True:
            car_state = client.getCarState()
            car_controls = client.getCarControls()

            captured_images = get_images()
            png_image_buf_0[0] = captured_images[0]
            state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
            steer_throttle_brake_model_output = steer_throttle_brake_model.predict([png_image_buf_0, state_buf])
            if steer_throttle_brake_model_output[0][0] > 0.5:
                steer_throttle_brake_model_output[0][0] = 0.5
            elif steer_throttle_brake_model_output[0][0] < -0.5:
                steer_throttle_brake_model_output[0][0] = -0.5
            if steer_throttle_brake_model_output[0][1] > 1:
                steer_throttle_brake_model_output[0][1] = 1.0
            elif steer_throttle_brake_model_output[0][1] < 0:
                steer_throttle_brake_model_output[0][1] = 0.0
            if steer_throttle_brake_model_output[0][2] > 1:
                steer_throttle_brake_model_output[0][2] = 1.0
            elif steer_throttle_brake_model_output[0][2] < 0:
                steer_throttle_brake_model_output[0][2] = 0.0

            car_controls.steering = round(float(steer_throttle_brake_model_output[0][0]), 2)
            car_controls.throttle = round(float(steer_throttle_brake_model_output[0][1]), 2)
            car_controls.brake = round(0.5 * float(steer_throttle_brake_model_output[0][2]), 2)
            print('Sending steering = {0}, throttle = {1}, brake = {2}'.format(car_controls.steering,
                                                                               car_controls.throttle,
                                                                               car_controls.brake))
            if client.isApiControlEnabled():
                client.setCarControls(car_controls)

            if car_state.speed == 0:
                client.enableApiControl(False)
                break
    else:
        print(predictions)
        print("Detected: Awake")

