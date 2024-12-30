from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image
# import sys
# import time
import numpy as np
# import glob
# import os
# from Simulator_E2EDL.AirSimClient import *
# from Simulator_E2EDL.airsim import *
import airsim

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# if '../../PythonClient/' not in sys.path:
#     sys.path.insert(0, '../../PythonClient/')

# If None, then the model with the lowest validation loss from training will be used
# BRAKE_MODEL_PATH = 'model/models/brake_model.01-0.0012584.h5'
STEER_THROTTLE_BRAKE_MODEL_PATH = 'model/models/steer_throttle_brake_00_model.17-0.0192344.h5'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


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


# if MODEL_PATH is None:
#     models = glob.glob('model/models/*.h5')
#     best_model = max(models, key=os.path.getctime)
#     MODEL_PATH = best_model

# brake_model = load_model(BRAKE_MODEL_PATH)
steer_throttle_brake_model = load_model(STEER_THROTTLE_BRAKE_MODEL_PATH)
# print('Using model {0} for brake testing.'.format(BRAKE_MODEL_PATH))
print('Using model {0} for steer & throttle testing.'.format(STEER_THROTTLE_BRAKE_MODEL_PATH))


# Connect to AirSim
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
print('Connection established!')

png_image_buf_0 = np.zeros((1, 59, 255, 3))
# png_image_buf_1 = np.zeros((1, 59, 255, 3))
# pfm_image_buf_0 = np.zeros((1, 59, 255))
# pfm_image_buf_1 = np.zeros((1, 59, 255))
state_buf = np.zeros((1, 4))

# Controlling the Vehicle
while True:
    car_state = client.getCarState()
    car_controls = client.getCarControls()
    if client.isApiControlEnabled() and car_state.speed < 0.1:
        client.enableApiControl(False)
    if not client.isApiControlEnabled() and car_controls.brake == 1:
        client.enableApiControl(True)

    captured_images = get_images()
    png_image_buf_0[0] = captured_images[0]
    # pfm_image_buf_0[0] = captured_images[1]
    # png_image_buf_1[0] = captured_images[2]
    # pfm_image_buf_1[0] = captured_images[3]
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    steer_throttle_brake_model_output = steer_throttle_brake_model.predict([png_image_buf_0, state_buf])
    # steer_throttle_brake_model_output[0][1] *= 1.5
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

    # if steer_throttle_brake_model_output[0][1] > steer_throttle_brake_model_output[0][2]:
    #     steer_throttle_brake_model_output[0][2] = 0.0
    # elif steer_throttle_brake_model_output[0][2] > steer_throttle_brake_model_output[0][1]:
    #     steer_throttle_brake_model_output[0][1] = 0.0
    car_controls.steering = round(float(steer_throttle_brake_model_output[0][0]), 2)
    car_controls.throttle = round(float(steer_throttle_brake_model_output[0][1]), 2)
    car_controls.brake = round(0.5 * float(steer_throttle_brake_model_output[0][2]), 2)
    # print('Sending brake = {0}'.format(brake_model_output[0][0]))
    print('Sending steering = {0}, throttle = {1}, brake = {2}'.format(car_controls.steering,
                                                                       car_controls.throttle,
                                                                       car_controls.brake))
    if client.isApiControlEnabled():
        client.setCarControls(car_controls)
