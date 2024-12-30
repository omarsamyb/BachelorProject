import airsim
# from Simulator_E2EDL.AirSimClient import *
from bachelor.Simulator_E2EDL import *
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time
from keras.preprocessing import image
import keras.backend as K
import cv2
import glob

IMG_PATH = 'Dataset-E2EDL/data_raw/'


def save_image():
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        airsim.ImageRequest("front_center", airsim.ImageType.DepthVis, pixels_as_float=True, compress=False),
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, pixels_as_float=False, compress=True),
        airsim.ImageRequest("back_center", airsim.ImageType.DepthVis, pixels_as_float=True, compress=False),
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, pixels_as_float=False, compress=False)])
    airsim.write_file(os.path.normpath(IMG_PATH + 'front%02d.png' % 0), responses[0].image_data_uint8)
    airsim.write_pfm(os.path.normpath(IMG_PATH + 'front%02d.pfm' % 3), airsim.get_pfm_array(responses[1]))
    airsim.write_file(os.path.normpath(IMG_PATH + 'front%02d.png' % 5), responses[2].image_data_uint8)
    airsim.write_pfm(os.path.normpath(IMG_PATH + 'back%02d.pfm' % 3), airsim.get_pfm_array(responses[3]))

    # sample_image_path = os.path.join(IMG_PATH, 'front00.png')
    # sample_image = Image.open(sample_image_path)

    image1d = np.frombuffer(responses[4].image_data_uint8, dtype=np.uint8)
    image_rgb = image1d.reshape(responses[0].height, responses[0].width, 3)
    image_rgb = image.image.flip_axis(image_rgb, 2)
    print(image_rgb[0, :20, 0])
    print(image_rgb[0, :20, 0]/255)
    print("$$$$$$$$$$$$$$$$$$$$$")
    pil_image = image.array_to_img(image_rgb, K.image_data_format(), scale=False)
    # pil_image = Image.fromarray(image_rgb)

    plt.imshow(pil_image)
    plt.show()
    # airsim.write_png(os.path.normpath(IMG_PATH + 'front%02d.png' % 999), image_rgb)
    # cv2.imwrite(os.path.normpath(IMG_PATH + 'front%d.png' % 999), image_rgb)
    img2 = np.asarray(Image.open(os.path.join(IMG_PATH, 'front00.png')))
    print(img2[0, :20, 0])
    print(img2[0, :20, 0] / 255)
    # print(img2.shape)
    # print("$$$$$$$$$$$$$$$$$$$$$")
    # img3 = np.asarray(Image.open(os.path.join(IMG_PATH, 'front999.png')))
    # print(img3[0, :20, 0])
    # print(img3[0, :20, 0]/255)
    # print(img3.shape)
    # print("$$$$$$$$$$$$$$$$$$$$$")
    # x = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    # print(len(x))
    # x = x.reshape(responses[0].height, responses[0].width, 4)
    # print(x[0, :20, 0])
    # pil_image = image.array_to_img(img3, K.image_data_format(), scale=False)
    # plt.imshow(pil_image)
    # plt.show()


# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
save_image()

# while True:
#     car_controls2 = client.getCarControls()
#     print(car_controls2)
#     car_state = client.getCarState()
#     car_controls2 = client.getCarControls()
#     # velocity = client.getVelocity()
#     car_controls = airsim.CarControls()
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#     print("throttle", car_controls.throttle)
#     print("brake", car_controls.brake)
#     print("steering", car_controls.steering)
#     print("speed", car_state.speed)
#     time.sleep(1)
