import os
import pickle
import numpy
import random

path = 'user_data/omar/'
characters_to_be_loaded = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'

for char in characters_to_be_loaded:
    
    file_name = str(random.randint(1000, 9999)) + '.pkl'
    while os.path.exists(path + file_name):
        file_name = str(random.randint(1000, 9999)) + '.pkl'
    
    data = {}
    data['character'] = char
    data['signal'] = numpy.loadtxt(path + '0/signal_' + char + '.txt')
    data['stimulus_code'] = numpy.loadtxt(path + '0/stimulus_code_' + char + '.txt')
    
    file = open(path + file_name, 'wb')
    pickle.dump(data, file)
    file.close()
    