import os.path
import random
import pickle
import numpy

class P300_FileLoader(object):
    
    def __init__(self, user):
        self.path = 'user_data/' + user + '/'
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)
    
    def save_file(self, character, signal, stimulus_code):
        file_name = str(random.randint(1000, 9999)) + '.pkl'
        while os.path.exists(self.path + file_name):
            file_name = random.randint(1000, 9999)
        
        f = open(self.path + file_name, 'wb')
        
        data = {}
        data['character'] = character
        data['signal'] = signal
        data['stimulus_code'] = stimulus_code
        pickle.dump(data, f)
        
        f.close()
    
    def load_files(self):
        
        self.signal = numpy.empty((0))
        self.stimulus_code = numpy.empty((0))
        self.loaded_characters = ''
        
        files = os.listdir(self.path)
        for file in files:
            f = open(self.path + file, 'rb')
            data = pickle.load(f)
            
            signal_epoch = data['signal']
            stimulus_code_epoch = data['stimulus_code']
            self.loaded_characters += data['character']
            self.append_epoch(signal_epoch, stimulus_code_epoch)
            
            f.close()
    
    def append_epoch(
            self,
            signal_epoch,
            stimulus_code_epoch
        ):
        
        if (self.signal.shape[0] == 0) and (self.stimulus_code.shape[0] == 0):
            self.signal = numpy.reshape(signal_epoch, (1, -1, signal_epoch.shape[1]))
            self.stimulus_code = numpy.reshape(stimulus_code_epoch, (1, -1))
        
        else:
            # Determine Minimum
            final_index = min(self.signal.shape[1], signal_epoch.shape[0])
            
            # Clip
            self.signal = self.signal[:, : final_index, :]
            self.stimulus_code = self.stimulus_code[:, : final_index]
            signal_epoch = signal_epoch[ : final_index, :]
            stimulus_code_epoch = stimulus_code_epoch[ : final_index]
            
            # Reshape
            signal_epoch = numpy.reshape(signal_epoch, (1, -1, self.signal.shape[2]))
            stimulus_code_epoch = numpy.reshape(stimulus_code_epoch, (1, -1))
            
            # Append
            self.signal = numpy.append(self.signal, signal_epoch, axis = 0)
            self.stimulus_code = numpy.append(self.stimulus_code, stimulus_code_epoch, axis = 0)