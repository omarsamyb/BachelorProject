import numpy
import socket
import _thread

class P300_SocketReceiver(object):
    
    def __init__(self, controller, BUFFER_SIZE = 1024, MAX_LEN = 10000):
        
        self.controller = controller
        self.BUFFER_SIZE = BUFFER_SIZE
        self.MAX_LEN = MAX_LEN
        
        self.signal = numpy.empty((0, self.controller.channels))
        self.stimulus_code = numpy.empty((0))
        self.is_record = False
        
        # Socket Establishment
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def connect(self, host, port):
        self.socket.connect((host, port))
        self.socket.send(b'\r\n')
        
        # Start Receiving In Parallel Thread
        _thread.start_new_thread(self.receive, ())
    
    def disconnect(self):
        self.socket.close()
    
    def start_record(self):
        self.signal = numpy.empty((0, self.controller.channels))
        self.stimulus_code = numpy.empty((0))
        self.is_record = True
    
    def end_record(self):
        self.is_record = False
        
        # In Case Of Clash In The Middle of The if Condition
        min_index = min(self.signal.shape[0] - 1, self.stimulus_code.shape[0] - 1)
        return self.signal[:min_index, :], self.stimulus_code[:min_index]
    
    def receive(self):
        
        while 1:
            
            # Add Message To Previous Message
            msg = self.socket.recv(self.BUFFER_SIZE).decode('utf-8')
            
            # Split Message & Read First Part
            msg = msg.split('\r\n')
            data = numpy.fromstring(msg[0], dtype = numpy.float, sep=',')[2 : -2]
            
            if data.shape[0] == self.controller.channels and self.is_record:
                self.signal = numpy.append(self.signal, data.reshape(1, self.controller.channels), axis = 0)
                self.stimulus_code = numpy.append(self.stimulus_code, self.controller.gui.stimulus_code + 1)
                
                if self.signal.shape[0] == self.MAX_LEN:
                    self.end_record()