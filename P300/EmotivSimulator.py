import socket
from time import sleep

class EmotivMessageSimulator(object):
    
    def __init__(self, HOST = '127.0.0.1', PORT = 54123, BUFFER_SIZE = 1024, TRANSMISSIONS_PER_SECOND = 128, signal = None, bit_size = 16):
        
        # Socket Info
        self.HOST = HOST
        self.PORT = PORT
        self.BUFFER_SIZE = BUFFER_SIZE
        
        # Delay To Send Every Tranmission At To Mimic Sampling Of Emotiv
        self.TRANSMISSIONS_PER_SECOND = TRANSMISSIONS_PER_SECOND
        self.DELAY = 1 / self.TRANSMISSIONS_PER_SECOND
        
        # Client's Socket & Address
        self.client_socket = None
        self.client_address = None
        
        # Message Info
        if not signal:  self.signal = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'
        else:           self.signal = signal
        self.bit_size = bit_size
        self.message_counter = 0
        
        # Starting Socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.HOST, self.PORT))
        
        print('Socket Sender Init Complete! Waiting For Connection!!!')
        
        self.wait_for_connection()
    
    def wait_for_connection(self):
        self.server_socket.listen(5)
        (self.client_socket, self.client_address) = self.server_socket.accept()
        print('Connection Established! Waiting For Initiation Message "b\'\\r\\n\'"!!!')
        
        while 1:
            data = self.client_socket.recv(self.BUFFER_SIZE)
            if data == b'\r\n':
                break
            else:
                print('Wrong Initiation Message! Waiting For Initiation Message "b\'\\r\\n\'"!!!')
        
        print('Connection Complete! Starting Transmission')
        
        self.transmit()
    
    def transmit(self):
        
        while 1:
            message = (str(self.message_counter) + ',' + str(self.bit_size) + ',' + self.signal + ',' + '\r\n')
            try:
                self.client_socket.send(message.encode('utf-8'))
            except:
                print('Connection Closed! Waiting For Connection!!!')
                self.wait_for_connection()
                break
            
            self.message_counter += 1
            if self.message_counter == self.TRANSMISSIONS_PER_SECOND:
                self.message_counter = 0
            
            sleep(self.DELAY)
EmotivMessageSimulator()