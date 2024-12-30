import tkinter
from win32api import GetSystemMetrics
import numpy
import random

class P300_GUI(object):
    
    def __init__(
            self,
            controller,
            matrices
        ):
        
        self.height = GetSystemMetrics(1)
        self.width = GetSystemMetrics(0)
        
        # Variables
        self.bd = 3
        self.bg = 'black'
        self.controller = controller
        self.fg = 'grey'
        self.font = 'Courier'
        self.state = 'halt'
        self.matrices = matrices
        self.scale = 3
        
        self.keyboard_matrix = self.matrices.get('train')
        self.color_array = numpy.array(['white'])
        self.start_delay = 2500
        self.state_0_delay = 100
        self.state_1_delay = 75
        self.end_delay = 2000
        
        self.repetitions = 15
        
        # Tkinter Initialization
        self.root = tkinter.Tk()
        self.root.title('On-Screen BCI Keyboard')
        self.root.attributes('-topmost', True)
        
        # Constructing GUI
        self.construct_connection()
        self.construct_user()
        self.construct_functionality()
        self.construct_keyboard()
        self.construct_log()
        
        # Disabling Order
        self.user_button.config(state = 'disabled')
        self.train_button.config(state = 'disabled')
        self.operation_button.config(state = 'disabled')
        self.halt_button.config(state = 'disabled')
        
        self.root.geometry('%dx%d+0+0' % (int(self.width / self.scale), self.height))
    
    def __del__(self):
        self.close()
        del self
    
    # Update GUI
    def update(self):
        self.root.update()
    
    # Close GUI
    def close(self):
        try: self.root.destroy()
        except: 1
    
    # Connection
    def construct_connection(self):
        self.connection_frame = tkinter.Frame(self.root, bd = self.bd, relief = 'groove')
        self.connection_frame.pack(expand = 'yes', fill = tkinter.X)
        
        self.host_entry = tkinter.Entry(self.connection_frame, font = self.font)
        self.host_entry.insert('end', '127.0.0.1')
        self.host_entry.pack(expand = 'yes', fill = 'both', side = 'left')
        
        self.port_entry = tkinter.Entry(self.connection_frame, font = self.font)
        self.port_entry.insert('end', '54123')
        self.port_entry.pack(expand = 'yes', fill = 'both', side = 'left')
        
        self.connection_button = tkinter.Button(self.connection_frame, command = self.connect, font = self.font, text = 'Try')
        self.connection_button.pack(side = 'right')
    
    def connect(self):
        if self.controller.connect(self.host_entry.get(), int(self.port_entry.get())):
            self.host_entry.config(state = 'disabled')
            self.port_entry.config(state = 'disabled')
            self.user_button.config(state = 'normal')
    
    # User
    def construct_user(self):
        self.user_frame = tkinter.Frame(self.root, bd = self.bd, relief = 'groove')
        self.user_frame.pack(expand = 'yes', fill = tkinter.X)
        
        self.user_entry = tkinter.Entry(self.user_frame, font = self.font)
        self.user_entry.insert('end', 'default')
        self.user_entry.pack(expand = 'yes', fill = 'both', side = 'left')
        
        self.user_button = tkinter.Button(self.user_frame, command = self.select_user, font = self.font, text = 'Select')
        self.user_button.pack(side = 'left')
    
    def select_user(self):
        if self.controller.select_user(self.user_entry.get()):
            self.operation_button.config(state = 'normal')
        
        self.user_entry.config(state = 'disabled')
        self.user_button.config(state = 'disabled')
        self.train_button.config(state = 'normal')
    
    # Funionality
    def construct_functionality(self):
        self.functionality_frame = tkinter.Frame(self.root, bd = self.bd, relief = 'groove')
        self.functionality_frame.pack(expand = 'yes', fill = tkinter.X)
        
        self.train_button = tkinter.Button(self.functionality_frame, command = self.train, font = self.font, text = 'Train')
        self.train_button.pack(expand = 'yes', fill = tkinter.X, side = 'left')
        
        self.operation_button = tkinter.Button(self.functionality_frame, command = self.operation, font = self.font, text = 'Operation')
        self.operation_button.pack(expand = 'yes', fill = tkinter.X, side = 'left')
        
        self.halt_button = tkinter.Button(self.functionality_frame, command = self.halt_f, font = self.font, text = 'Halt')
        self.halt_button.pack(expand = 'yes', fill = tkinter.X, side = 'left')
    
    def train(self):
        self.state = 'train'
        if not self.controller.calculate_accuracy() < 0.5:
            self.operation_button.config(state = 'normal')
        
        self.train_button.config(state = 'disabled')
        self.halt_button.config(state = 'normal')
        self.print_log('Train Mode!!!')
        
        self.start_session()
    
    def operation(self):
        self.state = 'operation'
        self.train_button.config(state = 'normal')
        self.operation_button.config(state = 'disabled')
        self.halt_button.config(state = 'normal')
        self.print_log('Operation Mode!!!')
        
        self.start_session()
    
    def halt_f(self):
        self.state = 'halt'
        if not self.controller.calculate_accuracy() < 0.5:
            self.operation_button.config(state = 'normal')
        
        self.train_button.config(state = 'normal')
        self.halt_button.config(state = 'disabled')
        self.print_log('Halt!!!')
    
    # Keyboard
    def construct_keyboard(self):
        self.keyboard_frame = tkinter.Frame(self.root, bd = self.bd, bg = self.bg, relief = 'groove')
        self.keyboard_frame.pack(expand = 'yes', fill = 'both')
        
        font_size = int(self.height / (self.scale * self.keyboard_matrix.shape[0]))
        self.keyboard_label = tkinter.Label(
                self.keyboard_frame,
                bd = self.bd,
                font = self.font + ' ' + str(font_size),
                relief = 'groove'
            )
        self.keyboard_label.grid(row = 0, column = 0, columnspan = self.keyboard_matrix.shape[1], sticky = 'news')
        
        cell_font_size = int(self.height / (self.scale * self.keyboard_matrix.shape[0]))
        self.keyboard_label_matrix = numpy.empty(self.keyboard_matrix.shape, dtype = tkinter.Label)
        for row in numpy.arange(self.keyboard_matrix.shape[0]):
            for column in numpy.arange(self.keyboard_matrix.shape[1]):
                self.keyboard_label_matrix[row, column] = tkinter.Label(
                        self.keyboard_frame,
                        bg = self.bg,
                        fg = self.fg,
                        font = self.font + ' ' + str(cell_font_size),
                        text = self.keyboard_matrix[row, column]
                    )
                self.keyboard_label_matrix[row, column].grid(row = row + 1, column = column)
        
        cell_size = int(self.width / (self.scale * self.keyboard_matrix.shape[1]))
        for row in numpy.arange(self.keyboard_matrix.shape[0] + 1):
            self.keyboard_frame.grid_rowconfigure(row, minsize = cell_size)    
        for column in numpy.arange(self.keyboard_matrix.shape[1]):
            self.keyboard_frame.grid_columnconfigure(column, minsize = cell_size)
    
    def reset_keyboard(self):
        for row in numpy.arange(self.keyboard_matrix.shape[0]):
            for column in numpy.arange(self.keyboard_matrix.shape[1]):
                self.keyboard_label_matrix[row, column].config(fg = self.fg)
    
    # Log
    def construct_log(self):
        self.log_frame = tkinter.Frame(self.root, bd = self.bd, relief = 'groove')
        self.log_frame.pack(expand = 'yes', fill = tkinter.X)
        
        self.log_text = tkinter.Text(self.log_frame, font = self.font)
        self.log_text.insert('end', 'Log Entries!')
        self.log_text.config(state = 'disabled')
        self.log_text.pack(expand = 'yes', fill = 'both')
    
    def print_log(self, message):
        self.log_text.config(state = 'normal')
        self.log_text.insert('end', '\n' + message)
        self.log_text.see('end')
        self.log_text.config(state = 'disabled')
    
    # Session
    def start_session(self):
        self.controller.start_session()
        
        # Resetting Temporary Repetitions
        self.temp_repetitions = self.repetitions - 1
        
        # Intensification Order Initialization
        self.intensification_order = numpy.arange(self.keyboard_matrix.shape[0] + self.keyboard_matrix.shape[1])
        numpy.random.shuffle(self.intensification_order)
        
        # Chosen Intensification Initialization
        self.intensification_order_index = 0
        self.stimulus_code = -1
        
        # Sleep for 2.5s Then Start
        self.root.after(self.start_delay, self.state_0)
    
    def repeat_session(self):
        # Decrementing Repetitions
        self.temp_repetitions -= 1
        
        # Intensification Order Initialization
        self.intensification_order = numpy.arange(self.keyboard_matrix.shape[0] + self.keyboard_matrix.shape[1])
        numpy.random.shuffle(self.intensification_order)
        
        # Chosen Intensification Initialization
        self.intensification_order_index = 0
        self.stimulus_code = -1
        
        # Start Immediately
        self.state_0()
    
    def end_session(self):
        if self.state == 'train':
            self.controller.end_session_as_train()
        elif self.state == 'operation':
            self.controller.end_session_as_operation()
        
        self.start_session()
    
    def state_0(self):
        
        self.reset_keyboard()
        
        if self.state == 'halt': return
        
        # Get Chosen Row / Column
        self.stimulus_code = self.intensification_order[self.intensification_order_index]
        if self.stimulus_code < self.keyboard_label_matrix.shape[1]:
            row_column_labels = self.keyboard_label_matrix[:, self.stimulus_code % self.keyboard_label_matrix.shape[1]]
        else:
            row_column_labels = self.keyboard_label_matrix[self.stimulus_code % self.keyboard_label_matrix.shape[0], :]
        
        color_index = random.randint(0, self.color_array.shape[0] - 1)
        for label in row_column_labels:
            label.config(fg = self.color_array[color_index])
        
        self.root.update()
        
        # Wait Then Move To Next State
        self.root.after(self.state_0_delay, self.state_1)
    
    def state_1(self):
        
        self.stimulus_code = -1
        self.reset_keyboard()
        
        if self.state == 'halt': return
        
        # Check If Final Index
        self.intensification_order_index += 1
        if self.intensification_order_index < self.intensification_order.shape[0]:
            self.root.after(self.state_1_delay, self.state_0)
        elif self.temp_repetitions != 0:
            self.root.after(self.state_1_delay, self.repeat_session)
        else:
            self.root.after(self.end_delay, self.end_session)