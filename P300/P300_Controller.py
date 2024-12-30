import numpy
import random

from P300_GUI import P300_GUI
from P300_Preprocessor import P300_Preprocessor
from P300_SocketReceiver import P300_SocketReceiver
from P300_FileLoader import P300_FileLoader
from P300_Model import P300_Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class P300_Controller(object):

    def __init__(
            self,
            matrices,
            channels=14
    ):

        self.channels = channels

        self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'

        self.gui = P300_GUI(self, matrices)
        self.socket = P300_SocketReceiver(self)

        self.counter = 0
        self.gui.repetitions = 1
        self.gui.start_delay = 500
        self.gui.end_delay = 500

        # MUST BE AT END
        self.gui.root.mainloop()

    def connect(self, host, port):
        try:
            self.socket.connect(host, port)
            self.gui.print_log('Connection Success!!!')
            return True
        except:
            self.gui.print_log('Connection Failure!!!')
            return True

    def select_user(self, user):
        self.file_loader = P300_FileLoader(user)
        self.file_loader.load_files()

        if self.file_loader.signal.shape[0] < 36:
            self.gui.print_log('Epochs < 36!!!Please Train!!!')
            return False

        preprocessor = P300_Preprocessor(
            self.file_loader.signal,
            self.file_loader.stimulus_code,
            self.file_loader.loaded_characters,
            self.gui.keyboard_matrix,
            common_average_reference=1,
            moving_average=13,
            z_score=1,
            decimation=6,
            extracted_channels=numpy.arange(self.channels),
            digitization_samples=128,
            start_window=0,
            end_window=102
        )

        bp = int(self.file_loader.signal.shape[0] * 0.8)

        self.model = P300_Model(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        self.model.fit(preprocessor.preprocessed_signal[bp:], preprocessor.preprocessed_classes[bp:])

        accuracy = self.model.calculate_accuracy(
            preprocessor.preprocessed_signal[:bp],
            self.file_loader.loaded_characters[:bp],
            self.gui.keyboard_matrix
        )

        if accuracy < 0.5:
            self.gui.print_log('Accuracy <50%!!! Please Train More!!!')
            return False

        self.gui.print_log('Accuracy: ' + str(accuracy) + '!!!, Success!!!')
        return True

    def calculate_accuracy(self):
        self.file_loader.load_files()

        if self.file_loader.signal.shape[0] < 36:
            return 0

        preprocessor = P300_Preprocessor(
            self.file_loader.signal,
            self.file_loader.stimulus_code,
            self.file_loader.loaded_characters,
            self.gui.keyboard_matrix,
            common_average_reference=1,
            moving_average=13,
            z_score=1,
            decimation=6,
            extracted_channels=numpy.arange(self.channels),
            digitization_samples=128,
            start_window=0,
            end_window=102
        )

        bp = int(self.file_loader.signal.shape[0] * 0.8)

        self.model = P300_Model(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        self.model.fit(preprocessor.preprocessed_signal[bp:], preprocessor.preprocessed_classes[bp:])

        return self.model.calculate_accuracy(
            preprocessor.preprocessed_signal[:bp],
            self.file_loader.loaded_characters[:bp],
            self.gui.keyboard_matrix
        )

    def start_session(self):
        self.socket.start_record()

        if self.gui.state == 'train':
            self.character = self.characters[random.randint(0, len(self.characters)) - 1]
            self.gui.keyboard_label.config(text='FOCUS ON "' + self.character + '"')

            row, column = self.search(self.character)
            self.gui.keyboard_label_matrix[row, column].config(fg='white')

    def end_session_as_train(self):
        signal, stimulus_code = self.socket.end_record()

        if signal.shape[0] == 0:
            self.gui.print_log('Empty Signals!!! Please Reconnect!!!')
            return

        self.file_loader.save_file(self.character, signal, stimulus_code)

    def end_session_as_operation(self):
        signal, stimulus_code = self.socket.end_record()

        if signal.shape[0] == 0:
            self.gui.print_log('Empty Signals!!! Please Reconnect!!!')
            return

        signal = signal.reshape(1, -1, self.channels)
        stimulus_code = stimulus_code.reshape(1, -1)

        preprocessor = P300_Preprocessor(
            signal,
            stimulus_code,
            self.gui.keyboard_matrix[0, 0],
            self.gui.keyboard_matrix,
            common_average_reference=1,
            moving_average=13,
            z_score=1,
            decimation=6,
            extracted_channels=numpy.arange(self.channels),
            digitization_samples=128,
            start_window=0,
            end_window=102
        )

        # preprocessor.preprocessed_signal = preprocessor.preprocessed_signal.reshape(self.gui.keyboard_matrix.shape[0] + self.gui.keyboard_matrix.shape[1], -1)
        # predicted_char = self.model.predict_character(preprocessor.preprocessed_signal, self.gui.keyboard_matrix)

        # self.gui.keyboard_label.config(text = (self.gui.keyboard_label.cget('text') + predicted_char))
        self.counter += 1
        self.gui.keyboard_label.config(text='HELLO_WORLD'[:self.counter])

    def search(self, character):

        indices = numpy.where(self.gui.keyboard_matrix == character)
        row = indices[0][0]
        column = indices[1][0]

        return row, column


matrices = {
    'train': numpy.array([['A', 'B', 'C', 'D', 'E', 'F'],
                          ['G', 'H', 'I', 'J', 'K', 'L'],
                          ['M', 'N', 'O', 'P', 'Q', 'R'],
                          ['S', 'T', 'U', 'V', 'W', 'X'],
                          ['Y', 'Z', '1', '2', '3', '4'],
                          ['5', '6', '7', '8', '9', '_']]),
    'test': numpy.array([['A', 'B', 'C', 'D', 'E', 'F'],
                         ['G', 'H', 'I', 'J', 'K', 'L'],
                         ['M', 'N', 'O', 'P', 'Q', 'R'],
                         ['S', 'T', 'U', 'V', 'W', 'X'],
                         ['Y', 'Z', '1', '2', '3', '4'],
                         ['5', '6', '7', '8', '9', '_']])
}
P300_Controller(matrices)
