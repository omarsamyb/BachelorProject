import numpy
import math
import matplotlib
from scipy import signal


class P300_Preprocessor(object):

    def __init__(
            self,
            signal,
            stimulus_code,
            target_char,
            matrix,
            common_average_reference=0,
            moving_average=0,
            z_score=0,
            decimation=0,
            extracted_channels=numpy.empty((0)),
            digitization_samples=240,
            start_window=0,
            end_window=240
    ):

        self.signal = signal
        self.stimulus_code = stimulus_code
        self.target_char = target_char
        self.matrix = matrix
        self.digitization_samples = digitization_samples
        self.start_window = start_window
        self.end_window = end_window

        self.window = self.end_window - self.start_window
        self.intensifications = self.matrix.shape[0] + self.matrix.shape[1]
        self.digitization_difference = math.floor(self.digitization_samples / 10)

        self.preprocessed_signal = self.signal
        self.preprocessed_classes = numpy.empty((self.signal.shape[0], self.intensifications))

        self.calculate_average()
        if common_average_reference: self.calulate_common_average_reference()
        if moving_average: self.calculate_moving_average(moving_average)
        if z_score: self.calculate_z_score()
        if decimation: self.calculate_decimation(decimation)
        if extracted_channels.shape[0]: self.calculate_concatenation(extracted_channels)

        self.calculate_classes()

    def reinit(
            self,
            signal=None,
            stimulus_code=None,
            target_char=None,
            matrix=None,
            common_average_reference=0,
            moving_average=0,
            z_score=0,
            decimation=0,
            extracted_channels=numpy.empty((0)),
            digitization_samples=240,
            start_window=0,
            end_window=240
    ):

        if signal: self.signal = signal
        if stimulus_code: self.stimulus_code = stimulus_code
        if target_char: self.target_char = target_char
        if matrix: self.matrix = matrix
        if digitization_samples: self.digitization_samples = digitization_samples
        if start_window: self.start_window = start_window
        if end_window: self.end_window = end_window

        self.window = self.end_window - self.start_window
        self.intensifications = self.matrix.shape[0] + self.matrix.shape[1]
        self.digitization_difference = math.floor(self.digitization_samples / 10)

        self.preprocessed_signal = self.signal
        self.preprocessed_classes = numpy.empty((self.signal.shape[0], self.intensifications))

        self.calculate_average()
        if common_average_reference: self.calulate_common_average_reference()
        if moving_average: self.calculate_moving_average(moving_average)
        if z_score: self.calculate_z_score()
        if decimation: self.calculate_decimation(decimation)
        if extracted_channels.shape[0]: self.calculate_concatenation(extracted_channels)

        self.calculate_classes()

    # Calculation of Intensification Average
    def calculate_average(self):

        preprocessed_signal = numpy.zeros(
            (self.preprocessed_signal.shape[0], self.intensifications, self.window, self.preprocessed_signal.shape[2]))

        # Looping Through Characters
        for epoch in range(self.preprocessed_signal.shape[0]):

            intensification_counter = numpy.zeros((self.intensifications))
            for n in range(1, self.preprocessed_signal.shape[1]):
                if self.stimulus_code[epoch, n] == 0 and self.stimulus_code[epoch, n - 1] != 0:
                    intensification_counter[int(self.stimulus_code[epoch, n - 1]) - 1] += 1
                    try:
                        preprocessed_signal[
                            epoch, int(self.stimulus_code[epoch, n - 1]) - 1] += self.preprocessed_signal[epoch,
                                                                                 n + self.start_window - self.digitization_difference: n + self.end_window - self.digitization_difference]
                    except:
                        intensification_counter[int(self.stimulus_code[epoch, n - 1]) - 1] -= 1
            for intensification in range(self.intensifications):
                preprocessed_signal[epoch, intensification] /= intensification_counter[intensification]

        self.preprocessed_signal = preprocessed_signal

    # Calculation of Common Average Reference - subtract from each sample in a channel the average of this sample's 14 channels values (channel avg)
    def calulate_common_average_reference(self):

        # Looping Through Characters
        for epoch in range(self.preprocessed_signal.shape[0]):

            mean_sample = numpy.mean(self.preprocessed_signal[epoch, :, :, :], axis=2)
            for channel in range(self.preprocessed_signal.shape[3]):
                self.preprocessed_signal[epoch, :, :, channel] = self.preprocessed_signal[epoch, :, :,
                                                                 channel] - mean_sample

    # Calculation of Moving Average - set each row with the average of the next "moving_average" rows
    def calculate_moving_average(self, moving_average):

        # Looping Through Characters
        for epoch in range(self.preprocessed_signal.shape[0]):

            for intensification in range(self.preprocessed_signal.shape[1]):
                for sample in range(self.preprocessed_signal.shape[2] - moving_average):
                    self.preprocessed_signal[epoch, intensification, sample, :] = \
                        numpy.mean(self.preprocessed_signal[epoch, intensification, sample: sample + moving_average, :],
                                   axis=0)

    # Calculation of Z-Score - subtract from each sample in each channel the average of all rows and div by std of all rows
    def calculate_z_score(self):

        for epoch in range(self.preprocessed_signal.shape[0]):

            mean_channel = numpy.mean(self.preprocessed_signal[epoch, :, :, :], axis=1)
            std_channel = numpy.std(self.preprocessed_signal[epoch, :, :, :], axis=1)
            for sample in range(self.preprocessed_signal.shape[2]):
                self.preprocessed_signal[epoch, :, sample, :] = (self.preprocessed_signal[epoch, :, sample,
                                                                 :] - mean_channel) / std_channel

    def calculate_notch(self):

        b, a = signal.iirnotch(5.71, 5.71 / 128, 128)

        for epoch in range(self.preprocessed_signal.shape[0]):
            for intensification in range(self.preprocessed_signal.shape[1]):
                for channel in range(self.preprocessed_signal.shape[3]):
                    self.preprocessed_signal[epoch, intensification, :, channel] = signal.lfilter(b, a,
                                                                                                  self.preprocessed_signal[
                                                                                                  epoch,
                                                                                                  intensification, :,
                                                                                                  channel])

    # Calculation of Decimation
    def calculate_decimation(self, decimation):

        preprocessed_signal = numpy.zeros((self.preprocessed_signal.shape[0], self.intensifications,
                                           math.floor(self.window / decimation), self.preprocessed_signal.shape[3]))

        # Looping Through Characters
        for epoch in range(self.preprocessed_signal.shape[0]):

            for intensification in range(self.preprocessed_signal.shape[1]):
                for i in range(math.floor(self.window / decimation)):
                    preprocessed_signal[epoch, intensification, i, :] = self.preprocessed_signal[epoch, intensification,
                                                                        (i * decimation), :]

        self.preprocessed_signal = preprocessed_signal

    # Calculation of Concatenation
    def calculate_concatenation(self, extracted_channels):

        preprocessed_signal = numpy.zeros((self.preprocessed_signal.shape[0], self.preprocessed_signal.shape[1],
                                           self.preprocessed_signal.shape[2] * extracted_channels.shape[0]))

        # Looping Through Characters
        for epoch in range(self.preprocessed_signal.shape[0]):

            for intensification in range(self.preprocessed_signal.shape[1]):

                for channel_index in range(extracted_channels.shape[0]):
                    preprocessed_signal[epoch, intensification, (self.preprocessed_signal.shape[2] * channel_index):(
                                                                                                                                self.preprocessed_signal.shape[
                                                                                                                                    2] * channel_index) +
                                                                                                                    self.preprocessed_signal.shape[
                                                                                                                        2]] = \
                        self.preprocessed_signal[epoch, intensification, :, extracted_channels[channel_index]]

        self.preprocessed_signal = preprocessed_signal

    # Plotter
    def plot(
            self,
            common_average_reference=0,
            moving_average=0,
            z_score=0,
            decimation=0,
            digitization_samples=0,
            start_window=0,
            end_window=0,
            plotted_channels=numpy.array([0]),
            title=''
    ):

        preprocessed_signal = self.preprocessed_signal
        self.reinit(
            common_average_reference=common_average_reference,
            moving_average=moving_average,
            z_score=z_score,
            decimation=decimation,
            digitization_samples=digitization_samples,
            start_window=start_window,
            end_window=end_window
        )

        # self.calculate_notch()

        sum_signal_success = numpy.zeros((self.preprocessed_signal.shape[2], self.preprocessed_signal.shape[3]))
        sum_signal_fail = numpy.zeros((self.preprocessed_signal.shape[2], self.preprocessed_signal.shape[3]))

        # Looping Through Characters
        for epoch in range(self.preprocessed_signal.shape[0]):

            # Getting Index Of Chosen Character
            chosen_row, chosen_column = self.search(self.target_char[epoch])

            for row_column in range(self.intensifications):
                if row_column == chosen_row or row_column == chosen_column:
                    sum_signal_success += self.preprocessed_signal[epoch, row_column]
                else:
                    sum_signal_fail += self.preprocessed_signal[epoch, row_column]

        average_signal_success = sum_signal_success / (self.preprocessed_signal.shape[0] * 2)
        average_signal_fail = sum_signal_fail / (self.preprocessed_signal.shape[0] * (self.intensifications - 2))

        X = numpy.linspace(
            (self.start_window / self.digitization_samples) * 1000,
            (self.end_window / self.digitization_samples) * 1000,
            average_signal_success[:, plotted_channels].shape[0]
        )
        matplotlib.pyplot.plot(X, average_signal_success[:, plotted_channels])
        matplotlib.pyplot.plot(X, average_signal_fail[:, plotted_channels])
        matplotlib.pyplot.legend(['P300', 'Non-P300'])
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.xlabel('Time (ms)')
        matplotlib.pyplot.ylabel('Signal (arbitary units)')
        matplotlib.pyplot.show()

        self.preprocessed_signal = preprocessed_signal

    # Class (y) Constructor
    def calculate_classes(self):

        # Looping Through Characters
        for epoch in range(len(self.target_char)):

            # Getting Index Of Chosen Character
            chosen_row, chosen_column = self.search(self.target_char[epoch])

            for row_column in range(self.intensifications):
                if row_column == chosen_row or row_column == chosen_column:
                    self.preprocessed_classes[epoch, row_column] = 1
                else:
                    self.preprocessed_classes[epoch, row_column] = -1

    # Search Function
    def search(self, char):

        indices = numpy.where(self.matrix == char)
        row = indices[0][0] + self.matrix.shape[1]
        column = indices[1][0]

        return row, column
