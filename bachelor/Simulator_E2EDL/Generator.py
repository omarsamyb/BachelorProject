from keras.preprocessing import image
import numpy as np
import keras.backend as K
import os
import cv2


# noinspection SpellCheckingInspection
class DriveDataGenerator(image.ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(DriveDataGenerator, self).__init__(featurewise_center,
                                                 samplewise_center,
                                                 featurewise_std_normalization,
                                                 samplewise_std_normalization,
                                                 zca_whitening,
                                                 zca_epsilon,
                                                 rotation_range,
                                                 width_shift_range,
                                                 height_shift_range,
                                                 brightness_range,
                                                 shear_range,
                                                 zoom_range,
                                                 channel_shift_range,
                                                 fill_mode,
                                                 cval,
                                                 horizontal_flip,
                                                 vertical_flip,
                                                 rescale,
                                                 preprocessing_function,
                                                 data_format)

    def flow(self,
             x_img_0_0=None,
             x_img_0_3=None,
             x_img_0_5=None,
             x_img_4_3=None,
             x_previous_state=None,
             y=None,
             x_previous_state_labels=None,
             y_labels=None,
             batch_size=32,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             zero_drop_percentage_steer=0.5,
             zero_drop_percentage_throttle=None,
             zero_drop_percentage_brake=None,
             one_drop_percentage_steer=None,
             one_drop_percentage_throttle=None,
             one_drop_percentage_brake=None,
             roi=None):
        return DriveIterator(
            x_img_0_0=x_img_0_0,
            x_img_0_3=x_img_0_3,
            x_img_0_5=x_img_0_5,
            x_img_4_3=x_img_4_3,
            x_previous_state=x_previous_state,
            y=y,
            x_previous_state_labels=x_previous_state_labels,
            y_labels=y_labels,
            image_data_generator=self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            zero_drop_percentage_steer=zero_drop_percentage_steer,
            zero_drop_percentage_throttle=zero_drop_percentage_throttle,
            zero_drop_percentage_brake=zero_drop_percentage_brake,
            one_drop_percentage_steer=one_drop_percentage_steer,
            one_drop_percentage_throttle=one_drop_percentage_throttle,
            one_drop_percentage_brake=one_drop_percentage_brake,
            roi=roi)

    def random_transform_with_states(self, x, flip=False, brighten=False):
        """
        Description:
            Randomly augment a single image tensor.
        Inputs:
            x: 3D tensor, single image.
            seed: random seed.
        Returns:
            A tuple. 0 -> randomly transformed version of the input (same shape). 1 -> true if image was horizontally
            flipped, false otherwise
        """

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1
        image_brighten_value = 0.0

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix
        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = image.image.transform_matrix_offset_center(transform_matrix, h, w)
            x = image.image.apply_affine_transform(x, transform_matrix, img_channel_axis,
                                                   fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = image.random_channel_shift(x,
                                           self.channel_shift_range,
                                           img_channel_axis)
        if self.horizontal_flip:
            if flip:
                x = image.image.flip_axis(x, img_col_axis)
        if self.vertical_flip:
            if flip:
                x = image.image.flip_axis(x, img_row_axis)
        if self.brightness_range is not None and brighten:
            random_bright = np.random.uniform(low=self.brightness_range[0], high=self.brightness_range[1])
            # TODO: Write this as an apply to push operations into C for performance
            img = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
            img[:, :, 2] = np.clip(img[:, :, 2] * random_bright, 0, 255)
            x = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            image_brighten_value = random_bright
        return x, image_brighten_value


class DriveIterator(image.Iterator):
    """
    Description:
        Iterator yielding data from a Numpy array.
    Inputs:
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator` to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self,
                 x_img_0_0=None,
                 x_img_0_3=None,
                 x_img_0_5=None,
                 x_img_4_3=None,
                 x_previous_state=None,
                 y=None,
                 x_previous_state_labels=None,
                 y_labels=None,
                 image_data_generator=None,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 zero_drop_percentage_steer=0.5,
                 zero_drop_percentage_throttle=None,
                 zero_drop_percentage_brake=None,
                 one_drop_percentage_steer=None,
                 one_drop_percentage_throttle=None,
                 one_drop_percentage_brake=None,
                 roi=None):
        self.x_img_0_0 = x_img_0_0
        self.x_img_0_3 = x_img_0_3
        self.x_img_0_5 = x_img_0_5
        self.x_img_4_3 = x_img_4_3
        if data_format is None:
            data_format = K.image_data_format()
        if y is not None and len(x_img_0_0) != len(y):
            raise ValueError('X (images tensor) and y (labels) should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x_img_0_0).shape, np.asarray(y).shape))
        if self.x_img_0_0.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` should have rank 4. You passed an array '
                             'with shape', self.x_img_0_0.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x_img_0_0.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('Incorrect channel axis of an RGB image')
        self.x_previous_state = x_previous_state
        self.y = y
        self.x_previous_state_labels = x_previous_state_labels
        self.y_labels = y_labels
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.zero_drop_percentage_steer = zero_drop_percentage_steer
        self.zero_drop_percentage_throttle = zero_drop_percentage_throttle
        self.zero_drop_percentage_brake = zero_drop_percentage_brake
        self.one_drop_percentage_steer = one_drop_percentage_steer
        self.one_drop_percentage_throttle = one_drop_percentage_throttle
        self.one_drop_percentage_brake = one_drop_percentage_brake
        self.roi = roi
        super(DriveIterator, self).__init__(x_img_0_0.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock. Only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self.__get_indexes(index_array)

    def __get_indexes(self, index_array):
        index_array = sorted(index_array)
        batch_x_img_0_0 = None
        batch_x_img_0_3 = None
        batch_x_img_0_5 = None
        batch_x_img_4_3 = None
        batch_x_previous_state = None
        if self.x_img_0_0 is not None:
            batch_x_img_0_0 = np.zeros(tuple([self.batch_size] + list(self.x_img_0_0.shape)[1:]), dtype=K.floatx())
            if self.roi is not None:
                batch_x_img_0_0 = batch_x_img_0_0[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3], :]
        if self.x_img_0_3 is not None:
            batch_x_img_0_3 = np.zeros(tuple([self.batch_size] + list(self.x_img_0_3.shape)[1:]), dtype=K.floatx())
            if self.roi is not None:
                batch_x_img_0_3 = batch_x_img_0_3[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        if self.x_img_0_5 is not None:
            batch_x_img_0_5 = np.zeros(tuple([self.batch_size] + list(self.x_img_0_5.shape)[1:]), dtype=K.floatx())
            if self.roi is not None:
                batch_x_img_0_5 = batch_x_img_0_5[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3], :]
        if self.x_img_4_3 is not None:
            batch_x_img_4_3 = np.zeros(tuple([self.batch_size] + list(self.x_img_4_3.shape)[1:]), dtype=K.floatx())
            if self.roi is not None:
                batch_x_img_4_3 = batch_x_img_4_3[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        if self.x_previous_state is not None:
            batch_x_previous_state = np.zeros(tuple([self.batch_size] + list(self.x_previous_state.shape)[1:]),
                                              dtype=K.floatx())

        used_indexes = []
        is_horizontally_flipped = []
        brighten_values = []
        # i = counter & j = value in index_array which is actual img index ex:0 -44k
        for i, j in enumerate(index_array):
            if np.random.random() < 0.5:
                flip = True
            else:
                flip = False
            is_horizontally_flipped.append(flip)

            if self.x_img_0_0 is not None:
                x_img_0_0 = self.x_img_0_0[j]
                if self.roi is not None:
                    x_img_0_0 = x_img_0_0[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3], :]
                transformed_x_img_0_0 = self.image_data_generator.random_transform_with_states(
                    x_img_0_0.astype(K.floatx()), flip=flip, brighten=True)
                brighten_values.append(transformed_x_img_0_0[1])
                transformed_x_img_0_0 = self.image_data_generator.standardize(transformed_x_img_0_0[0])
                batch_x_img_0_0[i] = transformed_x_img_0_0
            if self.x_img_0_3 is not None:
                x_img_0_3 = self.x_img_0_3[j]
                if self.roi is not None:
                    x_img_0_3 = x_img_0_3[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                transformed_x_img_0_3 = self.image_data_generator.random_transform_with_states(
                    x_img_0_3.astype(K.floatx()), flip=flip)
                transformed_x_img_0_3 = transformed_x_img_0_3[0] * 1. / 65504.
                batch_x_img_0_3[i] = transformed_x_img_0_3
            if self.x_img_0_5 is not None:
                x_img_0_5 = self.x_img_0_5[j]
                if self.roi is not None:
                    x_img_0_5 = x_img_0_5[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3], :]
                transformed_x_img_0_5 = self.image_data_generator.random_transform_with_states(
                    x_img_0_5.astype(K.floatx()), flip=flip)
                transformed_x_img_0_5 = self.image_data_generator.standardize(transformed_x_img_0_5[0])
                batch_x_img_0_5[i] = transformed_x_img_0_5
            if self.x_img_4_3 is not None:
                x_img_4_3 = self.x_img_4_3[j]
                if self.roi is not None:
                    x_img_4_3 = x_img_4_3[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                transformed_x_img_4_3 = self.image_data_generator.random_transform_with_states(
                    x_img_4_3.astype(K.floatx()), flip=flip)
                transformed_x_img_4_3 = transformed_x_img_4_3[0] * 1. / 65504.
                batch_x_img_4_3[i] = transformed_x_img_4_3
            if self.x_previous_state is not None:
                x_previous_state = self.x_previous_state[j]
                if flip:
                    if 'Steering' in self.x_previous_state_labels:
                        x_previous_state[self.x_previous_state_labels.index('Steering')] *= -1.0
                batch_x_previous_state[i] = x_previous_state
            used_indexes.append(j)

        if self.save_to_dir:
            for i in range(0, self.batch_size, 1):
                hash_value = np.random.randint(1e4)
                img = image.array_to_img(batch_x_img_0_0[i], self.data_format, scale=True)
                file_name = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=1,
                                                                      hash=hash_value,
                                                                      format=self.save_format)
                img.save(os.path.join(self.save_to_dir, file_name))

        batch_x = []
        if self.x_img_0_0 is not None:
            batch_x.append(batch_x_img_0_0)
        if self.x_img_0_3 is not None:
            batch_x.append(batch_x_img_0_3)
        if self.x_img_0_5 is not None:
            batch_x.append(batch_x_img_0_5)
        if self.x_img_4_3 is not None:
            batch_x.append(batch_x_img_4_3)
        if self.x_previous_state is not None:
            batch_x.append(batch_x_previous_state)
        batch_y = self.y[list(sorted(used_indexes))]
        idx = []
        for i in range(0, len(is_horizontally_flipped), 1):
            # if the label is not only steering but for example steering & throttle
            # if batch_y[i][int(len(batch_y[i]) / 2)] == 1:
            flag = True
            if 'Steering' in self.y_labels:
                if self.zero_drop_percentage_steer is not None:
                    if np.isclose(batch_y[i][self.y_labels.index('Steering')], 0):
                        if np.random.uniform(low=0, high=1) < self.zero_drop_percentage_steer:
                            flag = False
                if self.one_drop_percentage_steer is not None:
                    if np.isclose(batch_y[i][self.y_labels.index('Steering')], 1):
                        if np.random.uniform(low=0, high=1) < self.one_drop_percentage_steer:
                            flag = False
            if 'Throttle' in self.y_labels:
                if self.zero_drop_percentage_throttle is not None:
                    if np.isclose(batch_y[i][self.y_labels.index('Throttle')], 0):
                        if np.random.uniform(low=0, high=1) < self.zero_drop_percentage_throttle:
                            flag = False
                if self.one_drop_percentage_throttle is not None:
                    if np.isclose(batch_y[i][self.y_labels.index('Throttle')], 1):
                        if np.random.uniform(low=0, high=1) < self.one_drop_percentage_throttle:
                            flag = False
            if 'Brake' in self.y_labels:
                if self.zero_drop_percentage_brake is not None:
                    if np.isclose(batch_y[i][self.y_labels.index('Brake')], 0):
                        if np.random.uniform(low=0, high=1) < self.zero_drop_percentage_brake:
                            flag = False
                if self.one_drop_percentage_brake is not None:
                    if np.isclose(batch_y[i][self.y_labels.index('Brake')], 1):
                        if np.random.uniform(low=0, high=1) < self.one_drop_percentage_brake:
                            flag = False
            idx.append(flag)
            if is_horizontally_flipped[i]:
                if 'Steering' in self.y_labels:
                    batch_y[i][self.y_labels.index('Steering')] *= -1
                # batch_y[i] = batch_y[i][::-1]
                # no number before 1st : means start from index 0, no number
                # after 1st : means end at index len(), number after 2nd : is the increment, a -ve increment
                # means reverse
        batch_y = batch_y[idx]
        for index in range(0, len(batch_x), 1):
            batch_x[index] = batch_x[index][idx]
        return batch_x, batch_y  # , np.asarray(is_horizontally_flipped)[idx], np.asarray(brighten_values)[idx]

    def _get_batches_of_transformed_samples(self, index_array):
        return self.__get_indexes(index_array)
