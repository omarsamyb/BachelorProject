import random
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os
import errno
import h5py
import airsim


def checkAndCreateDir(full_path):
    """
    Description:
        Checks if a given path exists and if not, creates the needed directories.
    Inputs:
        full_path: path to be checked
    """

    if not os.path.exists(os.path.dirname(full_path)):
        try:
            os.makedirs(os.path.dirname(full_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def readImagesFromPath(image_paths):
    """
    Description:
        Takes in a list of lists of len chunk_size where each list contains image file paths to be loaded and returns
        a list of lists of all loaded images after resize.
    Inputs:
        image_names: list of lists of image paths
    Returns:
        all_image_values: List of lists of all loaded images
        image_types: List of strings of image type names
    """

    all_image_values = []
    image_types = []
    image_types_added = False
    for paths_list in image_paths:
        for index, image_path in enumerate(paths_list):
            if not image_types_added:
                image_name_start_index = len(image_path) - image_path[::-1].index('/')
                image_name = image_path[image_name_start_index:image_name_start_index + 7]
                image_types.append(image_name)
                all_image_values.append([])
            if str.endswith(image_path, '.png'):
                imgArr = np.asarray(Image.open(image_path))
                # Remove alpha channel if exists
                if len(imgArr.shape) == 3 and imgArr.shape[2] == 4:
                    if np.all(imgArr[:, :, 3] == imgArr[0, 0, 3]):
                        imgArr = imgArr[:, :, 0:3]
                if len(imgArr.shape) != 3 or imgArr.shape[2] != 3:
                    print('Error: Image', image_path, 'is not RGB.')
                    sys.exit()
                all_image_values[index].append(np.asarray(imgArr))
            elif str.endswith(image_path, '.pfm'):
                all_image_values[index].append(np.asarray(airsim.read_pfm(image_path)[0]))
        image_types_added = True
    return all_image_values, image_types


def generateDataMapAirSim(folders, required_previous_state, required_label):
    """
    Description:
        Data map generator for simulator(AirSim) data. Reads the driving_log csv file and returns a list of tuples
        of key<string> 'center camera of type scene image name' & value<tuple> 'current label<list>,
        previous_state<list>, image_filepaths<list><string>'
    Inputs:
        folders: list of folders paths to collect data from
        required_previous_state: list of names of to be included previous states
        required_label: list of names of to be included labels
    Returns:
        mappings: All data mappings as a list of dictionaries. Key is the front_center of type scene image filepath,
        the values are a 3-tuple:
            0 -> label(s) as a list of double
            1 -> previous state as a list of double
            2 -> images file paths as a list of string
            Ex: [(img[0], ([steering, throttle, brake], [prev_steering, prev_throttle, prev_brake, prev_speed],
                [img[0], img[1], img[2], img[3]]))] such that img[] could be any length, this Ex has 4 images
    """

    all_mappings = {}
    for folder in folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt').replace("\\", "/"), sep='\t')
        for i in range(1, current_df.shape[0] - 1, 1):
            previous_state = list(current_df.iloc[i - 1][required_previous_state])
            current_steering = list((current_df.iloc[i][['Steering']] + current_df.iloc[i - 1][['Steering']] +
                                     current_df.iloc[i + 1][['Steering']]) / 3.0)
            current_throttle = list((current_df.iloc[i][['Throttle']] + current_df.iloc[i - 1][['Throttle']] +
                                     current_df.iloc[i + 1][['Throttle']]) / 3.0)
            current_brake = list((current_df.iloc[i][['Brake']] + current_df.iloc[i - 1][['Brake']] +
                                  current_df.iloc[i + 1][['Brake']]) / 3.0)
            current_label = []
            if 'Steering' in required_label:
                current_label += current_steering
            if 'Throttle' in required_label:
                current_label += current_throttle
            if 'Brake' in required_label:
                current_label += current_brake
            image_names = current_df.iloc[i]['ImageFile'].split(";")
            image_paths = []
            for image_name in image_names:
                image_path = os.path.join(os.path.join(folder, 'images'), image_name).replace('\\', '/')
                image_paths.append(image_path)
            # Sanity check
            if image_paths[0] in all_mappings:
                print('Error: attempting to add image {0} twice.'.format(image_paths[0]))
            all_mappings[image_paths[0]] = (current_label, previous_state, image_paths)
    mappings = [(key, all_mappings[key]) for key in all_mappings]
    random.shuffle(mappings)
    return mappings


def splitTrainValidationAndTestData(all_data_mappings, split_ratio=(0.7, 0.2, 0.1)):
    """
    Description:
        Simple function to create train, validation and test splits on the data.
    Inputs:
        all_data_mappings: mappings from the entire dataset
        split_ratio: (train, validation, test) split ratio
    Returns:
        a list of 3 lists:
            train_data_mappings: list of mappings for training data
            validation_data_mappings: list of mappings for validation data
            test_data_mappings: list of mappings for test data
    """

    if round(sum(split_ratio), 5) != 1.0:
        print("Error: Your splitting ratio should add up to 1")
        sys.exit()
    train_split = int(len(all_data_mappings) * split_ratio[0])
    val_split = train_split + int(len(all_data_mappings) * split_ratio[1])
    train_data_mappings = all_data_mappings[0:train_split]
    validation_data_mappings = all_data_mappings[train_split:val_split]
    test_data_mappings = all_data_mappings[val_split:]

    return [train_data_mappings, validation_data_mappings, test_data_mappings]


def generatorForH5py(data_mappings, chunk_size=32):
    """
    Description:
        This helper function batches the data for saving to the H5 file,
        Data is expected to be a dict of (image, (label, previous_state, images))
    Input:
        data_mappings: a list of tuples of data
        chunk_size: an int to identify how many row are processed at a time.
    Yields:
        a tuple of 3 each is a list of len chunk_size ([[labels]], [[prev_state]], [[image_paths]]
    """
    for chunk_id in range(0, len(data_mappings), chunk_size):
        # Extract the parts
        data_chunk = data_mappings[chunk_id:chunk_id + chunk_size]
        # chunk_id + chunk_size can exceed len of mappings if len is not a multiple of chunk_size which is handled below
        # We will only consider equal sized chunks of len = chunk_size
        if len(data_chunk) == chunk_size:
            labels_chunk = np.asarray([b[0] for (a, b) in data_chunk])
            previous_state_chunk = np.asarray([b[1] for (a, b) in data_chunk])
            image_paths_chunk = np.asarray([b[2] for (a, b) in data_chunk])
            # Flatten and yield as tuple
            yield labels_chunk.astype(float), previous_state_chunk.astype(float), image_paths_chunk.astype(str)
            # will automatically raise a StopIteration when gen stops reaching yield


def saveH5pyData(data_mappings, target_file_path, chunk_size=32):
    """
    Description:
        Saves H5 data to file
    Inputs:
        data_mappings: a list of mapping tuples of key image data paths and value a corresponding
        (current_label, prev_state, img_paths)
        target_file_path: target path to save the file  as .H5
        chunk_size: size of chunk
    """

    gen = generatorForH5py(data_mappings, chunk_size)
    labels_chunk, previous_state_chunk, image_paths_chunk = next(gen)
    image_values_chunk, image_types = readImagesFromPath(image_paths_chunk)
    row_count = labels_chunk.shape[0]
    checkAndCreateDir(target_file_path)
    with h5py.File(target_file_path, 'w') as f:
        # Initialize a resizable dataset to hold the output
        labels_chunk_max_shape = (None,) + labels_chunk.shape[1:]
        previous_state_chunk_max_shape = (None,) + previous_state_chunk.shape[1:]
        dataset_label = f.create_dataset('label', shape=labels_chunk.shape, maxshape=labels_chunk_max_shape,
                                         chunks=labels_chunk.shape, dtype=labels_chunk.dtype)
        dataset_previous_state = f.create_dataset('previous_state', shape=previous_state_chunk.shape,
                                                  maxshape=previous_state_chunk_max_shape,
                                                  chunks=previous_state_chunk.shape, dtype=previous_state_chunk.dtype)
        dataset_label[:] = labels_chunk
        dataset_previous_state[:] = previous_state_chunk

        images_dataset = []
        for image_type_index, image_type_chunk in enumerate(image_values_chunk):
            image_type_chunk = np.asarray(image_type_chunk)
            image_max_shape = (None,) + image_type_chunk.shape[1:]
            dataset_image = f.create_dataset(image_types[image_type_index], shape=image_type_chunk.shape,
                                             maxshape=image_max_shape,
                                             chunks=image_type_chunk.shape,
                                             dtype=image_type_chunk.dtype)
            dataset_image[:] = image_type_chunk
            images_dataset.append(dataset_image)

        for labels_chunk, previous_state_chunk, image_paths_chunk in gen:
            image_values_chunk, image_types = readImagesFromPath(image_paths_chunk)
            # Resize the dataset to accommodate the next chunk of rows
            dataset_label.resize(row_count + labels_chunk.shape[0], axis=0)
            dataset_previous_state.resize(row_count + previous_state_chunk.shape[0], axis=0)
            # Write the next chunk
            dataset_label[row_count:] = labels_chunk
            dataset_previous_state[row_count:] = previous_state_chunk
            for image_type_index, image_type_chunk in enumerate(image_values_chunk):
                image_type_chunk = np.asarray(image_type_chunk)
                images_dataset[image_type_index].resize(row_count + image_type_chunk.shape[0], axis=0)
                images_dataset[image_type_index][row_count:] = image_type_chunk
            # Increment the row count
            row_count += labels_chunk.shape[0]


def cook(folders, output_directory, train_eval_test_split, chunk_size=32,
         previous_state=['Steering', 'Throttle', 'Brake', 'Speed'], label=['Steering', 'Throttle', 'Brake']):
    """
    Description:
        Primary function for data pre-processing. Reads and saves all data as h5 files.
    Inputs:
        folders: a list of all data folders
        output_directory: location for saving h5 files
        train_eval_test_split: dataset split ratio
    """

    output_files = [os.path.join(output_directory, f).replace("\\", "/") for f in ['train.h5', 'eval.h5', 'test.h5']]
    if any([os.path.isfile(f) for f in output_files]):
        print("Preprocessed data already exists at: {0}. Skipping pre-processing.".format(output_directory))
    else:
        all_data_mappings = generateDataMapAirSim(folders, previous_state, label)
        split_mappings = splitTrainValidationAndTestData(all_data_mappings, split_ratio=train_eval_test_split)
        for i in range(0, len(split_mappings), 1):
            print('Processing {0}...'.format(output_files[i]))
            saveH5pyData(split_mappings[i], output_files[i], chunk_size=chunk_size)
            print('Finished saving {0}.'.format(output_files[i]))
