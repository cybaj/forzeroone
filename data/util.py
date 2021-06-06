import os
import tensorflow as tf
import scipy.misc
import numpy as np
import imageio
import datetime
from PIL import Image
from .config import EXCLUDE_TRAIN, EXCLUDE_TEST

def get_log_path():
    now = datetime.datetime.now() 
    now = now.strftime('%Y-%m-%d-%H-%M-%S')

    log_path = os.path.join(__file__[:-len('util.py')], '_log', now)
    print(f'[image loader] log_path : {log_path}')
    return log_path

def imread(path, grayscale = False):
    try:
        if (grayscale):
            return imageio.imread(path, as_gray=True).astype(np.float)
        else:
            return imageio.imread(path, as_gray=False)
    except(TypeError):
        print(path)
    
def get_image(image_path, resize_height=64, 
              resize_width=64, input_height=None, 
              input_width=None, crop=True, 
              grayscale=False):
    try:
        image = imread(image_path, grayscale)
        return transform(image, input_height=input_height, input_width=input_width,
                   resize_height=resize_height, resize_width=resize_width, crop=False)
    except ValueError:
        print(" === Bad image ===")
        print(" filepath: ", image_path)
        
def search(dirname, log_path):
    fp = open(log_path, 'a') if os.path.exists(log_path) else open(log_path, 'w')
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                print(f'start {full_filename}')
                search(full_filename, log_path)
                print('finish')
            else:
                print(full_filename, file=fp)
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.jpg': 
                    get_image(full_filename, 64, 64, 64, 64, False, False)
    except PermissionError as e:
        print(f'permission error occurs {e}')
        
def get_data_path(dirname, log_path, mode='train'):
    fp = open(log_path, 'a') if os.path.exists(log_path) else open(log_path, 'w')
    try:
        filenames = os.listdir(dirname)
        paths = []
        for filename in filenames:
            full_filepath = os.path.join(dirname, filename)
            if os.path.isdir(full_filepath):
                print(f'start {full_filepath}')
                paths += get_data_path(full_filepath, log_path)
                print('finish')
            else:
                print(full_filepath, file=fp)
                ext = os.path.splitext(full_filepath)[-1]
                if ext == '.jpg': 
                    paths.append(full_filepath)
        return paths
    except PermissionError as e:
        print(f'permission error occurs {e}')

def center_crop(image, resize_height, resize_width):
    horizontal_diff = image.shape[1] // 4
    vertical_diff = image.shape[0] // 4
    box = (horizontal_diff, vertical_diff, horizontal_diff * 3, vertical_diff * 3)
    return np.array(Image.fromarray(image).crop(box).resize((resize_height, resize_width)))
    
def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, resize_height, resize_width)
    else:
        cropped_image = np.array(Image.fromarray(image).resize([resize_height, resize_width]))
        
    # normalized_data_n1p1 : from minus 1 to plus 1 normalized
    # normalized_data_zp1 : from zero to plus 1 normalized
    return {
        'normalized_data_n1p1': np.array(cropped_image)/127.5 - 1., 
        'normalized_data_zp1': np.array(cropped_image) / 256
    }