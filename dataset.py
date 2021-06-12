import os
import tensorflow as tf
import scipy.misc
import numpy as np
import imageio
import datetime
from PIL import Image
import data

class DataGenerator(object):
    def __init__(self):
        self.resize_size = (256, 256, 3)
        self.kaggle_paths = data.get_data_path(data.kaggle, data.get_log_path())
        self.wikiart_paths = data.get_data_path(data.wikiart, data.get_log_path())
        self.data_paths = self.kaggle_paths + self.wikiart_paths
        self.datasetlen = len(self.data_paths)
        self.id = 0
        print('dataset length : %s' % (self.datasetlen))
    
    def get_one_sample(self,only_n1p1=False,only_zp1=False,give_id=None):
        if self.id == self.datasetlen:
            self.id = 0
        if give_id==None:
            img_id = self.id
        else:
            img_id = give_id

        imgs = data.get_image(self.data_paths[img_id], 
                       resize_height=self.resize_size[0], 
                       resize_width=self.resize_size[1])

        if len(imgs)<=0:
            self.id +=1
            return None

        self.id += 1
        
        if only_n1p1:
            return imgs['normalized_data_n1p1']
        if only_zp1:
            return imgs['normalized_data_zp1']

        if not only_n1p1 and not only_zp1:
            # normalized_data_n1p1 : from negative 1 to positive 1 normalized
            # normalized_data_zp1 : from zero to positive 1 normalized
            return [imgs['normalized_data_n1p1'], imgs['normalized_data_zp1']]
                
    def gen_sample(self,batch_size=1):
        h,w,c = self.resize_size
        while True:
            for i in range(batch_size):
                sample = self.get_one_sample()
                while sample ==None: # not to train the images with no instance
                    sample = self.get_one_sample()           
                
                yield sample[0], sample[1]
    
    def gen_batch(self,batch_size=4):
        h,w,c = self.resize_size
        while True:
            img_negative_1_to_positive_1_batch = np.zeros((batch_size,h,w,c))
            img_zero_to_posivie_1_batch = np.zeros((batch_size,h,w,c))

            for i in range(batch_size):
                sample = self.get_one_sample()
                while sample ==None: # not to train the images with no instance
                    sample = self.get_one_sample()           
                img_negative_1_to_positive_1_batch[i] = sample[0]
                img_zero_to_posivie_1_batch[i] = sample[1]
             

            yield img_negative_1_to_positive_1_batch, img_zero_to_posivie_1_batch
                
    def gen_batch_only_n1p1(self,batch_size=4):
        h,w,c = self.resize_size
        while True:
            img_negative_1_to_positive_1_batch = np.zeros((batch_size,h,w,c))

            for i in range(batch_size):
                sample = self.get_one_sample()
                while sample ==None: # not to train the images with no instance
                    sample = self.get_one_sample(only_n1p1=True)           
                img_negative_1_to_positive_1_batch[i] = sample[0]
             
            yield img_negative_1_to_positive_1_batch
                
    def gen_batch_only_zp1(self,batch_size=4):
        h,w,c = self.resize_size
        while True:
            img_zero_to_posivie_1_batch = np.zeros((batch_size,h,w,c))

            for i in range(batch_size):
                sample = self.get_one_sample()
                while sample ==None: # not to train the images with no instance
                    sample = self.get_one_sample(only_zp1=True)           
                img_zero_to_posivie_1_batch[i] = sample[1]
             
            yield img_zero_to_posivie_1_batch
