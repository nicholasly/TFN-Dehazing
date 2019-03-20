# Codes mainly based on Xia Li's code.
# RESCAN: Recurrent Squeeze-and-Excitation Context Aggregation Net
# https://github.com/XiaLiPKU/RESCAN
import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from utils import transmission, atmospheric_light

import settings

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(40)
        self.patch_size = settings.patch_size
        self.root_dir = os.path.join(settings.data_dir, name)

        self.input_dir = os.path.join(self.root_dir, settings.input_dir)
        self.input_mat = os.listdir(self.input_dir)
        self.input_file_num = len(self.input_mat)

        self.target_dir = os.path.join(self.root_dir, settings.target_dir)
        self.target_mat = os.listdir(self.target_dir)
        self.target_file_num = len(self.target_mat)

        self.name = name


    def __len__(self):
        return self.input_file_num * 100

    def __getitem__(self, idx):
        input_file_name = self.input_mat[idx % self.input_file_num]
        input_file = os.path.join(self.input_dir, input_file_name)
        img_original = cv2.imread(input_file)
        img_original = img_original.astype(np.float32) / 255
        img_trans = transmission(img_original, omega=0.95, window=1)
        img_trans = cv2.resize(img_trans, (self.patch_size, self.patch_size))
        img_input = np.ndarray((1, img_trans.shape[0], img_trans.shape[1]))
        img_input[0] = img_trans
        img_input = np.transpose(img_input, (1, 2, 0))
        img_input = img_input.astype(np.float32)

        img_A = atmospheric_light(img_original, window=1)
        temp = np.ndarray((3, img_trans.shape[0], img_trans.shape[1]))
        temp[0] = img_A[0]
        temp[1] = img_A[1]
        temp[2] = img_A[2]
        temp = np.transpose(temp, (1, 2, 0))
        img_A = temp.astype(np.float32)

        end_idx1 = input_file_name.find('_')

        if (self.name == 'train'):
            if input_file_name[-3:] == 'png':
                target_file_name = input_file_name[:end_idx1] + '.png'
            elif input_file_name[-3:] == 'jpg':
                target_file_name = input_file_name[:end_idx1] + '.jpg'
        else:
            target_file_name = input_file_name[:end_idx1] + '.png'

        target_file = os.path.join(self.target_dir, target_file_name)

        img_target = cv2.imread(target_file).astype(np.float32) / 255.0

        img_target = cv2.resize(img_target, (self.patch_size, self.patch_size))
        img_original = cv2.resize(img_original, (self.patch_size, self.patch_size))
        img_A = cv2.resize(img_A, (self.patch_size, self.patch_size))
        
        img_input = np.transpose(img_input, (2, 0, 1))
        img_original = np.transpose(img_original, (2, 0, 1))
        img_A = np.transpose(img_A, (2, 0, 1))
        img_target = np.transpose(img_target, (2, 0, 1))

        sample = {'img_input': img_input, 'img_target': img_target, 'img_A': img_A, 'img_original': img_original}

        return sample

class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(23)
        self.root_dir = os.path.join(settings.data_dir, name)

        self.input_dir = os.path.join(self.root_dir, settings.input_dir)
        self.input_mat = os.listdir(self.input_dir)
        self.input_file_num = len(self.input_mat)


        self.target_dir = os.path.join(self.root_dir, settings.target_dir)
        self.target_mat = os.listdir(self.target_dir)
        self.target_file_num = len(self.target_mat)

        self.patch_size = settings.patch_size

    def __len__(self):
        return self.input_file_num

    def __getitem__(self, idx):
        input_file_name = self.input_mat[idx % self.input_file_num]
        input_file = os.path.join(self.input_dir, input_file_name)
        img_original = cv2.imread(input_file)
        img_original = img_original.astype(np.float32) / 255
        img_trans = transmission(img_original, omega=0.95, window=1)
        img_input = np.ndarray((1, img_trans.shape[0], img_trans.shape[1]))
        img_input[0] = img_trans
        img_input = np.transpose(img_input, (1, 2, 0))
        img_input = img_input.astype(np.float32)

        img_A = atmospheric_light(img_original, window=1)
        temp = np.ndarray((3, img_trans.shape[0], img_trans.shape[1]))
        temp[0] = img_A[0]
        temp[1] = img_A[1]
        temp[2] = img_A[2]
        temp = np.transpose(temp, (1, 2, 0))
        img_A = temp.astype(np.float32)

        end_idx1 = input_file_name.find('_')
        target_file_name = input_file_name[:end_idx1] + '.png'
        target_file = os.path.join(self.target_dir, target_file_name)
        img_target = cv2.imread(target_file).astype(np.float32) / 255

        img_input = np.transpose(img_input, (2, 0, 1))
        img_original = np.transpose(img_original, (2, 0, 1))
        img_A = np.transpose(img_A, (2, 0, 1))
        img_target = np.transpose(img_target, (2, 0, 1))
        sample = {'img_input': img_input, 'img_target': img_target, 'img_A': img_A, 'img_original': img_original}

        return sample
