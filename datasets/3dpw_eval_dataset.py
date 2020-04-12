import cv2
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import constants


class PW3DEvalDataset(Dataset):
    def __init__(self, pw3d_dir_path, img_wh):
        super(PW3DEvalDataset, self).__init__()

        # Paths
        cropped_frames_dir = os.path.join(pw3d_dir_path, 'cropped_frames')

        # Data
        data = np.load(os.path.join(pw3d_dir_path, '3dpw_test.npz'))
        self.frame_fnames = data['imgname']
        self.pose = data['pose']
        self.shape = data['shape']
        self.gender = data['gender']

        self.cropped_frames_dir = cropped_frames_dir
        self.img_wh = img_wh
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                                       std=constants.IMG_NORM_STD)

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Inputs
        fname = self.frame_fnames[index]
        frame_path = os.path.join(self.cropped_frames_dir, fname)

        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1])/255.0

        # Targets
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]

        img = torch.from_numpy(img).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        input = self.normalize_img(img)

        return {'input': input,
                'vis_img': img,
                'pose': pose,
                'shape': shape,
                'fname': fname,
                'gender': gender}








