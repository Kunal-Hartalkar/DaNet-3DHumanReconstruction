import cv2
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import constants

class H36MEvalDataset(Dataset):
    def __init__(self, h36m_dir_path, protocol, img_wh, use_subset=False):
        super(H36MEvalDataset, self).__init__()

        # Paths
        cropped_frames_dir = os.path.join(h36m_dir_path, 'cropped_frames')

        # Data
        data = np.load(os.path.join(h36m_dir_path,
                                    'h36m_with_smpl_valid_protocol{}.npz').format(str(protocol)))

        self.frame_fnames = data['imgname']
        self.joints3d = data['S']
        self.pose = data['pose']
        self.shape = data['betas']

        if use_subset:  # Evaluate on a subset of 200 samples
            all_indices = np.arange(len(self.frame_fnames))
            chosen_indices = np.random.choice(all_indices, 200, replace=False)
            self.frame_fnames = self.frame_fnames[chosen_indices]
            self.joints3d = self.joints3d[chosen_indices]
            self.pose = self.pose[chosen_indices]
            self.shape = self.shape[chosen_indices]

        self.cropped_frames_dir = cropped_frames_dir
        self.img_wh = img_wh
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)


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
        joints3d = self.joints3d[index]
        pose = self.pose[index]
        shape = self.shape[index]

        img = torch.from_numpy(img).float()
        joints3d = torch.from_numpy(joints3d).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        input = self.normalize_img(img)

        return {'input': input,
                'vis_img': img,
                'target_j3d': joints3d,
                'pose': pose,
                'shape': shape,
                'fname': fname}








