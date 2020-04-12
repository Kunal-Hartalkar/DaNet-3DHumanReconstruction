import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


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

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Inputs
        fname = self.frame_fnames[index]
        frame_path = os.path.join(self.cropped_frames_dir, fname)

        img = Image.open(frame_path).convert('RGB')  # (W, H, 3) - but is PIL Image and not array
        img = img.resize((self.img_wh, self.img_wh))
        input = ToTensor()(img)  # (3, 224, 224)
        vis_img = np.array(img).astype(np.float32)/255.0

        # Targets
        joints3d = self.joints3d[index]
        pose = self.pose[index]
        shape = self.shape[index]

        joints3d = torch.from_numpy(joints3d).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        return {'input': input,
                'vis_img': vis_img,
                'target_j3d': joints3d,
                'pose': pose,
                'shape': shape,
                'fname': fname}








