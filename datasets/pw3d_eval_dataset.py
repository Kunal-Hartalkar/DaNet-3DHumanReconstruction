import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


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

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Inputs
        fname = self.frame_fnames[index]
        frame_path = os.path.join(self.cropped_frames_dir, fname)

        img = Image.open(frame_path).convert(
            'RGB')  # (W, H, 3) - but is PIL Image and not array
        img = img.resize((self.img_wh, self.img_wh))
        input = ToTensor()(img)  # (3, 224, 224)
        vis_img = np.array(img).astype(np.uint8)

        # Targets
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]

        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        return {'input': input,
                'vis_img': vis_img,
                'pose': pose,
                'shape': shape,
                'fname': fname,
                'gender': gender}








