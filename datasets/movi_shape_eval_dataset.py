import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MoViShapeEvalDataset(Dataset):
    def __init__(self, movi_dir_path, img_wh, capture_round='f_pg1'):
        super(MoViShapeEvalDataset, self).__init__()

        # Paths
        cropped_frames_dir = os.path.join(movi_dir_path, 'cropped_frames')

        # Data
        data = np.load(os.path.join(movi_dir_path, 'movi_{}_eval.npz'.format(capture_round)))
        self.frame_fnames = data['imgname']
        self.shape = data['shape']
        self.genders = data['gender']

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
        shape = self.shape[index]
        gender = self.genders[index]

        shape = torch.from_numpy(shape).float()

        return {'input': input,
                'vis_img': vis_img,
                'shape': shape,
                'fname': fname,
                'gender': gender}

