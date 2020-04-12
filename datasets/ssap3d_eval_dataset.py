import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2])


class SSAP3DEvalDataset(Dataset):
    def __init__(self, dataset_path, img_wh, bbox_scale_factor=1.2,
                 path_correction=False):
        super(SSAP3DEvalDataset, self).__init__()

        # Data
        data = np.load(os.path.join(dataset_path, 'sports_videos_eval.npz'))

        self.frame_paths = data['frames_paths']
        self.vertices = data['vertices']
        self.body_shapes = data['shapes']
        self.genders = data['genders']
        self.bbox_centres = data['centres']  # Tight bounding box centre
        self.bbox_whs = data['whs']  # Tight bounding box width/height

        assert len(self.frame_paths) == len(self.vertices) == len(self.body_shapes) == len(self.genders)

        self.img_wh = img_wh
        self.bbox_scale_factor = bbox_scale_factor

        self.path_correction = path_correction

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Input image - need to crop and resize because annoyingly, for sports videos dataset,
        # the frames in the cropped_frames dir are not actually tightly cropped frames.
        frame_path = self.frame_paths[index]
        if self.path_correction:
            frame_path = frame_path.replace('/scratch2/', '/scratch/')
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        bbox_centre = self.bbox_centres[index]
        bbox_wh = self.bbox_whs[index] * self.bbox_scale_factor
        bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, bbox_wh, bbox_wh)
        top_left = bbox_corners[:2].astype(np.int16)
        bottom_right = bbox_corners[2:].astype(np.int16)
        top_left[top_left < 0] = 0
        bottom_right[bottom_right < 0] = 0
        img = img[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        img = cv2.resize(img, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        input = np.transpose(img, [2, 0, 1])/255.0

        # Targets
        vertices = self.vertices[index]
        shape = self.body_shapes[index]
        gender = self.genders[index]

        input = torch.from_numpy(input).float()
        vertices = torch.from_numpy(vertices).float()
        shape = torch.from_numpy(shape).float()

        return {'input': input,
                'vis_img': img,
                'shape': shape,
                'vertices': vertices,
                'frame_path': frame_path,
                'gender': gender}
