import torch


def orthographic_project_torch(points3D, cam_params):
    """
    Scaled orthographic projection (i.e. weak perspective projection).
    Should be going from SMPL 3D coords to [-1, 1] scaled image coords.
    cam_params are [s, tx, ty]  - i.e. scaling and 2D translation.
    """
    x = points3D[:, :, 0]
    y = points3D[:, :, 1]

    # Scaling
    s = torch.unsqueeze(cam_params[:, 0], dim=1)

    # Translation
    t_x = torch.unsqueeze(cam_params[:, 1], dim=1)
    t_y = torch.unsqueeze(cam_params[:, 2], dim=1)

    u = s * x + t_x
    v = s * y + t_y

    proj_points = torch.stack([u, v], dim=-1)

    return proj_points


def undo_keypoint_normalisation(normalised_keypoints, img_wh):
    """
    Converts normalised keypoints from [-1, 1] space to pixel space i.e. [0, img_wh]
    """
    keypoints = (normalised_keypoints + 1) * (img_wh/2.0)
    return keypoints