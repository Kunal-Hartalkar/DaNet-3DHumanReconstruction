from __future__ import absolute_import
from __future__ import division

import argparse
import os
import logging
from easydict import EasyDict
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')

from matplotlib.image import imsave
from matplotlib import pyplot as plt

import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from smplx import SMPL

import lib.nn as mynn
import lib.utils.net as net_utils
from lib.core.config import cfg, cfg_from_file
from lib.modeling.danet import DaNet
from lib.utils.logging import setup_logging
from skimage.transform import resize
from lib.utils.iuvmap import iuv_map2img
from datasets.ssap3d_eval_dataset import SSAP3DEvalDataset
from cam_utils import orthographic_project_torch, undo_keypoint_normalisation, \
    rotate_translate_verts_torch
from eval_utils import scale_and_translation_transform_batch, compute_similarity_transform_batch

# Set up logging and load config options
logger = setup_logging(__name__)

SMPL_MODEL_DIR = 'data/smpl_from_lib'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='DaNet for 3D Human Shape and Pose')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        default='configs/danet_demo.yaml',
        help='config file for training / testing')
    parser.add_argument(
        '--load_ckpt',
        help='checkpoint path to load',
        default='./data/pretrained_model/danet_model_h36m_cocodp.pth')

    parser.add_argument('--gpu', type=str)
    parser.add_argument('--img_wh', default=224, type=int)
    parser.add_argument('--path_correction', action='store_true')

    return parser.parse_args()


def evaluate_single_in_multitasknet_ssap3d(model,
                                           eval_dataset,
                                           metrics,
                                           device,
                                           save_path,
                                           num_workers=4,
                                           pin_memory=True,
                                           vis_every_n_batches=1000):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1)
    smpl_male = SMPL(SMPL_MODEL_DIR, batch_size=1, gender='male')
    smpl_female = SMPL(SMPL_MODEL_DIR, batch_size=1, gender='female')
    smpl.to(device)
    smpl_male.to(device)
    smpl_female.to(device)

    if 'pve' in metrics:
        pve_sum = 0.0
        pve_per_frame = []

    if 'pve_scale_corrected' in metrics:
        pve_scale_corrected_sum = 0.0
        pve_scale_corrected_per_frame = []

    if 'pve_pa' in metrics:
        pve_pa_sum = 0.0
        pve_pa_per_frame = []

    if 'pve-t' in metrics:
        pvet_sum = 0.0
        pvet_per_frame = []

    if 'pve-t_scale_corrected' in metrics:
        pvet_scale_corrected_sum = 0.0
        pvet_scale_corrected_per_frame = []

    frame_path_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []
    num_samples = 0
    num_vertices = 6890

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input']
        input = input.to(device)

        target_shape = samples_batch['shape']
        target_shape = target_shape.to(device)
        target_vertices = samples_batch['vertices']

        target_gender = samples_batch['gender'][0]
        if target_gender == 'm':
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_reposed_smpl_output = smpl_female(betas=target_shape)
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # ------------------------------- PREDICTIONS -------------------------------
        pred_results = model.infer_net(input)  # dict with keys 'visualisation' and 'para'
        para_pred = pred_results['para']
        pred_camera = para_pred[:, 0:3].contiguous()
        pred_betas = para_pred[:, 3:13].contiguous()
        pred_rotmat = para_pred[:, 13:].contiguous().view(-1, 24, 3, 3)
        # For some reason using the official SMPL implementation gives upside down results
        # Use this instead...
        smpl_pts = model.iuv2smpl.smpl(pred_betas, Rs=pred_rotmat, get_skin=True)
        pred_vertices = smpl_pts['verts']  # (1, 6890, 3)
        pred_vertices_projected2d = orthographic_project_torch(pred_vertices, pred_camera)
        pred_vertices_projected2d = undo_keypoint_normalisation(pred_vertices_projected2d,
                                                                input.shape[-1])

        pred_reposed_smpl_output = smpl(betas=pred_betas)
        pred_reposed_vertices = pred_reposed_smpl_output.vertices


        # Numpy-fying
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_vertices_projected2d = pred_vertices_projected2d.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
        pred_rotmat = pred_rotmat.cpu().detach().numpy()
        pred_betas = pred_betas.cpu().detach().numpy()
        pred_camera = pred_camera.cpu().detach().numpy()


        # ------------------------------- METRICS -------------------------------
        if 'pve' in metrics:
            pve_batch = np.linalg.norm(pred_vertices - target_vertices,
                                       axis=-1)  # (bs, 6890)
            pve_sum += np.sum(pve_batch)  # scalar
            pve_per_frame.append(np.mean(pve_batch, axis=-1))

        # Scale and translation correction
        if 'pve_scale_corrected' in metrics:
            pred_vertices_scale_corrected = scale_and_translation_transform_batch(pred_vertices,
                                                                                  target_vertices)
            pve_scale_corrected_batch = np.linalg.norm(pred_vertices_scale_corrected - target_vertices,
                                                       axis=-1)  # (bs, 6890)
            pve_scale_corrected_sum += np.sum(pve_scale_corrected_batch)  # scalar
            pve_scale_corrected_per_frame.append(np.mean(pve_scale_corrected_batch, axis=-1))

        # Procrustes analysis
        if 'pve_pa' in metrics:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bs, 6890)
            pve_pa_sum += np.sum(pve_pa_batch)  # scalar
            pve_pa_per_frame.append(np.mean(pve_pa_batch, axis=-1))

        if 'pve-t' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            pvet_sum += np.sum(pvet_batch)
            pvet_per_frame.append(np.mean(pvet_batch, axis=-1))

        # Scale and translation correction
        if 'pve-t_scale_corrected' in metrics:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                             target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices,
                                                        axis=-1)  # (bs, 6890)
            pvet_scale_corrected_sum += np.sum(pvet_scale_corrected_batch)  # scalar
            pvet_scale_corrected_per_frame.append(np.mean(pvet_scale_corrected_batch, axis=-1))

        num_samples += target_shape.shape[0]
        frame_path = samples_batch['frame_path']
        frame_path_per_frame.append(frame_path)
        pose_per_frame.append(pred_rotmat)
        shape_per_frame.append(pred_betas)
        cam_per_frame.append(pred_camera)

        # ------------------------------- VISUALISE -------------------------------
        if vis_every_n_batches is not None:
            if batch_num % vis_every_n_batches == 0:
                vis_img = samples_batch['vis_img'][0]

                # estimated global IUV
                ones_np = np.ones(vis_img.shape[:2]) * 255
                ones_np = ones_np[:, :, None]
                global_iuv = iuv_map2img(*pred_results['visualization']['iuv_pred'])[0].cpu().numpy()
                global_iuv = np.transpose(global_iuv, (1, 2, 0))
                global_iuv = resize(global_iuv, vis_img.shape[:2])
                global_iuv_rgba = np.concatenate((global_iuv, ones_np), axis=2)

                plt.figure(figsize=(16, 12))
                plt.subplot(341)
                plt.imshow(vis_img)

                plt.subplot(342)
                plt.imshow(global_iuv_rgba)

                plt.subplot(343)
                plt.imshow(vis_img)
                plt.scatter(pred_vertices_projected2d[0, :, 0], pred_vertices_projected2d[0, :, 1], s=0.1, c='r')

                plt.subplot(345)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(346)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1,
                            c='b')
                plt.scatter(pred_vertices_scale_corrected[0, :, 0],
                            pred_vertices_scale_corrected[0, :, 1], s=0.1,
                            c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(347)
                plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_vertices_pa[0, :, 0], pred_vertices_pa[0, :, 1], s=0.1, c='r')
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                plt.subplot(348)
                plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
                plt.scatter(pred_reposed_vertices_sc[0, :, 0], pred_reposed_vertices_sc[0, :, 1], s=0.1, c='r')
                plt.gca().set_aspect('equal', adjustable='box')

                # plt.show()
                split_path = frame_path[0].split('/')
                clip_name = split_path[-3]
                frame_num = split_path[-1]
                save_fig_path = os.path.join(save_path, clip_name + '_' + frame_num)
                plt.savefig(save_fig_path, bbox_inches='tight')
                plt.close()

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    frame_path_per_frame = np.concatenate(frame_path_per_frame, axis=0)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), frame_path_per_frame)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)

    if 'pve' in metrics:
        pve = pve_sum / (num_samples * num_vertices)
        pve_per_frame = np.concatenate(pve_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_per_frame.npy'), pve_per_frame)
        print('PVE: {:.5f}'.format(pve))

    if 'pve_scale_corrected' in metrics:
        pve_scale_corrected = pve_scale_corrected_sum / (num_samples * num_vertices)
        pve_scale_corrected_per_frame = np.concatenate(pve_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_scale_corrected_per_frame.npy'),
                pve_scale_corrected_per_frame)
        print('PVE SC: {:.5f}'.format(pve_scale_corrected))

    if 'pve_pa' in metrics:
        pve_pa = pve_pa_sum / (num_samples * num_vertices)
        pve_pa_per_frame = np.concatenate(pve_pa_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_pa_per_frame.npy'), pve_pa_per_frame)
        print('PVE PA: {:.5f}'.format(pve_pa))

    if 'pve-t' in metrics:
        pvet = pvet_sum / (num_samples * num_vertices)
        pvet_per_frame = np.concatenate(pvet_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_per_frame.npy'), pvet_per_frame)
        print('PVE-T: {:.5f}'.format(pvet))

    if 'pve-t_scale_corrected' in metrics:
        pvet_scale_corrected = pvet_scale_corrected_sum / (num_samples * num_vertices)
        pvet_scale_corrected_per_frame = np.concatenate(pvet_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_scale_corrected_per_frame.npy'), pvet_scale_corrected_per_frame)
        print('PVE-T SC: {:.5f}'.format(pvet_scale_corrected))


def main():
    """Main function"""
    args = parse_args()

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cfg_from_file(args.cfg_file)
    cfg.DANET.REFINEMENT = EasyDict(cfg.DANET.REFINEMENT)
    cfg.MSRES_MODEL.EXTRA = EasyDict(cfg.MSRES_MODEL.EXTRA)

    # Model
    model = DaNet().to(device)

    # Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint)
        del checkpoint

    model.eval()

    dataset_path = '/scratches/nazgul_2/as2562/datasets/sports_videos_smpl/final_dataset'
    dataset = SSAP3DEvalDataset(dataset_path, img_wh=args.img_wh,
                                path_correction=args.path_correction)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pve', 'pve_scale_corrected', 'pve_pa', 'pve-t', 'pve-t_scale_corrected']

    save_path = '/data/cvfs/as2562/DaNet-3DHumanReconstruction/evaluations/ssap3d'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    evaluate_single_in_multitasknet_ssap3d(model=model,
                                           eval_dataset=dataset,
                                           metrics=metrics,
                                           device=device,
                                           save_path=save_path,
                                           vis_every_n_batches=1)

if __name__ == '__main__':
    main()
