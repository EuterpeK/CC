import torch
from torch.utils.data import DataLoader
from collections import namedtuple
import sys
from datasets import get_dataset, load_images
from utils import instrinsics2focals, instrinsics2principal, eval_focal_or_principal, get_models, gt_intrinsics_scale, AverageMeter
sys.path.append("../dust3r/")
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dust3r.post_process import estimate_focal_knowing_depth

from torchvision.transforms import Compose
from tqdm.auto import tqdm
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES']='1'
# sys.path.append('../vivo/0-MyCode/')
# from utils import ignore_warnings

# ignore_warnings()
from transformers import logging
logging.set_verbosity_error()
import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dust3r_model_path', type=str, default='../0-Pretrained/Dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--focal_mode', type=str, default='weiszfeld')
    parser.add_argument('--data_root', type=str, default='../0-datasets')
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='KITTI', choices=['KITTI', 'NYUv2', 'CO3D', 'SUN3D', 'Waymo', 'ScanNet', 'ARKitScenes', 'NUScenes', 'Cityscapes'])
    parser.add_argument('--scene_mode', type=str, default='Pair', choices=['Pair', 'PointCloud'])
    parser.add_argument('--width', type=int, default=448)
    parser.add_argument('--height', type=int, default=448)
    # parser.add_argument('--method', type=str, default='dust3r')
    
    return parser.parse_args()


def infer_intrinsics_and_depths(model, imgs, args):
    model.eval()
    instrinsics = []
    depths = []
    outputs = []
    # if os.path.exists('./pred_pts3d/{}-{}-{}.pt'.format(args.dataset, args.width, args.height)):
    #     outputs = torch.load('./pred_pts3d/{}-{}-{}.pt'.format(args.dataset, args.width, args.height))
    #     print('{}-{}-{}.pt already exists!'.format(args.dataset, args.width, args.height))
    # else:
    for img in tqdm(imgs, colour='#0396ff', desc='inference batch {}-{}'.format(args.dataset,args.focal_mode), leave=False):
        images = load_images([img, img], args)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        with torch.no_grad():
            output = inference(pairs, model, args.device, batch_size=1)
        outputs.append(output)
        # torch.save(outputs, './pred_pts3d/{}-{}-{}.pt'.format(args.dataset, args.width, args.height))
    for output in tqdm(outputs, colour='#0396ff', desc='post_process', leave=False):
        if args.scene_mode == 'Pair':
            scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PairViewer)
        elif args.scene_mode == 'PointCloud':
            scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
            _ = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
        else:
            raise ValueError('scene_mode should be Pair or PointCloud')
        instrinsics.append(scene.get_intrinsics().cpu().numpy()[0])
        depths.append(scene.get_depthmaps()[0].cpu().numpy())
    return instrinsics, depths

def reconstruct_3d_points(intrinsics, depths, args):
    focals = instrinsics2focals(intrinsics)
    principals = instrinsics2principal(intrinsics)
    
    recon_pts3d = []
    for focal, principal, depth in tqdm(zip(focals, principals, depths), desc='Reconstruct 3D points', colour='#0396ff'):
        x, y = torch.meshgrid(torch.arange(args.width), torch.arange(args.height))
        x = x.float() - principal[0]
        y = y.float() - principal[1]
        camera_x = x / focal[0] * depth
        camera_y = y / focal[1] * depth
        camera_z = depth
        camera_pts = torch.stack([camera_x, camera_y, camera_z], dim=-1)
        recon_pts3d.append(camera_pts)
    return recon_pts3d


def pipe():
    args = get_params()
    
    # 获取数据    
    dataset = get_dataset(args)
    image_paths = dataset.load_image_paths()
    
    gt_intrinsics = dataset.load_intrinsics()
    # if args.dataset != 'KITTI':
    gt_intrinsics = gt_intrinsics_scale(gt_intrinsics, args)
    
    dust3r = get_models('dust3r', args)
    
    test_loader = DataLoader(list(zip(image_paths, gt_intrinsics)), batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    focal_avg = AverageMeter()
    principal_avg = AverageMeter()
    test_batches = tqdm(test_loader, desc='{}-{}'.format(args.dataset,args.focal_mode), colour='#0396ff')
    for img_paths, gt_intrinsic in test_batches:
        gt_focals = instrinsics2focals(gt_intrinsic)
        gt_principal = instrinsics2principal(gt_intrinsic)
        
        # 通过dust3r估计相机内参
        pred_intrinsics, depths = infer_intrinsics_and_depths(dust3r, img_paths, args)
    
        # 通过内参重建3D点云
        # recon_pts3d = reconstruct_3d_points(pred_intrinsics, depths, args)
        
        
        # 计算内参误差
        focal_avg.update(eval_focal_or_principal(instrinsics2focals(pred_intrinsics), gt_focals), len(img_paths))
        principal_avg.update(eval_focal_or_principal(instrinsics2principal(pred_intrinsics), gt_principal), len(img_paths))
        
        # 计算3D点云误差
        
        # 更新batch的误差
        test_batches.set_postfix(f=max(focal_avg.get_average()), b=max(principal_avg.get_average()))
    
    print('focal_error:', focal_avg.get_average())
    print('principal_error:', principal_avg.get_average())
    
    
    
    
if __name__ == "__main__":
    pipe()
    
    
    

        