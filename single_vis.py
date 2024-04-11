import torch
from torch.utils.data import DataLoader
from collections import namedtuple
import sys
from datasets import get_dataset, load_images
from utils import instrinsics2focals, instrinsics2principal, eval_focal_or_principal, get_models, gt_intrinsics_scale, AverageMeter, get_3D_model_from_scene
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
    parser.add_argument('--single', type=bool, default=True)
    parser.add_argument('--img_name', type=str, default='test.png')
    
    return parser.parse_args()


def get_single_scene(model, img_path, args):
    model.eval()
    images = load_images([img_path, img_path], args)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, args.device, batch_size=1)
    if args.scene_mode == 'Pair':
        scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PairViewer)
    elif args.scene_mode == 'PointCloud':
        scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        _ = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
    else:
        raise ValueError('scene_mode should be Pair or PointCloud')
    return scene

def reconstruct_3d_points(intrinsics, depths, args):
    focals = instrinsics2focals(intrinsics)
    principals = instrinsics2principal(intrinsics)
    
    recon_pts3d = []
    for focal, principal, depth in tqdm(zip(focals, principals, depths), desc='Reconstruct 3D points', colour='#0396ff'):
        x, y = torch.meshgrid(torch.arange(args.height), torch.arange(args.width), indexing='ij')
        x = x.float() - principal[0]
        y = y.float() - principal[1]
        camera_x = - x * depth / focal[0] 
        camera_y = y * depth / focal[1]
        camera_z = depth
        camera_pts = torch.stack([camera_x, camera_y, camera_z], dim=-1)
        recon_pts3d.append(camera_pts)
    return recon_pts3d


def to_device(aaa, d):
    bbb = []
    for a in aaa:
        a = a.to(d)
        bbb.append(a)
    return bbb

def vis_single_img(args=None):
    img_dir = './examples/'
    assert args != None, 'args should not be None'
    model = get_models('dust3r', args)
    scene = get_single_scene(model, img_dir + args.img_name, args)
    # print(len(scene.get_pts3d()))
    depths = to_device(scene.get_depthmaps(), 'cpu')
    intrinsics = to_device(scene.get_intrinsics(), 'cpu')
    # print(intrinsics.shape)
    # depths = [depths[0].cpu().numpy(), depths[1].cpu().numpy()]
    # dust3r = reconstruct_3d_points(scene.get_intrinsics().cpu().numpy(), depths, args)
    
    my_pts3d = reconstruct_3d_points(intrinsics, depths, args)
    my_pts3d = to_device(my_pts3d, 'cuda')
    pts3d = scene.get_pts3d()
    # depth2pts3d = scene.depth_to_pts3d()[0].cpu().numpy()
    # print(scene.imgs[0].shape)
    # print(pts3d.shape)
    # print(depth2pts3d.shape)
    # print(my_pts3d-depth2pts3d)
    
    # exit()
    
    glb_ori = get_3D_model_from_scene('./3D_ori/{}-ori.glb'.format(args.img_name.split(".")[0]), scene, pts3d)
    glb_refine = get_3D_model_from_scene('./3D_refine/{}-refine.glb'.format(args.img_name.split(".")[0]), scene, my_pts3d)
    return glb_ori, glb_refine
        
    

def vis_whole_dataset():
    args = get_params()
    
    # 获取数据    
    dataset = get_dataset(args)
    image_paths = dataset.load_image_paths()
    
    gt_intrinsics = dataset.load_intrinsics()
    # if args.dataset != 'KITTI':
    gt_intrinsics = gt_intrinsics_scale(gt_intrinsics, args)
    gt_focals = instrinsics2focals(gt_intrinsics)
    gt_principals = instrinsics2principal(gt_intrinsics)
    
    dust3r = get_models('dust3r', args)
    
    # test_loader = DataLoader(list(zip(image_paths, gt_intrinsics)), batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    focal_avg = AverageMeter()
    principal_avg = AverageMeter()
    vis_tqdm = tqdm(zip(image_paths, gt_focals, gt_principals), desc='{}-{}'.format(args.dataset,args.focal_mode), colour='#0396ff')
    for img_path, gt_focal, gt_principal in vis_tqdm:
        
        # 通过dust3r得到scene
        scene = get_single_scene(dust3r, img_path, args)
        glb_ori = get_3D_model_from_scene('./3D_ori/{}.glb'.format(img_path.split(".")[0]), scene)
        glb_refine = get_3D_model_from_scene('./3D_refine/{}.glb'.format(img_path.split(".")[0]), scene, refine=True)
        
        pred_intrinsic = scene.get_intrinsics().cpu().numpy()[0]
        
        # 计算内参误差
        focal_avg.update(eval_focal_or_principal(instrinsics2focals(pred_intrinsic), gt_focal), len(img_path))
        principal_avg.update(eval_focal_or_principal(instrinsics2principal(pred_intrinsic), gt_principal), len(img_path))
        
        
        # 更新batch的误差
        vis_tqdm.set_postfix(f=max(focal_avg.get_average()), b=max(principal_avg.get_average()))
    
    print('focal_error:', focal_avg.get_average())
    print('principal_error:', principal_avg.get_average())
    
    
    
    
if __name__ == "__main__":
    args = get_params()
    if args.single:
        glb_ori, glb_refine = vis_single_img(args)
    else:
        vis_whole_dataset()
        
    
    

        