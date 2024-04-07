import torch
from torch.utils.data import DataLoader
from collections import namedtuple
import sys
from datasets import get_dataset, DPDataset, load_dust3r_images
from utils import estimate_focal_knowing_depth
sys.path.append("../dust3r/")

from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
# from dust3r.post_process import estimate_focal_knowing_depth

sys.path.append("../depthanything/")
from depth_anything.dpt import DepthAnything
import cv2
from torchvision.transforms import Compose
from tqdm.auto import tqdm
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES']='1'
# sys.path.append('../vivo/0-MyCode/')
# from utils import ignore_warnings

# ignore_warnings()
from transformers import logging
logging.set_verbosity_error()
import argparse


def get_params():
    parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    parser.add_argument('--dust3r_model_path', type=str, default='../0-Pretrained/Dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--depthanything_model_path', type=str, default='../0-Pretrained/DepthAnything/pytorch_model.bin')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--image_path', type=str, default='dust3r/croco/assets/Chateau1.png')
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--focal_mode', type=str, default='weiszfeld')
    parser.add_argument('--dataset', type=str, default='KITTI', choices=['KITTI', 'NYUv2', 'CO3D', 'SUN3D', 'Waymo', 'ScanNet', 'ARKit'])
    parser.add_argument('--test_depthanything', type=bool, default=True)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    return parser.parse_args()


def get_models(model_name, args):
    if model_name == 'dust3r':
        dust3r = load_model(args.dust3r_model_path, args.device)
        return dust3r
    elif model_name == 'depthanything':
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        depthanything = DepthAnything(model_configs[args.encoder])
        depthanything.load_state_dict(torch.load(args.depthanything_model_path))
        depthanything.to(args.device)
        return depthanything
    else:
        raise ValueError('model_name should be dust3r or depthanything')


def get_focals():
    args = get_params()
    
    # 获取数据    
    image_paths = get_dataset(args.dataset).load_image_paths()
    gt_focals = get_dataset(args.dataset).load_image_focals()
    gt_focals = np.array(gt_focals)
    
    if args.dataset == 'CO3D':
        gt_focals[:,0] = gt_focals[:,0] * args.width
        gt_focals[:,1] = gt_focals[:,1] * args.height 
        
    dp_dataset = DPDataset(image_paths, args)     
    
    # 加载模型
    dust3r = get_models('dust3r', args).to(args.device)
    dust3r.eval()
    depthanything = get_models('depthanything', args)
    depthanything.eval()   
    
    dust3r_focals = []
    depthanything_focals = []
    ip_loader = DataLoader(image_paths, batch_size=args.batch_size, shuffle=False, num_workers=8)
    dp_img_loader = DataLoader(dp_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    for img_path, dp_img in tqdm(zip(ip_loader, dp_img_loader), total=len(ip_loader), colour='#0396ff', desc='{}-{}'.format(args.dataset,args.focal_mode)):
        dust3r_paths = []
        for ip in img_path:
            dust3r_paths.extend([ip, ip])
        dust3r_images = load_dust3r_images(dust3r_paths, args)
        dust3r_pairs = make_pairs(dust3r_images, scene_graph='pairs', prefilter=None, symmetrize=True)
        dust3r_output = inference(dust3r_pairs, dust3r, args.device, batch_size=args.batch_size)
        pp = torch.tensor((args.width/2, args.height/2))
        half = int(dust3r_output['pred1']['pts3d'].shape[0] / 2)
        pred1 = dust3r_output['pred1']['pts3d'][:half]
        dust3r_focal = estimate_focal_knowing_depth(pred1, pp, focal_mode=args.focal_mode)
        dust3r_focals.append(dust3r_focal)     
        
        if args.test_depthanything:
            # depthanything 3D point 构造
            # pred1_max = torch.amax(pred1[:,:,:,2], dim=(1, 2), keepdim=True).repeat(1, args.height, args.width)
            # pred1_min = torch.amin(pred1[:,:,:,2], dim=(1, 2), keepdim=True).repeat(1, args.height, args.width)
            # inter = pred1_max - pred1_min
            
            dp_img = dp_img.to(args.device)
            with torch.no_grad():
                depths = depthanything(dp_img) / 100 + 0.18582
            
            # depths_min = torch.amin(depths, dim=(1,2), keepdim=True).repeat(1, args.height, args.width)
            # depths_max = torch.amax(depths, dim=(1,2), keepdim=True).repeat(1, args.height, args.width)
            # print(pred1[0,:,:,2])
            # print(pred1_min[0])
            # print(pred1_max[0])
            # print(depths_min[0])
            # print(depths_max[0])
            # exit()
            
            # depths_max = torch.amax(depths, dim=(1,2), keepdim=True).repeat(1, args.height, args.width)
            # depths = depths / depths_max 
            # pred1[:,:,:,2] = depths * inter + pred1_min
            pred1[:,:,:,2] = depths
            depthanything_3d = pred1
            depthanything_focal = estimate_focal_knowing_depth(depthanything_3d, pp, focal_mode=args.focal_mode)
            depthanything_focals.append(depthanything_focal)
    
    
    dust3r_focals = torch.cat(dust3r_focals, dim=0).detach().cpu().numpy()
    dust3r_focals = dust3r_focals.reshape(-1, 1).repeat(2, axis=1)
    dust3r_error = (abs(gt_focals - dust3r_focals) / gt_focals).mean(axis=0)
    print('dust3r_error:', dust3r_error)
    
    if args.test_depthanything:
        depthanything_focals = torch.cat(depthanything_focals, dim=0).detach().cpu().numpy()
        depthanything_focals = depthanything_focals.reshape(-1, 1).repeat(2, axis=1)
        depthanything_error = (abs(gt_focals - depthanything_focals) /gt_focals).mean(axis=0)
        print('depthanything_error:', depthanything_error)
        # return gt_focals, dust3r_focals, depthanything_focals
    
    # return gt_focals, dust3r_focals
     

if __name__ == "__main__":
    get_focals()
    # a = get_testdata()
    # print(a[0])
    
    
    

        