import torch
from torch.utils.data import DataLoader
from collections import namedtuple
import sys
from datasets import get_dataset
from utils import estimate_focal_knowing_depth
sys.path.append("../dust3r/")

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
# from dust3r.post_process import estimate_focal_knowing_depth

sys.path.append("../depthanything/")
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
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
    parser.add_argument('--dataset', type=str, default='KITTI', choices=['KITTI', 'NYUv2', 'CO3D'])
    parser.add_argument('--test_depthanything', type=bool, default=True)
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
    
def load_da_image(img_paths, trans):
    return

def get_focals():
    args = get_params()
    
    # 加载模型
    dust3r = get_models('dust3r', args).to(args.device)
    dust3r.eval()
    depthanything = get_models('depthanything', args)
    depthanything.eval()
    
    # 获取数据    
    image_paths = get_dataset(args.dataset).load_image_paths()
    gt_focals = get_dataset(args.dataset).load_image_focals()
    
    if args.dataset == 'CO3D':
        gt_focals = gt_focals * args.image_size
    
    dust3r_focals = []
    depthanything_focals = []
    test_loader = DataLoader(image_paths, batch_size=args.batch_size, shuffle=False, num_workers=8)
    depthanything_transform = Compose([
            Resize(
                width=args.image_size,
                height=args.image_size,
                resize_target=False,
                # keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    # 获取dust3r和depthanything模型的输出
    # dust3r_images = []
    # depthanything_images = []
    # for img_path in tqdm(image_paths[:10], colour='#0396ff', desc='Prepare images'):
    #     dust3r_image = load_images([img_path, img_path], size=args.image_size)
    #     dust3r_images.append(dust3r_image)
    #     image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
    #     image = depthanything_transform({'image': image})['image']
    #     image = torch.from_numpy(image)
    #     depthanything_images.append(image)
    
    # test_loader = DataLoader(list(zip(dust3r_images, depthanything_images)), batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    
    for img_path in tqdm(test_loader, colour='#0396ff', desc='Processing dust3r images'):
        dust3r_paths = []
        for ip in img_path:
            dust3r_paths.extend([ip, ip])
        dust3r_images = load_images(dust3r_paths, size=args.image_size)
        dust3r_pairs = make_pairs(dust3r_images, scene_graph='pairs', prefilter=None, symmetrize=True)
        dust3r_output = inference(dust3r_pairs, dust3r, args.device, batch_size=args.batch_size)
        pp = torch.tensor((args.image_size/2, args.image_size/2))
        half = int(dust3r_output['pred1']['pts3d'].shape[0] / 2)
        pred1 = dust3r_output['pred1']['pts3d'][:half]
        dust3r_focal = estimate_focal_knowing_depth(pred1, pp, focal_mode=args.focal_mode)
        dust3r_focals.append(dust3r_focal)     
        
        if args.test_depthanything:
            # depthanything 3D point 构造
            images = []
            pred1_max = torch.amax(pred1[:,:,:,2], dim=(1, 2), keepdim=True).repeat(1, args.image_size, args.image_size)
            pred1_min = torch.amin(pred1[:,:,:,2], dim=(1, 2), keepdim=True).repeat(1, args.image_size, args.image_size)
            inter = pred1_max - pred1_min
            
            for ip in img_path:
                image = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB) / 255.0
                image = depthanything_transform({'image': image})['image']
                image = torch.from_numpy(image).unsqueeze(0).to(args.device)
                images.append(image)
            images = torch.cat(images, dim=0).to(args.device)
            with torch.no_grad():
                depths = depthanything(images).detach()
            depths_max = torch.amax(depths, dim=(1,2), keepdim=True).repeat(1, args.image_size, args.image_size)
            # depths_min = torch.amin(depths, dim=(1,2), keepdim=True).repeat(1, args.image_size, args.image_size)
            depths = depths / depths_max 
            # print(pred1[:,:,:,2]-depths)
            # exit()
            pred1[:,:,:,2] = depths * inter + pred1_min
            depthanything_3d = pred1
            depthanything_focal = estimate_focal_knowing_depth(depthanything_3d, pp, focal_mode=args.focal_mode)
            depthanything_focals.append(depthanything_focal)
    
    gt_focals = np.array(gt_focals)
    
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
    
    
    

        