import torch
from torch.utils.data import DataLoader
from collections import namedtuple
import sys
from datasets import get_dataset, load_images
from utils import instrinsics2focals, instrinsics2principal, get_models, estimate_focal_knowing_depth, eval_focal_or_principal
sys.path.append("../dust3r/")
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dust3r.post_process import estimate_focal_knowing_depth

from torchvision.transforms import Compose
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--focal_mode', type=str, default='weiszfeld')
    parser.add_argument('--data_root', type=str, default='../0-datasets')
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='KITTI', choices=['KITTI', 'NYUv2', 'CO3D', 'SUN3D', 'Waymo', 'ScanNet', 'ARKit', 'NUScenes', 'Cityscapes'])
    parser.add_argument('--scene_mode', type=str, default='Pair', choices=['Pair', 'PointCloud'])
    parser.add_argument('--width', type=int, default=448)
    parser.add_argument('--height', type=int, default=448)
    
    return parser.parse_args()


def infer_focal(model, imgs, args):
    model.eval()
    outputs = []
    for img in tqdm(imgs, colour='#0396ff', desc='inference {}-{}'.format(args.dataset,args.focal_mode)):
        images = load_images([img, img], args)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, args.device, batch_size=args.batch_size)
        outputs.append(output)
    
    focals = []
    fx_votes = []
    fy_votes = []
    for output in tqdm(outputs, colour='#0396ff', desc='post_process'):
        pred1 = output['pred1']['pts3d'][:1]
        f, fx_vote, fy_vote = estimate_focal_knowing_depth(pred1, None, focal_mode=args.focal_mode)
        fx_votes.append(fx_vote.reshape(-1))
        fy_votes.append(fy_vote.reshape(-1))
        focals.append(f)
        
    return f, fx_votes, fy_votes
    

def pipe():
    args = get_params()
    
    # 获取数据    
    dataset = get_dataset(args)
    image_paths = dataset.load_image_paths()
    gt_instrinsics = dataset.load_intrinsics()
    
    gt_focals = instrinsics2focals(gt_instrinsics)
    
    # 通过dust3r估计相机内参分布
    dust3r = get_models('dust3r', args)
    
    focals, fx_votes, fy_votes = infer_focal(dust3r, image_paths, args)
    
    error = eval_focal_or_principal(focals, gt_focals)
    print(error)
    
    
    
    # plot single image focal length distribution
    idx = 0
    plt.figure(figsize=(8, 6))
    plt.hist(fx_votes[idx], bins=100, alpha=0.75, label='fx-{}-{}'.format(args.dataset, idx))
    plt.legend()
    plt.savefig('fx-{}-{}.png'.format(args.dataset, idx))
    
if __name__ == "__main__":
    pipe()
    
    
    

        