import numpy as np
import sys
sys.path.append("../dust3r/")
from dust3r.inference import load_model

def instrinsics2focals(intrinsics):
    focals = []
    for intr in intrinsics:
        focal = [intr[0,0], intr[1,1]]
        focals.append(focal)
    return focals

def instrinsics2principal(intrinsics):
    principals = []
    for intr in intrinsics:
        principal = [intr[0,2], intr[1,2]]
        principals.append(principal)
    return principals


def eval_focal_or_principal(pred_focals, gt_focals):
    pred_focals = np.array(pred_focals)
    gt_focals = np.array(gt_focals)
    error = (abs(gt_focals - pred_focals) / gt_focals).mean(axis=0)
    return error


def get_models(model_name, args):
    if model_name == 'dust3r':
        dust3r = load_model(args.dust3r_model_path, args.device)
        return dust3r
    else:
        raise ValueError('model_name should be dust3r or depthanything')
    
    
    
def gt_intrinsics_scale(gt_intrinsics, args):
    if args.dataset == 'KITTI':
        w, h = 1224, 370
    elif args.dataset == 'NYUv2' or args.dataset == 'SUN3D':
        w, h = 640, 480
    elif args.dataset == 'Waymo':
        w, h = 1920, 1280
    elif args.dataset == 'ARKitScenes':
        w, h = 1920, 1440
    elif args.dataset == 'ScanNet':
        w, h = 1296, 968
    else:
        raise ValueError('dataset should be KITTI, NYUv2, SUN3D, Waymo, ARKitScenes or ScanNet')

    scale = np.eye(3)
    scale[0, 0] = args.width / w
    scale[1, 1] = args.height / h
    
    r_gt_intrinsics = []
    for intr in gt_intrinsics:
        r_gt_intrinsics.append(scale @ intr)

    return r_gt_intrinsics
