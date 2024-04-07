from .co3d import CO3D
from .kitti import KITTI
from .nyuv2 import NYUv2
from .sun3d import SUN3D
from .waymo import Waymo
from .ScanNet import ScanNet
from .arkit import ARKit
from .nuscenes import NUScenes
from .depthanything import DPDataset
from .dust3r import load_dust3r_images



def get_dataset(dataset_name):
    # assert dataset_name in ['NYU2v', 'CO3D', 'Kitti']
    if dataset_name == 'NYUv2':
        return NYUv2()
    elif dataset_name == 'CO3D':
        return CO3D()
    elif dataset_name == 'KITTI':
        return KITTI()
    elif dataset_name == 'SUN3D':
        return SUN3D()
    elif dataset_name == 'Waymo':
        return Waymo()
    elif dataset_name == 'ScanNet':
        return ScanNet()
    elif dataset_name == 'ARKit':
        return ARKit()
    elif dataset_name == 'NUScenes':
        return NUScenes()
    
    else:
        raise ValueError('dataset_name should be NYU2v, CO3D or Kitti')
    