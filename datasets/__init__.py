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



def get_dataset(dataset_name, data_dir):
    # assert dataset_name in ['NYU2v', 'CO3D', 'Kitti']
    if dataset_name == 'NYUv2':
        return NYUv2(data_dir)
    elif dataset_name == 'CO3D':
        return CO3D(data_dir)
    elif dataset_name == 'KITTI':
        return KITTI(data_dir)
    elif dataset_name == 'SUN3D':
        return SUN3D(data_dir)
    elif dataset_name == 'Waymo':
        return Waymo(data_dir)
    elif dataset_name == 'ScanNet':
        return ScanNet(data_dir)
    elif dataset_name == 'ARKit':
        return ARKit(data_dir)
    elif dataset_name == 'NUScenes':
        return NUScenes(data_dir)
    
    else:
        raise ValueError('dataset_name should be NYU2v, CO3D or Kitti')
    