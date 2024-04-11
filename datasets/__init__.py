from .General import GeneralDataset, CO3D, KITTI
from .dust3r import load_images



def get_dataset(args):
    # assert dataset_name in ['NYU2v', 'CO3D', 'Kitti']
    if args.dataset in ['ARKitScenes', 'NUScenes', 'NYUv2', 'ScanNet', 'SUN3D', 'Waymo', ]:
        return GeneralDataset(args.dataset, args.data_root, args.train)
    elif args.dataset == 'CO3D':
        return CO3D(args.dataset, args.data_root, args.train)
    elif args.dataset == 'KITTI':
        return KITTI(args.dataset, args.data_root, args.train)
    else:
        raise ValueError('dataset_name should be ARKitScenes, NUScenes, NYUv2, ScanNet, SUN3D, Waymo, CO3D, or KITTI.')
    