import os
import json
import gzip
import numpy as np

class GeneralDataset:
    def __init__(self, dataset, data_root='../0-datasets', train=False):
        assert dataset in ['ARKitScenes', 'NUScenes', 'NYUv2', 'ScanNet', 'SUN3D', 'Waymo', 'KITTI']
        self.data_dir = os.path.join(data_root, dataset)
    
    
    def load_image_paths(self):
        image_dir = os.path.join(self.data_dir, 'imgs')
        image_names = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir, name) for name in image_names]
        return image_paths
    
    def load_intrinsics(self):
        calib_dir = os.path.join(self.data_dir, 'intrinsics')
        calib_names = os.listdir(calib_dir)
        calib_paths = [os.path.join(calib_dir, name) for name in calib_names]
        
        intrinsics = []
        for cp in calib_paths:
            if cp.endswith('.npy'):
                intrinsics.append(np.load(cp))
        return intrinsics

    def load_dataset(self):
        image_paths = self.load_image_paths()
        intrinsics = self.load_intrinsics()
        dataset = []
        for i, ip in enumerate(image_paths):
            item = {'img_path': ip, 'focal': intrinsics[i]}
            dataset.append(item)
        return dataset
    
    
class CO3D:
    def __init__(self, dataset='CO3D', data_root='../0-datasets') -> None:
        self.data_dir = os.path.join(data_root, dataset)
        files_and_dirs = os.listdir(self.data_dir)
        self.class_names = [name for name in files_and_dirs if not name.count('_') and not name.count('zip') and not name.count('json')]
        
        
    def load_image_paths(self):
        img_paths = []
        for cn in self.class_names:
            frame_path = os.path.join(self.data_dir, cn, 'frame_annotations.jgz')
            with gzip.open(frame_path, 'r') as f:
                frame = json.loads(f.read())
            for f in frame:
                img_paths.append(os.path.join(self.data_dir, f['image']['path']))
        return img_paths
    
    def load_intrinsics(self):
        intrinsics = []
        for cn in self.class_names:
            frame_path = os.path.join(self.data_dir, cn, 'frame_annotations.jgz')
            with gzip.open(frame_path, 'r') as f:
                frame = json.loads(f.read())
            for f in frame:
                instrinsic = np.zeros((3, 3))
                instrinsic[0,0] = f['viewpoint']['focal_length'][0]
                instrinsic[1,1] = f['viewpoint']['focal_length'][1]
                instrinsic[0,2] = f['viewpoint']['principal_point'][0]
                instrinsic[1,2] = f['viewpoint']['principal_point'][1]
                instrinsic[2,2] = 1
                intrinsics.append(instrinsic)
        return intrinsics
        
    def load_image_focals(self):
        focals = []
        for cn in self.class_names:
            frame_path = os.path.join(self.data_dir, cn, 'frame_annotations.jgz')
            with gzip.open(frame_path, 'r') as f:
                frame = json.loads(f.read())
            for f in frame:
                focals.append(f['viewpoint']['focal_length'])
        return focals 
    
    
    def load_dataset(self):
        img_paths = self.load_image_paths()
        focals = self.load_image_focal()
        dataset = []
        for i, ip in enumerate(img_paths):
            item = {'img_path': ip, 'focal': focals[i]}
            dataset.append(item)
        return dataset
    
    
class KITTI:
    def __init__(self, dataset='KITTI', data_root='../0-datasets', train=False):
        self.data_dir = os.path.join(data_root, dataset)
        self.train_dir = os.path.join(self.data_dir, 'training')
        self.test_dir = os.path.join(self.data_dir, 'testing')
        self.train = train
        
    def load_image_paths(self):
        if self.train:
            image_dir = os.path.join(self.train_dir, 'image_2')
            image_names = os.listdir(image_dir)
            image_paths = [os.path.join(image_dir, name) for name in image_names]
        else:
            image_dir = os.path.join(self.test_dir, 'image_2')
            image_names = os.listdir(image_dir)
            image_paths = [os.path.join(image_dir, name) for name in image_names]
        return image_paths
    
    def load_image_focals(self):
        if self.train:
            calib_dir = os.path.join(self.train_dir, 'calib')
            calib_names = os.listdir(calib_dir)
            calib_paths = [os.path.join(calib_dir, name) for name in calib_names]
        else:
            calib_dir = os.path.join(self.test_dir, 'calib')
            calib_names = os.listdir(calib_dir)
            calib_paths = [os.path.join(calib_dir, name) for name in calib_names]
        
        focals = []
        for cp in calib_paths:
            with open(cp, 'r') as f:
                lines = f.readlines()
                line = lines[0]
                line = line.strip().split(' ')
                focal = [float(line[1]), float(line[6])]
            focals.append(focal)
        return focals
    
    def load_intrinsics(self):
        if self.train:
            calib_dir = os.path.join(self.train_dir, 'calib')
            calib_names = os.listdir(calib_dir)
            calib_paths = [os.path.join(calib_dir, name) for name in calib_names]
        else:
            calib_dir = os.path.join(self.test_dir, 'calib')
            calib_names = os.listdir(calib_dir)
            calib_paths = [os.path.join(calib_dir, name) for name in calib_names]
        
        intrinsics = []
        for cp in calib_paths:
            with open(cp, 'r') as f:
                lines = f.readlines()
                line = lines[0]
                line = line.strip().split(' ')
                intrinsic = np.zeros((3, 3))
                intrinsic[0,0] = float(line[1])
                intrinsic[1,1] = float(line[6])
                intrinsic[0,2] = float(line[3])
                intrinsic[1,2] = float(line[7])
                intrinsic[2,2] = 1
            intrinsics.append(intrinsic)
        return intrinsics
        
    
    def load_dataset(self):
        image_paths = self.load_image_paths()
        focals = self.load_image_focals()
        dataset = []
        for i, ip in enumerate(image_paths):
            item = {'img_path': ip, 'focal': focals[i]}
            dataset.append(item)
        return dataset