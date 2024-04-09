import os
import h5py
import numpy as np
from PIL import Image

class NYUv2:
    def __init__(self, data_dir='/home/fangbin/workspace/CATL/0-datasets/NYUv2') -> None:
        self.data_dir = data_dir
        self.h5_path = os.path.join(data_dir, 'nyu_depth_v2_labeled.mat')
        
    def load_image_paths(self):
        self.image_dir = os.path.join(self.data_dir, 'images')
        image_names = os.listdir(self.image_dir)
        image_paths = [os.path.join(self.image_dir, name) for name in image_names]
        return image_paths
    
    def load_image_focals(self):
        length = len(self.load_image_paths())
        focal = [5.1885790117450188e+02, 5.1946961112127485e+02] 
        focals = [focal for _ in range(length)]
        return focals
    
    def load_dataset(self):
        image_paths = self.load_image_paths()
        focals = self.load_image_focals()
        dataset = []
        for i, ip in enumerate(image_paths):
            item = {'img_path': ip, 'focal': focals[i]}
            dataset.append(item)
        return dataset
    
    def load_images(self):
        nyu = h5py.File(self.h5_path, 'r')
        images = nyu['images']
        images = np.array(images)
        image_rgb = []
        for image in images:
             # print(image.shape)
            r = Image.fromarray(image[0]).convert("L")
            g = Image.fromarray(image[1]).convert("L")
            b = Image.fromarray(image[2]).convert("L")
            image = Image.merge("RGB", (r, g, b))
            image = image.transpose(Image.ROTATE_270)
            image_rgb.append(image)
        return image_rgb
    
    def load_depths(self):
        nyu = h5py.File(self.h5_path, 'r')
        depths = nyu['depths']
        depths = np.array(depths)
        
        max_value = depths.max(axis=0)
        depths = depths / max_value * 255
        depths = depths.transpose(0, 2, 1)
        
        depth_matric = []
        for depth in depths:
            depth_img = Image.fromarray(np.uint8(depth))
            depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
            depth_matric.append(depth_img)
        return depth_matric
            
        