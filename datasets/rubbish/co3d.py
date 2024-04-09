import os
import json
import gzip

class CO3D:
    def __init__(self,  data_dir='/home/fangbin/workspace/CATL/0-datasets/CO3D') -> None:
        self.data_dir = data_dir
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