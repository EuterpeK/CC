import os

class City:
    def __init__(self, data_dir='/home/fangbin/workspace/CATL/0-datasets/Cityscapes', train=False):
        self.data_dir = data_dir
        
        
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
    
    def load_dataset(self):
        image_paths = self.load_image_paths()
        focals = self.load_image_focals()
        dataset = []
        for i, ip in enumerate(image_paths):
            item = {'img_path': ip, 'focal': focals[i]}
            dataset.append(item)
        return dataset



if __name__ == "__main__":
    # Instantiate the KITTIDataset class
    dataset = KITTIDataset(data_dir='/home/fangbin/workspace/CATL/0-datasets/Kitti', train=True)
    
    # Load the dataset
    data = dataset.load_dataset()
    
    # Print the first item in the dataset
    print(data[0])