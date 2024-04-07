import os

class NUScenes:
    def __init__(self, data_dir='/home/fangbin/workspace/CATL/0-datasets/NUScenes', train=False):
        self.data_dir = data_dir
        
    def load_image_paths(self):
        image_dir = os.path.join(self.data_dir, 'imgs')
        image_names = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir, name) for name in image_names]
        return image_paths
    
    def load_image_focals(self):
        calib_dir = os.path.join(self.data_dir, 'intrinsics')
        calib_names = os.listdir(calib_dir)
        calib_paths = [os.path.join(calib_dir, name) for name in calib_names]
        focals = []
        for cp in calib_paths:
            with open(cp, 'r') as f:
                lines = f.readlines()
                line1 = lines[0]
                line1 = line1.strip().split(' ')
                line2 = lines[1]
                line2 = line2.strip().split(' ')
                focal = [float(line1[0]), float(line2[1])]
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
    dataset = NUScenes(data_dir='/home/fangbin/workspace/CATL/0-datasets/NUScenes', train=False)
    
    # Load the dataset
    data = dataset.load_dataset()
    
    # Print the first item in the dataset
    print(data[0])