import numpy as np
import pandas as pd
import scipy.sparse as sp
import torchvision.transforms as transforms

import torch.utils.data as data
import torch
import glob
import os

class melData(data.Dataset):

    def __init__(self, data_path):
        super(melData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
                        add them in the ng_sample() function.
                """
        self.dir_list = glob.glob(os.path.join(data_path,"*"))
        self.transform = transforms.Compose([ transforms.ToPILImage(),transforms.Resize(192,449), transforms.ToTensor() ])
        self.image_paths = []

        for dir in self.dir_list :
            files = glob.glob(os.path.join(dir,"*"))
            self.image_paths.extend(files)

        print(len(self.image_paths))
        print(len(self.dir_list))

    def dir_max(self, num) :
        maxN = -9999
        minN = 9999
        files = glob.glob(os.path.join(self.dir_list[num],"*"))
        print(len(files))
        for path in files :
            image = np.load(path)
            maxN = max(maxN, image.max())
            minN = min(minN, image.min())

        return minN, maxN, len(files)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = torch.from_numpy(np.load(self.image_paths[idx]))
        image = np.load(self.image_paths[idx])

        if image.shape[1] != 576:
            image = np.resize(image, (48,576))

        image= np.repeat(image[np.newaxis,:,:],1,axis=2)


        # image = self.transform(image)

        label = torch.tensor(int(os.path.basename(self.image_paths[idx]).split(".")[0]))
        # print(image.shape)
        return image, label

