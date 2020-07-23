import numpy as np
import pandas as pd
import scipy.sparse as sp
import torchvision.transforms as transforms

import torch.utils.data as data
import torch
import glob
import os

class melData(data.Dataset):

    def __init__(self, data_path, is_training=True):
        super(melData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
                        add them in the ng_sample() function.
                """
        self.dir_list = glob.glob(os.path.join(data_path,"*"))
        self.transform = transforms.Compose([ transforms.ToPILImage(),transforms.Resize(192,449), transforms.ToTensor() ])
        self.image_paths = []
        self.is_training = is_training
        self.minVal = -100
        self.maxVal = 26.924

        for dir in self.dir_list :
            files = glob.glob(os.path.join(dir,"*"))
            self.image_paths.extend(files)

        self.image_paths = np.array(sorted(self.image_paths, key=self.natural_keys))

    def natural_keys(self, text):
        text = text.split("/")[-1]
        text = text.split(".")[0]
        return int(text)
    

    def load_image(self, idx):
       
        def MinMaxScale(array) :
            return (array - self.minVal) / (self.maxVal - self.minVal)
       
        image = np.load(self.image_paths[idx])
        if image.shape[1] != 576:
            # image = MinMaxScale(np.resize(image, (48,576)))
            image = np.resize(image, (48,576))

        #return np.repeat(image[np.newaxis,:,:],1,axis=2)
        return np.reshape(image, (1,)+image.shape)

    def make_user(self, user) :
        image = self.load_image(int(user))
        return torch.tensor(np.reshape(image, (1,)+image.shape))

    def make_batch(self, items, batch_size=128):
        self.images = []
        
        cnt = 0
        tmp = []
        for item in items[0] :
            cnt += 1
            tmp.append(self.load_image(int(item)))

            if cnt == batch_size :
                self.images.append(torch.tensor(tmp))
                tmp = []
        
        if tmp :
            self.images.append(torch.tensor(tmp))


    def __len__(self):
        if self.is_training :
            return len(self.image_paths)
        else :
            return len(self.images)

    def __getitem__(self, idx):
        if self.is_training :
            return self.load_image(idx)
        else :
            return self.images[idx]

