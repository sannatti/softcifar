
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

class SoftCifar10(data.Dataset):
    
    def __init__(self, images, labels, image_transforms=None):
        self.image_transforms = image_transforms
        
        # Remove examples with no soft label
        inds = []
        for i, (soft_label, hard_label) in enumerate(labels):
            if np.sum(soft_label) > 0:
                inds.append(i)

        self.images = images[inds]
        self.labels = [labels[i] for i in inds]
        assert self.images.shape[0] == len(self.labels)
        self.N = self.images.shape[0]
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, i):
        
        image = self.images[i]
        soft_label, hard_label = self.labels[i]
        soft_label = torch.tensor(soft_label).float()

        hard_label_onehot = torch.zeros(10)
        hard_label_onehot[hard_label] = 1

        if soft_label.sum() == 0:
            raise Exception('No Label')
        else:
            soft_label = soft_label/soft_label.sum()

        if self.image_transforms is not None:
            image = self.image_transforms(image)
        
        # out = {
        #     'image' : image,
        #     'soft_label' : soft_label,
        #     'hard_label' : hard_label
        # }
        # return out

        return image, (soft_label, hard_label_onehot)
