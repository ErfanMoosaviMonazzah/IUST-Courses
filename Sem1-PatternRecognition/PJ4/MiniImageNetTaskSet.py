import torch
from torchvision import transforms
import random
from PIL import Image
import matplotlib.pyplot as plt

import os


class MiniImageNetTaskSet():
    '''A task set helper class
    
    '''
    def __init__(self, root='miniimagenet/', mode='train', nway=5, support_samples=1, query_samples=10) -> None:
        self.root = root
        self.data_dir = os.path.join(root, 'data')
        self.mode = mode
        self.nway = nway
        self.sup_count = support_samples
        self.que_count = query_samples

        with open(os.path.join(root, 'splits', f'{mode}.txt')) as file:
            self.C = [line.strip().split(',') for line in file.readlines()]

        self.transform = transforms.Compose([
            #transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])



    def __len__(self):
        'Denotes the total number of classes'
        return len(self.C)

    
    def sample_episode(self):
        # sample nway class from all classes avaialabe to this task set
        C = random.sample(self.C, k=self.nway)
        
        ls_tensors_sup = []
        ls_tensors_que = []
        ls_labels_sup = []
        ls_labels_que = []

        # load all nway * (sup_count + que_count) samples
        for cc, ci in zip(C, range(len(C))):
            c = cc[0]
            # get all images of class c
            all_c_images = os.listdir(os.path.join(self.data_dir, c))
            # get a sup_count + que_count sample out of all
            selected_images = random.sample(all_c_images, k=self.sup_count+self.que_count)
            # load images to torch tensors
            tensors = [self.transform(Image.open(os.path.join(self.data_dir, c, image_name))).unsqueeze(0) for image_name in selected_images]
            ls_tensors_sup += tensors[:self.sup_count]
            ls_labels_sup += [ci] * self.sup_count
            ls_tensors_que += tensors[self.sup_count:]
            ls_labels_que += [ci] * self.que_count

        
        # 
        X_sup = torch.cat(ls_tensors_sup, 0)
        y_sup = torch.tensor(ls_labels_sup)
        # 
        X_que = torch.cat(ls_tensors_que, 0)
        y_que = torch.tensor(ls_labels_que)

        # shuffling Support and Query data
        shuff_indices_sup = random.sample(range(X_sup.shape[0]), X_sup.shape[0])
        X_sup = X_sup[shuff_indices_sup]
        y_sup = y_sup[shuff_indices_sup]

        shuff_indices_que = random.sample(range(X_que.shape[0]), X_que.shape[0])
        X_que = X_que[shuff_indices_que]
        y_que = y_que[shuff_indices_que]

        D_sup = (X_sup, y_sup) # nway * sup_count
        D_que = (X_que, y_que) # nway * que_count

        return (D_sup, D_que) # a sampled task
    
    
    def sample_episodes(self, batch):
        return [self.sample_episode() for _ in range(batch)]

    def visualize_episode(self, task):
        D_sup, D_que = task
        X_sup, y_sup = D_sup
        X_que, y_que = D_que
        
        X = torch.cat([X_sup, X_que], 0)

        noi = 4
        num_of_images = noi * noi
        for index in range(1, num_of_images+1):
            img = X[index-1]
            
            _ = plt.subplot(noi,noi, index)
            _ = plt.axis('off')
            _ = plt.imshow(img.moveaxis(0, -1))
    
