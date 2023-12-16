import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import torch
import random 
from random import randint
import os

class siamese_Dataset_no_mode(Dataset):
    """
    Create dataset representation of ROP data
    - This class returns image pairs with a change label (i.e. change vs no change in a categorical disease severity label) and other metadata
    - Image pairs are sampled so that there are an equal number of change vs no change labels
    - Epoch size can be set for empirical testing

    Concepts adapted from:
    - https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, patient_table, image_dir,  transform=None):
        """
        Args:
            patient_table (pd.dataframe): dataframe table containing image names, disease severity category label, and other metadata
            image_dir (string): directory containing all of the image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.patient_table = patient_table
        self.image_dir = image_dir
        self.transform = transform
        self.epoch_size = len(self.patient_table)
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        #you should change this part-------------------------------------------------------------------
        name_list = self.patient_table
        num_entries = len(name_list)
        # goal is 50:50 distribution of change vs no change
        change_binary = random.randint(0,1)
        label = None
        # keep on looping until no change pair created
        while change_binary == 0:
            while True:
                #random_image = random.choice(name_list).split('.')[0]+'.png' # note that processed images are all .png type, while patient_table has different types
               
                random_num = random.randint(0, num_entries-1)
                #random_image_row = random.choice(name_list)

                random_image_row = name_list.iloc[random_num]
                random_image = random_image_row['File_name']
                #if random_image in os.listdir(os.path.join(self.image_dir, self.mode, random_image_row['class'])):

                paired_image = random_image.split(".")[0][:-5] + 'out.png'
                if (random_image in os.listdir(os.path.join(self.image_dir, "truth"))) and (paired_image in os.listdir(os.path.join(self.image_dir,  "out"))):
                    break
                elif paired_image in os.listdir(os.path.join(self.image_dir, "out")):
                    print('attempted to get following image, but missing: ' + random_image)
                elif random_image in os.listdir(os.path.join(self.image_dir,  "truth")):
                    print('attempted to get following image, but missing: ' + paired_image)
                else:
                    print ('both_images are not there')
                    print ("random_img_dir ", os.listdir(os.path.join(self.image_dir, "truth")))
                    print ('paired_img_dir', os.listdir(os.path.join(self.image_dir, "out")))
            #img_name1 = random_image.split('.')[0].split('_')[:2]
            #img_name2 = paired_image.split('.')[0].split('_')[:2]
            img_name1 = random_image.split('.')[0].split('_')[0]
            img_name2 = paired_image.split('.')[0].split('_')[0]
            if img_name1 == img_name2:
                label = 0
                break

        # keep on looping until change pair created
        while change_binary == 1:
            # pick random image from folder
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:

                random_num = random.randint(0, num_entries - 1)
                pair_num = random.randint(0, num_entries - 1)
                if random_num == pair_num:
                    continue

                # random_image_row = random.choice(name_list)

                random_image_row = name_list.iloc[random_num]
                random_image = random_image_row['File_name']

                paired_image_row = name_list.iloc[pair_num]
                paired_image = paired_image_row['File_name'].split(".")[0][:-5] + 'out.png'

                if (random_image in os.listdir(os.path.join(self.image_dir,  "truth"))) and (paired_image in os.listdir(os.path.join(self.image_dir,"out"))):
                    break
                elif paired_image in os.listdir(os.path.join(self.image_dir,  "out")):
                    print('attempted to get following image, but missing: ' + random_image)
                elif random_image in os.listdir(os.path.join(self.image_dir,  "truth")):
                    print('attempted to get following image, but missing: ' + paired_image)
            #img_name1 = random_image.split('.')[0].split('_')[:2]
            #img_name2 = paired_image.split('.')[0].split('_')[:2]
            img_name1 = random_image.split('.')[0].split('_')[0]
            img_name2 = paired_image.split('.')[0].split('_')[0]
            if img_name1 != img_name2:
                label = 1
                break

        random_img_dir = os.path.join(self.image_dir,  "truth")
        paired_img_dir = os.path.join(self.image_dir,  "out")
        img0 = Image.open(random_img_dir +'/' +random_image).convert("RGB")
        img1 = Image.open(paired_img_dir +'/' +paired_image).convert("RGB")

        #img0 = img0.convert("L")            #converting to grayscale
        #img1 = img1.convert("L")            #converting to grayscale
        k1, k2 = img0.size, img1.size
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        #print ('bhghjagfhafgdhsfgdhjgfhghghjfgdhjg',img0.shape)
        return img0, img1, label, random_image, paired_image

class ContrastiveLossk(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super(ContrastiveLossk, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive, euclidean_distance


    
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
    
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import torch
import random 
from random import randint
import os
import torchvision.transforms as T

class siamese_Dataset_filter_same_img(Dataset):
    """
    Create dataset representation of ROP data
    - This class returns image pairs with a change label (i.e. change vs no change in a categorical disease severity label) and other metadata
    - Image pairs are sampled so that there are an equal number of change vs no change labels
    - Epoch size can be set for empirical testing

    Concepts adapted from:
    - https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, patient_table, image_dir,  transform=None):
        """
        Args:
            patient_table (pd.dataframe): dataframe table containing image names, disease severity category label, and other metadata
            image_dir (string): directory containing all of the image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.patient_table = patient_table
        self.image_dir = image_dir
        self.transform = transform
        self.epoch_size = len(self.patient_table)
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        #you should change this part-------------------------------------------------------------------
        name_list = self.patient_table
        num_entries = len(name_list)
        # goal is 50:50 distribution of change vs no change
        change_binary = random.randint(0,1)
        label = None
        # keep on looping until no change pair created
        while change_binary == 0:
            while True:
                #random_image = random.choice(name_list).split('.')[0]+'.png' # note that processed images are all .png type, while patient_table has different types
               
                random_num = random.randint(0, num_entries-1)
                #random_image_row = random.choice(name_list)

                random_image_row = name_list.iloc[random_num]
                random_image_name = random_image_row['File_name']
                #if random_image in os.listdir(os.path.join(self.image_dir, self.mode, random_image_row['class'])):

                paired_image_name = random_image_name.split(".")[0][:-5] + 'out.png'
                if (random_image_name in os.listdir(os.path.join(self.image_dir, "truth"))) and (paired_image_name in os.listdir(os.path.join(self.image_dir,  "out"))):
                    break
                elif paired_image_name in os.listdir(os.path.join(self.image_dir, "out")):
                    print('attempted to get following image, but missing: ' + random_image_name)
                elif random_image_name in os.listdir(os.path.join(self.image_dir,  "truth")):
                    print('attempted to get following image, but missing: ' + paired_image_name)
                else:
                    print ('both_images are not there')
                    print ("random_img_dir ", os.listdir(os.path.join(self.image_dir, "truth")))
                    print ('paired_img_dir', os.listdir(os.path.join(self.image_dir, "out")))
            #img_name1 = random_image.split('.')[0].split('_')[:2]
            #img_name2 = paired_image.split('.')[0].split('_')[:2]
            img_name1 = random_image_name.split('.')[0].split('_')[0]
            img_name2 = paired_image_name.split('.')[0].split('_')[0]
            
            random_img_dir = os.path.join(self.image_dir, "truth")
            paired_img_dir = os.path.join(self.image_dir, "out")
            random_img = Image.open(random_img_dir +'/' +random_image_name).convert("RGB")
            paired_img = Image.open(paired_img_dir +'/' +paired_image_name).convert("RGB")
            
            
            random_transform = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2),
                           T.RandomResizedCrop(size=(256, 256), scale=(0.85, 1.0)),
                           #T.RandomRotation(5),
                           T.RandomHorizontalFlip(p=0.5),
                           T.ToTensor(),
                           #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
            
            paired_transform = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2),
                           T.RandomResizedCrop(size=(256, 256), scale=(0.85, 1.0)),
                           #T.RandomRotation(5),
                           T.RandomHorizontalFlip(p=0.5),
                           T.ToTensor(),
                           #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
            paired_img = paired_transform(paired_img)
            random_img = random_transform(random_img)
            
            if img_name1 == img_name2:
                label = 0
                break

        # keep on looping until change pair created
        while change_binary == 1:
            # pick random image from folder
            # check to see if the image exists and can be loaded, if not move to another random image
            random_img_dir = os.path.join(self.image_dir,  "truth")
            label = 1
            
            while True:
                random_num = random.randint(0, num_entries - 1)
                random_image_row = name_list.iloc[random_num]
                
                random_image_name = random_image_row['File_name']
                paired_image_name = random_image_row['File_name']

                if (random_image_name in os.listdir(random_img_dir)):
                    break
            

        

            random_img = Image.open(random_img_dir +'/' +random_image_name).convert("RGB")
            paired_img = Image.open(random_img_dir +'/' +paired_image_name).convert("RGB")

            paired_transform = T.Compose([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue = 0.1),
                           T.RandomResizedCrop(size=(256, 256), scale=(0.75, 0.85)),
                           T.RandomRotation(15),
                           T.RandomHorizontalFlip(p=0.5),
                           T.ToTensor(),
                           #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
            random_transform = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2),
                           T.RandomResizedCrop(size=(256, 256), scale=(0.85, 1.0)),
                           #T.RandomRotation(5),
                           T.RandomHorizontalFlip(p=0.5),
                           T.ToTensor(),
                           #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])

            paired_img = paired_transform(paired_img)
            random_img = random_transform(random_img)
            if label == 1:
                break

        #print ('bhghjagfhafgdhsfgdhjgfhghghjfgdhjg',img0.shape)
        return paired_img, random_img, label, random_image_name
