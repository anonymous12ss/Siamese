
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from torch.utils import data
import pandas as pd
from random import randint

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # aim to train to be invariant to laterality of eye
    transforms.RandomRotation(5),  # rotate +/- 5 degrees around center    # pixel crop
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # brightness and color variation of +/- 5%
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.ToTensor()
])
'''
validation_transforms = transforms.Compose([ # pixel crop
    transforms.ToTensor()
])
'''

validation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # aim to train to be invariant to laterality of eye
    transforms.RandomRotation(5),  # rotate +/- 5 degrees around center    # pixel crop
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # brightness and color variation of +/- 5%
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.ToTensor()
])


class Config():
    train_batch_size = 1
    train_number_epochs = 10
    training_table = pd.read_csv(
        '/mnt/recsys/daniel/datasets/ffhq_gan_generated_siamese_training/full_img_folders/csv_files/train_file.csv')
    testing_table = pd.read_csv(
        '/mnt/recsys/daniel/datasets/ffhq_gan_generated_siamese_training/full_img_folders/csv_files/test_file.csv')
    image_dir = '/mnt/recsys/daniel/datasets/ffhq_gan_generated_siamese_training/full_img_folders'


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()




class siamese_Dataset_no_mode(torch.utils.data.Dataset):
    """
    Create dataset representation of ROP data
    - This class returns image pairs with a change label (i.e. change vs no change in a categorical disease severity label) and other metadata
    - Image pairs are sampled so that there are an equal number of change vs no change labels
    - Epoch size can be set for empirical testing

    Concepts adapted from:
    - https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, patient_table, image_dir, transform=None):
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
        # you should change this part-------------------------------------------------------------------
        name_list = self.patient_table
        num_entries = len(name_list)
        # goal is 50:50 distribution of change vs no change
        change_binary = random.randint(0, 1)
        label = None
        # keep on looping until no change pair created
        while change_binary == 0:
            print('here0')
            while True:
                # random_image = random.choice(name_list).split('.')[0]+'.png' # note that processed images are all .png type, while patient_table has different types
                print('inside true')
                random_num = randint(0, num_entries - 1)
                # random_image_row = random.choice(name_list)

                random_image_row = name_list.iloc[random_num]
                random_image = random_image_row['File_name']
                # if random_image in os.listdir(os.path.join(self.image_dir, self.mode, random_image_row['class'])):

                paired_image = random_image.split(".")[0][:-5] + 'out.png'
                print(paired_image)
                if (random_image in os.listdir(os.path.join(self.image_dir, "truth"))) and (
                        paired_image in os.listdir(os.path.join(self.image_dir, "out"))):
                    break
                elif paired_image in os.listdir(os.path.join(self.image_dir, "out")):
                    print('attempted to get following image, but missing: ' + random_image)
                elif random_image in os.listdir(os.path.join(self.image_dir, "truth")):
                    print('attempted to get following image, but missing: ' + paired_image)
                else:
                    print('both_images are not there')
                    print("random_img_dir ", os.listdir(os.path.join(self.image_dir, "truth")))
                    print('paired_img_dir', os.listdir(os.path.join(self.image_dir, "out")))
                print('looping here')
            img_name1 = random_image.split('.')[0].split('_')[:1]
            img_name2 = paired_image.split('.')[0].split('_')[:1]

            if img_name1 == img_name2:
                label = 0
                break

        # keep on looping until change pair created
        while change_binary == 1:
            print('here1')
            # pick random image from folder
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:

                random_num = randint(0, num_entries - 1)
                pair_num = randint(0, num_entries - 1)
                if random_num == pair_num:
                    continue

                # random_image_row = random.choice(name_list)

                random_image_row = name_list.iloc[random_num]
                random_image = random_image_row['File_name']

                paired_image_row = name_list.iloc[pair_num]
                paired_image = paired_image_row['File_name'].split(".")[0][:-5] + 'out.png'

                if (random_image in os.listdir(os.path.join(self.image_dir, "truth"))) and (
                        paired_image in os.listdir(os.path.join(self.image_dir, "out"))):
                    break
                elif paired_image in os.listdir(os.path.join(self.image_dir, "out")):
                    print('attempted to get following image, but missing: ' + random_image)
                elif random_image in os.listdir(os.path.join(self.image_dir, "truth")):
                    print('attempted to get following image, but missing: ' + paired_image)
                print('looping here2')
            img_name1 = random_image.split('.')[0].split('_')[:2]
            img_name2 = paired_image.split('.')[0].split('_')[:2]

            if img_name1 != img_name2:
                label = 1
                break

        random_img_dir = os.path.join(self.image_dir, "truth")
        paired_img_dir = os.path.join(self.image_dir, "out")
        img0 = Image.open(random_img_dir + '/' + random_image).convert("RGB")
        img1 = Image.open(paired_img_dir + '/' + paired_image).convert("RGB")

        # img0 = img0.convert("L")            #converting to grayscale
        # img1 = img1.convert("L")            #converting to grayscale
        k1, k2 = img0.size, img1.size
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        # print ('bhghjagfhafgdhsfgdhjgfhghghjfgdhjg',img0.shape)
        return img0, img1, label


training_siamese_dataset = siamese_Dataset_no_mode(patient_table=Config.training_table,
                                                   image_dir=Config.image_dir,
                                                   transform=training_transforms,
                                                   )

train_dataloader = torch.utils.data.DataLoader(training_siamese_dataset,
                                               batch_size=Config.train_batch_size,
                                               shuffle=True,
                                               num_workers=0)

testing_siamese_dataset = siamese_Dataset_no_mode(patient_table=Config.testing_table,
                                                  image_dir=Config.image_dir,
                                                  transform=validation_transforms,
                                                  )

test_dataloader = torch.utils.data.DataLoader(testing_siamese_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)


class SiameseNetworkbasic(nn.Module):
    def __init__(self):
        super(SiameseNetworkbasic, self).__init__()

        # Shared feature extraction layers (twin branches)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 30 * 30, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),

            nn.Linear(256, 2)  # 2 output dimensions for similarity scoring
        )

    def forward_one(self, x):
        # Forward pass through one branch of the Siamese network
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # Forward pass through both branches
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


Config.train_batch_size = 1
Config.train_number_epochs = 1

net = SiameseNetworkbasic().cuda()

criterion = ContrastiveLoss(margin=2.9)
optimizer = optim.Adam(net.parameters(), lr=0.000001)

counter = []
training_loss_history = []
validation_loss_history = []

iteration_number = 0

checkpoints_dir = "./ckpt"

for epoch in range(0, Config.train_number_epochs):
    training_loss = 0
    net.train()
    count = 0
    print('Starting training')
    for i, data in enumerate(train_dataloader, 0):
        print('hi')
        img0, img1, label = data
        # print (img0.shape, img1.shape)
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = net.forward(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        training_loss += loss_contrastive.item()
        count += 1
        print(training_loss)
        if count % 10 == 0:
            print(training_loss)
    else:
        validation_loss = 0
        with torch.no_grad():
            net.eval()
        for i, data in enumerate(test_dataloader, 0):
            img0, img1, label = data
            # print (img0.shape, img1.shape)
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output_v1, output_v2 = net.forward(img0, img1)
            loss_contrastive = criterion(output1, output2, label.float())
            validation_loss += loss_contrastive.item()

    print(
        "Epoch number {}\n Training loss {}\n Validation loss {}\n".format(epoch, training_loss / len(train_dataloader),
                                                                           validation_loss / len(test_dataloader)))
    counter.append(epoch)
    training_loss_history.append(training_loss / len(train_dataloader))
    validation_loss_history.append(validation_loss / len(test_dataloader))
    torch.save(net.state_dict(), checkpoints_dir + "/base_model_epoch{}.pth".format(epoch))

plt.plot(training_loss_history)
plt.plot(validation_loss_history)
plt.show()