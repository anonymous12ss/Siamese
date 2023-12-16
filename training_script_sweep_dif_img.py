
import matplotlib.pyplot as plt

import numpy as np
import random
from PIL import Image
import os
import pandas as pd
from random import randint
import statistics
import torchvision
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable  
from torch.utils import data
import torchvision.datasets as dset
from torch.utils.data import DataLoader,Dataset
from torch import optim

from siamese_new_classes import ContrastiveLossk, siamese_Dataset_no_mode
from siamese_new_network import PretrainedNet

import wandb




class Config_ours():
    training_table = pd.read_csv('/mnt/recsys/daniel/datasets/ffhq_gan_generated_siamese_training/full_img_folders/csv_files/train_file.csv')
    testing_table = pd.read_csv('/mnt/recsys/daniel/datasets/ffhq_gan_generated_siamese_training/full_img_folders/csv_files/test_file.csv')
    image_dir = '/mnt/recsys/daniel/datasets/ffhq_gan_generated_siamese_training/full_img_folders'
    contrastive_margin = 2.4
    euclidean_distance_threshold = 1.0
    pretrained_model = 'resnet18'


def build_dataset(batch_size, Config_ours):
   
    training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # aim to train to be invariant to laterality of eye
    transforms.RandomRotation(15), # rotate +/- 5 degrees around center    # pixel crop
    transforms.ColorJitter(brightness = 0.5, contrast = 0.5), # brightness and color variation of +/- 5%
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.ToTensor()
    ])

    validation_transforms = transforms.Compose([ # pixel crop
        transforms.ToTensor()
    ])
    
    
    training_siamese_dataset = siamese_Dataset_no_mode(patient_table = Config_ours.training_table, image_dir = Config_ours.image_dir, transform = training_transforms)

    train_dataloader = torch.utils.data.DataLoader(training_siamese_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    testing_siamese_dataset = siamese_Dataset_no_mode(patient_table = Config_ours.testing_table, image_dir = Config_ours.image_dir, transform = validation_transforms)

    test_dataloader = torch.utils.data.DataLoader(testing_siamese_dataset, batch_size=1, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        train_dataloader, test_dataloader = build_dataset(config.batch_size, Config_ours)
        net = PretrainedNet(model= Config_ours.pretrained_model).cuda()
        optimizer = build_optimizer(net, config.optimizer, config.learning_rate)
        criterion = ContrastiveLossk(margin = config.contrastive_margin)
        
        
        for epoch in range(0,config.epochs):
            training_loss = 0

            #count = 0
            training_accuracy_history = []
            training_euclidean_distance_history = []
            training_label_history = []



            validation_accuracy_history = []
            validation_euclidean_distance_history = []
            validation_label_history = []

            net.train()
            print ('Starting training')
            for i, data in enumerate(train_dataloader,0):
                #print ('hi')
                img0, img1 , label, img_name1, img_name2 = data
                #print (img_name1[0], img_name2[0], label[0].item())
                #print (img0.shape, img1.shape)
                img0, img1 , label = img0.cuda(), img1.cuda(), label.cuda(),
                optimizer.zero_grad()
                output1, output2 = net.forward(img0,img1)
                loss_contrastive, euclidean_distance = criterion(output1,output2,label)
                loss_contrastive.backward()
                optimizer.step()
                training_loss += loss_contrastive.item()

                #print (training_loss)
                #if count % 100 == 0:
                #    print (training_loss/count)

                training_label = euclidean_distance > Config_ours.euclidean_distance_threshold
                equals = training_label.int().detach().cpu().numpy().flatten() == label.int().cpu().numpy()
                #acc_tmp = torch.Tensor.numpy(equals.cpu())
                training_accuracy_history.extend(equals)

                # save euclidean distance and label history 
                euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # detach gradient, move to CPU
                training_euclidean_distance_history.extend(euclid_tmp)
                label_tmp = torch.Tensor.numpy(label.cpu())
                training_label_history.extend(label_tmp)

                #count += 1

            else:
                validation_loss = 0
                count_correct_valid = 0

                label_0_distance_valid = 0; label_1_distance_valid = 0
                label_0_count_valid = 0;  label_1_count_valid = 0 


                net.eval() 
                print ("Starting Validation")
                with torch.no_grad():
                    for i, data_v in enumerate(test_dataloader, 0):
                        img0_v, img1_v , label_v, _, _ = data_v
                        img0_v, img1_v , label_v = img0_v.cuda(), img1_v.cuda() , label_v.cuda()
                        output_v1, output_v2 = net.forward(img0_v,img1_v)
                        loss_contrastive_v, euclidean_distance_v = criterion(output_v1, output_v2, label_v.float())
                        validation_loss += loss_contrastive_v.item()

                        testing_label = euclidean_distance_v > Config_ours.euclidean_distance_threshold
                        equals = testing_label.int().detach().cpu().numpy().flatten() == label_v.int().cpu().numpy()
                        validation_accuracy_history.extend(equals)

                        euclid_tmp = torch.Tensor.numpy(euclidean_distance_v.detach().cpu()) # detach gradient, move to CPU
                        validation_euclidean_distance_history.extend(euclid_tmp)
                        label_tmp = torch.Tensor.numpy(label_v.cpu())
                        validation_label_history.extend(label_tmp)




            training_accuracy = statistics.mean(np.array(training_accuracy_history).tolist())        
            euclid_if_0_t = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 0]
            euclid_if_1_t= [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 1]
            euclid_if_0_t = np.array(euclid_if_0_t).flatten().tolist()
            euclid_if_1_t = np.array(euclid_if_1_t).flatten().tolist()

            # summary statistics for euclidean distances
            mean_euclid_0t = statistics.mean(euclid_if_0_t) 
            std_euclid_0t = statistics.pstdev(euclid_if_0_t)       
            mean_euclid_1t = statistics.mean(euclid_if_1_t)
            std_euclid_1t = statistics.pstdev(euclid_if_1_t)


            validation_accuracy = statistics.mean(np.array(validation_accuracy_history).tolist())        
            euclid_if_0_v = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 0]
            euclid_if_1_v= [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 1]
            euclid_if_0_v = np.array(euclid_if_0_v).flatten().tolist()
            euclid_if_1_v = np.array(euclid_if_1_v).flatten().tolist()

            # summary statistics for euclidean distances
            mean_euclid_0v = statistics.mean(euclid_if_0_v) 
            std_euclid_0v = statistics.pstdev(euclid_if_0_v)       
            mean_euclid_1v = statistics.mean(euclid_if_1_v)
            std_euclid_1v = statistics.pstdev(euclid_if_1_v)

            training_loss_avg = training_loss/len(train_dataloader)
            validation_loss_avg = validation_loss/len(test_dataloader)
            #print (same_class_loss, count_same_class, other_class_loss, count_other_class)
            print("Epoch number {}\n Training loss {}\n Validation loss {}\n".format(epoch, training_loss_avg, validation_loss_avg))
            #print ("Same class loss{}\n same class count {} \n".format(same_class_loss.item()/count_same_class, count_same_class))
            #print ("Other class loss{}\n other class count {} \n".format(other_class_loss.item()/count_other_class, count_other_class))

            print ("Training Details")
            print ("training Accuracy = {}".format(training_accuracy))
            print ("Average Distance for 0 label = {}".format(mean_euclid_0t))
            print ("Average Distance for 1 label = {} \n".format(mean_euclid_1t))

            print ("Validation Details")
            print ("Validation Accuracy = {}".format(validation_accuracy))
            print ("Average Distance for 0 label = {}".format(mean_euclid_0v))
            print ("Average Distance for 1 label = {} \n".format(mean_euclid_1v))

            wandb.log({"Training_accuracy": training_accuracy, "Training loss":training_loss_avg, "Validation loss":validation_loss_avg, "Validation accuracy":validation_accuracy, 
                      "mean_euclid_0v": mean_euclid_0v, "mean_euclid_1v": mean_euclid_1v, "mean_euclid_0t": mean_euclid_0t, "mean_euclid_1t": mean_euclid_1t,
                       "std_euclid_0v": std_euclid_0v, "std_euclid_1v": std_euclid_1v, "std_euclid_0t": std_euclid_0t, "std_euclid_1t":std_euclid_1t
                      })
            wandb.watch(net)


            #counter.append(epoch)
            #Config.euclidean_distance_threshold = (mean_euclid_0v + mean_euclid_1v) / 2
            #training_loss_history.append(training_loss/len(train_dataloader))
            #validation_loss_history.append(validation_loss/len(test_dataloader))
            torch.save(net.state_dict(), checkpoints_dir + "/base_model_epoch{}.pth".format(config.epochs))


if __name__ == "__main__":
    sweep_config = {
        'method': 'grid'
        }

    metric = {
        'name': 'Training_accuracy',
        'goal': 'maximize'   
        }

    

    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'batch_size': {
            'values': [1, 4, 8, 16]
            },
        'learning_rate': {
              'values': [0.001, 0.0001, 0.00001]
            },
        'contrastive_margin': {
              'values': [1.0, 1.5, 2.0, 2.5, 3.0]
            },
        'epochs': {
            'values': [10, 15, 20]
            }
        }
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="siamese_alexnet_random_imgs")


    os.environ['CUDA_VISIBLE_DEVICES']='1' # pick GPU to use
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    checkpoints_dir = "./ckpt/{}".format(Config_ours.pretrained_model)
    CkptisExist = os.path.exists(checkpoints_dir)
    if not CkptisExist:
       os.makedirs(checkpoints_dir)

    wandb.agent(sweep_id, train)
