import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x




class PretrainedNet(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """
    def __init__(self, model):
        super(PretrainedNet, self).__init__()
        #model = 'alexnet'
        if model == 'resnet18':
            self.cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            #self.cnn.fc = Identity()
        if model == 'alexnet':
            self.cnn = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
        if model == 'mobilenet_v2':
            self.cnn = torchvision.models.mobilenet_v2(weights=torchvision.models.ALEXNET_Weights.DEFAULT)
        if model == "vgg16":
            self.cnn = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        if model == "vgg16_bn":
            self.cnn = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
        if model == "vgg_19":
            self.cnn = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.DEFAULT)
        if model == "vgg19_bn":
            self.cnn = torchvision.models.vgg19_bn(weights = torchvision.models.VGG19_BN_Weights.DEFAULT)
        #se
        #for name, para in self.cnn1.named_parameters():
        #    if para.requires_grad:
        #        print (name)
        #alex_load.classifier[4] = nn.Linear(in_features = alex_load.classifier[1].out_features, out_features = 1000, bias = True)
        #self.cnn1.classifier[6] = nn.Linear(in_features = self.cnn1.classifier[4].out_features, out_features = 2, bias = True)
        
        '''
        self.eff = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        self.eff.classifier[0] = Identity()
        self.eff.classifier[1] = Identity()
        '''
        #vgg16
    
        #self.vgg16 = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
        
        #self.vgg16.classifier[0] = nn.Linear(in_features = 25088, out_features = 4096)
        #self.vgg16.classifier[3] = nn.Linear(in_features = 4096, out_features = 2048)
        #self.vgg16.classifier[6] = nn.Linear(in_features = 2048, out_features = 1024)
        
        '''
        self.fc1 = nn.Linear(in_features = 2048, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 2)
        '''
        # LAST 3 CNN LAYERS / LAST BLOCK OF LAYERS ARE UNFROZEN
        #CONV-CONV-POOL - CONV-CONV-POOL - CONV-CONV-CONV-POOL - CONV-CONV-CONV-POOL
        #for param in self.vgg16.features[0:22].parameters():
        #      param.requires_grad = False
        
        #for name, param in vgg16.named_parameters():
        #    print(f'Parameter name: {name}, Requires gradient: {param.requires_grad}')
        '''
        self.alex = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.DEFAULT)
        self.alex.classifier[1] = nn.Linear(in_features = 9216, out_features = 4096)
        self.alex.classifier[4] = nn.Linear(in_features = 4096, out_features = 2048)
        self.alex.classifier[6] = nn.Linear(in_features = 2048, out_features = 1024)
        '''
    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        #concat = torch.cat([output1, output2], dim=1)
        #x = self.fc1(concat)
        #x = self.fc2(x)
        #out = self.fc3(x)
        return output1, output2


#x = torch.randn(1, 3, 224, 224)
#output = vgg(x)
#print(output.shape)