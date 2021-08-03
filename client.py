from collections import OrderedDict
from typing import Dict, List, Tuple

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import numpy as np

import flwr as fl

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    transform = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                ])
    
    trainset = MNIST(root = './data', 
                     train = True, 
                     transform = transform, 
                     download = True)
    testset = MNIST(root = './data', 
                    train = False, 
                    transform = transform, 
                    download = True)

    #victim_idx = random.sample(range(trainset.data.shape[0]), k=2000) # seleciona quais dados vão ser passados para o cliente
    #victim_train_idx = victim_idx[:1000] # seleciona os 1000 primeiros dados para treinamento 
    #attack_idx = victim_idx[1000:] # a segunda metade dos dados será destinada ao atacante
    #victim_test_idx = random.sample(range(testset.data.shape[0]), k=15) # seleciona 15 dados para o teste

    #victim_train_dataset = Subset(trainset, victim_train_idx)
    #attack_dataset = Subset(trainset, attack_idx)
    #victim_test_dataset = Subset(testset, victim_test_idx)

    trainloader = DataLoader(trainset, 
                             batch_size = 32, 
                             shuffle = True)
    #attack_dataloader = torch.utils.data.DataLoader(attack_dataset, 
    #                                                batch_size = 64, 
    #                                                shuffle = True)
    testloader = DataLoader(testset, 
                            batch_size = 32, 
                            shuffle = False)
    
    return trainloader, testloader

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    #criterion = NoPeekLoss(alpha = 0.8)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            
def test(net, testloader):
    """Validate the network on the entire test set."""
    #criterion = NoPeekLoss(alpha = 0.8) #Analisar como vai ficar o criterio
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


class Net(nn.Module):
    """Model for MNIST Classification."""
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 64,
                               kernel_size = 3, 
                               padding = 1, 
                               stride = 1)        
        self.conv2 = nn.Conv2d(in_channels = 64, 
                               out_channels = 128,
                               kernel_size = 3, 
                               padding = 1)        
        self.conv3 = nn.Conv2d(in_channels = 128, 
                               out_channels = 256,
                               kernel_size = 3, 
                               padding = 1)        
        self.conv4 = nn.Conv2d(in_channels = 256, 
                               out_channels = 512,
                               kernel_size = 3, 
                               padding = 1)
        
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.L1 = nn.Linear(512, 10) # Temos 10 classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3ch > 64ch, shape 32 x 32 -> 16 x 16
        x = self.conv1(x) # [64,32,32]
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # [64,16,16]
        
        # 64ch > 128ch, shape 16 x 16 -> 8 x 8
        x = self.conv2(x) # [128,16,16]
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # [128,8,8]
        
        x = self.conv3(x) # [256,8,8]
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # [256,4,4]   

        # 256ch > 512ch, shape 4 x 4 -> 2 x 2
        x = self.conv4(x) # [512,4,4]
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # [512,2,2]
        
        # camada totalmente conectada
        x = x.view(-1, 512)
        x = self.L1(x)
        return x
    
class ClientFL(fl.client.NumPyClient):        
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(), len(trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader), {"accuracy":float(accuracy)}
    
    
net = Net()
#net.to(DEVICE)
trainloader, testloader = load_data()

fl.client.start_numpy_client("[::]:8080", ClientFL())