import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from params import *

class TrainBERT:
    def __init__(self, model, train_dataloader, epochs, test_dataloader=None, learning_rate=0.001, threshold=0.5, device='cuda'):
        
        # hyperparameters for optimization
        self.device = device
        self.bar = None
        self.model = model
        self.epochs = epochs
        self.training_data = train_dataloader
        self.testing_data = test_dataloader

        # optimizer: Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        # cost function cross entropy loss for predicting classes of toxicity
        self.criterion = nn.CrossEntropyLoss()
        
        # predictions threshold above which predictions are set True
        self.threshold = threshold 
        
        # run training
        for epoch in range(self.epochs):
            self.training(epoch)

    def training(self, epoch):
        # init stats
        avg_loss = 0.0
        corrects_sum = 0
        trues_sum = 0
        
        # set back progress bar
        self.bar = None
        # create new progress bar
        self.bar = tqdm(total=len(self.training_data.dataset), desc=f'Training epoch {epoch+1}', leave=True, position=0)

        for i, data in enumerate(self.training_data):
            
            # send data to GPU/CPU
            data ={key: value.to(self.device) for key, value in data.items()}
            
            # labels convert to float()
            labels = data['labels'].float()
            
            # forward pass: comments trough model
            output = self.model.forward(data['input'])
            
            # compute loss with labels (input, target)
            loss = self.criterion(output, labels)
            
            # backward pass for training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # average loss per batch
            avg_loss += loss.item()
            
            # compute accuracy 
            # softmax the output vector to get probabilites
            predictions = nn.functional.softmax(output, dim=1)
            # use threshold to determine which of the outputs are considered True
            predictions = torch.ge(predictions, self.threshold).int()
            # compare with the label and count correct classifications
            corrects_sum += (predictions == labels).sum().item()
            # sump up total number of Trues in labels for batch
            trues_sum += labels.nelement()
            
            # update progress bar
            self.bar.update(self.training_data.batch_size)
        
        # update learning rate scheduler
        self.scheduler.step() 
        # print stats
        print('\Training epoch: {}\nAvg. training loss: {:.2f}\nAccuracy: {:.2f}'.format(epoch+1, avg_loss / len(self.training_data), corrects_sum * 100.0 / trues_sum))
 