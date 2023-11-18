import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from params import *

class TrainBERT:
    def __init__(self, model, train_dataloader, epochs, test_dataloader=None, learning_rate=0.001, threshold=0.5, device=DEVICE):
        
        # hyperparameters for optimization
        self.device = device
        self.bar = None
        self.model = model
        self.epochs = epochs
        self.training_data = train_dataloader
        self.testing_data = test_dataloader

        # model to device
        self.model.to(self.device)

        # optimizer: Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        # cost function binary cross entropy loss for predicting each beloning to a class seperately
        self.criterion = nn.BCELoss()
        
        # predictions threshold above which predictions are set True
        self.threshold = threshold 
        
        # create progress bar
        self.bar = tqdm(total=len(train_dataloader.dataset), desc=f'Training', position=0)

        # run training
        for epoch in range(self.epochs):
            self.training(epoch)
    
    def write_results(self, output, file):
        with open(file, "a") as file:
            file.write(output)

    def training(self, epoch):
        # init stats
        avg_loss = 0.0
        corrects_sum = 0
        total = 0
        zero_prediction = 0

        for i, data in enumerate(self.training_data):
            
            # update progress bar
            self.bar.update(self.training_data.batch_size)

            # send data to GPU/CPU
            data ={key: value.to(self.device) for key, value in data.items()}
            
            # labels convert to float()
            labels = data['labels'].float()
            
            # forward pass: comments trough model
            output = self.model.forward(data['input'])
            
            # compute loss with labels (input, target)
            loss = self.criterion(output, labels)

            # average loss per batch
            avg_loss += loss.item()
            
            # backward pass for training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # compute accuracy 
            # use threshold to determine which of the outputs are considered True
            predictions = torch.ge(output, self.threshold).int()
            # compare with the label and count correct classifications
            corrects_sum += (predictions == labels).sum().item()
            # check whether model is only predicting 0
            zero_prediction += ((predictions == 0) & (labels == 0)).sum().item()
            # sump up total number of labels in batch
            total += labels.nelement()
        
        # update learning rate scheduler
        self.scheduler.step() 
        # print stats
        output ="\nTraining epoch: {}\nAvg. training loss: {:.2f}\nAccuracy: {:.2f}\nCorrect predictions: {} of which the model predicted 'False': {}".format(epoch+1, avg_loss / len(self.training_data), corrects_sum / total, corrects_sum, zero_prediction)
        print(output)

        # write in results
        self.write_results(output, "training_results")

        # reset progress bar
        self.bar.n = 0
        self.bar.last_print_n = 0
        self.bar.refresh()


