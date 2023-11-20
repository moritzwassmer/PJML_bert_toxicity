import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from params import *

class TrainBERT:
    def __init__(self, model, train_dataloader, epochs, test_dataloader=None, learning_rate=0.0001, threshold=0.5, device=DEVICE, class_weights=WEIGHTS_LIST):
        
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

        # loss treats every output as an own random variable
        self.criterion = nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.Tensor(class_weights).cuda()
        )
        
        # predictions threshold above which predictions are set True
        self.threshold = threshold 
        
        # create progress bar
        self.bar = tqdm(total=len(train_dataloader.dataset), desc=f'Training', position=0)

        # run training
        for epoch in range(self.epochs):
            self.training(epoch)
            self.testing(epoch)
    
    def write_results(self, output, file):
        with open(file, "a") as file:
            file.write(output)

    def training(self, epoch):
        # init stats
        avg_loss = 0.0
        T = 0
        total = 0
        TN = 0
        TP = 0
        P = 0

        for i, data in enumerate(self.training_data):
            
            # update progress bar
            self.bar.update(self.training_data.batch_size)

            # send data to GPU/CPU
            data ={key: value.to(self.device) for key, value in data.items()}
            
            # labels convert to float()
            labels = data['labels'].to(torch.float)

            # forward pass: comments trough model
            preds = self.model.forward(data['input'])
            loss = self.criterion(preds, labels)
            avg_loss += loss.item()

            # backward pass for training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # COMPUTE METRICS
            # Since we removed the last sigmoid activation function due to using the BCE loss with logits,
            # we need to pass the model output through a sigmoid activation to obtain probabilities
            sigmoid = torch.nn.Sigmoid()
            preds = sigmoid(preds)  # TODO for inference due to BCE with logits we need to apply sigmoid manually
            preds_th = torch.ge(preds, self.threshold).int()

            # compare with the label and count correct classifications
            T += (preds_th == labels).sum().item()
            # check whether model is only predicting 0
            TN += ((preds_th == 0) & (labels == 0)).sum().item()
            TP += ((preds_th == 1) & (labels == 1)).sum().item()
            P += (labels == 1).sum().item()
            # sump up total number of labels in batch
            total += labels.nelement()
        
        # update learning rate scheduler
        self.scheduler.step() 
        # print stats
        message ="\nTraining epoch: {}\nAvg. training loss: {:.2f}\nAccuracy: {:.2f}\nCorrect predictions: {} of which the model predicted 'False': {}, true positives={}, positives={}".format(epoch+1, avg_loss / len(self.training_data), T / total, T, TN, TP, P)
        print(message)

        # write in results
        self.write_results(message, "training_results")

        # reset progress bar
        self.bar.n = 0
        self.bar.last_print_n = 0
        self.bar.refresh()

    def testing(self, epoch):
        # model to evaluation mode
        self.model.eval()


        avg_loss = 0.0
        corrects_sum = 0
        total = 0
        TF = 0
        TP = 0
        P = 0

        with torch.no_grad():
            for i, data in enumerate(self.testing_data):
                # send data to GPU/CPU
                data = {key: value.to(self.device) for key, value in data.items()}

                # labels convert to float()
                labels = data['labels'].to(torch.float)

                # forward pass: comments through model
                preds = self.model.forward(data['input'])

                loss = self.criterion(preds, labels)

                avg_loss += loss.item()

                # compute measures
                # use threshold to determine which of the outputs are considered True
                sigmoid = torch.nn.Sigmoid() # TODO required due to BCEwithLogits
                preds =  sigmoid(preds)
                predictions = torch.ge(preds, self.threshold).int()

                # compare with the label and count correct classifications
                corrects_sum += (predictions == labels).sum().item()

                # check whether model is only predicting 0
                TF += ((predictions == 0) & (labels == 0)).sum().item()
                TP += ((predictions == 1) & (labels == 1)).sum().item()
                P += (labels == 1).sum().item()

                # sum up total number of labels in batch
                total += labels.nelement()

        # print stats for testing
        message = "\nTesting epoch: {}\nAvg. testing loss: {:.2f}\nAccuracy: {:.2f}\nCorrect predictions: {} of which the model predicted 'False': {}, true positives={}, positives = {}".format(
            epoch + 1, avg_loss / len(self.testing_data), corrects_sum / total, corrects_sum, TF, TP, P)
        print(message)

        # write results
        self.write_results(message, "testing_results")

        # Set the model back to training mode
        self.model.train()


