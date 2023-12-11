import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from params import *


class TrainBERT:
    """
    Class to perform a training and a testing processes (if test_dataloader is set) for the toxic comment classification model.

        Args: 
            model (nn.Module): BERT-based toxic comment classification model
            train_dataloader (torch.utils.data.DataLoader): Dataloader for training 
            test_dataloader (torch.utils.data.DataLoader): Dataloader for testing

        Attributes:
            model (nn.Module): BERT-based toxic comment classification model
            training_data (torch.utils.data.Dataloader): Dataloader for training 
            testing_data (torch.utils.data.DataLoader): Dataloader for testing
            bar (tqdm.tqdm): Progress bar for training
            test_bar (tqdm.tqdm): Progress bar for testing
            optimizer (torch.optim.Adam): Adam optimizer
            scheduler (torch.optim.lr_scheduler.StepLR): Scheduler for learning rate
            criterion (nn.BCEWithLogitsLoss): Binary cross-entropy loss with logits
        
        Methods:
            write_results(output, file): Writes the output measures of training and testing loop into a txt. file
            training(epoch): Performs a training of the model for the epoch
            testing(epoch): Performs a test of the model for the epoch

 
    """

    def __init__(self, model, train_dataloader=None, test_dataloader=None, epochs=EPOCHS, learning_rate=LEARNING_RATE, mode='train_test', info=None):
        """
        Initializes a training and a testing processes.

        Args: 
            model (nn.Module): BERT-based toxic comment classification model
            train_dataloader (torch.utils.data.DataLoader): Dataloader for training 
            test_dataloader (torch.utils.data.DataLoader): Dataloader for testing
        """

        # parameters
        self.model = model
        self.epochs = epochs
        self.training_data = train_dataloader
        self.testing_data = test_dataloader
        self.bar = None
        # model to device
        self.model.to(DEVICE)

        # optimizer: Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        # loss treats every output as an own random variable
        self.criterion = nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.Tensor(WEIGHTS_LIST).to(DEVICE)
        )
        self.info = info
        self.train_res = "training_results"
        self.test_res = "testing_results"
        self.mode = mode

    def run(self):  
        auc_list = []  
        # write hyperparameters in output
        if self.mode != "train_test":
            self.write_results(self.info, self.train_res)
            self.write_results(self.info, self.test_res)

        if self.mode == "validation":
            self.bar = tqdm(total=len(self.testing_data.dataset), desc=f'Validation', position=0)
            self.bar.total = len(self.testing_data.dataset)
            auc_list.append(self.testing(epoch))
            self.bar.close()

        # run training, testing
        else:
            self.bar = tqdm(total=len(self.training_data.dataset), desc=f'Training', position=0)
            
            for epoch in range(self.epochs):
            # training case 
                self.bar.set_description(f"Training epoch {epoch+1}")
                self.bar.total = len(self.training_data.dataset)

                self.training(epoch)

                # reset progress bar
                self.bar.n = 0
                self.bar.last_print_n = 0
                self.bar.refresh()

                # test case
                self.bar.set_description(f"Testing epoch {epoch+1}")
                self.bar.total = len(self.testing_data.dataset)
                
                auc_list.append(self.testing(epoch))

                # reset progress bar
                self.bar.n = 0
                self.bar.last_print_n = 0
                self.bar.refresh()
                
        return auc_list

    def write_results(self, output, file):
        """
        Helper function that writes the output of the training/testing loop into a .txt file in the output_folder.

        Args:
            output (str): Message
            file (str): File path
        """
        os.makedirs(OUTPUT, exist_ok=True)
        
        with open(os.path.join(OUTPUT, f'{file}.txt'), "a") as file:
            file.write(output)

    def training(self, epoch):
        """
        Runs a training process for a given epoch on the training set. It saves the performance metrics to a file.

        Args:
            epoch (int): Current epoch of training
        """
        # init stats
        avg_loss = 0.0
        T = 0
        total = 0
        TN = 0
        TP = 0
        P = 0
        N = 0

        for i, data in enumerate(self.training_data):
            # update progress bar
            self.bar.update(data['input'].size(0))

            # send data to GPU/CPU
            data = {key: value.to(DEVICE) for key, value in data.items()}

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
            preds = sigmoid(preds)
            preds_th = torch.ge(preds, THRESHOLD).int()

            # compare with the label and count correct classifications
            T += (preds_th == labels).sum().item()
            # check whether model is only predicting 0
            TN += ((preds_th == 0) & (labels == 0)).sum().item()
            TP += ((preds_th == 1) & (labels == 1)).sum().item()
            P += (labels == 1).sum().item()
            N += (labels == 0).sum().item()
            # sump up total number of labels in batch
            total += labels.nelement()

        # update learning rate scheduler
        self.scheduler.step()

        # print stats
        message = f"\nTraining epoch {epoch+1}\nAvg. training loss: {avg_loss / len(self.training_data):.2f}, Accuracy: {T / total:.2f}, TPR: {TP/P:.2f}, TNR: {TN/N:.2f}\n"
        print(message)

        # write in results
        self.write_results(message, self.train_res)

    def testing(self, epoch):
        """
        Runs a testing process for a given epoch on the training set. It saves the performance metrics to a file.

        Args:
            epoch (int): Current epoch of testing
        """

        # model to evaluation mode
        self.model.eval()

        auc = None
        avg_loss = 0.0
        T = 0
        total = 0
        TN = 0
        TP = 0
        P = 0
        N = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for i, data in enumerate(self.testing_data):
                # update progress bar
                self.bar.update(data['input'].size(0))

                # send data to GPU/CPU
                data = {key: value.to(DEVICE)
                        for key, value in data.items()}

                # labels convert to float()
                labels = data['labels'].to(torch.float)

                # forward pass: comments through model
                preds = self.model.forward(data['input'])

                loss = self.criterion(preds, labels)

                avg_loss += loss.item()

                # compute measures
                # use THRESHOLD to determine which of the outputs are considered True
                sigmoid = torch.nn.Sigmoid()  
                preds = sigmoid(preds)
                predictions = torch.ge(preds, THRESHOLD).int()

                # compare with the label and count correct classifications
                T += (predictions == labels).sum().item()

                # check whether model is only predicting 0
                TN += ((predictions == 0) & (labels == 0)).sum().item()
                TP += ((predictions == 1) & (labels == 1)).sum().item()
                P += (labels == 1).sum().item()
                N += (labels == 0).sum().item()

                # sum up total number of labels in batch
                total += labels.nelement()

                # labels and predictions for AUC
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

            # average AUC-score over all classes, for individual one vs. rest score: average=None
            auc = roc_auc_score(all_labels, all_predictions)

        # print stats for testing
        message = f"\nTesting epoch {epoch+1}\nAvg. testing loss: {avg_loss / len(self.testing_data):.2f}, avg. ROC-AUC: {auc:.2f}, Accuracy: {T / total:.2f}, TPR: {TP/P:.2f}, TNR: {TN/N:.2f}\n"
        print(message)

        # write results
        self.write_results(message, self.test_res)

        # Set the model back to training mode
        self.model.train()

        return auc
