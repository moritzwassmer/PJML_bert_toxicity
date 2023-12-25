import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from params import *


def write_results(output, file):
    """
    Helper function that writes the output of the training/testing loop into a .txt file in the output_folder.

    Args:
        output (str): Message
        file (str): File path
    """
    os.makedirs(OUTPUT, exist_ok=True)

    with open(os.path.join(OUTPUT, f'{file}.txt'), "a") as file:
        file.write(output)


def calc_metrics(labels, predictions, avg_loss, len_dataset, epoch=0):
    T, TN, TP, FP, FN, P, N = 0, 0, 0, 0, 0, 0, 0
    total = 0

    # COMPUTE METRICS
    sigmoid = torch.nn.Sigmoid()
    preds = sigmoid(predictions)
    preds_th = torch.ge(preds, THRESHOLD).int()

    T += (preds_th == labels).sum().item()
    TP += ((preds_th == 1) & (labels == 1)).sum().item()
    FP += ((preds_th == 1) & (labels == 0)).sum().item()
    TN += ((preds_th == 0) & (labels == 0)).sum().item()
    FN += ((preds_th == 0) & (labels == 1)).sum().item()
    P += (labels == 1).sum().item()
    N += (labels == 0).sum().item()
    # sump up total number of labels in batch
    total += labels.nelement()

    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    metrics = {
        'epoch': epoch+1,
        'avg_loss': avg_loss / len_dataset,
        'roc_auc': roc_auc_score(labels, predictions, average='macro', multi_class='ovr'),
        'accuracy': T / total,
        'TPR': TP/P,
        'FPR': FP/(FP+TN),
        'TNR': TN/N,
        'FNR': FN/(FN+TP),
        'toxic': roc_auc_score(np.array(labels)[:, 0], np.array(predictions)[:, 0], average='macro', multi_class='ovr'),
        'severe_toxic': roc_auc_score(np.array(labels)[:, 1], np.array(predictions)[:, 1], average='macro', multi_class='ovr'),
        'obscene': roc_auc_score(np.array(labels)[:, 2], np.array(predictions)[:, 2], average='macro', multi_class='ovr'),
        'threat': roc_auc_score(np.array(labels)[:, 3], np.array(predictions)[:, 3], average='macro', multi_class='ovr'),
        'insult': roc_auc_score(np.array(labels)[:, 4], np.array(predictions)[:, 4], average='macro', multi_class='ovr'),
        'identity_hate': roc_auc_score(np.array(labels)[:, 5], np.array(predictions)[:, 5], average='macro', multi_class='ovr')
    }
    return metrics


class DiscriminativeLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, start_lr, last_epoch=-1, decay=DECAY):
        self.decay = decay
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        learning_rate = self.start_lr
        # apply discriminative layer rate layer-wise
        decay_lrs = [learning_rate * (self.decay**i)
                     for i in range(len(self.optimizer.param_groups))]
        """for param_group, decay_lr in zip(self.optimizer.param_groups, decay_lrs):
            param_group['lr'] = decay_lr"""
            # print(param_group['lr'])
        return decay_lrs


class SlantedLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, iterations, start_lr, ratio=32, eta_max=LEARNING_RATE, cut_frac=0.1, last_epoch=-1, decay=DECAY, discriminative = True):
        self.eta_max = eta_max
        self.cut = iterations * cut_frac
        self.cut_frac = cut_frac
        self.iterations = iterations
        self.ratio = ratio
        self.decay = decay
        self.start_lr = start_lr
        self.t = last_epoch
        self.discriminative = discriminative
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        p = 0
        self.t += 1
        if self.t < self.cut:
            learning_rate = (self.eta_max - self.start_lr) / \
                self.cut*self.t + self.start_lr
        else:
            p = 1 - ((self.t - self.cut) /
                     (self.cut * (1 / self.cut_frac - 1)))
            learning_rate = self.eta_max * \
                ((1 + p * (self.ratio - 1)) / self.ratio)
        #learning_rate = self.start_lr
        # apply discriminative layer rate layer-wise
        if self.discriminative:
            decay_lrs = [learning_rate * (self.decay**i)
                        for i in range(len(self.optimizer.param_groups))]
            return decay_lrs
        else:
            no_decay_lrs = [learning_rate
                        for i in range(len(self.optimizer.param_groups))]
            #print(no_decay_lrs)
            return no_decay_lrs
"""for param_group, decay_lr in zip(self.optimizer.param_groups, decay_lrs):
            param_group['lr'] = decay_lr"""
# print(param_group['lr'])
#print(decay_lrs)
        

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

    def __init__(self, model, method='bert_base', train_dataloader=None, test_dataloader=None, epochs=EPOCHS, learning_rate=LEARNING_RATE, validate=False, info=None):
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
        self.metrics = None
        # scheduler method
        self.method = method
        # model to device
        self.model.to(DEVICE)

        if self.method == 'bert_discr_lr':
            self.train_res = DISCR_TRAIN
            self.test_res = DISCR_TEST

            # optimizer: Adam
            self.optimizer = optim.Adam(
                self.create_param_groups(), lr=learning_rate)

            # check number of parameter groups
            # num_param_groups = len(self.optimizer.param_groups)
            # print(f"Anzahl der Parametergruppen: {num_param_groups}")

            # learning rate scheduler
            iterations = self.epochs*len(train_dataloader)
            self.scheduler = DiscriminativeLRScheduler(
                self.optimizer, iterations, learning_rate)

            # Print the current learning rate
            # current_lr = self.optimizer.param_groups[0]['lr']
            # check learning rates (write in 'learning_rates' in output_folder)
            # write_results(str(current_lr) +'\n', "learning_rates")
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="mean",
                pos_weight=torch.Tensor(WEIGHTS_LIST).to(DEVICE)
            )

        elif self.method == 'bert_slanted_lr':
            self.train_res = SLANTED_TRAIN
            self.test_res = SLANTED_TEST

            # optimizer: Adam
            self.optimizer = optim.Adam(
                self.create_param_groups(), lr=learning_rate)

            # check number of parameter groups
            # num_param_groups = len(self.optimizer.param_groups)
            # print(f"Anzahl der Parametergruppen: {num_param_groups}")

            # learning rate scheduler
            iterations = self.epochs*len(train_dataloader)
            
            """self.scheduler = SlantedLRScheduler(
                self.optimizer, iterations, learning_rate)"""
            self.scheduler = SlantedLRScheduler(self.optimizer, iterations,0, eta_max=learning_rate)
            """
            __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):

            def __init__(self, optimizer, iterations, start_lr, ratio=32, eta_max=ETA_MAX, cut_frac=0.1, last_epoch=-1, decay=DECAY):"""
            
            
            """self.scheduler = STLR(
                self.optimizer, max_mul=2, ratio=4, steps_per_cycle=iterations, decay=DECAY, last_epoch=-1) # ratio = batch_size?"""
            
            # Print the current learning rate
            #current_lr = self.optimizer.param_groups[0]['lr']
            #print(current_lr)
            # check learning rates (write in 'learning_rates' in output_folder)
            # write_results(str(current_lr) +'\n', "learning_rates")
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="mean",
                pos_weight=torch.Tensor(WEIGHTS_LIST).to(DEVICE)
            )

        # default
        elif self.method == 'bert_base':
            self.train_res = BASE_TRAIN
            self.test_res = BASE_TEST
            # optimizer: Adam

            """self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=0.01)"""
            
            self.optimizer = optim.Adam(
                self.create_param_groups(), lr=learning_rate, weight_decay=0.01)

            # lr scheduler
            iterations = self.epochs*len(train_dataloader)
            self.scheduler = SlantedLRScheduler(self.optimizer, iterations, 0, eta_max=learning_rate, discriminative=False, cut_frac=10000/len(train_dataloader))

            # loss function
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="mean",
                pos_weight=torch.Tensor(WEIGHTS_LIST).to(DEVICE)
            )
        else:
            raise NotImplementedError()
        self.info = info
        self.validate = validate

    def create_param_groups(self): # TODO SIMPLIFY
        toxic_comment = []
        encoders = [[] for _ in range(12)]
        embedding = []

        # extract layers
        for idx, (name, module) in enumerate(self.model.named_parameters()):
            parts = name.split('.')
            # group layers
            if parts[0] == 'toxic_comment':
                # group parameters in list
                toxic_comment.append(module)
                # group encoders
            elif len(parts) > 3 and parts[3].isdigit():
                encoder_num = int(parts[3])
                encoders[encoder_num].append(module)
            elif parts[1] == 'embedding':
                embedding.append(module)

        param_groups = [
            {'params': toxic_comment, 'lr': 0},
            *[{'params': encoder, 'lr': 0} for encoder in encoders[::-1]],
            {'params': embedding, 'lr': 0}
        ]
        # print(param_groups)
        return param_groups

    def run(self):
        auc_list = []
        # write hyperparameters in output
        write_results(self.info, self.train_res)
        write_results(self.info, self.test_res)

        if self.validate:
            self.bar = tqdm(position=0, total=len(
                self.testing_data.dataset), desc="Validation")

            labels, predictions, avg_loss, len_data = self.testing(self.epochs)

            self.bar.close()
            return labels, predictions, avg_loss, len_data

        # run training, testing
        else:
            self.bar = tqdm(position=0, total=len(
                self.training_data.dataset), leave=False)
            for epoch in range(self.epochs):
                # training case
                self.bar.set_description(f"Training epoch {epoch+1}")
                self.bar.n = 0
                self.bar.last_print_n = 0
                self.bar.refresh()

                self.training(epoch)

                # reset progress bar
                self.bar.n = 0
                self.bar.last_print_n = 0
                self.bar.refresh()

                # test case
                self.bar.set_description(f"Testing epoch {epoch+1}")
                self.bar.total = len(self.testing_data.dataset)
                auc_list.append(self.testing(epoch))
            self.bar.close()
        return auc_list

    def training(self, epoch):
        """
        Runs a training process for a given epoch on the training set. It saves the performance metrics to a file.

        Args:
            epoch (int): Current epoch of training
        """
        avg_loss = 0.0
        all_labels = torch.tensor([], device=DEVICE)
        all_predictions = torch.tensor([], device=DEVICE)

        for i, data in enumerate(self.training_data):
            # update progress bar
            self.bar.update(data['input'].size(0))

            # send data to GPU/CPU
            data = {key: value.to(DEVICE) for key, value in data.items()}

            # labels convert to float()
            labels = data['labels'].to(torch.float)

            # forward pass: comments trough model
            preds = self.model.forward(data['input'])
            loss = self.criterion(preds, labels)#/RESCALING_FACTOR # TODO ALSO CHANGE IN TEST VALIDATION
            avg_loss += loss.item()

            # backward pass for training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_labels = torch.cat((all_labels, labels.detach()))
            all_predictions = torch.cat((all_predictions, preds.detach()))

            # update slanted triangular learning rate scheduler after every batch
            if  self.method == 'bert_slanted_lr' or self.method == 'bert_base':
                self.scheduler.step()
                # Print the current learning rate
                # current_lr = self.optimizer.param_groups[0]['lr']
                # check learning rates (write in 'learning_rates' in output_folder)
                # write_results(str(current_lr) + '\n', "learning_rates")

        # update normal learning rate scheduler
        if  self.method == 'bert_discr_lr':
            self.scheduler.step()

        self.metrics = calc_metrics(
            all_labels, all_predictions, avg_loss, len(self.training_data), epoch)

        # print metrics
        message = f"\nTraining epoch {self.metrics['epoch']}\nAvg. training loss: {self.metrics['avg_loss']:.2f}, ROC-AUC: {self.metrics['roc_auc']:.2f}, Accuracy: {self.metrics['accuracy']:.2f}, TPR: {self.metrics['TPR']:.2f}, TNR: {self.metrics['TNR']:.2f}\n"
        print(message)

        # write in results
        write_results(message, self.train_res)

    def testing(self, epoch):
        """
        Runs a testing process for a given epoch on the training set. It saves the performance metrics to a file.

        Args:
            epoch (int): Current epoch of testing
        """

        # model to evaluation method
        self.model.eval()

        avg_loss = 0.0
        all_labels = torch.tensor([], device=DEVICE)
        all_predictions = torch.tensor([], device=DEVICE)

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

                loss = self.criterion(preds, labels)#//RESCALING_FACTOR # TODO

                avg_loss += loss.item()

                # all_labels.extend(labels.cpu().detach().numpy()) # TODO move to GPU
                # all_predictions.extend(preds.cpu().detach().numpy()) # TODO move to GPU

                all_labels = torch.cat((all_labels, labels.detach()))
                all_predictions = torch.cat((all_predictions, preds.detach()))

            self.metrics = calc_metrics(
                all_labels, all_predictions, avg_loss, len(self.testing_data), epoch)

        if not self.validate:
            message = f"\nTesting epoch {self.metrics['epoch']}\nAvg. testing loss: {self.metrics['avg_loss']:.2f}, avg. ROC-AUC: {self.metrics['roc_auc']:.2f}, Accuracy: {self.metrics['accuracy']:.2f}, TPR: {self.metrics['TPR']:.2f}, FPR: {self.metrics['FPR']:.2f}, TNR: {self.metrics['TNR']:.2f}, FNR: {self.metrics['FNR']:.2f}\n"
            auc_classes = '\n'.join(
                [f'ROC-AUC for {label}: {self.metrics[label]:.2f}'for label in ORDER_LABELS])
            message = message + auc_classes + '\n'
            print(message)

            # write results
            write_results(message, self.test_res)

            # Set the model back to training method
            self.model.train()

            return self.metrics['roc_auc']
        else:
            return all_labels, all_predictions, avg_loss, len(self.testing_data)
