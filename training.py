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


def calc_metrics(labels, predictions, loss, len_dataset, epoch=0):
    """
    Calculates various metrics based on passed true labels and predictions.

    Args:
        labels (torch.Tensor): True labels for the data
        predictions (torch.Tensor): Predictions of the model for the same data
        loss (float): summed loss per epoch
        len_dataset (int): Length of the input dataset
        epoch (int): Current epoch number (optional, default: 0)

    Returns:
        dict: dictionary containing all the computed metrics, which are: epoch, abg_loss, roc_auc, accuracy, TPR, FPR, TNR, FNR, toxic (ROC-AUC), severe_toxic (ROC-AUC), 
        obscene (ROC-AUC), threat (ROC-AUC), insult (ROC-AUC), identity_hate (ROC-AUC)
    """
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
        'avg_loss': loss / len_dataset,
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
    """
    Discriminative learning rate scheduler, that adjusts the learning rate with a decay term for the layers of the model, so they decrease in depth. 

    Attributes:
        optimizer (torch.optim.Optimizer): Optimizer for which to define the learning rate
        start_lr (float): Initial learning rate
        last_epoch (int): INdex of the last epoch 
        decay (float): Decay rate applied per layer to the learning rate 
    """

    def __init__(self, optimizer, start_lr, last_epoch=-1, decay=DECAY):
        """
        Initializes the DiscriminativeLRScheduler.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer for which to define the learning rate
            start_lr (float): Initial learning rate
            last_epoch (int): Index of the last epoch (optional, default: -1)
            decay (float): Decay rate applied per layer to the learning rate (default: DECAY)
        """
        self.decay = decay
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Computes decaying learning rate for each parameter group. 

        Returns:
            list: A list of decaying learning rates for all the layers of the model
        """
        learning_rate = self.start_lr
        # apply discriminative layer rate layer-wise
        decay_lrs = [learning_rate * (self.decay**i)
                     for i in range(len(self.optimizer.param_groups))]
        return decay_lrs


class SlantedLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Slanted triangular learning rate scheduler that increases the learning rate until eta_max for a given ratio and then decreases it again . 
    It can also optionally apply a layer-wise decay term on the learning rate (discriminative=True).

    Attributes:
        optimizer (torch.optim.Optimizer): Optimizer for which to define the learning rate
        iterations (int): Number of iterations over which the learning rate has to be calculated 
        ratio (int): Change ratio for learning rate 
        eta_max (float, optional): Maximum learning rate 
        cut_frac (float, optional): Fraction of iteration at which the learning rate reaches the peak 
        last_epoch (int, optional): Index of the last epoch 
        decay (float): Decay term for optional discriminative layer-wise learning rate 
        discriminative (bool): Flag to apply discriminative layer 
    """

    def __init__(self, optimizer, iterations, ratio=32, eta_max=1e-05, cut_frac=0.1, last_epoch=-1, decay=DECAY, discriminative=True):
        """
        Initializes the SLantedLRScheduler. 

        Args: 
            optimizer (torch.optim.Optimizer): Optimizer for which to define the learning rate
            iterations (int): Number of iterations over which the learning rate has to be calculated (epochs*batches per epoch)
            ratio (int): Change ratio for learning rate (default: 32)
            eta_max (float, optional): Maximum learning rate (default: 1e-05)
            cut_frac (float, optional): Fraction of iteration at which the learning rate reaches the peak (default: 0.1)
            last_epoch (int, optional): Index of the last epoch (default: -1)
            decay (float): Decay term for optional discriminative layer-wise learning rate (default: DECAY)
            discriminative (bool): Flag to apply discriminative layer (default: True)
        """
        self.eta_max = eta_max
        self.cut = iterations * cut_frac
        self.cut_frac = cut_frac
        self.iterations = iterations
        self.ratio = ratio
        self.decay = decay
        self.t = last_epoch
        self.discriminative = discriminative
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Computes the slanted triangular learning rates for each layer. 

        Returns:
            list: List of computed learning rates for each layer of the model
        """
        p = 0
        self.t += 1
        if self.t < self.cut:
            p = (self.t/self.cut)
        else:
            p = 1 - ((self.t - self.cut) /
                     (self.cut * (1 / self.cut_frac - 1)))
        learning_rate = self.eta_max * \
            ((1 + p * (self.ratio - 1)) / self.ratio)
        # apply discriminative layer rate layer-wise (if discriminative is set)
        if self.discriminative:
            self.decay = 1
        decay_lrs = [learning_rate * (self.decay**i)
                     for i in range(len(self.optimizer.param_groups))]
        return decay_lrs


class TrainBERT:
    """
    Class to perform a training and a testing processes as well as a validation for the toxic comment classification model.
    This can be performed for different learning rate methods like "base", "slanted_discriminative" and "discriminative".

        Attributes:
            model (nn.Module): BERT-based toxic comment classification model
            epochs (int): Overall number of epochs
            method (str): Method for learning rate scheduling
            train_res (str): output name for the train results file
            test_res (str): output name for the test results file
            training_data (torch.utils.data.Dataloader): Dataloader for training 
            testing_data (torch.utils.data.DataLoader): Dataloader for testing
            iterations (int): Number of iterations over which the learning rate has to be calculated (epochs*batches per epoch)
            bar (tqdm.tqdm): Progress bar 
            metrics (dict): Dictionary to store the current metrics of the model run
            optimizer (torch.optim.Adam): Adam optimizer
            scheduler (torch.optim.lr_scheduler.StepLR): Scheduler for learning rate
            criterion (nn.BCEWithLogitsLoss): Binary cross-entropy loss with logits

        Methods:
            create_param_groups(): Creates list of dictionaries for layers of the model and learning rates
            run(): Runs either validation or training and testing
            training(epoch): Performs a training of the model for the epoch
            testing(epoch): Performs a test of the model for the epoch
    """

    def __init__(self, model, method='base', train_dataloader=None, test_dataloader=None, epochs=4, learning_rate=1e-05, validate=False, info=None):
        """
        Initializes a validation or a training and testing processes for a BERT-based toxic comment classification model.
        Allows to choose between the methods "base", "slanted_discriminative" and "discriminative" for learning rate scheduling.

        Args: 
            model (nn.Module): BERT-based toxic comment classification model
            method (str): Method for learning rate scheduling (default: 'base')
            train_dataloader (torch.utils.data.DataLoader): Dataloader for training (default: None)
            test_dataloader (torch.utils.data.DataLoader): Dataloader for testing (default: None)
            epochs (int): Number of epochs for training (default: 4)
            learning_rate (float): Initial learning rate for optimization (default: 1e-05)
            validate (bool): Flag to determine if validation should be performed (default: false)
            info (str): Additional information or metadata (default: None)
        """

        # parameters
        self.model = model
        self.epochs = epochs
        self.training_data = train_dataloader
        self.testing_data = test_dataloader
        if train_dataloader is None:
            self.iterations = self.epochs*len(test_dataloader)
        else:
            self.iterations = self.epochs*len(train_dataloader)

        self.bar = None
        self.metrics = None
        # scheduler method
        self.method = method
        # model to device
        self.model.to(DEVICE)

        if self.method == 'discriminative':
            self.train_res = DISCR_TRAIN
            self.test_res = DISCR_TEST

            # optimizer: Adam
            self.optimizer = optim.Adam(
                self.create_param_groups(), lr=learning_rate)

            # learning rate scheduler
            self.scheduler = DiscriminativeLRScheduler(
                self.optimizer, learning_rate)

            # loss function
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="mean",
                pos_weight=torch.Tensor(WEIGHTS_LIST).to(DEVICE)
            )

        elif self.method == 'slanted_discriminative':
            self.train_res = SLANTED_TRAIN
            self.test_res = SLANTED_TEST

            # optimizer: Adam
            self.optimizer = optim.Adam(
                self.create_param_groups(), lr=learning_rate)

            # check number of parameter groups
            # num_param_groups = len(self.optimizer.param_groups)
            # print(f"Anzahl der Parametergruppen: {num_param_groups}")

            # lr scheduler
            """self.scheduler = SlantedLRScheduler(
                self.optimizer, iterations, learning_rate)"""
            self.scheduler = SlantedLRScheduler(
                self.optimizer, self.iterations, eta_max=learning_rate, discriminative=True)

            self.criterion = nn.BCEWithLogitsLoss(
                reduction="mean",
                pos_weight=torch.Tensor(WEIGHTS_LIST).to(DEVICE)
            )
        elif self.method == 'base':
            self.train_res = BASE_TRAIN
            self.test_res = BASE_TEST

            # optimizer: Adam
            self.optimizer = optim.Adam(
                self.create_param_groups(), lr=learning_rate, weight_decay=0.01)

            # lr scheduler
            self.scheduler = SlantedLRScheduler(
                self.optimizer, self.iterations, eta_max=learning_rate, discriminative=False)

            # loss function
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="mean",
                pos_weight=torch.Tensor(WEIGHTS_LIST).to(DEVICE)
            )
        else:
            raise NotImplementedError()
        self.info = info
        self.validate = validate

    def create_param_groups(self):  
        """
        Groups the parameters of the BERT-toxic comment classification model by layer into parameter groups in reverse order.
        The parameters a grouped into "toxic_comment", the encoders and an embedding layer.

        Returns:
            list: A list of dictionaries, each dictionary representing a parameter group with its associated learning rate
        """
        toxic_comment = []
        encoders = [[] for _ in range(12)]
        embedding = []

        # extract layers
        for i,(name, module) in enumerate(self.model.named_parameters()):
            parts = name.split('.')
            # group layers
            if parts[0] == 'toxic_comment':
                # group parameters in list
                toxic_comment.append(module)
                # group encoders
            elif len(parts) > 3 and parts[3].isdigit():
                encoders[int(parts[3])].append(module)
            elif parts[1] == 'embedding':
                embedding.append(module)

        return [
            {'params': toxic_comment, 'lr': 0},
            *[{'params': encoder, 'lr': 0} for encoder in encoders[::-1]],
            {'params': embedding, 'lr': 0}
        ]

    def run(self):
        """
        Executes the training and testing process or validation, if the "validate"-flag is set.

        Returns:
            list or tuple: 
                If validation is performed, it returns a tuple comprising:
                    labels: Labels of the validation set
                    predictions: Model predictions for validation set
                    avg_loss: Loss sum of the validation run
                    len_data: Length of the validation set
                Otherwise, it returns a list of ROC-AUC scores computed for each epoch during testing
        """
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
        Runs a training process for a given (single) training epoch on the training set. It saves the performance metrics to a specified file.

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
            # /RESCALING_FACTOR # TODO ALSO CHANGE IN TEST VALIDATION
            loss = self.criterion(preds, labels)
            avg_loss += loss.item()

            # backward pass for training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_labels = torch.cat((all_labels, labels.detach()))
            all_predictions = torch.cat((all_predictions, preds.detach()))

            # update slanted triangular learning rate scheduler after every batch
            if self.method == 'slanted_discriminative' or self.method == 'base':
                self.scheduler.step()

        # update normal learning rate scheduler
        if self.method == 'discriminative':
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
        Runs a testing process for a given epoch on the training set. It saves the performance metrics to a file. In the "validation"-case it returns 
        labels, predictions and loss sum als well as the dataset length. If it is set to test, it returns the average ROC-AUC value for all classes. 

        Args:
            epoch (int): Current epoch of testing

        Returns:
            If validation is performed:
                Union[Dict, float, torch.Tensor, int]: tuple containing labels, predictions, average loss and length of testing dataset
            Otherwise:
                float: ROC-AUC value of the validation set 
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

                # //RESCALING_FACTOR # TODO
                loss = self.criterion(preds, labels)

                avg_loss += loss.item()

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
