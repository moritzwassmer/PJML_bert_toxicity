from torch.utils.data import DataLoader
import custom_datasets
import models
from params import *
import training
from torch.utils.data import random_split

def load_data(dataset: str, transformation=None, n_train: int = None, n_test: int = None, n_val: int = None, batch_size=BATCH_SIZE, shuffle=True):  
    """
    Function to load dataset in path indicated by name, specify number of training and testing samples, apply transformations and return dataloader.

    Args:
        dataset (str): name of dataset
        transformation (callable): transformation to apply to the data (default: None)
        att_head_dim (int): dimension of each attention head
        n_train (int): number of training samples (default: None)
        n_test (int): number of testing samples (default: None)

    Returns:
        Tuple[torch.utils.data.DataLoader]: training and testing dataloader
    """
        
    if dataset == "jigsaw_toxicity_pred":
        train = custom_datasets.ToxicComment(
            tokenizer=transformation,
            seq_len=SEQ_LEN,
            split="train",
            n_rows=n_train
        )

        test = custom_datasets.ToxicComment(
            tokenizer=transformation,
            seq_len=SEQ_LEN,
            split="test",
            n_rows=n_test
        )
        # initialize dataloader for training
        train = DataLoader(train, batch_size, shuffle)

        # if validation set required
        if n_val is not None:
            test, val = random_split(test, [n_test-n_val, n_val])

            # initialize dataloader for testing and validation
            test = DataLoader(test, batch_size, shuffle)
            val = DataLoader(test, batch_size, shuffle)

            return train, test, val
        
        # if no validation set required
        else:
            # initialize dataloader for testing
            test = DataLoader(test, batch_size, shuffle)

            return train, test

    else:
        raise NotImplementedError("Dataset not implemented")
    
# Task sheet function: method ={"train_test", "hyperparameter"}
def train_apply(method="method_name", dataset="jigsaw_toxicity_pred"):
    if method == "train_test":
        # load the entire training data (length dataset_length) into train, test
        train_loader, test_loader = load_data(dataset, transformation=TOKENIZER, n_train=TRAIN_LENGTH, n_test=TEST_LENGTH, batch_size=BATCH_SIZE, shuffle=True)

        # set up BERT model with toxic multi-label classification head
        berti = models.Model()

        # train model (device to be updated according to cluster GPU)
        training.TrainBERT(berti, train_loader, test_loader)

    elif method == "hyperparameter":
        
        for batch_size in HYPER_PARAMS['batch_size']:
            # define batch size
            train_loader, test_loader, val_loader = load_data(dataset, transformation=TOKENIZER, n_train=TRAIN_LENGTH, n_test=TEST_LENGTH, n_val=VAL_LENGTH, batch_size=batch_size, shuffle=True)

            for learning_rate in HYPER_PARAMS['learning_rate']:
                for epochs in HYPER_PARAMS['epochs']:
                    # set up new model
                    berti = models.Model()
                    
                    # assign epochs and learning rate
                    training.TrainBERT(berti, train_loader, test_loader, epochs=epochs, learning_rate=learning_rate)

                    # TODO returns predicted labels for the test data
                    # TODO save test metrics to compare
                    # TODO select best model and validate