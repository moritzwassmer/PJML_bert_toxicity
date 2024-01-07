from torch.utils.data import DataLoader
import custom_datasets
import models
from params import *
import training
from torch.utils.data import random_split


def load_data(dataset: str, transformation=None, n_train: int = None, n_test: int = None, n_val: int = None, batch_size=32, shuffle=True):
    """
    Function to load dataset specified by name, apply transformations and return dataloaders for training, testing and validation if required. 
    The test set is split in half into test and validation set, if validation set is required (half the size of n_test each).

    Args:
        dataset (str): name of dataset
        transformation (callable): transformation to apply to the data (default: None)
        n_train (int): Size of training set to load (default: None) 
        n_test (int): Size of testing set to load (default: None)
        n_val (int): Size of validation set to load (default: None)
        batch_size (int): Batch size (default: 32)
        shuffle (bool): Whether data is loaded in randomized order (default: True)


    Returns:
        If validation set is required:
            Tuple[torch.utils.data.DataLoader]: training, testing and validation dataloader
        Otherwise:
            Tuple[torch.utils.data.DataLoader]: training and testing dataloader

    Raises:
        NotImplementedError: If the specified dataset does not exist
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
            val = DataLoader(val, batch_size, shuffle)

            return train, test, val

        # if no validation set required
        else:
            # initialize dataloader for testing
            test = DataLoader(test, batch_size, shuffle)

            return train, test

    else:
        raise NotImplementedError("Dataset not implemented")


# Task sheet function: method: {"base", "slanted_discriminative"}
def train_apply(method="base", dataset="jigsaw_toxicity_pred"):
    """
    TASK SHEET: Trains a BERT-based model for toxic comment classification with the specified learning rate scheduling method on a given dataset and performs model selection 
    based on the given hyperparameters. It returns the validation results. The learning rate scheduler methods are "base", "slanted_discriminative".

    Args:
        method (str): Learning rate method to use for training. Available are "base" and "slanted_discriminative" (default: "base")
        dataset (str): Name of the dataset to train on (default: "jigsaw_toxicity_pred")
    
    Returns:
        tuple (labels, predictions, avg_loss, len_data): Validation results after training the model
    """


    for batch_size in HYPER_PARAMS['batch_size']:
        train_loader, test_loader, val_loader = load_data(
            dataset, transformation=TOKENIZER, n_train=TRAIN_LENGTH, n_test=TEST_LENGTH, n_val=VAL_LENGTH, batch_size=batch_size, shuffle=True)
        best_model = [None, 0, None]

        for learning_rate in HYPER_PARAMS['learning_rate']:
            berti = models.Model()
            
            # hyperparameter stats
            info = f"\nHyperparameters: batch size: {batch_size}, learning rate: {learning_rate}\n"
            print(info)

            # assign epochs and learning rate
            if method == 'base':
                trainer = training.TrainBERT(berti,  method=method, train_dataloader=train_loader,
                                             test_dataloader=test_loader, epochs=HYPER_PARAMS['epochs'], learning_rate=learning_rate, info=info)

            elif method == 'slanted_discriminative':
                trainer = training.TrainBERT(berti, method=method, train_dataloader=train_loader,
                                             test_dataloader=test_loader, epochs=HYPER_PARAMS['epochs'], learning_rate=learning_rate, info=info)
            # default
            else:
                trainer = training.TrainBERT(berti, method=method, train_dataloader=train_loader,
                                             test_dataloader=test_loader, epochs=HYPER_PARAMS['epochs'], learning_rate=learning_rate, info=info)

            auc_list = trainer.run()
            # select best performing model
            for i in range(len(auc_list)):
                if auc_list[i] > best_model[1]:
                    # save: [model, auc value, hyperparameter info, epochs]
                    best_model = [berti, auc_list[i], info, i]
    message = f'\nOptimal hyperparameters are: {best_model[2][:-1]}, epochs: {best_model[3]+1} with an avg. ROC-AUC of: {best_model[1]:.2f}\n'
    print(message)

    # validate 
    validator = training.TrainBERT(
        best_model[0], method=method, test_dataloader=val_loader, epochs=1, validate=True, info=message)
    return validator.run()
