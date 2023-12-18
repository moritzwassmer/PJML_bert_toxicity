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
            val = DataLoader(val, batch_size, shuffle)

            return train, test, val
        
        # if no validation set required
        else:
            # initialize dataloader for testing
            test = DataLoader(test, batch_size, shuffle)

            return train, test

    else:
        raise NotImplementedError("Dataset not implemented")
    
# Task sheet function: method ={"bert_base", "bert_discr_lr"}
def train_apply(method="bert_base", dataset="jigsaw_toxicity_pred"):

    # default: BERTBase
    berti = models.Model()

    # define batch size
    for batch_size in HYPER_PARAMS['batch_size']:
        train_loader, test_loader, val_loader = load_data(dataset, transformation=TOKENIZER, n_train=TRAIN_LENGTH, n_test=TEST_LENGTH, n_val=VAL_LENGTH, batch_size=batch_size, shuffle=True)
        best_model = [None, 0, None]

        for learning_rate in HYPER_PARAMS['learning_rate']:
            # hyperparameter stats
            info = f"\nHyperparameters: batch size: {batch_size}, learning rate: {learning_rate}\n"
            print(info)
                    
            # assign epochs and learning rate
            if method == 'bert_base':
                trainer = training.TrainBERT(berti, train_dataloader=train_loader, test_dataloader=test_loader, epochs=HYPER_PARAMS['epochs'], learning_rate=learning_rate, info=info)
            elif method == 'bert_discr_lr':
                trainer = training.TrainBERT(berti, scheduler=method, train_dataloader=train_loader, test_dataloader=test_loader, epochs=HYPER_PARAMS['epochs'], learning_rate=learning_rate, info=info)
            # default
            else:
                trainer = training.TrainBERT(berti, train_loader, test_loader, epochs=HYPER_PARAMS['epochs'], learning_rate=learning_rate, info=info)

            auc_list = trainer.run()
            # select best performing model
            for i in range(len(auc_list)):
                if auc_list[i] > best_model[1]:
                    # save: [model, auc value, hyperparameter info, epochs]
                    best_model = [berti, auc_list[i], info, i]
    print(f'Optimal hyperparameters are: {best_model[2][:-1]}, epochs: {best_model[3]+1}, with an avg. ROC-AUC of: {best_model[1]:.2f}\n')

    # validate
    validator = training.TrainBERT(best_model[0],test_dataloader=val_loader, epochs=1, validate=True, info="Validiation\n" + best_model[2])
    # auc_val = validator.run()
    # print(f'Optimal hyperparameters are: {best_model[2][:-1]}, epochs: {best_model[3]+1}, with a ROC-AUC on validation set of: {auc_val[0]:.2f}')
    return validator.run() 