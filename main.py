from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertModel, BertConfig

import custom_datasets
import models
from params import *
import training

# tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Choose an appropriate tokenizer

# Download pretrained weights from huggingface (for the base BERT)
bert_base = "bert-base-uncased"
configuration = BertConfig.from_pretrained(bert_base)
pretrained_model = BertModel.from_pretrained(bert_base, config=configuration)

# TASK SHEET: load_data
def load_data(dataset:str, transformation=None, n_train:int=None, n_test:int=None, batch_size=BATCH_SIZE, shuffle=True): # transformation callable
    """
    TASK SHEET: Function to load dataset in path indicated by name, specify number of training and testing samples, apply transformations and return dataloader

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

        # initialize dataloader for training and testing
        train = DataLoader(train, batch_size, shuffle)
        test = DataLoader(test, batch_size, shuffle)
        return train, test
    
    else:
        raise NotImplementedError("Dataset not implemented")
    
# TASK SHEET: show
def show(x, outfile:str=None): # can have more args
    """
    TASK SHEET: Function to visualize stuff

    Args:
        dataset (str): name of dataset
        transformation (callable): transformation to apply to the data (default: None)
        att_head_dim (int): dimension of each attention head
        n_train (int): number of training samples (default: None)
        n_test (int): number of testing samples (default: None)

    Returns:
        Tuple[torch.utils.data.DataLoader]: training and testing dataloader
    """   
    pass

def main():
    """
    Main function to load the data for toxic comment classification, set up a BERT model and run a training 

    - loads the dataset of specified length, batch size and transformations into a dataloader
    - sets up a BERT model for specified vocabulary size, model dimension, pretrained BERT base model, number of encoders and number of attention heads per encoder
    - starts training the BERT model

    """
    # Training (for cluster)

    # load the entire training data (length dataset_length) into train
    train_loader, _ = load_data("jigsaw_toxicity_pred", transformation=tokenizer, n_train=TRAIN_LENGTH, n_test=None, batch_size=BATCH_SIZE, shuffle=True)
    # train, _ = load_data("jigsaw_toxicity_pred", transformation=tokenizer, n_train=128, n_test=None)

    # set up BERT model with toxic multilabel classification head
    berti = models.Model(vocab_size=VOCAB_SIZE, model_dimension=EMBED_SIZE, pretrained_model=pretrained_model, number_layers=12, number_heads=12)

    # train model (device to be updated according to cluster GPU)
    training.TrainBERT(berti, train_loader, EPOCHS, device=DEVICE)


if __name__ == "__main__":
    main()



