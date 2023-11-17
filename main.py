from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertModel, BertConfig

import custom_datasets
import model
from params import *
import training

# tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Choose an appropriate tokenizer

# Download pretrained weights from huggingface (for the base BERT)
bert_base = "bert-base-uncased"
configuration = BertConfig.from_pretrained(bert_base)
pretrained_model = BertModel.from_pretrained(bert_base, config=configuration)

# TASK SHEET: load_data
def load_data(dataset:str, transformation=None, n_train:int=None, n_test:int=None): # transformation callable
    """
    if dataset == "bookcorpus":
        train = Bookcorpus(
            tokenizer=transformation,
            seq_len=SEQ_LEN,
            split="train",
            n_rows=n_train
        )
        return train, None
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
        return train, test
    
    else:
        raise NotImplementedError("Dataset not implemented")
    
# TASK SHEET: show
def show(x, outfile:str=None): # can have more args
    pass

def main():
    # Training (for cluster)

    # load the entire training data (length dataset_length) into train
    train, _ = load_data("jigsaw_toxicity_pred", transformation=tokenizer, n_train=TRAIN_LENGTH, n_test=None)
    # train, _ = load_data("jigsaw_toxicity_pred", transformation=tokenizer, n_train=128, n_test=None)

    # set up dataloader
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    # set up BERT model
    berti = model.Model(vocab_size=VOCAB_SIZE, model_dimension=EMBED_SIZE, pretrained_model=pretrained_model, number_layers=12, number_heads=12)

    # train model (device to be updated according to cluster GPU)
    bert_trainer = training.TrainBERT(berti, train_loader, EPOCHS, device=DEVICE)


if __name__ == "__main__":
    main()



