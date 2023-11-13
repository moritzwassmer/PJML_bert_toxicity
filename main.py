from datasets import load_dataset
from itertools import chain
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertModel, BertConfig

import customdataset
import embedding 
import model
from params import *
import training

# tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Choose an appropriate tokenizer

# dataset
toxic_dataset = load_dataset("jigsaw_toxicity_pred", data_dir=TOXIC)
dataset_length = len(toxic_dataset["train"])

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
        train = customdataset.ToxicComment(
            tokenizer=transformation,
            seq_len=SEQ_LEN,
            split="train",
            n_rows=n_train
        )
        
        test = customdataset.ToxicComment(
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
    train, _ = load_data("jigsaw_toxicity_pred", transformation=tokenizer, n_train=dataset_length, n_test=None)

    # set up dataloader
    train_loader = DataLoader(train, batch_size=128, shuffle=True)

    # set up BERT model
    berti = model.Model(vocab_size=VOCAB_SIZE, model_dimension=EMBED_SIZE, pretrained_model=pretrained_model, number_layers=12, number_heads=12)

    # number of epochs
    epochs = 10

    # train model (device to be updated according to cluster GPU)
    bert_trainer = training.TrainBERT(berti, train_loader, epochs, device='cpu')

"""
if __name__ == "__main__":
    main()
"""

##########################################  TEST STUFF  #####################################################
"""
# tokenizer
print(tokenizer.truncation_side)
print(tokenizer.model_max_length) # we might need to fixate this
print(tokenizer.mask_token)
print(tokenizer.vocab['[MASK]'])

# example usage
text = "hi i am moritz, who are you ?"#["hi i am moritz", "no you are not moritz, you are kevin"]
encoded_input = tokenizer(text)#,padding=True, truncation=True)
# , return_tensors='pt') use this for pt tensors
print(encoded_input)
print(encoded_input["input_ids"])
print(tokenizer.decode(encoded_input["input_ids"]))

# test dataset
from torch.utils.data import DataLoader
dataloader = DataLoader(toxic_dataset["train"], batch_size=1, shuffle = True)
dataset_length = len(toxic_dataset["train"])
print("Length of dataset:", dataset_length)
batch = next(iter(dataloader))
print(batch)

# test tokenizer
encoded_input = tokenizer(batch["comment_text"])
print(encoded_input)
flattened = list(chain(*(encoded_input["input_ids"])))
print(tokenizer.decode(flattened))

# test customized dataset
test2 = customdataset.ToxicComment(tokenizer=tokenizer, seq_len=SEQ_LEN, split = "train", n_rows = 100)
print(len(test2))
dl2 = DataLoader(test2,batch_size=1,shuffle=False)
batch = next(iter(dl2))
print(batch)
print(len(batch))
print(len(batch["input"][0]))

# embedding test: tokenized sequence
sample_seq = batch['input'][0] 
print(f'sample_seq size {sample_seq.size()}')
print(sample_seq)
bert = embedding.BERTEmbedding(VOCAB_SIZE, SEQ_LEN)
batch_embed = bert(batch['input'][0].long())
print(batch_embed.size())
"""
# Training test
# set up tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# load test dataset
train, test = load_data("jigsaw_toxicity_pred", transformation=tokenizer, n_train=128, n_test=None)

# set up dataloader
train_loader = DataLoader(train, batch_size=32, shuffle=True)

# set up BERT model (pass the pretrained weights(model))
bert = model.Model(vocab_size=VOCAB_SIZE, model_dimension=EMBED_SIZE, pretrained_model=pretrained_model, number_layers=12, number_heads=12)

# number of epochs
epochs = 5

# train model
bert_trainer = training.TrainBERT(bert, train_loader, epochs, device='cpu')

####################################################################################################
