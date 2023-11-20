from datasets import load_dataset
import torch
from torch.utils.data import Dataset

from params import *

class ToxicComment(Dataset):
    """
        Dataset class for Toxic Comment Classification.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text data.
            seq_len (int, optional): Maximum sequence length. Default is SEQ_LEN.
            split (str, optional): Dataset split, either 'train' or 'test'. Default is 'train'.
            n_rows (int, optional): Number of rows to load from the dataset. Default is None.

        Raises:
            ValueError: If the 'split' parameter is not 'train' or 'test'.

        Attributes:
            n_rows (int): Number of rows to load from the dataset.
            split (str): Dataset split, either 'train' or 'test'.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text data.
            seq_len (int): Maximum sequence length.
            dataset Loaded dataset from the Jigsaw Toxicity Prediction dataset.

        Methods:
            __len__(): Returns the number of samples in the dataset.
            __getitem__(item): Retrieves a sample from the dataset.

        """
    
    def __init__(self, tokenizer, seq_len=SEQ_LEN, split="train", n_rows:int=None):
        
        if not split in ["train","test"]:
            raise ValueError("Parameter has to be 'train' or 'test'")  
        
        self.n_rows = n_rows
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
            
        if self.n_rows is not None:
            n_rows_str = f"[0:{self.n_rows}]" if self.n_rows is not None else ""
            self.dataset = load_dataset("jigsaw_toxicity_pred", data_dir=TOXIC, split=f"{self.split}{n_rows_str}")#[split]
        else:
            self.dataset = load_dataset("jigsaw_toxicity_pred", data_dir=TOXIC)#[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        # Step 1: get row
        output = self.dataset[item]

        # Step 2: tokenize comment
        output["input"] = self.tokenizer(
            output["comment_text"],
            max_length=self.seq_len ,
            padding="max_length", 
            truncation=True, 
            return_tensors='pt'
        )["input_ids"].to(DEVICE)
        
        # flatten output
        output["input"] = output["input"].squeeze().to(torch.long)
        
        output.pop("comment_text") # remove otherwise non tensor in dictionary

        
        # Step 4: collect different labels to one tensor
        labels = [output[key] if isinstance(output[key], torch.Tensor) else torch.tensor([output[key]]) for key in
                  ORDER_LABELS]

        labels = torch.cat(labels, dim=-1)
        output["labels"] = labels.to(DEVICE)
        
        return output

        