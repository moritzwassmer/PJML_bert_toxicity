from datasets import load_dataset
import torch
from torch.utils.data import Dataset

from params import *

class ToxicComment(Dataset):
    
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
        )["input_ids"]#.to(DEVICE) # TODO
        
        # flatten output
        output["input"] = output["input"].squeeze().to(torch.long) # TODO
        
        output.pop("comment_text") # remove otherwise non tensor in dictionary
        
        # Step 3: add segment_label like in pretraining task for consistency  TODO 1s for non padded elements only
        output["segment"] = torch.ones(self.seq_len,dtype=torch.long)
        
        # Step 4: collect different labels to one tensor # TODO created one more class if nothing of the following classes
        asdf = [output[key] if isinstance(output[key], torch.Tensor) else torch.tensor([output[key]]) for key in
                ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] # TODO watch out for order in params
        all_zero = torch.sum(torch.cat(asdf)) == 0
        if all_zero:
            labels = torch.cat([torch.ones(1)] + asdf,dim=-1)
        else:
            labels = torch.cat([torch.zeros(1)] + asdf,dim=-1)
        output["labels"] = labels.to(DEVICE)
        
        return output

        