import math
import torch
import torch.nn as nn

from params import *

class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_size, seq_len, device=DEVICE):
        super().__init__()
        n = 10000 # scalar for pos encoding
        # create embedding matrix dim(seq_len  x embed_size)
        self.embed_matrix = torch.zeros(seq_len, embed_size, device=device).float()
        # positional encoding not to be updated while gradient descent
        self.embed_matrix.require_grad = False
        
        # compute embedding for each position in input
        for position in range(seq_len):
            # run trough every component of embedding vector for each position with stride 2
            for c in range(0, embed_size, 2):
                # even 
                self.embed_matrix[position,c] = math.sin(position/(n**(2*c/embed_size)))
                # uneven
                self.embed_matrix[position,c+1] = math.cos(position/(n**(2*c/embed_size)))
        
        # self.embed_matrix =  embed_matrix.unsqueeze(0) 
    def forward(self, x):
        return self.embed_matrix
    

class BERTEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, seq_len, embed_size=EMBED_SIZE, device=DEVICE):
        super().__init__()
        # token embedding: transforms (vocabulary size, number of tokens) into (vocabulary size, number of tokens, length of embdding vector)
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0).to(device) # padding remains 0 during training
        # embedding of position
        self.position = PositionEmbedding(embed_size, seq_len)
        
    def forward(self, sequence):
        return self.token(sequence) + self.position(sequence)    
    