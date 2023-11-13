import math
import torch
import torch.nn as nn

from params import *
import embedding

# attention heads
class MultiHeadAttention(nn.Module):
    """
    Module for multi-headed Attention

    Args:
        number_heads (int): The number of attention heads
        model_dimension (int): Input dimension of the model

    Attributes:
        number_heads (int): Total number of attention heads
        att_head_dim (int): Input dimension of each attention head
        query (nn.Linear): Query layer for Attention
        key (nn.Linear): Key layer for Attention
        value (nn.Linear): Value layer for Attention
        lin_output (nn.Linear): Linear layer for the output of Attention
    """
    def __init__(self, number_heads, model_dimension):
        """
        Initializing MultiHeadAttention

        Args:
            number_heads (int): Total number of attention heads
            model_dimension (int): Input dimension of the model
        """
        super(MultiHeadAttention, self).__init__()
        
        # model dimension must be divideable into equal parts for the attention heads
        self.number_heads = number_heads
        self.att_head_dim = int(model_dimension/number_heads)
        
        # attention mechanism: query, key, value are linear embeddings -> embedding matrix dim: (model_dimension x model_dimension)
        self.query = nn.Linear(model_dimension, model_dimension)
        self.key = nn.Linear(model_dimension, model_dimension)
        self.value = nn.Linear(model_dimension, model_dimension)
        self.lin_output = nn.Linear(model_dimension, model_dimension)
    
    def forward(self, query, key, value, mask):
        """
        Forward pass trough MultiHeadAttention

        Args: 
            query (torch.Tensor): Input for query
            key (torch.Tensor): Input for key
            value (torch.Tensor): Input for value
            mask (torch.Tensor): Mask for the padded tokens
        
        Returns:
            torch.Tensor: Weighted embedding of input after multi-head Attention
        """
        # output dim (batch_size x seq_len x model_dimension) 
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value) 
        
        # transform q,k,v to fit attention heads:(batch_size x seq_len x model_dimension) -> (batch_size x number_heads x seq_len x att_head_dim)
        query = query.view(query.shape[0], query.shape[1], self.number_heads, self.att_head_dim)
        query = query.permute(0,2,1,3)
        key = key.view(key.shape[0], key.shape[1], self.number_heads, self.att_head_dim)
        key = key.permute(0,2,1,3)
        value = value.view(value.shape[0], value.shape[1], self.number_heads, self.att_head_dim)
        value = value.permute(0,2,1,3)
        
        # calculate dot product between each query and each key and normaliz the output, output dim: (batch_size x number_heads x seq_len x seq_len)
        score = torch.matmul(query, key.permute(0, 1, 3, 2)) 
        score_n = score / math.sqrt(self.att_head_dim) # normalize: <q,k>/sqrt(d_k)
        
        # mask 0 with -infinity so it becomes 0 after softmax, output dim: (batch_size x number_heads x seq_len x seq_len)
        score_m = score_n.masked_fill(mask == 0, -10000000000)    
        
        # softmax scores along each query, output dim: (batch_size x number_heads x seq_len x seq_len)
        score_w = nn.functional.softmax(score_m, dim=-1) 
        
        # multiply with value matrix: output weighted sum for each query, output dim: (batch_size x number_heads x seq_len x att_head_dim)
        weighted_sum = torch.matmul(score_w, value)
        
        # concatenate attention heads to 1 output, output dim: (batch_size x seq_len x model_dimension)
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).reshape(weighted_sum.shape[0], -1, self.number_heads * self.att_head_dim)
        
        # linear embedding for output, output dim: (batch_size x seq_len x model_dimension)
        out = self.lin_output(weighted_sum)      
        return out    
    

    # feedforward layer
class FeedForwardLayer(nn.Module):
    """
    Module for a feedforward layer

    Args:
        model_dimension (int): Dimension of input vector
        hidden_dimension (int): Dimension of the hidden layer

    Attributes:
        linear1 (nn.Linear): Transforms to hidden layer linearly
        linear2 (nn.Linear): Transforms from hidden layer to output linearly
        non_linear (nn.ReLU): Non-linear layer in the middle
    """
    def __init__(self, model_dimension, hidden_dimension):
        """
        Initializing FeedForwardLayer

        Args:
            model_dimension (int): Dimension of input vector
            hidden_dimension (int): Dimension of the hidden layer

        """
        super(FeedForwardLayer, self).__init__()
        
        # linear layer
        self.linear1 = nn.Linear(model_dimension, hidden_dimension)
        self.linear2 = nn.Linear(hidden_dimension, model_dimension)
        # non-linearity
        self.non_linear = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass trough FeedForwardLayer

        Args: 
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output of FeedForward layer
        """
        return self.linear2(self.non_linear(self.linear1(x)))       
    

    # encoder stacks together all the previous modules
class Encoder(nn.Module):
    """
    Puts together an encoder from: MultiHeadAttention + feedforward layer + normalization layer

    Args: 
        model_dimension (int): Input dimension (default: EMBED_SIZE)
        number_heads (int): Number of attention heads (default: 12).
        ff_hidden_dim (int): Dimension of hidden layer in feedforward (default: EMBED_SIZE*4).

    """
    def __init__(self, model_dimension=EMBED_SIZE, number_heads=12, ff_hidden_dim=EMBED_SIZE*4):
        super(Encoder, self).__init__()
        # attention heads
        self.multihead_attention = MultiHeadAttention (number_heads, model_dimension)
        # normalisation layer
        self.normlayer = nn.LayerNorm(model_dimension)
        self.feedforward_layer = FeedForwardLayer(model_dimension, hidden_dimension=ff_hidden_dim)
    
    # also residuals possible here
    def forward(self, x, mask):
        """
        Forward pass through Encoder 

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask padded tokens

        Returns:
            torch.Tensor: Output of encoder.
        """
        # input x 3x to generate query, key, value
        x = self.normlayer(self.multihead_attention(x, x, x, mask))
        return self.normlayer(self.feedforward_layer(x))
    

    # base class for BERT
class BERTBase(nn.Module):
    # __init__ function takes hyperparameters, initializes the model accordingly and sets up trainable parameters
    def __init__(self, vocab_size, model_dimension, pretrained_model, number_layers, number_heads):
        super().__init__()
        self.model_dimension=model_dimension
        self.number_layers=number_layers
        self.number_heads=number_heads
        # hidden layer dimenion of FF is 4*model_dimension (see paper)
        self.ff_hidden_layer = 4*model_dimension
        # embedding of input 
        self.embedding = embedding.BERTEmbedding(vocab_size=vocab_size, seq_len=SEQ_LEN, embed_size=model_dimension)
        
        # stack encoders
        self.encoders = torch.nn.ModuleList() # create empty module list
        for _ in range(self.number_layers):
            self.encoders.append(Encoder(model_dimension=model_dimension, number_heads=number_heads, ff_hidden_dim=4*model_dimension))
        
        # apply the pretrained weights to the layers of the encoders
        for i in range(number_layers):
            self.encoders[i].load_state_dict(pretrained_model.encoder.layer[i].state_dict(), strict=False)
        
        
    def forward(self, x):
        # mask to mark the padded (0) tokens
        mask = (x > 0).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)
        x = self.embedding(x) 
        # run trough encoders
        for encoder in self.encoders:
            x =encoder.forward(x, mask)
        return x
    

    # finetuning
class ToxicityPrediction(nn.Module):
    """
    class to predict multivariate class of toxicity
    """
    def __init__(self, bert_out):
        super().__init__()
        self.tox_classes = 6 # there are 6 classes of toxicity in the dataset
        self.linear = nn.Linear(bert_out, self.tox_classes)
        self.softmax = nn.LogSoftmax(dim=-1) 
        
    def forward(self, x):
        # recieve output dimension (batch_size, self.tox_classes)
        return self.softmax(self.linear(x[:, 0]))


# TASK SHEET: model class    
class Model(nn.Module):
    """
    Model class according to Milestone 1 task sheet
    """
    def __init__(self, vocab_size, model_dimension, pretrained_model, number_layers=12, number_heads=12):
        super().__init__()
        # base BERT model
        self.base_model = BERTBase(vocab_size, model_dimension, pretrained_model, number_layers, number_heads)
        # toxic comment classfication layer
        self.toxic_comment = ToxicityPrediction(self.base_model.model_dimension)
    
    def forward(self, x):
        x = self.base_model(x)
        return self.toxic_comment(x)