import math
import torch
import torch.nn as nn

from params import *

import numpy as np



class BERTBase(nn.Module):
    """
    Base class for BERT model without a Task specific head

    Args:
        vocab_size (int): size of vocabulary
        model_dimension (int): dimensionality of the model
        pretrained_model (str): path of the pretrained model
        number_layers (int): number of transformer layers
        number_heads (int): number of attention heads

    Attributes:
        model_dimension (int): dimensionality of the model
        number_layers (int): number of transformer layers
        number_heads (int): number of attention heads
        ff_hidden_layer (int): hidden layer dimension of the feedforward network (4 * model_dimension)
        embedding (BERTEmbedding): BERT embedding layer 
        encoders (torch.nn.ModuleList): list of encoder modules
    """

    class Encoder(nn.Module):
        """
        Puts together an encoder: MultiHeadAttention + feedforward layer + normalization layer

        Args:
            model_dimension (int): input dimension (default: EMBED_SIZE)
            number_heads (int): number of attention heads (default: 12)
            ff_hidden_dim (int): dimension of hidden layer in feedforward (default: EMBED_SIZE*4)

        """

        class FeedForwardLayer(nn.Module):
            """
            Module for a feedforward layer

            Args:
                model_dimension (int): dimension of input vector
                hidden_dimension (int): dimension of the hidden layer

            Attributes:
                linear1 (nn.Linear): transforms to hidden layer linearly
                linear2 (nn.Linear): transforms from hidden layer to output linearly
                non_linear (nn.ReLU): non-linear layer in the middle
            """

            def __init__(self, model_dimension=EMBED_SIZE, hidden_dimension=EMBED_SIZE * 4):
                """
                Initializing FeedForwardLayer

                Args:
                    model_dimension (int): dimension of input vector
                    hidden_dimension (int): dimension of the hidden layer

                """
                super(BERTBase.Encoder.FeedForwardLayer, self).__init__()

                # linear layer
                self.linear1 = nn.Linear(model_dimension, hidden_dimension)
                self.linear2 = nn.Linear(hidden_dimension, model_dimension)
                # non-linearity
                self.non_linear = nn.ReLU()

            def forward(self, x):
                """
                Forward pass trough FeedForwardLayer

                Args:
                    x (torch.Tensor): input tensor

                Returns:
                    torch.Tensor: output of FeedForward layer
                """
                return self.linear2(self.non_linear(self.linear1(x)))

        class MultiHeadAttention(nn.Module):
            """
            Module for multi-headed Attention

            Args:
                number_heads (int): number of attention heads
                model_dimension (int): input dimension of the model

            Attributes:
                number_heads (int): total number of attention heads
                att_head_dim (int): input dimension of each attention head
                Q (nn.Linear): query layer for Attention
                K (nn.Linear): key layer for Attention
                V (nn.Linear): value layer for Attention
                lin_output (nn.Linear): linear layer for the output of Attention
            """

            def __init__(self, number_heads, model_dimension, seq_len):
                """
                Initializing MultiHeadAttention

                Args:
                    number_heads (int): total number of attention heads
                    model_dimension (int): input dimension of the model
                """
                super(BERTBase.Encoder.MultiHeadAttention, self).__init__()

                self.seq_len = seq_len

                # model dimension must be divideable into equal parts for the attention heads
                self.number_heads = number_heads
                self.att_head_dim = int(model_dimension / number_heads)

                # attention mechanism: Q, K, V are linear embeddings -> embedding matrix dim: (model_dimension x model_dimension)
                self.Q = nn.Linear(model_dimension, model_dimension)
                self.K = nn.Linear(model_dimension, model_dimension)
                self.V = nn.Linear(model_dimension, model_dimension)
                self.lin_output = nn.Linear(model_dimension, model_dimension)

            def forward(self, Q, K, V, mask):
                """
                Forward pass trough MultiHeadAttention

                Args:
                    Q (torch.Tensor): query
                    K (torch.Tensor): key
                    V (torch.Tensor): value
                    mask (torch.Tensor): mask for the padded tokens

                Returns:
                    torch.Tensor: weighted embedding of input after multi-head Attention
                """
                batch_size = Q.shape[0]  # infer batch_size dynamically

                def fit_attention_head(t, number_heads, att_head_dim, batch_size):
                    """
                    Transform from (batch_size x seq_len x model_dimension)
                    to
                    (batch_size x number_heads x seq_len x att_head_dim)

                    Args:
                        t (torch.Tensor): input tensor
                        number_heads (int): number of attention heads
                        att_head_dim (int): dimension of each attention head

                    Returns:
                        torch.Tensor: reshaped tensor

                    """

                    t = t.view(batch_size, number_heads, self.seq_len, att_head_dim)
                    t = t.transpose(2, 3)
                    return t

                Q = fit_attention_head(self.Q(Q), self.number_heads, self.att_head_dim, batch_size)
                K = fit_attention_head(self.K(K), self.number_heads, self.att_head_dim, batch_size)
                V = fit_attention_head(self.V(V), self.number_heads, self.att_head_dim, batch_size)

                # calculate dot product between each Q and each K and normaliz the output, output dim: (batch_size x number_heads x seq_len x seq_len)
                score = torch.matmul(Q, K.transpose(2, 3))
                score_n = score / math.sqrt(self.att_head_dim)  # normalize: <q,k>/sqrt(d_k)

                # mask 0 with -infinity so it becomes 0 after softmax, output dim: (batch_size x number_heads x seq_len x seq_len)
                score_m = score_n.masked_fill(mask == 0,
                                              -np.inf)  # -> DELETE COMMENT: this is not for the pretraining masks but the padded tokens, they are 0 (see line 253)

                # softmax scores along each Q, output dim: (batch_size x number_heads x seq_len x seq_len)
                score_w = nn.functional.softmax(score_m, dim=-1)

                # multiply with V matrix: output weighted sum for each Q, output dim: (batch_size x number_heads x seq_len x att_head_dim)
                weighted_sum = torch.matmul(score_w, V)

                # concatenate attention heads to 1 output, output dim: (batch_size x seq_len x model_dimension)
                weighted_sum = weighted_sum.transpose(2, 3).reshape(batch_size, -1,
                                                                    self.number_heads * self.att_head_dim)

                # linear embedding for output, output dim: (batch_size x seq_len x model_dimension)
                out = self.lin_output(weighted_sum)
                return out

            # feedforward layer

        def __init__(self, seq_len=SEQ_LEN, model_dimension=EMBED_SIZE, number_heads=12, ff_hidden_dim=EMBED_SIZE * 4):
            super(BERTBase.Encoder, self).__init__()
            # attention heads
            self.multihead_attention = BERTBase.Encoder.MultiHeadAttention(number_heads, model_dimension, seq_len=seq_len)
            # normalisation layer
            self.normlayer = nn.LayerNorm(model_dimension)
            self.feedforward_layer = BERTBase.Encoder.FeedForwardLayer(model_dimension, hidden_dimension=ff_hidden_dim)

        def forward(self, x, mask):
            """
            Forward pass through Encoder

            Args:
                x (torch.Tensor): input tensor
                mask (torch.Tensor): mask padded tokens

            Returns:
                torch.Tensor: output of encoder
            """
            # input x 3x to generate Q, K, V
            x = self.normlayer(self.multihead_attention(x, x, x, mask))
            return self.normlayer(self.feedforward_layer(x))

        # base class for BERT

    class BERTEmbedding(torch.nn.Module):

        class PositionEmbedding(torch.nn.Module):
            def __init__(self, embed_size, seq_len, device=DEVICE):
                super().__init__()
                n = 10000  # scalar for pos encoding
                # create embedding matrix dim(seq_len  x embed_size)
                self.embed_matrix = torch.zeros(seq_len, embed_size, device=device).float()
                # positional encoding not to be updated while gradient descent
                self.embed_matrix.require_grad = False

                # compute embedding for each position in input
                for position in range(seq_len):
                    # run trough every component of embedding vector for each position with stride 2
                    for c in range(0, embed_size, 2):
                        # even
                        self.embed_matrix[position, c] = math.sin(position / (n ** (2 * c / embed_size)))
                        # uneven
                        self.embed_matrix[position, c + 1] = math.cos(position / (n ** (2 * c / embed_size)))

                # self.embed_matrix =  embed_matrix.unsqueeze(0)

            def forward(self, x):
                return self.embed_matrix

        def __init__(self, vocab_size, seq_len, embed_size=EMBED_SIZE, device=DEVICE):
            super().__init__()
            # token embedding: transforms (vocabulary size, number of tokens) into (vocabulary size, number of tokens, length of embdding vector)
            self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0).to(
                device)  # padding remains 0 during training
            # embedding of position
            self.position = BERTBase.BERTEmbedding.PositionEmbedding(embed_size, seq_len)

        def forward(self, sequence):
            return self.token(sequence) + self.position(sequence)

    def __init__(self, vocab_size, model_dimension, pretrained_model, number_layers, number_heads, seq_len=SEQ_LEN):
        """
        Initializes a the BERTBase model

        Parameters:
            vocab_size (int): size of the vocabulary
            model_dimension (int): model dimension
            pretrained_model (str): path of the pretrained model
            number_layers (int): number of transformer layers
            number_heads (int): number of attention heads

        Attributes:
            model_dimension (int): dimensionality of the model
            number_layers (int): number of transformer layers
            number_heads (int): number of attention heads
            ff_hidden_layer (int): hidden layer dimension of feedforward module (4 * model_dimension)
            embedding (BERTEmbedding): BERT embedding 
        """
        super().__init__()
        self.model_dimension=model_dimension
        self.number_layers=number_layers
        self.number_heads=number_heads
        # hidden layer dimenion of FF is 4*model_dimension 
        self.ff_hidden_layer = 4*model_dimension
        # embedding of input

        self.seq_len = seq_len

        self.embedding = BERTBase.BERTEmbedding(vocab_size=vocab_size, seq_len=seq_len, embed_size=model_dimension)

        
        # stack encoders and apply the pretrained weights to the layers of the encoders
        self.encoders = torch.nn.ModuleList() # create empty module list
        for i in range(self.number_layers):
            encoder = BERTBase.Encoder(model_dimension=model_dimension, number_heads=number_heads, ff_hidden_dim=4*model_dimension)
            encoder.load_state_dict(pretrained_model.encoder.layer[i].state_dict(), strict=False)
            self.encoders.append(encoder)
        
    def forward(self, x):
        """
        Forward pass of the BERTBase model

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
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
    Head for toxicity classification

    Args:
        bert_out (int): dimension of the BERT base model output

    Attributes:
        tox_classes (int): number of toxicity classes 
        linear (nn.Linear): linear layer for classification
        sigmoid (nn.Sigmoid): sigmoid function for multi-label classification

    """
    def __init__(self, bert_out):
        """
        Initializes the ToxicityPrediction model

        Parameters:
            bert_out (int): dimension of the BERT output

        Attributes:
            tox_classes (int): number of toxicity classes 
            linear (nn.Linear): linear layer for classification
            sigmoid (nn.Sigmoid): sigmoid activation for multi-label classification

        """
        super().__init__()
        self.tox_classes = 6 # there are 6 classes of toxicity in the dataset
        self.linear = nn.Linear(bert_out, self.tox_classes)
        # multilabel classification taks output is probiability of beloning to a class for each component of the output vector seperately 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass of ToxicityPrediction 

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        """
        # recieve output dimension (batch_size, self.tox_classes)
        return self.sigmoid(self.linear(x[:, 0]))


# TASK SHEET: model class    
class Model(nn.Module):
    """
    BERT-based model for toxic comment classification, consists of a base BERT model for feature extraction and a task-specific head for toxic comment classification

    Args:
        vocab_size (int): size of the vocabulary
        model_dimension (int): input dimension for the BERT model
        pretrained_model (str): path of pretrained BERT model
        number_layers (int, optional): number of transformer layers in BERT (default=12)
        number_heads (int, optional): number of attention heads in BERT (default=12)

    Attributes:
        base_model (BERTBase): base BERT model 
        toxic_comment (ToxicityPrediction): head for toxic comment classification

    """
    def __init__(self, vocab_size, model_dimension, pretrained_model, number_layers=12, number_heads=12):
        """
        Initializes the model

        Args:
            vocab_size (int): size of the vocabulary
            model_dimension (int): input dimension for the BERT model
            pretrained_model (str): path of the pretrained BERT model
            number_layers (int, optional): number of transformer layers in BERT (default=12)
            number_heads (int, optional): number of attention heads in BERT (default=12)

        Attributes:
            base_model (BERTBase): base BERT model 
            toxic_comment (ToxicityPrediction): head for toxic comment classification

        """
        super().__init__()
        # base BERT model
        self.base_model = BERTBase(vocab_size, model_dimension, pretrained_model, number_layers, number_heads)
        # toxic comment classfication layer
        self.toxic_comment = ToxicityPrediction(self.base_model.model_dimension)
    
    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor 

        """
        x = self.base_model(x)
        return self.toxic_comment(x)