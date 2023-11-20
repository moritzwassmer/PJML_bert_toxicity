import math
import torch
import torch.nn as nn

from params import *

import numpy as np

from transformers import BertModel, BertConfig # ONLY USED TO GET PRETRAINED CLASS_WEIGHTS!



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
        embedding (BertEmbedding): BERT embedding layer
        encoders (torch.nn.ModuleList): list of encoder modules
    """

    class BertEncoder(nn.Module):

        class BertLayer(nn.Module):
            """
            Puts together an encoder: BertSelfAttention + feedforward layer + normalization layer

            Args:
                model_dimension (int): input dimension (default: EMBED_SIZE)
                number_heads (int): number of attention heads (default: 12)
                ff_hidden_dim (int): dimension of hidden layer in feedforward (default: EMBED_SIZE*4)

            """

            class BertAttention(nn.Module):

                class BertSelfAttention(nn.Module):
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
                        Initializing BertSelfAttention

                        Args:
                            number_heads (int): total number of attention heads
                            model_dimension (int): input dimension of the model
                        """
                        super(BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfAttention, self).__init__()

                        self.seq_len = seq_len

                        # model dimension must be divideable into equal parts for the attention heads
                        self.number_heads = number_heads
                        self.att_head_dim = int(model_dimension / number_heads)

                        # attention mechanism: Q, K, V are linear embeddings -> embedding matrix dim: (model_dimension x model_dimension)
                        self.Q = nn.Linear(model_dimension, model_dimension)
                        self.K = nn.Linear(model_dimension, model_dimension)
                        self.V = nn.Linear(model_dimension, model_dimension)
                        self.dropout = nn.Dropout(0.1)  # TODO hardcoded

                    def forward(self, Q, K, V, mask):
                        """
                        Forward pass trough BertSelfAttention

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
                        out = self.dropout(weighted_sum)
                        return out

                    # feedforward layer

                class BertSelfOutput(nn.Module):
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
                        Initializing BertSelfOutput

                        Args:
                            model_dimension (int): dimension of input vector
                            hidden_dimension (int): dimension of the hidden layer

                        """
                        super(BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfOutput, self).__init__()

                        # linear layer
                        self.linear = nn.Linear(model_dimension, model_dimension)
                        self.normlayer = nn.LayerNorm(model_dimension)
                        # non-linearity
                        self.dropout = nn.Dropout(0.1)  # TODO hardcoded

                    def forward(self, x):
                        """
                        Forward pass trough BertSelfOutput

                        Args:
                            x (torch.Tensor): input tensor

                        Returns:
                            torch.Tensor: output of FeedForward layer
                        """
                        return self.dropout(self.normlayer(self.linear(x)))

                def __init__(self, seq_len=SEQ_LEN, model_dimension=EMBED_SIZE, number_heads=NUMBER_HEADS, ff_hidden_dim=EMBED_SIZE * 4):

                    super(BERTBase.BertEncoder.BertLayer.BertAttention, self).__init__()
                    self.bert_self_attention = BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfAttention(number_heads, model_dimension, seq_len=seq_len)
                    #self.bert_intermediate = BERTBase.BertEncoder.BertLayer.BertIntermediate(model_dimension=EMBED_SIZE, hidden_dimension=EMBED_SIZE * 4)
                    self.bert_self_output = BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfOutput(model_dimension=EMBED_SIZE, hidden_dimension=EMBED_SIZE * 4)

                def forward(self, x, mask): # TODO unsure about this forward pass

                    x = self.bert_self_attention(x,x,x, mask)
                    x = self.bert_self_output(x)
                    return x

            class BertIntermediate(nn.Module):

                def __init__(self, model_dimension=EMBED_SIZE, hidden_dimension=EMBED_SIZE * 4):
                    super(BERTBase.BertEncoder.BertLayer.BertIntermediate, self).__init__()
                    self.linear = nn.Linear(model_dimension, hidden_dimension)
                    # non-linearity
                    self.non_linear = nn.GELU()

                def forward(self, x):
                    """
                    Forward pass trough BertSelfOutput

                    Args:
                        x (torch.Tensor): input tensor

                    Returns:
                        torch.Tensor: output of FeedForward layer
                    """
                    return self.non_linear(self.linear(x))

            class BertOutput(nn.Module):
                def __init__(self, model_dimension=EMBED_SIZE, hidden_dimension=EMBED_SIZE * 4):
                    super(BERTBase.BertEncoder.BertLayer.BertOutput, self).__init__()
                    self.linear = nn.Linear(hidden_dimension, model_dimension)
                    # non-linearity
                    self.normlayer = nn.LayerNorm(model_dimension)
                    self.dropout = nn.Dropout(0.1) # TODO hardcoded

                def forward(self, x):
                    """
                    Forward pass trough BertSelfOutput

                    Args:
                        x (torch.Tensor): input tensor

                    Returns:
                        torch.Tensor: output of FeedForward layer
                    """
                    return self.dropout(self.normlayer(self.linear(x)))


            def __init__(self, seq_len=SEQ_LEN, model_dimension=EMBED_SIZE, number_heads=NUMBER_HEADS,
                         ff_hidden_dim=EMBED_SIZE * 4):
                super(BERTBase.BertEncoder.BertLayer, self).__init__()
                # attention heads
                self.bert_attention = BERTBase.BertEncoder.BertLayer.BertAttention() # TODO params

                # normalisation layer
                # self.normlayer = nn.LayerNorm(model_dimension)
                self.bert_intermediate = BERTBase.BertEncoder.BertLayer.BertIntermediate() # TODO params
                # self.dropout = nn.Dropout(0.1)

                self.bert_output = BERTBase.BertEncoder.BertLayer.BertOutput() # TODO Params



            def forward(self, x, mask):
                """
                Forward pass through BertLayer

                Args:
                    x (torch.Tensor): input tensor
                    mask (torch.Tensor): mask padded tokens

                Returns:
                    torch.Tensor: output of encoder
                """

                x = self.bert_attention(x,mask)

                x = self.bert_intermediate(x)

                x = self.bert_output(x)


                """x = self.normlayer(self.multihead_attention(x, x, x, mask))
                x = self.normlayer(self.feedforward_layer(x)) # TODO WRONG
                x = self.dropout(x)"""
                return x

            # base class for BERT

        def __init__(self, model_dimension=EMBED_SIZE, number_layers=NUMBER_LAYERS, number_heads=NUMBER_HEADS):

            super().__init__()

            # INIT ENCODERS
            self.encoders = torch.nn.ModuleList()  # create empty module list
            for i in range(number_layers):
                encoder = BERTBase.BertEncoder.BertLayer(model_dimension=model_dimension, number_heads=number_heads,
                                             ff_hidden_dim=4 * model_dimension)
                self.encoders = self.encoders.append(encoder)

        def forward(self, x, mask):
            # run trough encoders
            for encoder in self.encoders:
                x = encoder.forward(x, mask)
            return x


    class BertEmbedding(torch.nn.Module):
        """
            BertEmbedding is a module that combines token embeddings and positional embeddings for input sequences.

            Parameters:
            - vocab_size (int): Size of the vocabulary.
            - seq_len (int): Length of the input sequence.
            - embed_size (int): Dimensionality of the embedding vector.
            - device (str): Device on which the module is instantiated.
            """

        class PositionEmbedding(torch.nn.Module):
            """
                Generates positional embeddings for input sequences.
                The positional embeddings are created only once during initialization and are not updated during gradient descent.

                Parameters:
                - embed_size (int): Dimensionality of the embedding vector. Default is EMBED_SIZE.
                - seq_len (int): Length of the input sequence. Default is SEQ_LEN.
            """

            def create_embedding_matrix(self, embed_size, seq_len):
                embed_matrix = torch.zeros(seq_len, embed_size, device=DEVICE).float()  # TODO DEVICE

                # positional encoding not to be updated during gradient descent
                embed_matrix.requires_grad = False

                # compute embedding for each position in input
                for position in range(seq_len):
                    embed_matrix[position, ::2] = torch.sin(
                        position / (10000  ** (2 * torch.arange(0, embed_size, 2) / embed_size)))
                    embed_matrix[position, 1::2] = torch.cos(
                        position / (10000  ** (2 * torch.arange(0, embed_size, 2) / embed_size)))

                return embed_matrix

            def __init__(self, embed_size=EMBED_SIZE, seq_len=SEQ_LEN):
                super().__init__()
                self.pos_embedding = self.create_embedding_matrix(embed_size, seq_len)

            def forward(self,x):
                return self.pos_embedding

        def __init__(self, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, embed_size=EMBED_SIZE, device=DEVICE, dropout=0.1):
            super().__init__()
            # token embedding: transforms (vocabulary size, number of tokens) into (vocabulary size, number of tokens, length of embdding vector)
            self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0).to( # are we sure padding is 0? -> yes
                device)  # padding remains 0 during training
            # embedding of position
            self.position = BERTBase.BertEmbedding.PositionEmbedding(embed_size, seq_len)
            self.segment = nn.Embedding(2, embed_size, padding_idx=0)
            self.normlayer = nn.LayerNorm(embed_size)
            self.dropout = torch.nn.Dropout(p=dropout)


        def forward(self, sequence, segments):

            total_embedding = self.token(sequence) + self.position(sequence) + self.segment(segments)
            norm_embedding = self.normlayer(total_embedding)
            return self.dropout(norm_embedding)

    def __init__(self, vocab_size=VOCAB_SIZE, model_dimension=EMBED_SIZE, use_pretrained=True,
                 number_layers=NUMBER_LAYERS, number_heads=NUMBER_HEADS, seq_len=SEQ_LEN):
        """
        Initializes a the BERTBase model

        Parameters:
            vocab_size (int): size of the vocabulary
            model_dimension (int): model dimension
            pretrained_model (str): path of the pretrained model # TODO wrong
            number_layers (int): number of transformer layers
            number_heads (int): number of attention heads

        Attributes:
            model_dimension (int): dimensionality of the model
            number_layers (int): number of transformer layers
            number_heads (int): number of attention heads
            ff_hidden_layer (int): hidden layer dimension of feedforward module (4 * model_dimension)
            embedding (BertEmbedding): BERT embedding
        """
        super().__init__()
        """self.model_dimension=model_dimension
        self.number_layers=number_layers
        self.number_heads=number_heads
        # hidden layer dimenion of FF is 4*model_dimension 
        self.ff_hidden_layer = 4*model_dimension
        # embedding of input

        self.seq_len = seq_len"""

        self.embedding = BERTBase.BertEmbedding(vocab_size=vocab_size, seq_len=seq_len, embed_size=model_dimension)
        self.encoder = BERTBase.BertEncoder(model_dimension=model_dimension,number_layers=number_layers,number_heads=number_heads)

        # TODO implement load model weights
        """
        if use_pretrained:
            self.load_from_pretrained()
        """

    def load_from_pretrained(self):
        # TODO finish

        # Download pretrained weights from huggingface (for the base BERT)
        bert_base = "bert-base-uncased"
        configuration = BertConfig.from_pretrained(bert_base)
        pretrained_model = BertModel.from_pretrained(bert_base, config=configuration)

        # TODO Embedding layers


        # stack encoders and apply the pretrained weights to the layers of the encoders
        self.encoders = torch.nn.ModuleList()  # create empty module list
        for i in range(self.number_layers):
            pretrained_encoder = pretrained_model.encoder.layer[i].state_dict()
            encoder = self.encoders[i]
            encoder = encoder.load_state_dict(pretrained_encoder, strict=False)
            self.encoders.insert(i,encoder)

    def forward(self, words, segments):
        """
        Forward pass of the BERTBase model

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # mask to mark the padded (0) tokens TODO understood correctly?
        mask = (words > 0).unsqueeze(1).repeat(1,words.size(1),1).unsqueeze(1) # TODO commented out

        x = self.embedding(words, segments)
        # run trough encoders
        x = self.encoder(x, mask)

        return x
    

    # finetuning

class ToxicityPredictionHead(nn.Module):
    """
    Head for toxicity classification

    Args:
        bert_out (int): dimension of the BERT base model output

    Attributes:
        tox_classes (int): number of toxicity classes 
        linear (nn.Linear): linear layer for classification
        sigmoid (nn.Sigmoid): sigmoid function for multi-label classification

    """
    def __init__(self, bert_out=EMBED_SIZE):
        """
        Initializes the ToxicityPredictionHead model

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
        #self.sigmoid = nn.Sigmoid() # TODO sigmoid not used for BCE with logits
        
    def forward(self, x):
        """
        Forward pass of ToxicityPredictionHead

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        """
        # recieve output dimension (batch_size, self.tox_classes)
        x = self.linear(x[:, 0]) # only extract cls embedding (at beginning)

        #x = self.sigmoid(x) # TODO commented out due to BCELossWithLogits, does combine sigmoid with BCE loss in one class, also multilabel classification
        return x


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
        toxic_comment (ToxicityPredictionHead): head for toxic comment classification

    """
    def __init__(self, vocab_size=VOCAB_SIZE, model_dimension=EMBED_SIZE, use_pretrained=True, number_layers=NUMBER_LAYERS, number_heads=NUMBER_HEADS):
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
            toxic_comment (ToxicityPredictionHead): head for toxic comment classification

        """
        super().__init__()
        # base BERT model
        self.base_model = BERTBase(vocab_size, model_dimension, use_pretrained, number_layers, number_heads)
        # toxic comment classfication layer
        self.toxic_comment = ToxicityPredictionHead(model_dimension)
    
    def forward(self, words, segments):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor 

        """
        x = self.base_model(words, segments)

        return self.toxic_comment(x)