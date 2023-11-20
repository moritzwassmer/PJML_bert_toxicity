import math
import torch
import torch.nn as nn

from params import *

import numpy as np

from transformers import BertModel, BertConfig # ONLY USED TO GET PRETRAINED CLASS_WEIGHTS!



class BERTBase(nn.Module):
    """
    Base class for BERT model. We made sure to implement the same nested structure such as the state dict of bert-base-uncased
    from huggingface to have an easier time loading the weights. https://huggingface.co/bert-base-uncased

    check params.py for constants

    Args:
        use_pretrained (bool, optional): Whether to load pretrained weights from Hugging Face. Default is True.
    """

    class BertEncoder(nn.Module):

        class BertLayer(nn.Module):

            class BertAttention(nn.Module):
                """
                BERT Attention module, including multi-headed self-attention and an output layer

                Attributes:
                    bert_self_attention (BertSelfAttention): Multi-headed self-attention mechanism.
                    bert_self_output (BertSelfOutput): Output layer for the self-attention mechanism.
                """

                class BertSelfAttention(nn.Module):
                    """
                    Module for multi-headed Attention.

                    Attributes:
                        number_heads (int): Total number of attention heads.
                        att_head_dim (int): Input dimension of each attention head.
                        Q (nn.Linear): Query layer for Attention.
                        K (nn.Linear): Key layer for Attention.
                        V (nn.Linear): Value layer for Attention.
                        lin_output (nn.Linear): Linear layer for the output of Attention.
                    """

                    def __init__(self):

                        super().__init__()

                        # attention mechanism: Q, K, V are linear embeddings -> embedding matrix dim: (model_dimension x model_dimension)
                        self.Q = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.K = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.V = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.dropout = nn.Dropout(DROPOUT)  # TODO hardcoded

                    def forward(self, Q, K, V, mask):

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

                            t = t.view(batch_size, number_heads, SEQ_LEN, att_head_dim)
                            t = t.transpose(2, 3)
                            return t

                        att_head_dim = int(EMBED_SIZE / NUMBER_HEADS)

                        Q = fit_attention_head(self.Q(Q), NUMBER_HEADS, att_head_dim, batch_size)
                        K = fit_attention_head(self.K(K), NUMBER_HEADS, att_head_dim, batch_size)
                        V = fit_attention_head(self.V(V), NUMBER_HEADS, att_head_dim, batch_size)

                        # calculate dot product between each Q and each K and normaliz the output, output dim: (batch_size x number_heads x seq_len x seq_len)
                        score = torch.matmul(Q, K.transpose(2, 3))
                        score_n = score / math.sqrt(att_head_dim)  # normalize: <q,k>/sqrt(d_k)

                        # mask 0 with -infinity so it becomes 0 after softmax, output dim: (batch_size x number_heads x seq_len x seq_len)
                        print(mask.shape)
                        print(score_n.shape)
                        score_m = score_n.masked_fill(mask == 0,
                                                      -np.inf)  # -> DELETE COMMENT: this is not for the pretraining masks but the padded tokens, they are 0 (see line 253)

                        # softmax scores along each Q, output dim: (batch_size x number_heads x seq_len x seq_len)
                        score_w = nn.functional.softmax(score_m, dim=-1)

                        # multiply with V matrix: output weighted sum for each Q, output dim: (batch_size x number_heads x seq_len x att_head_dim)
                        weighted_sum = torch.matmul(score_w, V)

                        # concatenate attention heads to 1 output, output dim: (batch_size x seq_len x model_dimension)
                        weighted_sum = weighted_sum.transpose(2, 3).reshape(batch_size, -1,
                                                                            NUMBER_HEADS * att_head_dim)

                        # linear embedding for output, output dim: (batch_size x seq_len x model_dimension)
                        out = self.dropout(weighted_sum)
                        return out

                class MultiHeadedAttention(torch.nn.Module):

                    def __init__(self, heads=NUMBER_HEADS, d_model=EMBED_SIZE, dropout=DROPOUT):
                        super().__init__()

                        assert d_model % heads == 0
                        self.d_k = d_model // heads
                        self.heads = heads
                        self.dropout = torch.nn.Dropout(dropout)

                        self.query = torch.nn.Linear(d_model, d_model)
                        self.key = torch.nn.Linear(d_model, d_model)
                        self.value = torch.nn.Linear(d_model, d_model)
                        self.output_linear = torch.nn.Linear(d_model, d_model)

                    def forward(self, query, key, value, mask):
                        """
                        query, key, value of shape: (batch_size, max_len, d_model)
                        mask of shape: (batch_size, 1, 1, max_words)
                        """
                        # (batch_size, max_len, d_model)
                        query = self.query(query)
                        key = self.key(key)
                        value = self.value(value)

                        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
                        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
                        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
                        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

                        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
                        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

                        # fill 0 mask with super small number so it wont affect the softmax weight
                        # (batch_size, h, max_len, max_len)
                        scores = scores.masked_fill(mask == 0, -1e9)

                        # (batch_size, h, max_len, max_len)
                        # softmax to put attention weight for all non-pad tokens
                        # max_len X max_len matrix of attention

                        weights = torch.nn.functional.softmax(scores, dim=-1)#nn.Softmax(scores, dim=-1)
                        weights = self.dropout(weights)

                        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
                        context = torch.matmul(weights, value)

                        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
                        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1,
                                                                                self.heads * self.d_k)

                        # (batch_size, max_len, d_model)
                        return self.output_linear(context)
                # TODO take out

                class BertSelfOutput(nn.Module):
                    """
                    Output layer for the self-attention mechanism.

                    Attributes:
                        linear (nn.Linear): Linear layer
                        normlayer (nn.LayerNorm): Layer normalization
                        dropout (nn.Dropout): Dropout layer
                    """

                    def __init__(self):

                        super().__init__()

                        self.linear = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
                        self.dropout = nn.Dropout(DROPOUT)

                    def forward(self, x):

                        return self.dropout(self.normlayer(self.linear(x)))

                def __init__(self):

                    super().__init__()
                    #self.bert_self_attention = BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfAttention()
                    self.bert_self_attention = BERTBase.BertEncoder.BertLayer.BertAttention.MultiHeadedAttention() # TODO
                    self.bert_self_output = self.BertSelfOutput()

                def forward(self, x, mask): # TODO unsure about this forward pass

                    x = self.bert_self_attention(x,x,x, mask)
                    x = self.bert_self_output(x)
                    return x

            class BertIntermediate(nn.Module):
                """
                Intermediate layer for the FeedForward mechanism in the BERT model.

                Attributes:
                    linear (nn.Linear): Linear layer
                    non_linear (nn.GELU): GELU activation function
                """

                def __init__(self, ):
                    super().__init__()
                    self.linear = nn.Linear(EMBED_SIZE, EMBED_SIZE*4)
                    self.non_linear = nn.GELU()

                def forward(self, x):

                    return self.non_linear(self.linear(x))

            class BertOutput(nn.Module):
                """
                Output layer

                Attributes:
                    linear (nn.Linear): Linear layer
                    normlayer (nn.LayerNorm): Layer normalization
                    dropout (nn.Dropout): Dropout layer
                """

                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(EMBED_SIZE*4, EMBED_SIZE)
                    self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
                    self.dropout = nn.Dropout(DROPOUT)

                def forward(self, x):

                    return self.dropout(self.normlayer(self.linear(x)))


            def __init__(self):
                super().__init__()
                self.bert_attention = BERTBase.BertEncoder.BertLayer.BertAttention()
                self.bert_intermediate = BERTBase.BertEncoder.BertLayer.BertIntermediate()
                self.bert_output = BERTBase.BertEncoder.BertLayer.BertOutput()



            def forward(self, x, mask):
                """
                Forward pass through BertLayer

                Args:
                    x (torch.Tensor): input tensor
                    mask (torch.Tensor): mask padded tokens

                Returns:
                    torch.Tensor: output of encoder
                """

                x = self.bert_attention(x, mask)

                x = self.bert_intermediate(x)

                x = self.bert_output(x)


                """x = self.normlayer(self.multihead_attention(x, x, x, mask))
                x = self.normlayer(self.feedforward_layer(x)) # TODO WRONG
                x = self.dropout(x)"""
                return x

            # base class for BERT

        def __init__(self):

            super().__init__()

            # init encoder layers
            self.encoders = torch.nn.ModuleList()  # create empty module list
            for i in range(NUMBER_LAYERS):
                encoder = BERTBase.BertEncoder.BertLayer()
                self.encoders = self.encoders.append(encoder)

        def forward(self, x, mask):
            # run trough encoders
            for encoder in self.encoders:
                x = encoder.forward(x, mask)
            return x

    class BertEmbedding(torch.nn.Module):

        """
        BERT embedding layer for token, position, and segment embeddings.

        Attributes:
            token (nn.Embedding): Token embedding layer.
            position (PositionEmbedding): Positional embedding layer.
            segment (nn.Embedding): Segment embedding layer.
            normlayer (nn.LayerNorm): Layer normalization
            dropout (torch.nn.Dropout): Dropout layer
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

        def __init__(self):
            super().__init__()
            # token embedding: transforms (vocabulary size, number of tokens) into (vocabulary size, number of tokens, length of embdding vector)
            self.token = nn.Embedding(VOCAB_SIZE, EMBED_SIZE, padding_idx=0).to( # are we sure padding is 0? -> yes
                DEVICE)  # padding remains 0 during training
            # embedding of position
            self.position = BERTBase.BertEmbedding.PositionEmbedding()
            self.segment = nn.Embedding(2, EMBED_SIZE, padding_idx=0)
            self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
            self.dropout = torch.nn.Dropout(p=DROPOUT)


        def forward(self, sequence, segments):

            total_embedding = self.token(sequence) + self.position(sequence) + self.segment(segments)
            norm_embedding = self.normlayer(total_embedding)
            return self.dropout(norm_embedding)

    def __init__(self, use_pretrained=True):

        super().__init__()

        self.embedding = BERTBase.BertEmbedding()
        self.encoder = BERTBase.BertEncoder()

        # TODO implement load model weights
        """
        if use_pretrained:
            self.load_from_pretrained()
        """

    def load_from_pretrained(self):
        # TODO implement
        pass
        """# Download pretrained weights from huggingface (for the base BERT)
        bert_base = "bert-base-uncased"
        configuration = BertConfig.from_pretrained(bert_base)
        pretrained_model = BertModel.from_pretrained(bert_base, config=configuration)

        # stack encoders and apply the pretrained weights to the layers of the encoders
        self.encoders = torch.nn.ModuleList()  # create empty module list
        for i in range(self.number_layers):
            pretrained_encoder = pretrained_model.encoder.layer[i].state_dict()
            encoder = self.encoders[i]
            encoder = encoder.load_state_dict(pretrained_encoder, strict=False)
            self.encoders.insert(i,encoder)"""

    def forward(self, words):

        # mask to mark the padded (0) tokens
        mask = (words > 0).unsqueeze(1).repeat(1,words.size(1),1).unsqueeze(1)
        #print(mask.shape)
        segments = (words > 0).to(torch.int).cuda() # create segment embeddings (1s if words exist else padding so 0) # TODO check if 1 is correct

        x = self.embedding(words, segments)
        # run trough encoders
        x = self.encoder(x, mask)



        return x
    

    # finetuning

class BERTMultiLabelClassifcation(nn.Module):
    """
        Head for Multilabel Classification. Expects the usage of BCEWithLogits (no sigmoid layer).

        Attributes:
            linear (nn.Linear): Linear layer for classification using the CLS Embedding
        """

    def __init__(self):

        super().__init__()
        bert_out = EMBED_SIZE
        self.linear = nn.Linear(bert_out, len(ORDER_LABELS)) # TODO hardcoded
        # multilabel classification taks output is probiability of beloning to a class for each component of the output vector seperately 
        #self.sigmoid = nn.Sigmoid() # TODO sigmoid not used for BCE with logits
        
    def forward(self, x):

        # recieve output dimension (batch_size, self.tox_classes)
        x = self.linear(x[:, 0]) # only extract cls embedding (at beginning)

        #x = self.sigmoid(x) # TODO commented out due to BCELossWithLogits, does combine sigmoid with BCE loss in one class, also multilabel classification
        return x


# TASK SHEET: model class    
class Model(nn.Module):

    """
    Model for Toxic Comment Classification using BERT.

    Attributes:
        base_model (BERTBase): Base BERT model for feature extraction.
        toxic_comment (BERTMultiLabelClassification): Multilabel classification layer for toxicity prediction.
    """

    def __init__(self):

        super().__init__()
        # base BERT model
        self.base_model = BERTBase()
        # toxic comment classfication layer
        self.toxic_comment = BERTMultiLabelClassifcation()
    
    def forward(self, words):

        x = self.base_model(words)

        return self.toxic_comment(x)