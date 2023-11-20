import math
import torch
import torch.nn as nn

from params import *

import numpy as np

from transformers import BertModel, BertConfig # ONLY USED TO GET PRETRAINED CLASS_WEIGHTS!



class BERTBase(nn.Module):

    class BertEncoder(nn.Module):

        class BertLayer(nn.Module):

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

                    def __init__(self):
                        # TODO
                        """
                        Initializing BertSelfAttention

                        Args:
                            number_heads (int): total number of attention heads
                            model_dimension (int): input dimension of the model
                        """
                        super().__init__()

                        # attention mechanism: Q, K, V are linear embeddings -> embedding matrix dim: (model_dimension x model_dimension)
                        self.Q = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.K = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.V = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.dropout = nn.Dropout(DROPOUT)  # TODO hardcoded

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

                class BertSelfOutput(nn.Module):

                    def __init__(self):

                        super().__init__()

                        self.linear = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
                        self.dropout = nn.Dropout(DROPOUT)

                    def forward(self, x):

                        return self.dropout(self.normlayer(self.linear(x)))

                def __init__(self, seq_len=SEQ_LEN, model_dimension=EMBED_SIZE, number_heads=NUMBER_HEADS):

                    super().__init__()
                    self.bert_self_attention = BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfAttention()
                    self.bert_self_output = BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfOutput()

                def forward(self, x, mask): # TODO unsure about this forward pass

                    x = self.bert_self_attention(x,x,x, mask)
                    x = self.bert_self_output(x)
                    return x

            class BertIntermediate(nn.Module):

                def __init__(self, ):
                    super().__init__()
                    self.linear = nn.Linear(EMBED_SIZE, EMBED_SIZE*4)
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
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(EMBED_SIZE*4, EMBED_SIZE)
                    self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
                    self.dropout = nn.Dropout(DROPOUT)

                def forward(self, x):
                    """
                    Forward pass trough BertSelfOutput

                    Args:
                        x (torch.Tensor): input tensor

                    Returns:
                        torch.Tensor: output of FeedForward layer
                    """
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
        """
        Parameters:
            words (torch.Tensor): word tokens

        Returns:
            torch.Tensor: output tensor
        """

        # mask to mark the padded (0) tokens
        mask = (words > 0).unsqueeze(1).repeat(1,words.size(1),1).unsqueeze(1)
        segments = (words > 0).to(torch.int).cuda() # create segment embeddings (1s if words exist else padding so 0) # TODO check if 1 is correct

        x = self.embedding(words, segments)
        # run trough encoders
        x = self.encoder(x, mask)



        return x
    

    # finetuning

class ToxicityPredictionHead(nn.Module):

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

    def __init__(self):

        super().__init__()
        # base BERT model
        self.base_model = BERTBase()
        # toxic comment classfication layer
        self.toxic_comment = ToxicityPredictionHead()
    
    def forward(self, words):

        x = self.base_model(words)

        print(x.shape)

        return self.toxic_comment(x)