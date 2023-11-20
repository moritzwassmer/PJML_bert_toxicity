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

                class BertSelfAttention(torch.nn.Module):

                    def __init__(self):
                        super().__init__()

                        if EMBED_SIZE % NUMBER_HEADS != 0:
                            raise ValueError(f"EMBED_SIZE:{EMBED_SIZE} % NUMBER_HEADS:{NUMBER_HEADS} != 0")

                        self.dropout = nn.Dropout(DROPOUT)

                        self.query = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.key = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.value = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.output_linear = nn.Linear(EMBED_SIZE, EMBED_SIZE)

                    def forward(self, Q, K, V, mask):

                        Q, K, V = self.query(Q), self.key(K), self.value(V)

                        d_k = EMBED_SIZE // NUMBER_HEADS
                        batch_size = Q.shape[0]

                        def prep_attention(t):
                            # TODO Doc
                            t = t.view(batch_size, -1, NUMBER_HEADS, d_k).permute(0, 2, 1, 3)
                            return t

                        Q, K, V  = prep_attention(Q), prep_attention(K), prep_attention(V)

                        # z = Q*K / sqrt(d_k)
                        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(d_k)
                        scores = scores.masked_fill(mask == 0, -np.inf)

                        # softmax(z)
                        soft_scores = nn.functional.softmax(scores, dim=-1)
                        soft_scores = self.dropout(soft_scores)

                        # softmax(z) * V
                        output = torch.matmul(soft_scores, V)
                        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1,
                                                                                NUMBER_HEADS * d_k)

                        return self.output_linear(output)

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
                    self.bert_self_attention = BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfAttention() # TODO
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
            x=x
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