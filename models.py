import math
import torch
import torch.nn as nn

from params import *

import numpy as np

# ONLY USED TO GET PRETRAINED CLASS_WEIGHTS!
from transformers import BertModel, BertConfig


class BERTBase(nn.Module):
    """
    Base class for BERT model. We made sure to implement the same nested structure such as the state dict of bert-base-uncased
    from huggingface to have an easier time loading the weights. https://huggingface.co/bert-base-uncased

    check params.py for constants

    Args:
        use_pretrained (bool, optional): Whether to load pretrained weights from Hugging Face. Default is True.
    """

    class BertEncoder(nn.Module):
        """
        Module to comprise the entire base for BERT including a pipeline of encoders.

        Attributes:
            encoders (nn.ModuleList): List of BertLayer modules (encoders)
        """

        class BertLayer(nn.Module):
            """
            Initializes BertLayer module which comprises an encoder: BertAttention module (the multi-headed Attention and Add & Norm), a BertIntermediate module (the FeedForward layer) and a BertOutput (Add & Norm).
            """
            class BertAttention(nn.Module):
                """
                BERT Attention module, including multi-headed self-attention and an output layer with a residual connection

                Attributes:
                    bert_self_attention (BertSelfAttention): Multi-headed self-attention mechanism.
                    bert_self_output (BertSelfOutput): Output layer for the self-attention mechanism.
                """

                class BertSelfAttention(torch.nn.Module):
                    """
                    BERT Attention module to perform multi-headed Self-Attention.

                    Attributes:
                        dropout (nn.Dropout): Dropout layer 
                        query (nn.Linear): Learned linear embedding of the input into query
                        key (nn.Linear): Learned linear embedding of the input into key
                        value (nn.Linear): Learned linear embedding of the input into value
                        out_linear (nn.Linear): Linear embedding of the final output
                    """

                    def __init__(self):
                        """
                        Initializes the BertSelfAttention module (multi-headed Attention).
                        The module performs multi-headed Self-Attention on the input sequence and outputs an Attention-weighted embedding. 

                        Raises:
                            ValueError: If EMBED_SIZE is not divisible by NUMBER_HEADS
                        """
                        super().__init__()

                        if EMBED_SIZE % NUMBER_HEADS != 0:
                            raise ValueError(
                                f"EMBED_SIZE:{EMBED_SIZE} % NUMBER_HEADS:{NUMBER_HEADS} != 0")

                        self.dropout = nn.Dropout(DROPOUT)

                        self.query = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.key = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.value = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.output_linear = nn.Linear(EMBED_SIZE, EMBED_SIZE)

                    def forward(self, Q, K, V, mask):
                        """
                        Forward pass through BertSelfAttention module (multi-headed Attention).

                        Args:
                            Q (torch.Tensor): Query
                            K (torch.Tensor): Key
                            V (torch.Tensor): Value
                            mask (torch.Tensor): Mask the padded tokens
                        """
                        d_k = EMBED_SIZE // NUMBER_HEADS  # dimension of a single attention head
                        batch_size = Q.shape[0]

                        def prep_attention(t):
                            """
                            Reshapes the input of the multi-headed Attention to match the dimensions of the attention heads.
                            Input dimension: (batch_size, sequence_length, EMBED_SIZE) --> output dimension: (batch_size, NUMBER_HEADS, sequence_length, d_k)
                            Args:
                                t (torch.Tensor): Input tensor

                            Returns:
                                torch.Tensor:  Tensor with dimensions suitable for multi-headed Self-Attention
                            """
                            t = t.view(batch_size, -1, NUMBER_HEADS,
                                       d_k).permute(0, 2, 1, 3)
                            return t

                        Q, K, V = self.query(Q), self.key(K), self.value(V)
                        Q, K, V = prep_attention(
                            Q), prep_attention(K), prep_attention(V)

                        # z = Q*K / sqrt(d_k)
                        scores = torch.matmul(Q, K.permute(
                            0, 1, 3, 2)) / math.sqrt(d_k)
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
                    Output layer for the self-attention mechanism, comes after the Attention-Head (Add & Norm)

                    Attributes:
                        linear (nn.Linear): Linear layer
                        normlayer (nn.LayerNorm): Layer normalization
                        dropout (nn.Dropout): Dropout layer
                    """

                    def __init__(self):
                        """
                        Initializes the BerSelfOutput layer. This layer applies a linear embedding (linear), 
                        a normalization (normlayer) and a dropout (dropout) to the output of the multi-headed Self-Attention.
                        """

                        super().__init__()

                        self.linear = nn.Linear(EMBED_SIZE, EMBED_SIZE)
                        self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
                        self.dropout = nn.Dropout(DROPOUT)

                    def forward(self, x):
                        """
                        Performs a forward pass with ths output of the Attention-Head 

                        Args:
                            x (torch.Tensor): Input tensor

                        Returns:
                            torch.Tensor: Output of the Add & Norm layer
                        """

                        return self.dropout(self.normlayer(self.linear(x)))

                def __init__(self):
                    """
                    Initializes the BertAttention module which comprises a BertSelfAttention module and a BertSelfOutput, which is the first part of the encoder.
                    """

                    super().__init__()
                    self.bert_self_attention = BERTBase.BertEncoder.BertLayer.BertAttention.BertSelfAttention()
                    self.bert_self_output = self.BertSelfOutput()

                def forward(self, x, mask):
                    """
                    Forward pass through the BertAttention module, which is the first part of the encoder.

                    Args:
                        x (torch.Tensor): Input tensor
                        mask (torch.Tensor): Masking padded tokens

                    Returns:
                        torch.Tensor: Output of BertAttention, including a residual connection from after the BertSelfAttention
                    """

                    att_out = self.bert_self_attention(x, x, x, mask)
                    # residual connection
                    residual = self.bert_self_output(x + att_out)
                    return residual

            class BertIntermediate(nn.Module):
                """
                Intermediate layer for the FeedForward mechanism in the BERT model (after BertAttention).
                This layer consits of a linear transformation and a GELU activation function. 

                Attributes:
                    linear (nn.Linear): Linear layer
                    non_linear (nn.GELU): GELU activation function
                """

                def __init__(self, ):
                    """
                    Initializes BertIntermediate layer. 
                    """
                    super().__init__()
                    self.linear = nn.Linear(EMBED_SIZE, EMBED_SIZE*4)
                    self.non_linear = nn.GELU()

                def forward(self, x):
                    """
                    Forward pass through intermediate layer.

                    Args:
                        x (torch.Tensor): Input tensor

                    Returns:
                        torch.Tensor: Output tensor after intermediate (FeedForward) layer                   
                    """
                    return self.non_linear(self.linear(x))

            class BertOutput(nn.Module):
                """
                Output layer of the encoder. 

                Attributes:
                    linear (nn.Linear): Linear layer
                    normlayer (nn.LayerNorm): Layer normalization
                    dropout (nn.Dropout): Dropout layer
                """

                def __init__(self):
                    """
                    Initializes the BeryOutput layer.
                    This layer consists of a linear layer, a layer normalization and a dropout layer. 
                    """
                    super().__init__()
                    self.linear = nn.Linear(EMBED_SIZE*4, EMBED_SIZE)
                    self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
                    self.dropout = nn.Dropout(DROPOUT)

                def forward(self, x, residual):
                    """
                    Forward pass through the output layer. 

                    Args:
                        x (torch.Tensor): Input tensor

                    Returns:
                        torch.Tensor: Output tensor after application of linear embedding, normalization and dropout (output of the entire encoder)
                    """
                    x = self.linear(x)
                    x = self.dropout(x)
                    x = self.normlayer(x + residual)  # residual connection
                    return x

            def __init__(self):
                super().__init__()
                self.bert_attention = BERTBase.BertEncoder.BertLayer.BertAttention()
                self.bert_intermediate = BERTBase.BertEncoder.BertLayer.BertIntermediate()
                self.bert_output = BERTBase.BertEncoder.BertLayer.BertOutput()

            def forward(self, x, mask):
                """
                Forward pass through BertLayer (encoder)

                Args:
                    x (torch.Tensor): input tensor
                    mask (torch.Tensor): mask padded tokens

                Returns:
                    torch.Tensor: output of the entire encoder
                """

                x = self.bert_attention(x, mask)
                residual = x

                x = self.bert_intermediate(x)

                x = self.bert_output(x, residual)

                return x

        # base class for BERT
        def __init__(self):
            """
            Initializes the BertEncoder module. 
            """
            super().__init__()

            # init encoder layers
            self.encoders = torch.nn.ModuleList()  # create empty module list
            for i in range(NUMBER_LAYERS):
                encoder = BERTBase.BertEncoder.BertLayer()
                self.encoders = self.encoders.append(encoder)

        def forward(self, x, mask):
            """
            Forward pass through BertEncoder (through a pipeline of encoders).

            Args:
                x (torch.Tensor): Input tensor
                mask (torch.Tensor): Mask for padded tokens

            Returns:
                torch.Tensor: Output tensor after passing through the encoder pipeline (the BERT-base)
            """
            # run trough encoders
            x = x
            for encoder in self.encoders:
                x = encoder.forward(x, mask)
            return x

    class BertEmbedding(torch.nn.Module):

        """
        BERT embedding layer for token, position, and segment embeddings of the raw input.

        Attributes:
            token (nn.Embedding): Token embedding layer
            position (PositionEmbedding): Positional embedding layer
            segment (nn.Embedding): Segment embedding layer
            normlayer (nn.LayerNorm): Layer normalization
            dropout (torch.nn.Dropout): Dropout layer
        """

        class PositionEmbedding(torch.nn.Module):
            """
                Generates positional embeddings for input sequences.
                The positional embeddings are created only once during initialization and are not updated during gradient descent.

                Parameters:
                    embed_size (int): Dimensionality of the embedding vector (default is EMBED_SIZE).
                    seq_len (int): Length of the input sequence (default is SEQ_LEN).
            """

            def create_embedding_matrix(self, embed_size, seq_len):
                """
                Creates a matrix that applies a positional embedding to the input.

                Args:
                    embed_size (int): Dimension of input vector
                    seq_len (int): Length of the sequence

                Returns:
                    torch.Tensor: Matrix for positional embedding
                """
                embed_matrix = torch.zeros(
                    seq_len, embed_size, device=DEVICE).float()  

                # positional encoding not to be updated during gradient descent
                embed_matrix.requires_grad = False

                # compute embedding for each position in input
                for position in range(seq_len):
                    embed_matrix[position, ::2] = torch.sin(
                        position / (10000 ** (2 * torch.arange(0, embed_size, 2) / embed_size)))
                    embed_matrix[position, 1::2] = torch.cos(
                        position / (10000 ** (2 * torch.arange(0, embed_size, 2) / embed_size)))

                return embed_matrix

            def __init__(self, embed_size=EMBED_SIZE, seq_len=SEQ_LEN):
                """
                Initializes a positional embedding (PositionEmbedding module).
                """
                super().__init__()
                self.pos_embedding = self.create_embedding_matrix(
                    embed_size, seq_len)

            def forward(self, x):
                """
                Forward pass for positional embedding. 

                Args: 
                    x (torch.Tensor): Input tensor

                Returns:
                    torch.Tensor: Positional embeding of the input
                """
                return self.pos_embedding

        def __init__(self):
            """
            Initializes the BertEmbedding for token, position, and segment embeddings of the raw input.

            Attributes:
                token (nn.Embedding): Token embedding
                position (PositionEmbedding): Positional embedding
                segment (nn.Embedding): Segment embedding
                normlayer (nn.LayerNorm): Normalization layer
                dropout (torch.nn.Dropout): Dropout layer
            """
            super().__init__()
            # token embedding: transforms (vocabulary size, number of tokens) into (vocabulary size, number of tokens, length of embedding vector)
            self.token = nn.Embedding(VOCAB_SIZE, EMBED_SIZE, padding_idx=0).to(
                DEVICE)  # padding remains 0 during training
            # embedding of position
            self.position = BERTBase.BertEmbedding.PositionEmbedding()
            self.segment = nn.Embedding(2, EMBED_SIZE, padding_idx=0)
            self.normlayer = nn.LayerNorm(EMBED_SIZE, eps=EPS)
            self.dropout = torch.nn.Dropout(p=DROPOUT)

        def forward(self, sequence, segments):
            """
            Forward pass through Embedding, applies BertEmbedding to input to be processed by the model.

            Args:
                sequence (torch.Tensor): Input tensor
                segments (torch.Tensor): Segmentation mask tensor

            Returns:
                torch.Tensor: Embedding 
            """

            total_embedding = self.token(
                sequence) + self.position(sequence) + self.segment(segments)
            norm_embedding = self.normlayer(total_embedding)
            return self.dropout(norm_embedding)

    def __init__(self, use_pretrained=True):
        """
        Initializes a BERTBase model to which pre-trained weights can be applied.

        Args:
            use_pretrained (bool): States if pre-trained weights are to be used for the BERTBase model
        """

        super().__init__()

        self.embedding = BERTBase.BertEmbedding()
        self.encoder = BERTBase.BertEncoder()

        if use_pretrained:
            self.load_from_pretrained()

    def load_from_pretrained(self):
        """
        Method to load pretrained weights from https://huggingface.co/bert-base-uncased into the BERTBase model.

        TODO: implement method
        """
        pass
        """
        # Download pretrained weights from huggingface (for the base BERT)
        bert_base = "bert-base-uncased"
        pretrained_model = BertModel.from_pretrained(bert_base)

        # stack encoders and apply the pretrained weights to the layers of the encoders
        self.encoders = torch.nn.ModuleList()  # create empty module list
        for i in range(NUMBER_LAYERS):
            pretrained_encoder = pretrained_model.encoder.layer[i].state_dict()
            encoder = self.encoders[i]
            encoder = encoder.load_state_dict(pretrained_encoder, strict=False)
            self.encoders.insert(i,encoder)
        
        # freeze the pre-trained parameters
        for param in self.encoders.parameters():
            param.requires_grad = False
        """
        

    def forward(self, words):
        """
        Forward pass through the BERTBase model. Applies a mask to the padded characters so they remain 0 during training, applies a segment mask for non-padded characters.

        Args:
            words (torch.Tensor): Input line tensor

        Returns:
            torch.Tensor: Output of the BERT base model
        """

        # mask to mark the padded (0) tokens
        mask = (words > 0).unsqueeze(1).repeat(
            1, words.size(1), 1).unsqueeze(1)
        segments = (words > 0).to(torch.int).to(DEVICE)

        x = self.embedding(words, segments)
        # run trough encoders
        x = self.encoder(x, mask)

        return x

# finetuning
class BERTMultiLabelClassification(nn.Module):
    """
        Head for multi-label classification. Expects the usage of BCEWithLogits (no sigmoid layer).

        Attributes:
            linear (nn.Linear): Linear layer for classification using the CLS Embedding
        """

    def __init__(self):
        """
        Initializes the BERTMultiLabelCLassification head. 

        Attributes:
            linear (nn.Linear): Linear layer for classification using CLS embedding
        """

        super().__init__()
        bert_out = EMBED_SIZE
        self.linear = nn.Linear(bert_out, len(ORDER_LABELS))  

    def forward(self, x):

        # receive output dimension (batch_size, self.tox_classes)
        x = self.linear(x[:, 0])  # only extract CLS embedding (at beginning)

        return x


# TASK SHEET: model class
class Model(nn.Module):

    """
    TASK SHEET: model class. Model for Toxic Comment Classification using BERT.

    Attributes:
        base_model (BERTBase): Base BERT model for feature extraction.
        toxic_comment (BERTMultiLabelClassification): Multi-label classification layer for toxicity prediction.
    """

    def __init__(self):
        """
        Initializes the entire Model for toxic comment classification.

        Attributes:
            base_model (BERTBase): BERT Base model
            toxic_comment (BERTMultiLabelClassification): Toxic comment classification layer
        """
        super().__init__()
        # base BERT model
        self.base_model = BERTBase()
        # toxic comment classification layer
        self.toxic_comment = BERTMultiLabelClassification()

    def forward(self, words):
        """
        Performs the forward pass through the entire model.

        Args:
            words (torch.Tensor): raw input tensor (comments)

        Returns:
            torch.Tensor: Output of the model
        """

        x = self.base_model(words)

        return self.toxic_comment(x)
