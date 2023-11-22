import torch
from transformers import BertTokenizer
"""
Configuration Parameters:
    TOXIC (str): Path to the dataset.
    TRAIN_LENGTH (int): Length of the training set.
    TEST_LENGTH (int): Length of the test set.
    DEVICE (str): Device for training ('cuda' if available, else 'cpu').
    BATCH_SIZE (int): Batch size for training.
    EPOCHS (int): Number of training epochs.
    LEARNING_RATE (float): Learning rate for optimization.
    THRESHOLD (float): Classification threshold.
    ORDER_LABELS (list): Order of label classes.
    CLASS_WEIGHTS (dict): Weights assigned to each label class.
    WEIGHTS_LIST (list): List of weights corresponding to ORDER_LABELS.
    TOKENIZER (BertTokenizer): Tokenizer for processing text data.
    N_SEGMENTS (int): Number of segmentation labels.
    NUMBER_LAYERS (int): Number of layers in the model.
    NUMBER_HEADS (int): Number of attention heads in the model.
    EMBED_SIZE (int): Size of the embedding vector.
    VOCAB_SIZE (int): Vocabulary size.
    SEQ_LEN (int): Maximum sequence length.
    DROPOUT (float): Dropout probability.
    PS (float): Small constant for numerical stability.
"""


### DATASET PATH

TOXIC = r"C:\Users\Johannes\Project Machine Learning\datasets\finetuning\toxic_comment" # Johannes
#TOXIC = r"/home/space/datasets/toxic_comment" # cluster path
#TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment" # Moritz

### RUN SPECIFIC

TRAIN_LENGTH = 100000# 159571 # length of training set
TEST_LENGTH = 10000
DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128 # 512
EPOCHS = 2 # 10
LEARNING_RATE = 0.00001
THRESHOLD = 0.5

ORDER_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

CLASS_WEIGHTS = {
 'toxic': 10.433568719759382,
 'severe_toxic': 100.04451410658307,
 'obscene': 18.886377086045687,
 'threat': 333.8305439330544,
 'insult': 20.257839278913295,
 'identity_hate': 113.57366548042704}

WEIGHTS_LIST = [CLASS_WEIGHTS[key] for key in ORDER_LABELS]

### MODEL OR TOKENIZER SPECIFIC
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased") 
N_SEGMENTS = 2 # number of segmentation labels
NUMBER_LAYERS = 12 # 12 
NUMBER_HEADS = 12 # 12
EMBED_SIZE = 768 # size of embedding vector
VOCAB_SIZE = 30522  # len(tokenizer.vocab)
SEQ_LEN = 64 # maximum sequence length
DROPOUT = 0.1 #0.1
EPS =  1e-12


