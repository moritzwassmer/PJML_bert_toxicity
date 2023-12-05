import torch
from transformers import BertTokenizer
"""
Parameters for configuration:

    - Dataset path:
        TOXIC (str): Path of dataset

    - Output folder:
        OUTPUT (str): directory where output is saved

    - Run specific:
        TRAIN_LENGTH (int): Training set length 
        TEST_LENGTH (int): Test set length
        DEVICE (str): Device to run code
        BATCH_SIZE (int): Batch size
        EPOCHS (int): Number of training epochs
        LEARNING_RATE (float): Optimization learning rate
        THRESHOLD (float): Threshold for classification
        ORDER_LABELS (list): Order of the labels
        CLASS_WEIGHTS (dict): Weights for each class for loss function-balancing
        WEIGHTS_LIST (list): List of weights in the order of the labels

    - Model or tokenizer specific:
        TOKENIZER (BertTokenizer): Tokenizer for the raw input comments
        N_SEGMENTS (int): Number of segment labels
        NUMBER_LAYERS (int): Number of model layers
        NUMBER_HEADS (int): Number of attention heads
        EMBED_SIZE (int): Dimension of the embedding vector of the model
        VOCAB_SIZE (int): Vocabulary size
        SEQ_LEN (int): Maximum length of the input sequence
        DROPOUT (float): Dropout probability
        EPS (float): Small constant for layer normalization
"""


# DATASET PATH

TOXIC = r"C:/Users/Johannes/Project Machine Learning/datasets/finetuning/toxic_comment/"  # Johannes
# TOXIC = r"/home/space/datasets/toxic_comment/" # cluster path
# TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment/" # Moritz

# OUTPUT PATH

OUTPUT = "output_folder" # output folder

# RUN SPECIFIC

TRAIN_LENGTH = 12800 # 159571  # length of training set
TEST_LENGTH = 12800 # 63978  # length of test set
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64 # 512
EPOCHS = 10
LEARNING_RATE = 0.00001
THRESHOLD = 0.5

ORDER_LABELS = ['toxic', 'severe_toxic',
                'obscene', 'threat', 'insult', 'identity_hate']

CLASS_WEIGHTS = {
    'toxic': 10.433568719759382,
    'severe_toxic': 100.04451410658307,
    'obscene': 18.886377086045687,
    'threat': 333.8305439330544,
    'insult': 20.257839278913295,
    'identity_hate': 113.57366548042704}

WEIGHTS_LIST = [CLASS_WEIGHTS[key] for key in ORDER_LABELS]

# MODEL OR TOKENIZER SPECIFIC
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
BERT_BASE = "bert-base-uncased"
N_SEGMENTS = 2  # number of segmentation labels
NUMBER_LAYERS = 12  # 12
NUMBER_HEADS = 12  # 12
EMBED_SIZE = 768  # size of embedding vector
VOCAB_SIZE = 30522  # len(tokenizer.vocab)
SEQ_LEN = 512 # 64  # maximum sequence length
DROPOUT = 0.1
EPS = 1e-12
