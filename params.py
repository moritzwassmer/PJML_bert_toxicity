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

# TOXIC = r"C:/Users/Johannes/Project Machine Learning/datasets/finetuning/toxic_comment/"  # Johannes
TOXIC = r"/home/space/datasets/toxic_comment/" # cluster path
# TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment/" # Moritz

# OUTPUT PATH

OUTPUT = "output_folder" # output folder

# RUN SPECIFIC
METHOD = 'bert_slanted_lr' # 'bert_base', 'bert_discr_lr', 'bert_slanted_lr'
TRAIN_LENGTH =159571  # length of training set
TRAIN_TOTAL = 159571 # length of total training set
TEST_LENGTH =63978  # length of test set
VAL_LENGTH = TEST_LENGTH//2 # length of validation set
NUM_CLASSES = 6 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# default hyperparameters (not in use)
BATCH_SIZE = 32 
EPOCHS = 10
LEARNING_RATE = 0.00001

THRESHOLD = 0.5
# see paper p. 13 f. (changed)
HYPER_PARAMS = {
    'batch_size': [16,32],
    'learning_rate': [1e-5, 5e-6, 1e-6], 
    'epochs': 4
}
ORDER_LABELS = ['toxic', 'severe_toxic',
                'obscene', 'threat', 'insult', 'identity_hate']

# weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
CLASS_WEIGHTS = {
 'toxic': (TRAIN_TOTAL/(15294*NUM_CLASSES)),
 'severe_toxic': (TRAIN_TOTAL/(1595*NUM_CLASSES)),
 'obscene':  TRAIN_TOTAL/(8449*NUM_CLASSES),
 'threat':  TRAIN_TOTAL/(478*NUM_CLASSES),
 'insult': TRAIN_TOTAL/(7877*NUM_CLASSES),
 'identity_hate':  TRAIN_TOTAL/(1405*NUM_CLASSES)
}
WEIGHTS_LIST = [CLASS_WEIGHTS[key] for key in ORDER_LABELS]

# maximum lr for slanted triangular discriminative learning rate
ETA_MAX = 0.01
# decay factor for slanted triangular discriminative learning rate (Sun et al., 2020, p.6)
DECAY = 0.95

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
