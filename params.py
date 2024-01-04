import torch
from transformers import BertTokenizer

"""
Configuration parameters:

    - Dataset path:
        TOXIC (str): Path of dataset

    - Output directory:
        OUTPUT (str): directory where output is saved
        BASE_TEST, BASE_TRAIN, DISCR_TEST, DISCR_TRAIN, SLANTED_TEST, SLANTED_TRAIN (str): Specific output directories for different training methods 

    - Run specific:
        METHOD (str): Training method ('bert_base', 'bert_discr_lr', 'bert_slanted_lr')
        TRAIN_LENGTH (int): Training set length 
        TEST_LENGTH (int): Test set length
        VAL_LENGTH (int): Validation set length
        NUM_CLASSES (int): Number of classes in the dataset
        DEVICE (str): Device to run code ('cuda' or 'cpu')
        THRESHOLD (float): Threshold for classification
        ORDER_LABELS (list): Order of the labels
        CLASS_WEIGHTS (dict): Weights for each class for loss function-balancing
        WEIGHTS_LIST (list): List of weights in the order of the labels
        DECAY (float): Decay factor for discriminative layer
        RESCALING_FACTOR (float): average of class weights for class weight normalization (not in use)
        HYPER_PARAMS (dict): Hyper parameters for model selection

    - Model or tokenizer specific:
        TOKENIZER (BertTokenizer): Tokenizer for the raw input comments
        BERT_BASE (str): Name of the bert model used for the pretrained weights
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
TOXIC = r"/home/space/datasets/toxic_comment/"  # cluster path
# TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment/" # Moritz

# OUTPUT DIRECTORY DEFINITIONS
OUTPUT = "output_folder"
BASE_TEST = 'test_base'  
BASE_TRAIN = 'train_base'  
SLANTED_TEST = 'test_slanted_discriminative'  
SLANTED_TRAIN = 'train_slanted_discriminative'  

# RUN SPECIFIC
METHOD = 'base'  # "slanted_discriminative", 'base'
TRAIN_LENGTH =159571
TEST_LENGTH = 63978
VAL_LENGTH = TEST_LENGTH//2
NUM_CLASSES = 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5


ORDER_LABELS = ['toxic', 'severe_toxic',
                'obscene', 'threat', 'insult', 'identity_hate']
CLASS_WEIGHTS = {
    'toxic': TRAIN_LENGTH/(15294*NUM_CLASSES),
    'severe_toxic': TRAIN_LENGTH/(1595*NUM_CLASSES),
    'obscene':  TRAIN_LENGTH/(8449*NUM_CLASSES),
    'threat':  TRAIN_LENGTH/(478*NUM_CLASSES),
    'insult': TRAIN_LENGTH/(7877*NUM_CLASSES),
    'identity_hate':  TRAIN_LENGTH/(1405*NUM_CLASSES)
}
WEIGHTS_LIST = [CLASS_WEIGHTS[key] for key in ORDER_LABELS]
RESCALING_FACTOR = torch.tensor(
    sum(CLASS_WEIGHTS.values())/len(CLASS_WEIGHTS.keys()), device=DEVICE) #TODO
WARMUP = 10000
DECAY = 0.95

HYPER_PARAMS = {
    'batch_size': [16],
    'learning_rate': [3e-5, 2e-5, 1e-5, 1e-6],
    'epochs': 4
}

# MODEL OR TOKENIZER SPECIFIC
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
BERT_BASE = "bert-base-uncased"
N_SEGMENTS = 2
NUMBER_LAYERS = 12
NUMBER_HEADS = 12
EMBED_SIZE = 768
VOCAB_SIZE = 30522
SEQ_LEN = 512
DROPOUT = 0.1
EPS = 1e-12
