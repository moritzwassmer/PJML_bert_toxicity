import torch
from transformers import BertTokenizer

"""
Configuration parameters:

    - Dataset path:
        TOXIC (str): Path of dataset

    - Output directory:
        OUTPUT (str): directory where output is saved

    - Run specific:
        METHOD (str): Training method ('bert_base', 'bert_discr_lr', 'bert_slanted_lr')
        TRAIN_LENGTH (int): Training set length 
        TEST_LENGTH (int): Test set length
        VAL_LENGTH (int): Validation set length
        NUM_CLASSES (int): Number of classes in the dataset
        DEVICE (str): Device to run code ('cuda' or 'cpu')
        BATCH_SIZE (int): Batch size
        EPOCHS (int): Number of training epochs
        LEARNING_RATE (float): Optimization learning rate
        THRESHOLD (float): Threshold for classification
        HYPER_PARAMS (dict): Hyper parameters for model selection
        ORDER_LABELS (list): Order of the labels
        CLASS_WEIGHTS (dict): Weights for each class for loss function-balancing
        WEIGHTS_LIST (list): List of weights in the order of the labels
        ETA_MAX (float): Peak learning rate for slanted triangular learning rate
        DECAY (float): Decay factor for discriminative layer

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
#TOXIC = r"/home/space/datasets/toxic_comment/"  # cluster path
TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment/" # Moritz

# OUTPUT DIRECTORY DEFINITIONS
OUTPUT = "output_folder"  
BASE_TEST = 'test_base' # 'testing_bert_base'
BASE_TRAIN = 'train_base' # 'training_bert_base'
DISCR_TEST = 'testing_bert_discr_lr'
DISCR_TRAIN = 'training_bert_discr_lr'
SLANTED_TEST = 'test_slanted' #'testing_bert_slanted_lr'
SLANTED_TRAIN = 'train_slanted' #'training_bert_slanted_lr'

# RUN SPECIFIC
METHOD = 'bert_base' #"bert_slanted_lr" #'bert_base' 
TRAIN_LENGTH = 159571//4 #12800 #159571
TRAIN_TOTAL= 159571  
TEST_LENGTH = 63978//4 #25600 #63978  
VAL_LENGTH = TEST_LENGTH//2  
NUM_CLASSES = 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4 #32
EPOCHS = 4
LEARNING_RATE = 0.00001
THRESHOLD = 0.5


ORDER_LABELS = ['toxic', 'severe_toxic',
                'obscene', 'threat', 'insult', 'identity_hate']
CLASS_WEIGHTS = {
    'toxic': TRAIN_TOTAL/(15294*NUM_CLASSES),
    'severe_toxic': TRAIN_TOTAL/(1595*NUM_CLASSES),
    'obscene':  TRAIN_TOTAL/(8449*NUM_CLASSES),
    'threat':  TRAIN_TOTAL/(478*NUM_CLASSES),
    'insult': TRAIN_TOTAL/(7877*NUM_CLASSES),
    'identity_hate':  TRAIN_TOTAL/(1405*NUM_CLASSES)
}
WEIGHTS_LIST = [CLASS_WEIGHTS[key] for key in ORDER_LABELS]
RESCALING_FACTOR = torch.tensor(sum(CLASS_WEIGHTS.values())/len(CLASS_WEIGHTS.keys()), device=DEVICE)

DECAY = 0.9

HYPER_PARAMS = {
    'batch_size': [4],
    'learning_rate': [2.5e-5], # proposed 5e-5, 3e-5, 2e-5 # 2e-5, 1e-5, 1e-6  #  1e-5 looked promising
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
