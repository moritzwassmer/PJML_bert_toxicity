import torch

### DATASET PATH
#TOXIC = r"C:\Users\Johannes\Project Machine Learning\datasets\finetuning\toxic_comment" # Johannes
#TOXIC = r"/home/space/datasets/toxic_comment" # cluster path
TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment" # Moritz

### RUN SPECIFIC

TRAIN_LENGTH = 50000# 159571 # length of training set
TEST_LENGTH = 1024
DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128 # 512
EPOCHS = 2 # 10

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
N_SEGMENTS = 2 # number of segmentation labels
NUMBER_LAYERS = 1 # 12 # TODO does only work for 1 layer
NUMBER_HEADS = 12 # 12
EMBED_SIZE = 768 # size of embedding vector
VOCAB_SIZE = 30522  # = len(tokenizer.vocab)
SEQ_LEN = 64 # maximum sequence length
DROPOUT = 0.1 #0.1
EPS =  1e-12


