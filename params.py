import torch

### DATASET PATH
#TOXIC = r"C:\Users\Johannes\Project Machine Learning\datasets\finetuning\toxic_comment" # Johannes
#TOXIC = r"/home/space/datasets/toxic_comment" # cluster path
TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment" # Moritz

### RUN SPECIFIC
N_SEGMENTS = 3 # number of segmentation labels
TRAIN_LENGTH = 64# 159571 # length of training set
DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64 # 512
EPOCHS = 1 # 10

### MODEL OR TOKENIZER SPECIFIC
NUMBER_LAYERS = 12
NUMBER_HEADS = 12
EMBED_SIZE = 768 # size of embedding vector
VOCAB_SIZE = 30522  # = len(tokenizer.vocab)
SEQ_LEN = 64 # maximum sequence length