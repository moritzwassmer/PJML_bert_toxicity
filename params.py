import torch

# global constants
#TOXIC = r"C:\Users\Johannes\Project Machine Learning\datasets\finetuning\toxic_comment" # local path
#TOXIC = r"/home/space/datasets/toxic_comment" # cluster path
TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment"
SEQ_LEN = 64 # maximum sequence length
VOCAB_SIZE = 30522  # = len(tokenizer.vocab)
N_SEGMENTS = 3 # number of segmentation labels
EMBED_SIZE = 768 # size of embedding vector
TRAIN_LENGTH = 2500# 159571 # length of training set
DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64 # 512
# number of epochs
EPOCHS = 3 # 10