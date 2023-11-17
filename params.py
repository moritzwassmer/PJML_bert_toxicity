import torch

# global constants
TOXIC = r"C:\Users\Johannes\Project Machine Learning\datasets\finetuning\toxic_comment" # local path
#TOXIC = r"/home/space/datasets/toxic_comment" # cluster path
SEQ_LEN = 64 # maximum sequence length
VOCAB_SIZE = 30522  # = len(tokenizer.vocab)
N_SEGMENTS = 3 # number of segmentation labels
EMBED_SIZE = 768 # size of embedding vector
TRAIN_LENGTH = 159571 # length of training set
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
