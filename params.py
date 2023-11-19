import torch

### DATASET PATH
#TOXIC = r"C:\Users\Johannes\Project Machine Learning\datasets\finetuning\toxic_comment" # Johannes
#TOXIC = r"/home/space/datasets/toxic_comment" # cluster path
TOXIC = r"C:\Users\morit\OneDrive\UNI\Master\WS23\PML\repo\bert_from_scratch.toxic_comment\datasets\finetuning\kaggle-toxic_comment" # Moritz

### RUN SPECIFIC
N_SEGMENTS = 3 # number of segmentation labels
TRAIN_LENGTH = 100# 159571 # length of training set
TEST_LENGTH = 100
DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64 # 512
EPOCHS = 3 # 10

### MODEL OR TOKENIZER SPECIFIC
NUMBER_LAYERS = 12
NUMBER_HEADS = 12
EMBED_SIZE = 768 # size of embedding vector
VOCAB_SIZE = 30522  # = len(tokenizer.vocab)
SEQ_LEN = 64 # maximum sequence length

ORDER = ['nothing','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


"""
WEIGHTS = {'nothing': 0.013027581601785167,
 'toxic': 0.1221035512154764,
 'severe_toxic': 1.170816120557678,
 'obscene': 0.22102635960344377,
 'threat': 3.9068027453755154,
 'insult': 0.23707651546140618,
 'identity_hate': 1.3291471261846948}

WEIGHTS = {'nothing': 1,
 'toxic': 1,
 'severe_toxic': 1,
 'obscene': 1,
 'threat': 1,
 'insult': 1,
 'identity_hate': 1}
  """

WEIGHTS ={'nothing': 5.623175898321041,
 'toxic': 0.5999529264082849,
 'severe_toxic': 0.06256864898791778,
 'obscene': 0.3314373136670328,
 'threat': 0.018750980699827394,
 'insult': 0.30899890161619326,
 'identity_hate': 0.055115330299701865}

WEIGHTS_LIST = [WEIGHTS[key] for key in ORDER]
