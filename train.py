# download from http://nlp.stanford.edu/data/glove.twitter.27B.zip
# WORD_VECTORS = "../embeddings/glove.twitter.27B.50d.txt"
from torch.utils.data import DataLoader

from modules.dataloaders import SentenceDataset
from utils.load_embeddings import load_word_vectors

########################################################
# PARAMETERS
########################################################

EMBEDDINGS = "../embeddings/glove.twitter.27B.50d.txt"
EMB_DIM = 50
BATCH_SIZE = 128
EPOCHS = 50

########################################################
# Define the datasets/dataloaders
########################################################

# 1 - load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# you can load the raw data like this:
# train = load_semeval2017A("datasets/Semeval2017A/train_dev")
# val = load_semeval2017A("datasets/Semeval2017A/gold")

# 2 - define the datasets
train_set = SentenceDataset...
test_set = SentenceDataset...

loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

# define a simple model, loss function and optimizer


#############################################################################
# Training Pipeline
#############################################################################

# loop the dataset with the dataloader that you defined and train the model
# for each batch return by the dataloader
