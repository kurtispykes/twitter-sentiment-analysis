# data
DATA_DIR = "../inputs/"
ORIGINAL_TRAIN = DATA_DIR + "train.csv"
MODIFIED_TRAIN = DATA_DIR + "modified_train.csv"
TEST_DATA = DATA_DIR + "test.csv"
MODIFIED_TEST = DATA_DIR + "modified_test.csv"
SUBMISSION = DATA_DIR + "sample_submission.csv"
MODEL_DIR = "../models/"
IMAGES = "../images/"

# features
ID = "id"
TEXT = "text"
KEYWORD = "keyword"
LOCATION = "location"
FOLD = "kfold"
TOKENS = "tokens"

# created features
ALL_TEXT = "all_text"
CLEANED_TEXT = "cleaned_text"

# target
TARGET = "target"
RELABELED_TARGET = "relabeled_target"

# Pretrained Word2Vec
PRETRAINED_WORD2VEC = "word2vec-google-news-300"
EMBED_SIZE = 300

# TRAINING
HIDDEN_DIM = 256
TARGET_DIM = 1
BATCH_SIZE = 32
N_EPOCHS = 8
N_SPLITS = 5
LEARNING_RATE = 1e-3
MAXLEN = 202
VOCAB_SIZE = 172901

