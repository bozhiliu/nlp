import dynet as dy
from model.data_utils import CoNLLDataset, load_vocab, get_processing_word, get_trimmed_glove_vectors
from model.config import Config

LAYERS_WORD = 1
LAYERS_CHAR = 1
HIDDEN_DIM_CHAR = 100
HIDDEN_DIM_WORD = 200


config = Config()
DIM_WORD = config.dim_word
DIM_CHAR = config.dim_char


vocab = get_trimmed_glove_vectors(config.filename_trimmed)
vocab_char = load_vocab(config.filename_chars)
VOCAB_SIZE = len(vocab)
VOCAB_CHAR_SIZE = len(vocab_char)

train = CoNLLDataset(config.filename_train, config.processing_word, config.processing_tag, config.max_iter)

model_word = dy.Model()
model_char = dy.Model()
wordBiLSTM = dy.BiLSTMBuilder(LAYERS_WORD, DIM_WORD, HIDDEN_DIM_WORD, model)
charBiLSTM = dy.BiLSTMBuilder(LAYERS_CHAR, DIM_CHAR, HIDDEN_DIM_CHAR, model)


def build_graph(sentence):
    dy.renew_cg()
    charInit = charBiLSTM.initial_state()
    for word in sentence:
        
    wordInit = wordBiLSTM.initial_state()
    ws = wordInit.transduce(words)
    
