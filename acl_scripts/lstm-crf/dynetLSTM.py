import dynet as dy
import numpy as np
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
vocab_tag = load_vocab(config.filename_tags)

VOCAB_SIZE = len(vocab)
VOCAB_CHAR_SIZE = len(vocab_char)
VOCAB_TAG_SIZE = len(vocab_tag)

train = CoNLLDataset(config.filename_train, config.processing_word, config.processing_tag, config.max_iter)

pc = dy.ParameterCollection()

wordBiLSTM = dy.BiLSTMBuilder(LAYERS_WORD, DIM_WORD, HIDDEN_DIM_WORD, pc)
charBiLSTM = dy.BiLSTMBuilder(LAYERS_CHAR, DIM_CHAR, HIDDEN_DIM_CHAR/2, pc)

LOOKUP_CHAR = pc.add_lookup_parameters((VOCAB_CHAR_SIZE, DIM_CHAR/2))
LOOKUP_WORD = pc.add_lookup_parameters((VOCAB_WORD_SIZE, DIM_WORD-DIM_CHAR), init=np.delete(vocab, np.s_[DIM_WORD-DIM_CHAR:DIM_WORD], axis=1))

pLSTM = pc.add_parameters((VOCAB_TAG_SIZE, DIM_WORD))
bLSTM = pc.add_parameters((VOCAB_TAG_SIZE))
pMLP = pc.add_parameters((VOCAB_TAG_SIZE, VOCAB_TAG_SIZE))
bMLP = pc.add_parameters(VOCAB_TAG_SIZE)

transCRF = pc.add_lookup_parameters((VOCAB_TAG_SIZE, VOCAB_TAG_SIZE))
beginCRF = pc.add_lookup_parameters((VOCAB_TAG_SIZE, 1))
endCRF = pc.add_lookup_parameters((VOCAB_TAG_SIZE, 1))


def build_graph(sentence):
    dy.renew_cg()
    charInit = charBiLSTM.initial_state()
    word_embs = []
    for word in sentence:        
        char_embs = [LOOKUP_CHAR[cid] for cid in word[0]] 
        charExpr = charInit.transduce(char_embs)
        wordExpr = dy.const_parameter(LOOKUP_WORD[word[1]])
        word_embs.append(dy.concatenate(charExpr, wordExpr))
    
    wordInit = wordBiLSTM.initial_state()
    sentExpr = wordInit.transduce(word_embs)
    paramLSTM = dy.parameter(pLSTM)
    biasLSTM = dy.parameter(bLSTM)
    paramMLP = dy.parameter(pMLP)
    biasMLP = dy.parameter(bMLP)
    mlps = []
    for sent in sentExpr:
        score = paramMLP * dy.tanh(paramLSTM * sent + biasLSTM) + biasMLP
        mlps.append(score)

    return mlps


def CRF_score(MLPs, tags):
    assert len(MLPs) == len(tags)
    score = dy.scalarInput(0)
    score = score + dy.pick(beginCRF, tags[0])
    score = score + dy.pick(endCRF, tag[-1])
    for i, obs in enumerate(MLPs):
        score = score + dy.pick(obs, tags[i])
    for i, obs in enumerate(MLPs[:len(MLPs)-1]):
        score = score + dy.pick(transCRF[tags[i+1]], tags[i])
    return dy.exp(score)


def CRF_estimate(MLPs):
    best_tags = []
    return best_tags


def CRF_partition():
    z = dy.scalarInput(0)
    return z
