#!/usr/bin/python
from tqdm import tqdm
import numpy as np
import random
import dynet as dy
import numpy as np
import math
import argparse
from model.data_utils import CoNLLDataset, load_vocab, get_processing_word, get_trimmed_glove_vectors
from model.config import Config



parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch_size',type=float, action='store', default=10.0, help='set batch size')
options = parser.parse_args()


config = Config()
LAYERS_WORD = 1
LAYERS_CHAR = 1
HIDDEN_DIM_CHAR = config.hidden_size_char
HIDDEN_DIM_WORD = config.hidden_size_lstm
DIM_WORD = config.dim_word
DIM_CHAR = config.dim_char


vocab = get_trimmed_glove_vectors(config.filename_trimmed)
vocab_char = load_vocab(config.filename_chars)
vocab_tag = load_vocab(config.filename_tags)

VOCAB_WORD_SIZE = len(vocab)
VOCAB_CHAR_SIZE = len(vocab_char)
VOCAB_TAG_SIZE = len(vocab_tag)

trainData = CoNLLDataset(config.filename_train, config.processing_word, config.processing_tag, config.max_iter)
train = [i for i in trainData]    
developData = CoNLLDataset(config.filename_dev, config.processing_word, config.processing_tag, config.max_iter)
dev = [i for i in developData]
testData = CoNLLDataset(config.filename_test, config.processing_word, config.processing_tag, config.max_iter)
test = [i for i in testData]
    
pc = dy.ParameterCollection()

LSTM = dy.VanillaLSTMBuilder
wordFwdLSTM = LSTM(LAYERS_WORD, DIM_WORD + HIDDEN_DIM_CHAR * 2, HIDDEN_DIM_WORD, pc)
wordBwdLSTM = LSTM(LAYERS_WORD, DIM_WORD + HIDDEN_DIM_CHAR * 2, HIDDEN_DIM_WORD, pc)
charFwdLSTM = LSTM(LAYERS_CHAR, DIM_CHAR, HIDDEN_DIM_CHAR, pc)
charBwdLSTM = LSTM(LAYERS_CHAR, DIM_CHAR, HIDDEN_DIM_CHAR, pc)

LOOKUP_CHAR = pc.add_lookup_parameters((VOCAB_CHAR_SIZE, DIM_CHAR))
LOOKUP_WORD = pc.add_lookup_parameters((VOCAB_WORD_SIZE, DIM_WORD), init=vocab)
LOOKUP_WORD.set_updated(False)

paramLSTM = pc.add_parameters((VOCAB_TAG_SIZE, HIDDEN_DIM_WORD*2))
biasLSTM = pc.add_parameters((VOCAB_TAG_SIZE))
paramMLP = pc.add_parameters((VOCAB_TAG_SIZE, VOCAB_TAG_SIZE))
biasMLP = pc.add_parameters(VOCAB_TAG_SIZE)


def softmax(vec, dim = 0):
    row,col = np.matrix(vec).shape
    if row == 1 or col == 1:
        return np.exp(vec) / np.sum(np.exp(vec))
    else:
        if dim == 0:
            for i in range(col):
                vec[:,i] = softmax(vec[:,i])
        else:
            for i in range(row):
                vec[i,:] = softmax(vec[i,:])

transCRF = pc.add_parameters((VOCAB_TAG_SIZE, VOCAB_TAG_SIZE), init = softmax(np.random.rand(VOCAB_TAG_SIZE, VOCAB_TAG_SIZE), dim=1))
beginCRF = pc.add_parameters((VOCAB_TAG_SIZE, 1), init = softmax(np.random.rand(VOCAB_TAG_SIZE, 1)))
endCRF = pc.add_parameters((VOCAB_TAG_SIZE, 1), init = softmax(np.random.rand(VOCAB_TAG_SIZE, 1)))


def build_graph(sentence):
#    dy.renew_cg()
    charFwdInit = charFwdLSTM.initial_state()
    charBwdInit = charBwdLSTM.initial_state()
    wordFwdInit = wordFwdLSTM.initial_state()
    wordBwdInit = wordBwdLSTM.initial_state()
    
    word_embs = []
    for word in sentence:
        char_embs = [LOOKUP_CHAR[cid] for cid in word[0]]
        charFwdExpr = charFwdInit.transduce(char_embs)
        charBwdExpr = charBwdInit.transduce(reversed(char_embs))
        charExpr = dy.concatenate([charFwdExpr[-1], charBwdExpr[-1]])        
        word_embs.append(dy.concatenate([charExpr, LOOKUP_WORD[word[1]]]))
        
    wordFwdInit = wordFwdLSTM.initial_state()
    wordBwdInit = wordBwdLSTM.initial_state()
    wordFwdExpr = wordFwdInit.transduce(word_embs)
    wordBwdExpr = wordBwdInit.transduce(reversed(word_embs))
    sentExpr = [dy.concatenate([f,b]) for f,b in zip(wordFwdExpr, reversed(wordBwdExpr))]

    mlps = [paramMLP * dy.tanh(paramLSTM * sent + biasLSTM) + biasMLP for sent in sentExpr]
                    
    return mlps


def CRF_score(MLPs, tags):
    assert len(MLPs) == len(tags)
    score = dy.scalarInput(0)
    score = score + beginCRF[tags[0]]
    score = score + endCRF[tags[-1]]
    for i, obs in enumerate(MLPs):
        score = score + obs[tags[i]]
    for i, obs in enumerate(MLPs[:len(MLPs)-1]):
        score = score + transCRF[tags[i]][tags[i+1]]
    return dy.exp(score)


def CRF_estimate(MLPs):
    best_tags = []
    alpha = [0] * len(MLPs)
    back = [0] * len(MLPs)
    for i in range(len(MLPs)):
        alpha[i] = [0] * VOCAB_TAG_SIZE
        back[i] = [0] * VOCAB_TAG_SIZE
        for j in range(VOCAB_TAG_SIZE):
            if i == 0:
                alpha[i][j] = beginCRF[j] + MLPs[i][j]
            if i != 0:
                alpha[i][j] = alpha[i][j] + MLPs[i][j]
                if i == VOCAB_TAG_SIZE -1 :
                    alpha[i][j] = alpha[i][j] + endCRF[j]

                candidates = [transCRF[idx][j] for idx in range(VOCAB_TAG_SIZE)]                
                alpha[i][j] = alpha[i][j] + max(candidates)
                back[i][j] = candidates.index(max(candidates))

    best_tags.append(alpha[len(MLPs)-1].index(max(alpha[len(MLPs)-1])))
    for i in reversed(range(len(MLPs))):
        if i != 0:
            best_tags.append(back[i][best_tags[len(MLPs)-1-i]])
    best_tags = list(reversed(best_tags))
#    print best_tags
    return best_tags


def CRF_partition(MLPs):
    z = [0] * len(MLPs)
    z[len(MLPs)-1] = dy.exp(endCRF).value()
    for i in reversed(range(len(MLPs)-1)):
        z[i] = [0] * VOCAB_TAG_SIZE
        for j in range(VOCAB_TAG_SIZE):
            for jj in range(VOCAB_TAG_SIZE):
                z[i][j] = z[i][j] + z[i+1][jj]*dy.exp(MLPs[i][j]+transCRF[j][jj].value()).value()
    
    zout = 0
    for j in range(VOCAB_TAG_SIZE):
        zout = zout + (z[0][j])
    return zout[0]


def sentence_feed(sentence):
    dy.renew_cg()
    MLPs = build_graph(sentence)
    tags = CRF_estimate(MLPs)
    return (MLPs, tags)


def get_loss(sentence, tags):
    MLPs = build_graph(sentence)
    gold_score = CRF_score(MLPs, tags)
    test_score = CRF_score(MLPs, CRF_estimate(MLPs))
    z = CRF_partition(MLPs)
#    print test_score.value()
#    print gold_score.value()
#    print z
    return (gold_score / z) - (test_score / z)


num_tagged = cum_loss = 0
max_iter = 50
trainer = dy.AdamTrainer(pc)
trainer.set_sparse_updates(False)

def train_one(sentence, tags):    
    global num_tagged
    global cum_loss
    dy.renew_cg()
    loss_exp = get_loss(sentence,tags)
#    print loss_exp.scalar_value()
    loss_exp.forward()
    cum_loss = cum_loss + loss_exp.scalar_value()
    num_tagged = num_tagged + len(tags)
    loss_exp.backward()
    try:
        trainer.update()
        return 0
    except:
        return 1


def train_batch(batch):
    dy.renew_cg()
    fail = 0
    losses = []
    for (sentence, tags) in batch:
        losses.append(get_loss(sentence,tags))
    loss = sum(losses)
    loss.forward()
    loss.backward()
    try:
        trainer.update()
        return 0
    except:
        return 1
    
def print_status():
    global num_tagged
    global cum_loss
    print trainer.status()
    print cum_loss / num_tagged
    cum_loss = num_tagged = 0


def evaluate(data):
    tp = tn = fp = fn = 0.0
    for idx, (sentence, gold) in enumerate(data):
        try:
            MLPS, tags = sentence_feed(sentence)
        except:
            continue
        for i in range(len(tags)):
            if gold[i] == tags[i]:
                if gold[i] == 0:
                    tn = tn + 1
                else:
                    tp = tp + 1
            else:
                if gold[i] == 0:
                    fp = fp + 1
                else:
                    fn = fn + 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


if options.batch_size:
    batch_size = options.batch_size
else:
    batch_size = 20.0

def main():
    prev_p = prev_recall = prev_f1 = 0
    stable_count = 0
    for iter in range(max_iter):
        random.shuffle(train)
        fails = 0
#        for idx in tqdm(range(len(train))):
#            sentence,tags = train[idx]
#            fails = fails + train_one(sentence,tags)
#        for idx in tqdm(range(int(math.ceil(len(train)/batch_size)))):
        for idx in tqdm(range(10)):
            start = int(idx*batch_size)
            end = int(idx*batch_size + min(batch_size, len(train)-idx*batch_size))
            curr_batch = train[start:end]
            fails = fails + train_batch(curr_batch)
        print 'Failed batches {:02d} {:02.2f}'.format(fails, float(fails)/math.ceil(len(train)/batch_size))
        (p,r,f1) = evaluate(dev[0:5])
        print 'current iteration {:02d} precision {:02.2f} recall {:02.2f} F1 {:02.2f}'.format(iter, p,r,f1)
        if p == prev_p and r == prev_r and f1 == prev_f1:
            stable_count = stable_count + 1
        else:
            prev_p = p
            prev_r = r
            prev_f1 = f1
        if stable_count == 3:
            print 'Converged! Start evalutation on test sets'
            break

    (p,r,f1) = evaluate(test)
    print 'Test {:2d} precision {:2.2f} recall {:2.2f} F1 {:2.2f}'.format(iter, p,r,f1)



if __name__ == '__main__':
    main()
