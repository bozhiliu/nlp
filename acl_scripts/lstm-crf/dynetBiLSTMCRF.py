#!/usr/bin/python

import dynet_config

dynet_config.set(mem=10000)

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
parser.add_argument('--batch', dest='batch_size',type=float, action='store', default=2.0, help='set batch size')
parser.add_argument('--load', dest='load', action='store_true', help='whether if load previous network')
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


transCRF = pc.add_lookup_parameters((VOCAB_TAG_SIZE, VOCAB_TAG_SIZE))
beginCRF = pc.add_lookup_parameters((1, VOCAB_TAG_SIZE))
endCRF = pc.add_lookup_parameters((1, VOCAB_TAG_SIZE))

num_tagged = cum_loss = 0
max_iter = 50
trainer = dy.AdamTrainer(pc)
trainer.set_sparse_updates(False)

if options.batch_size:
    batch_size = options.batch_size
else:
    batch_size = 20.0
paramsFile = './data/dynet_params'


def save_network():
    dy.save(paramsFile, [wordFwdLSTM, wordBwdLSTM, charFwdLSTM, charBwdLSTM, LOOKUP_CHAR, LOOKUP_WORD, paramLSTM, biasLSTM, paramMLP, biasMLP, transCRF, beginCRF, endCRF])


def load_network():
    global wordFwdLSTM, wordBwdLSTM, charFwdLSTM, charBwdLSTM, LOOKUP_CHAR, LOOKUP_WORD, paramLSTM, biasLSTM, paramMLP, biasMLP, transCRF, beginCRF, endCRF
    wordFwdLSTM, wordBwdLSTM, charFwdLSTM, charBwdLSTM, LOOKUP_CHAR, LOOKUP_WORD, paramLSTM, biasLSTM, paramMLP, biasMLP, transCRF, beginCRF, endCRF = dy.load(paramsFile, pc)


def build_graph(sentence):
#    dy.renew_cg()
#    print 'build graph ' + str(len(sentence))
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
#    print 'crf score mlp ' + str(len(MLPs))
#    print 'crf score tags ' + str(len(tags))
    assert len(MLPs) == len(tags)
    score = dy.scalarInput(0)
    score = score + dy.pick(beginCRF, tags[0])
    for i, obs in enumerate(MLPs):
        if i != len(MLPs)-1:
            score = score + dy.pick(transCRF[tags[i+1]], tags[i]) + dy.pick(obs, tags[i])
        else:
            score = score + dy.pick(obs, tags[i]) + dy.pick(endCRF, tags[i])
    return score


def CRF_estimate(MLPs):
    backpointers = []
    for_exp = dy.inputVector([0.0] * VOCAB_TAG_SIZE)
    trans_exprs = [transCRF[idx] for idx in range(VOCAB_TAG_SIZE)]
    for_exp = for_exp + beginCRF
    for obs in MLPs:
        bptrs_t = []
        vvars_t = []
        for next_tag in range(VOCAB_TAG_SIZE):
            next_tag_expr = for_exp + trans_exprs[next_tag]
            next_tag_arr = next_tag_expr.npvalue()
            best_tag_id = np.argmax(next_tag_arr)
            bptrs_t.append(best_tag_id)
            vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
        if MLPs.index(obs) == 0:
            for_exp = beginCRF + obs
        else:                
            for_exp = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)

    terminal_expr = for_exp + endCRF
    terminal_arr = terminal_expr.npvalue()
    best_tag_id = np.argmax(terminal_arr)
    path_score = dy.pick(terminal_expr, best_tag_id)

    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
        best_tag_id = bptrs_t[best_tag_id]
        best_path.append(best_tag_id)
    best_path.reverse()
    return best_path
    

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
    MLPs = build_graph(sentence)
    tags = CRF_estimate(MLPs)
    return (MLPs, tags)


def get_loss(sentence, tags):
#    print 'get loss ' + str(len(sentence))
    MLPs = build_graph(sentence)
    gold_score = CRF_score(MLPs, tags)
    test_score = CRF_score(MLPs, CRF_estimate(MLPs))
#    z = CRF_partition(MLPs)
    return test_score - gold_score


def train_one(sentence, tags):    
    global num_tagged
    global cum_loss
#    print 'Train one ' + str(len(sentence))
    loss_exp = get_loss(sentence,tags)
#    print loss_exp.scalar_value()
    loss_exp.forward()
    cum_loss = cum_loss + loss_exp.scalar_value()
    num_tagged = num_tagged + len(tags)
    loss_exp.backward()
    trainer.update()
    

def train_batch(batch):
    global num_tagged
    global cum_loss
    fail = 0
    losses = []
    for (sentence, tags) in batch:
        losses.append(get_loss(sentence,tags))
        num_tagged = num_tagged + len(tags)
    loss = dy.esum(losses)
    cum_loss = cum_loss + loss.value()
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
    print 'averaged training loss {:02.2f}'.format((cum_loss / num_tagged)*100)


def evaluate(data):
    stats = {t:[0.0,0.0,0.0] for t in vocab_tag.values()}   # tp fp fn
    tp = tn = fp = fn = 0.0
    for idx, (sentence, gold) in enumerate(data):
        try:
            dy.renew_cg()
            MLPS, tags = sentence_feed(sentence)
        except:
            continue
        for i in range(len(tags)):
            if gold[i] == tags[i]:
                stats[gold[i]][0] = stats[gold[i]][0] + 1
                if gold[i] == 0:
                    tn = tn + 1
                else:
                    tp = tp + 1
            else:
                stats[tags[i]][1] = stats[tags[i]][1] + 1
                stats[gold[i]][2] = stats[gold[i]][2] + 1
                if gold[i] == 0:
                    fp = fp + 1
                else:
                    fn = fn + 1
    print stats
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    precision_results = {t:stats[t][0] / (stats[t][0] + stats[t][1]) for t in vocab_tag.values()}
    recall_results = {t:stats[t][0] / (stats[t][0] + stats[t][2]) for t in vocab_tag.values()}
    f1_results = {t:2*precision_results[t]*recall_results[t] / (precision_results[t]+recall_results[t]) for t in vocab_tag.values()}
    return (precision, recall, f1, precision_results, recall_results, f1_results)

    
def main():
    global cum_loss
    global num_tagged
    prev_loss = prev_p = prev_recall = prev_f1 = 0
    stable_count = 0
    if options.load:
        load_network()
    for iter in range(max_iter):
        random.shuffle(train)
        save_network()
        fails = 0
#        for idx in tqdm(range(len(train))):
#            sentence,tags = train[idx]
#            fails = fails + train_one(sentence,tags)
#        for idx in tqdm(range(int(math.ceil(len(train)/batch_size)))):
        for idx in tqdm(range(100)):
            dy.renew_cg()
            start = int(idx*batch_size)
            end = int(idx*batch_size + min(batch_size, len(train)-idx*batch_size))
            curr_batch = train[start:end]
            fails = fails + train_batch(curr_batch)
        print 'Failed batches {:02d} {:02.2f}'.format(fails, float(fails)/math.ceil(len(train)/batch_size))
        print_status()
        random.shuffle(dev)
        dy.renew_cg()
        (p,r,f1, pr,rr,f1r) = evaluate(dev[:])
        print 'current iteration {:02d} precision {:02.2f} recall {:02.2f} F1 {:02.2f}'.format(iter+1, p,r,f1)
        for t in vocab_tag:
            print '{:s}'.format(t).rjust(20) + ' precision {:02.2f} recall {:02.2f} F1 {:02.2f}'.format(pr[vocab_tag[t]],rr[vocab_tag[t]],f1r[vocab_tag[t]])
        if prev_f1 == f1:
            stable_count = stable_count + 1
        else:
            print 'previous cumulated loss: {:02f} current cummulated loss: {:02f}'.format(prev_loss, cum_loss)
            prev_loss = cum_loss
            prev_p = p
            prev_r = r
            prev_f1 = f1
            cum_loss = num_tagged = 0
        if stable_count == 3:
            print 'Converged! Start evalutation on test sets'
            break

    (p,r,f1, pr, rr, f1r) = evaluate(test)
    print 'Test {:2d} precision {:2.2f} recall {:2.2f} F1 {:2.2f}'.format(iter, p,r,f1)
    for t in vocab_tag:
        print '{:s}'.format(t).rjust(8) + ' precision {:02.2f} recall {:02.2f} F1 {:02.2f}'.format(pr[vocab_tag[t]],rr[vocab_tag[t]],f1r[vocab_tag[t]])



if __name__ == '__main__':
    main()
