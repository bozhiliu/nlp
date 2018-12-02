import tensorflow as tf
import numpy as np
import os

from model.data_utils import minibatches, pad_sequences, CoNLLDataset
from model.general_utils import Progbar
from model.config import Config


config = Config()
config.dir_prefix = 'char_embed_adv'


#################################################################################
# batch size, max sentence length
word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

# batch size
sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

# batch size, max length sentence, max length of word
char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

# batch size, max length sentence
word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

# batch size, max length sentence
labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

dropout = tf.placeholder(tf.float32, shape=[], name="dropout")
    
lr = tf.placeholder(tf.float32, shape=[], name="lr")


#################################################################################
with tf.variable_scope('word'):
    _word_embeddings = tf.Variable(config.embeddings, name="raw_word_embeddings",dtype=tf.float32, trainable=config.train_embeddings)
    
    word_embeddings = tf.nn.embedding_lookup(_word_embeddings, word_ids, name="word_embedings")

    
#################################################################################
with tf.variable_scope('char'):
    _char_embeddings = tf.get_variable(name="raw_char_embeddings", dtype=tf.float32, shape=[config.nchars, config.dim_char])
    
    char_embeddings = tf.nn.embedding_lookup(_char_embeddings, char_ids, name="char_embeddings")
    
    # flatten words in batch into a single array
    s = tf.shape(char_embeddings)
    char_embeddings = tf.reshape(char_embeddings, shape=[s[0]*s[1], s[-2], config.dim_char])
    word_lengths_flat = tf.reshape(word_lengths, shape=[s[0]*s[1]])
    
    char_cell_fwd = tf.contrib.rnn.LSTMCell(config.hidden_size_char, state_is_tuple=True)
    char_cell_bwd = tf.contrib.rnn.LSTMCell(config.hidden_size_char, state_is_tuple=True)
    _, ((_, char_output_fwd), (_, char_output_bwd)) = tf.nn.bidirectional_dynamic_rnn(char_cell_fwd, char_cell_bwd, char_embeddings, sequence_length=word_lengths_flat, dtype=tf.float32)
    char_output = tf.concat([char_output_fwd, char_output_bwd], axis=-1)
    char_output = tf.reshape(char_output, shape=[s[0],s[1],2*config.hidden_size_char])
    word_embeddings = tf.concat([word_embeddings, char_output], axis=-1)
    
    word_embeddings = tf.nn.dropout(word_embeddings, dropout)
#################################################################################

with tf.variable_scope('bi_lstm'):
    
    word_cell_fwd = tf.contrib.rnn.LSTMCell(config.hidden_size_lstm)
    word_cell_bwd = tf.contrib.rnn.LSTMCell(config.hidden_size_lstm)
    (word_output_fwd, word_output_bwd),_ = tf.nn.bidirectional_dynamic_rnn(word_cell_fwd, word_cell_bwd, word_embeddings, sequence_length = sequence_lengths, dtype=tf.float32)
    word_output = tf.concat([word_output_fwd, word_output_bwd], axis=-1)

#################################################################################

with tf.variable_scope('proj'):
    W = tf.get_variable('W', dtype=tf.float32, shape=[2*config.hidden_size_lstm, config.ntags])
    b = tf.get_variable('b', dtype=tf.float32, shape=[config.ntags], initializer = tf.zeros_initializer())
    nsteps = tf.shape(word_output)[1]
    word_output = tf.reshape(word_output, shape=[-1, 2*config.hidden_size_lstm])
    pred = tf.matmul(word_output, W) + b
    logits = tf.reshape(pred, [-1, nsteps, config.ntags])

log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, labels, sequence_lengths)
loss = tf.reduce_mean(-log_likelihood)


#################################################################################

if config.lr_method.lower() == 'adam':
    optimizer = tf.train.AdamOptimizer(lr)

char_gradient = tf.gradients(loss, _char_embeddings)[0]
char_gradient = tf.stop_gradient(char_gradient)
#print tf.shape(char_gradient)
char_gradient = 0.01 * tf.math.l2_normalize(char_gradient, axis=1) * tf.math.sqrt(float(config.dim_char))
_new_char_embeddings = _char_embeddings

#gradient_update_op = optimizer.apply_gradients([(char_gradient, _new_char_embeddings)])
_new_char_embeddings = _new_char_embeddings + char_gradient
new_char_embeddings = tf.nn.embedding_lookup(_new_char_embeddings, char_ids, name='new_char_embeddings')


new_s = tf.shape(new_char_embeddings)
new_char_embeddings = tf.reshape(new_char_embeddings, shape=[new_s[0]*new_s[1], new_s[-2], config.dim_char])
new_word_lengths_flat = tf.reshape(word_lengths, shape=[new_s[0]*new_s[1]])
_, ((_, new_char_output_fwd), (_, new_char_output_bwd)) = tf.nn.bidirectional_dynamic_rnn(char_cell_fwd, char_cell_bwd, new_char_embeddings, sequence_length = new_word_lengths_flat, dtype=tf.float32)
new_char_output = tf.concat([new_char_output_fwd, new_char_output_bwd], axis=-1)
new_char_output = tf.reshape(new_char_output, shape=[new_s[0], new_s[1], 2*config.hidden_size_char])

new_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, word_ids, name="word_embeddings")
new_word_embeddings = tf.concat([new_word_embeddings, new_char_output], axis=-1)
new_word_embeddings = tf.nn.dropout(new_word_embeddings, dropout)

(new_word_output_fwd, new_word_output_bwd), _ = tf.nn.bidirectional_dynamic_rnn(word_cell_fwd, word_cell_bwd, new_word_embeddings, sequence_length = sequence_lengths, dtype=tf.float32)
new_word_output = tf.concat([new_word_output_fwd, new_word_output_bwd], axis=-1)

new_nsteps = tf.shape(new_word_output)[1]
new_word_output = tf.reshape(new_word_output, shape=[-1, 2*config.hidden_size_lstm])
new_pred = tf.matmul(new_word_output, W)+b
new_logits = tf.reshape(new_pred, shape=[-1, new_nsteps, config.ntags])

new_log_likelihood , new_trans_params = tf.contrib.crf.crf_log_likelihood(new_logits, labels, sequence_lengths,trans_params)
assert new_trans_params == trans_params
new_loss = tf.reduce_mean(-new_log_likelihood)

complete_loss = new_loss + loss
train_op = optimizer.minimize(complete_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


#################################################################################

dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                     config.processing_tag, config.max_iter)
dev = [i for i in dev]
train = CoNLLDataset(config.filename_train, config.processing_word,
                     config.processing_tag, config.max_iter)
train = [i for i in train]
test = CoNLLDataset(config.filename_test, config.processing_word,
                    config.processing_tag, config.max_iter)


idx_to_tag = {idx:tag for tag, idx in config.vocab_tags.items()}
tags = idx_to_tag.values()
run_size_default = 100
batch_size = config.batch_size
nbatches = (len(train) + batch_size -1) // batch_size
best_score = 0
no_improvement = 0


def div_or_zero(num, den):
    return num/den if den else 0.0


def get_feed_dict(dwords, dlabels=None, dlr=None, ddropout=None):
    data_char_ids , data_word_ids = zip(*dwords)
    data_word_ids , data_sequence_lengths = pad_sequences(data_word_ids, 0)
    data_char_ids , data_word_lengths = pad_sequences(data_char_ids, pad_tok=0, nlevels=2)

    feed = {
        word_ids:data_word_ids,
        sequence_lengths:data_sequence_lengths,
        char_ids: data_char_ids,
        word_lengths: data_word_lengths
        }
    if dlabels is not None:
        data_labels, _ = pad_sequences(dlabels, 0)
        feed[labels] = data_labels

    if dlr is not None:
        feed[lr] = dlr

    if ddropout is not None:
        feed[dropout] = ddropout

    return feed, data_sequence_lengths


def run_evaluate(_test, run_size = run_size_default):
    results = [{metric:{} for metric in ['f1','p','r']} for _ in range(run_size)]
    dev = _test
    for runs in range(run_size):
        if run_size == 1:
            replace_value = False
        else:
            replace_value = True
            
        curr_dev = [dev[i] for i in np.random.choice(len(dev), len(dev)/run_size, replace=replace_value)]
        stats = {tag:{'n_correct':0., 'n_pred':0., 'n_true':0.} for tag in tags}
        for i, (dev_words, dev_labels) in enumerate(minibatches(curr_dev, batch_size)):

            feed, dev_sequence_lengths = get_feed_dict(dev_words, dev_labels, config.lr, config.dropout)            
            viterbi_sequences = []
            dev_logits, dev_trans_params = sess.run([logits, trans_params], feed_dict = feed)
        
            for ite_logit, ite_sequence_length in zip(dev_logits, dev_sequence_lengths):
                ite_logit = ite_logit[:ite_sequence_length]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(ite_logit, dev_trans_params)
                viterbi_sequences += [viterbi_seq]

            for lab, lab_pred, l3 in zip(dev_labels, viterbi_sequences, dev_sequence_lengths):
                lab = lab[:l3]
                lab_pred = lab_pred[:l3]

                for l_true, l_pred in zip(lab, lab_pred):
                    if l_true == l_pred:
                        stats[idx_to_tag[l_true]]['n_correct'] += 1
                    stats[idx_to_tag[l_true]]['n_true'] += 1
                    stats[idx_to_tag[l_pred]]['n_pred'] += 1

        for curr_tag,curr_counts in stats.items():
            tag_p = div_or_zero(curr_counts['n_correct'], curr_counts['n_pred'])
            tag_r = div_or_zero(curr_counts['n_correct'], curr_counts['n_true'])
            results[runs]['p'][curr_tag] = tag_p
            results[runs]['r'][curr_tag] = tag_r
            results[runs]['f1'][curr_tag] = div_or_zero(2.0*tag_p*tag_r, tag_p+tag_r)
    print('')
    for t in tags:
        print ('               %s: %s' %(t, '  '.join(['%s=%02.3f/%02.3f' %(metric, np.mean([results[j][metric][t] for j in range(run_size)]), np.var([results[j][metric][t] for j in range(run_size)])) for metric in ['p', 'r', 'f1']])))

    average_results = { metric: np.mean([results[j][metric].values() for j in range(run_size)]) for metric in ['p', 'r', 'f1'] }
    print ('          average p {:02.3f} r {:02.3f} f1 {:02.3f}'.format(average_results['p'], average_results['r'], average_results['f1']))
    return average_results



def train_epoch():
    for epoch in range(config.nepochs):

        #################################################################################

        print ('Epoch {:02d} out of {:02d}'.format(epoch+1, config.nepochs))

        prog = Progbar(target=nbatches)

        np.random.shuffle(train)
    
        for i, (epoch_words, epoch_labels) in enumerate(minibatches(train[:], batch_size)):

            feed, _ = get_feed_dict(epoch_words, epoch_labels, config.lr, config.dropout)
        
            _,train_loss = sess.run([train_op, complete_loss], feed_dict = feed)
            prog.update(i+1, [("train loss", train_loss)])

            
        #################################################################################
        average_results = run_evaluate(dev, run_size=100)
        config.lr *= config.lr_decay
        if average_results['f1'] >= best_score:
            no_improvement = 0
            best_score = average_results['f1']
            if not os.path.exists(config.dir_model):
                os.makedirs(config.dir_model)
            saver.save(sess, config.dir_model)
            print ('New best score {:.02f}!'.format(average_results['f1']))
        else:
            no_improvement += 1
            if no_improvement >= config.nepoch_no_imprv:
                print ('Early stop at #{:02d} epoch without improvement'.format(epoch))
                break
    
    

#################################################################################

def test_epoch():
    print ('Test')
    saver.restore(sess, config.dir_model)
    run_evaluate(test, run_size=1)    



if __name__ == '__main__':
    if len(sys.argv) == 1:
        train_epoch()
        test_epoch()
    else if sys.argv[1] == 'test':
        test_epoch()
    else if sys.argv[1] == 'train':
        train_epoch()
