'''
Created on Dec 17, 2014

@author: tal
'''


from __future__ import unicode_literals, division
import os
import json
import time
import random
import numpy as np
# pylint:disable=no-member
from sklearn.cross_validation import train_test_split
import subprocess
from metrics.accuracy import conlleval
from utils.tools import shuffle, minibatch, contextwin
from rnn.elman import model as regular_elman
from core.infra.play_sound import beep_please
import atexit
atexit.register(beep_please)


SESSION_PATH = "/home/tal/workspace/sessions"


def session_to_text0(session):
    """Get the first text utterence in a session"""
    utterance = session["utterances"][0]
    return utterance["text"]


def get_char_to_idx(sentences):
    """Get an index for each char in the input"""
    # Read all the text (first utterance only for now
    chars = {c for sentence in sentences for c in sentence} # All the characters
    char2idc = {c: idx for idx, c in enumerate(list(chars))} # Turn the set into a list and enumerate
#     char2idc["_"] = len(char2idc)
    return char2idc

def get_session_files(number_of_files=None, random_seed=None):
    """Get shuffled session files"""
    _session_files = [os.path.join(SESSION_PATH, f) for f in os.listdir(SESSION_PATH) if f.endswith("json")]
    session_files = [f for f in _session_files if os.path.isfile(f)]
    print "len(session_files)", len(session_files)
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(session_files)
    if number_of_files:
        session_files = session_files[:number_of_files]
    return session_files


CONF = {
    'learning_rate': 0.1, # 0.0627142536696559,
    'verbose': True,
    'decay': True, # decay on the learning rate if improvement stops
    'win': 15, # number of characters in the context window
    'batch_size': 5, # number of back-propagation through time steps
    'nhidden': 200, # number of hidden units
    'seed': 345,
    'emb_dimension': 10, # dimension of character embedding
    'nepochs': 1000,
    'number_of_files': None, # All of them. Limit the scope To speed things up...
    'error_probability': 0.1
    }


def play_with_spelling():
    """Play with spelling mistakes"""
    print CONF
    np.random.seed(CONF['seed'])
    random.seed(CONF['seed'])
    print "Calculate output"
    session_files = get_session_files(number_of_files=CONF['number_of_files'], random_seed=CONF['seed'])
    sentences = get_sentences(session_files)
    print len(sentences)
    labels2idx = char2idx = get_char_to_idx(sentences)

    print "Prepare train, validation and test sets"
    train_valid_sentences, test_sentences = train_test_split(sentences, test_size=0.15, random_state=CONF['seed'])
    print len(train_valid_sentences), len(test_sentences)
    test_lex, test_y = create_tests(test_sentences, CONF['error_probability'], labels2idx, char2idx)
    train_valid_idxes = []
    train_valid_labels_idxes = []
    for error_probability in (CONF['error_probability'], CONF['error_probability'] / 10, CONF['error_probability'] / 100, 0):
        _train_valid_idxes, _train_valid_labels_idxes = create_tests(train_valid_sentences, error_probability, labels2idx, char2idx)
        train_valid_idxes.extend(_train_valid_idxes)
        train_valid_labels_idxes.extend(_train_valid_labels_idxes)
    train_lex, valid_lex, train_y, valid_y = train_test_split(train_valid_idxes, train_valid_labels_idxes, test_size=0.2, random_state=CONF['seed'])
    print len(train_lex), len(valid_lex), len(train_y), len(valid_y)

    print "Some more prep"
    idx2label = dict((k, v) for v, k in labels2idx.iteritems()) # Reverse the dictionary
    idx2word = dict((k, v) for v, k in char2idx.iteritems()) # Reverse the dictionary
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    windowed_test_lex = [np.asarray(contextwin(x, CONF['win'])).astype('int32') for x in test_lex]
    windowed_valid_lex = [np.asarray(contextwin(x, CONF['win'])).astype('int32') for x in valid_lex]

    words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]
    groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
    words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]
    vocsize = 1 + len(set(item for lex in (train_lex, valid_lex, test_lex) for sublist in lex for item in sublist))
    nclasses = 1 + len(set(item for _y in (train_y, test_y, valid_y) for sublist in _y for item in sublist))
    nsentences = len(train_lex)

    words_lex = []
    for i in xrange(nsentences):
        cwords = contextwin(train_lex[i], CONF['win'])
        words = [np.asarray(x).astype(np.int32) for x in minibatch(cwords, CONF['batch_size'])]
        words_lex.append(words)

    print "Some file os calls"
    folder = os.path.basename(__file__).split('.')[0] + "_3"
    if not os.path.exists(folder):
        os.mkdir(folder)
    print "Create a Neural Network"
    rnn = regular_elman(nh=CONF['nhidden'],
                        nc=nclasses,
                        ne=vocsize,
                        de=CONF['emb_dimension'],
                        cs=CONF['win'],)

    # train with early stopping on validation set
    best_f1 = -np.inf
    CONF['current_learning_rate'] = CONF['learning_rate']
    print "Start training"
    start_time = time.time()
    for epoch in xrange(CONF['nepochs']):
        # shuffle
        shuffle([words_lex, train_y], CONF['seed'])
        CONF['ce'] = epoch
        tic = time.time()
        print "tic"
        for i in xrange(nsentences):
            words = words_lex[i]
            labels = train_y[i]
            for word_batch, label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, CONF['current_learning_rate'])
                rnn.normalize()
            if CONF['verbose']:
                print '[learning] epoch %i >> %2.2f%%' % (epoch, (i + 1) * 100. / nsentences), 'completed in %.2f (sec) <<\r' % (time.time() - tic),

        # evaluation // back into the real world : idx -> words
        if CONF['verbose']:
            print "Classify test"
        predictions_test = [[idx2label[x] for x in rnn.classify(windowed_test_lex_item)]
                            for windowed_test_lex_item in windowed_test_lex]

        if CONF['verbose']:
            print "Classify validation"
        predictions_valid = [[idx2label[x] for x in rnn.classify(windowed_valid_lex_item)]
                             for windowed_valid_lex_item in windowed_valid_lex]
        # evaluation // compute the accuracy using conlleval.pl
        if CONF['verbose']:
            print "Evaluate test and validation"
        res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            print 'NEW BEST: epoch', epoch, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' ' * 20
            CONF['vf1'], CONF['vp'], CONF['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            CONF['tf1'], CONF['tp'], CONF['tr'] = res_test['f1'], res_test['p'], res_test['r']
            CONF['be'] = epoch
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print '        : epoch', epoch, 'valid F1', res_valid['f1'], '     test F1', res_test['f1'], ' ' * 20
#             rnn.load(folder)

        # learning rate decay if no improvement in 10 epochs
        if CONF['decay'] and abs(CONF['be'] - CONF['ce']) >= 10:
            CONF['current_learning_rate'] *= 0.5
        if CONF['current_learning_rate'] < 1e-5:
            break

    print 'BEST RESULT: epoch', CONF['be'], 'valid F1', best_f1, 'best test F1', CONF['tf1'], 'with the model', folder
    print "total time = {} seconds".format(time.time() - start_time)


def get_sentences(session_files):
    """Read the sentences from the sessions files"""
    sentences = []
    for session_file in session_files:
        session = json.loads(open(session_file, "rb").read())
        sentence = session_to_text0(session)
        if not sentence.strip():
            continue
        sentences.append(sentence)
    return sentences

def create_tests(sentences, error_probability, labels2idx, char2idx):
    """Create a batch of tests"""
    idxes = []
    labels_idxes = []
    for sentence in sentences:
        sentence_out = create_test(sentence, probability=error_probability)
        labels_idxes.append(np.fromiter((labels2idx[l] for l in sentence), dtype=np.uint32))
        idxes.append(np.fromiter((char2idx[char] for char in sentence_out), dtype=np.uint32))
    return idxes, labels_idxes


def create_test(sentence, probability):
    """Concatenate some words"""
    w_errors = ""
    for char in sentence:
        if random.random() > probability:
            w_errors += char
        else: # Inject a substitution error
            err_char = random.choice(list(set("abcdefghijklmnopqrstuvwxyz ") - set([char.lower()])))
            if char.isupper():
                err_char = err_char.upper()
            w_errors += err_char
    return w_errors


if __name__ == '__main__':
    play_with_spelling()
