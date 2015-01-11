'''
Created on Dec 17, 2014

@author: tal
'''


from __future__ import unicode_literals, division
import os
import json
import time
import sys
import random
import re
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import Counter
import subprocess
from metrics.accuracy import conlleval
from utils.tools import shuffle, minibatch, contextwin
from rnn.elman import model as regular_elman
import theano
from core.infra.play_sound import beep_please
import atexit
from cherrypy.test.modpy import conf_cpmodpy
atexit.register(beep_please)


SESSION_PATH = "/home/tal/workspace/sessions"
# LABELS2IDX = {"O":0, "B-when": 1, "I-when": 2, "B-duration": 3, "I-duration": 4}


def session_to_text0(session):
    """Get the first text utterence in a session"""
    utterance = session["utterances"][0]
    return utterance["text"]

# def get_words2idx(session_files):
#     """Get the indexes of words"""
#     tokens = Counter()
#     for session_file in session_files:
#         session = json.loads(open(session_file, "rb").read())
#         sentence = session_to_text0(session)
#         token_list = tokenize(sentence)
#         for token in token_list:
#             tokens[token.lower()] += 1
#             tokens[re.sub(r"\d", "DIGIT", token.lower())] += 1
#     tokens_clean = dict((k, v) for k, v in tokens.iteritems()) # if v > 1) - words that happen only once -> unknown
#     words2idx = dict((k, i) for i, (k, v) in enumerate(tokens_clean.iteritems()))
#     words2idx["<UNK>"] = max(index for index in words2idx.itervalues()) + 1
#     return words2idx

def get_char_to_idx(session_files):
    """Get an index for each char in the input"""
    # Read all the text (first utterance only for now
    chars = {c for session_file in session_files for c in session_to_text0(json.loads(open(session_file, "rb").read()))}
    char2idc = {c: idx for idx, c in enumerate(list(chars))} # Turn the set into a list and enumerate
#     char2idc["_"] = len(char2idc)
    return char2idc

def get_session_files(number_of_files=None, random_seed=None):
    """Get shuffled session files"""
    _session_files = [os.path.join(SESSION_PATH, f) for f in os.listdir(SESSION_PATH) if f.endswith("json")]
    session_files = [f for f in _session_files if os.path.isfile(f)]
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(session_files)
    if number_of_files:
        session_files = session_files[:number_of_files]
    return session_files

def play_with_spelling():
    """Play with spelling mistakes"""
    conf = {
        'lr': 0.0627142536696559,
        'verbose': False,
        'decay': True, # decay on the learning rate if improvement stops
        'win': 15, # number of characters in the context window
        'bs': 5, # number of back-propagation through time steps
        'nhidden': 100, # number of hidden units
        'seed': 345,
        'emb_dimension': 50, # dimension of character embedding
        'nepochs': 10}
    number_of_files = 50000
    print conf
    print "number_of_files=", number_of_files
    np.random.seed(conf['seed'])
    random.seed(conf['seed'])
    print "Calculate output"
    session_files = get_session_files(number_of_files=number_of_files, random_seed=conf['seed']) # Limit the scope To speed things up...
#     labels2idx = {"O": 0, "X": 1}
    sentences = []
    idxes = []
    labels_idxes = []
    labels = []
    labels2idx = char2idx = get_char_to_idx(session_files) 
    for session_file in session_files:
        session = json.loads(open(session_file, "rb").read())
        sentence = session_to_text0(session)
        if not sentence.strip():
            continue
        sentence_out, label = create_test(sentence, probability=0.1)
        sentences.append(sentence_out)
        labels.append(label)
        labels_idxes.append(np.fromiter((labels2idx[l] for l in label), dtype=np.uint32))
        idxes.append(np.fromiter((char2idx[char] for char in sentence_out), dtype=np.uint32))

    print "Prepare train, validation and test sets"
    train_valid_lex, test_lex, train_valid_y, test_y = train_test_split(idxes, labels_idxes, test_size=0.15, random_state=42)
    train_lex, valid_lex, train_y, valid_y = train_test_split(train_valid_lex, train_valid_y, test_size=0.2, random_state=42)
    print "Some more prep"
    idx2label = dict((k, v) for v, k in labels2idx.iteritems()) # Reverse the dictionary
    idx2word = dict((k, v) for v, k in char2idx.iteritems()) # Reverse the dictionary

#     vocsize = 1 + len(set(reduce(\
#                                  lambda x, y: list(x)+list(y),\
#                                  train_lex+valid_lex+test_lex)))
    vocsize = 1 + len(set(item for lex in (train_lex, valid_lex, test_lex) for sublist in lex for item in sublist))
    nclasses = 1 + len(set(item for _y in (train_y, test_y, valid_y) for sublist in _y for item in sublist))
    nsentences = len(train_lex)
    print "Some file os calls"
    folder = os.path.basename(__file__).split('.')[0] + "_1"
    if not os.path.exists(folder):
        os.mkdir(folder)
    print "Create a Neural Network"
    rnn = regular_elman(nh=conf['nhidden'],
                        nc=nclasses,
                        ne=vocsize,
                        de=conf['emb_dimension'],
                        cs=conf['win'],)

    # train with early stopping on validation set
    best_f1 = -np.inf
    conf['clr'] = conf['lr']
    print "Start training"
    start_time = time.time()
    for epoch in xrange(conf['nepochs']):
        # shuffle
        shuffle([train_lex, train_y], conf['seed'])
        conf['ce'] = epoch
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(train_lex[i], conf['win'])
            words = [np.asarray(x).astype(np.int32) for x in minibatch(cwords, conf['bs'])]
            labels = train_y[i]
            for word_batch , label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, conf['clr'])
                rnn.normalize()
            if conf['verbose']:
                print '[learning] epoch %i >> %2.2f%%' % (epoch, (i + 1) * 100. / nsentences), 'completed in %.2f (sec) <<\r' % (time.time() - tic),
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [ map(lambda x: idx2label[x], \
                         rnn.classify(np.asarray(contextwin(x, conf['win'])).astype('int32')))\
                         for x in test_lex ]
        groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [ map(lambda x: idx2label[x], \
                             rnn.classify(np.asarray(contextwin(x, conf['win'])).astype('int32')))\
                             for x in valid_lex ]
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            print 'NEW BEST: epoch', epoch, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' ' * 20
            conf['vf1'], conf['vp'], conf['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            conf['tf1'], conf['tp'], conf['tr'] = res_test['f1'], res_test['p'], res_test['r']
            conf['be'] = epoch
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print '        : epoch', epoch, 'valid F1', res_valid['f1'], '     test F1', res_test['f1'], ' ' * 20

        # learning rate decay if no improvement in 10 epochs
        if conf['decay'] and abs(conf['be'] - conf['ce']) >= 10:
            conf['clr'] *= 0.5
        if conf['clr'] < 1e-5:
            break

    print 'BEST RESULT: epoch', conf['be'], 'valid F1', best_f1, 'best test F1', conf['tf1'], 'with the model', folder
    print "total time = {} seconds".format(time.time() - start_time)

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
    return w_errors, sentence


if __name__ == '__main__':
    play_with_spelling()
