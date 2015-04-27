'''
Created on Dec 17, 2014

@author: tal
'''


from __future__ import unicode_literals, division
from os import listdir
from os.path import isfile, join
import os
import json
import time
import sys
import random
import re
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import Counter
from core.factory import get_time_parser
from core.locales import Locales
from core.infra.tokenizer import tokenize
import subprocess
from metrics.accuracy import conlleval
from utils.tools import shuffle, minibatch, contextwin
from rnn.elman2vec import model as elman2vec
from rnn.elman import model as regular_elman
import theano
from gensim.models.word2vec import Word2Vec
from core.infra.play_sound import beep_please
import atexit
atexit.register(beep_please)


WORD2VEC_FILENAME = "/media/beka/evature/word2vec/GoogleNews-vectors-negative300.bin"
DTP = get_time_parser()
SESSION_PATH = "/home/beka/workspace/sessions"
LABELS2IDX = {"O":0, "B-when": 1, "I-when": 2, "B-duration": 3, "I-duration": 4}

def session_to_input(session):
    """Extract all the relevant info from a session"""
    result = []
    for utterance in session["utterances"]:
        context = utterance.get("context", "")
        result.append("<s{}> {} </s>".format(context, utterance["text"]))
        api_reply = utterance.get('api_reply')
        if not api_reply or not isinstance(api_reply, basestring):
            return None
        api_reply = json.loads(api_reply)
        if not api_reply or not isinstance(api_reply, dict):
            return None
        api_reply = api_reply.get('api_reply')
        if not api_reply or not isinstance(api_reply, dict):
            return None
        flow = api_reply.get("Flow")
        if not flow or not isinstance(flow, list):
            return None
        flow_action = flow[-1]
        flow_action_type = flow_action["Type"]
        flow_action_type += flow_action.get("QuestionCategory", "")
        flow_action_type = flow_action_type.replace(" ", "")
        result.append(flow_action_type)
    return " ".join(result)


def dtp_search(sentence, expected_meaning):
    """Date time parser search"""
    return DTP.search(sentence, enable_typos=False, locale=Locales.english_us,
                      expected_meaning=expected_meaning, from_speech=True, search_within_results=False)

def to_iob(token_list, dtp_search_res):
    """Change DTP search results to IOB"""
    whens = [range(res[1], res[1] + res[2] + 1) for res in dtp_search_res if res[0] == "When"]
    whens_start = {res[1] for res in dtp_search_res if res[0] == "When"}
    durations = [range(res[1], res[1] + res[2]) for res in dtp_search_res if res[0] == "Duration"]
    durations_start = {res[1] for res in dtp_search_res if res[0] == "Duration"}
    def token_to_iob(token):
        """Token to IOB"""
        result = "O"
        if token.start in whens_start:
            result = "B-when"
        elif token.start in durations_start:
            result = "B-duration"
        elif any(token.start in when for when in whens):
            result = "I-when"
        elif any(token.start in duration for duration in durations):
            result = "I-duration"
        return result
    return [token_to_iob(token) for token in token_list]

def session_to_text0(session):
    """Get the first text utterence in a session"""
    utterance = session["utterances"][0]
    return utterance["text"]

def get_words2idx(session_files):
    """Get the indexes of words"""
    tokens = Counter()
    for session_file in session_files:
        session = json.loads(open(session_file, "rb").read())
        sentence = session_to_text0(session)
        token_list = tokenize(sentence)
        for token in token_list:
            tokens[token.lower()] += 1
            tokens[re.sub(r"\d", "DIGIT", token.lower())] += 1
    tokens_clean = dict((k, v) for k, v in tokens.iteritems()) # if v > 1) - words that happen only once -> unknown
    words2idx = dict((k, i) for i, (k, v) in enumerate(tokens_clean.iteritems()))
    words2idx["<UNK>"] = max(index for index in words2idx.itervalues()) + 1
    return words2idx

def get_char_to_idx(session_files):
    """Get an index for each char"""
    chars = {c for session_file in session_files for c in session_to_text0(json.loads(open(session_file, "rb").read()))}
    char2idc = {c: idx for idx, c in enumerate(list(chars))}
    char2idc["_"] = len(char2idc)
    return char2idc

def get_session_files(number_of_files=None, random_seed=None):
    """Get shuffled session files"""
    _session_files = [join(SESSION_PATH, f) for f in listdir(SESSION_PATH) if f.endswith("json")]
    session_files = [f for f in _session_files if isfile(f)]
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(session_files)
    if number_of_files:
        session_files = session_files[:number_of_files]
    return session_files

def prepare_data():
    """Prepare the data"""
    conf = {'fold': 3, # 5 folds 0,1,2,3,4
            'lr': 0.0627142536696559,
            'verbose': True,
            'decay': True, # decay on the learning rate if improvement stops
            'win': 7, # number of words in the context window
            'bs': 9, # number of back-propagation through time steps
            'nhidden': 100, # number of hidden units
            'seed': 345,
            'emb_dimension': 300, # dimension of word embedding
            'nepochs': 50}
    np.random.seed(conf['seed'])
    random.seed(conf['seed'])
    session_files = get_session_files(number_of_files=None, random_seed=conf['seed']) # Limit the scope To speed things up...
    sentences = []
    idxes = []
    labels = []
    labels_idxes = []
    print "Calculate words2idx"
    words2idx = get_words2idx(session_files)
    unknown = words2idx["<UNK>"]
    print "Calculate output"
    for session_file in session_files:
        session = json.loads(open(session_file, "rb").read())
        sentence = session_to_text0(session)
        if not sentence.strip():
            continue
        sentences.append(sentence)
        token_list = tokenize(sentence.lower())
        dtp_search_res = dtp_search(sentence, None)
        iobes = to_iob(token_list, dtp_search_res)
        labels.append(iobes)
        labels_idxes.append(np.fromiter((LABELS2IDX[iob] for iob in iobes), dtype=np.int32))
#         token_list = [re.sub(r"\d", "DIGIT", token) for token in token_list]
        idxes.append(np.fromiter((words2idx.get(token, unknown) for token in token_list), dtype=np.int32))



    print "Prepare train, validation and test sets"
    train_valid_lex, test_lex, train_valid_y, test_y = train_test_split(idxes, labels_idxes, test_size=0.15, random_state=42)
    train_lex, valid_lex, train_y, valid_y = train_test_split(train_valid_lex, train_valid_y, test_size=0.2, random_state=42)

    idx2label = dict((k, v) for v, k in LABELS2IDX.iteritems()) # Reverse the dictionary
    idx2word = dict((k, v) for v, k in words2idx.iteritems()) # Reverse the dictionary

    vocsize = len(idx2word)

    nclasses = len({label for labels in labels_idxes for label in labels})
    # nclasses = len(set(reduce(lambda x, y: list(x) + list(y), train_y + test_y + valid_y)))

    nsentences = len(train_lex)
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder):
        os.mkdir(folder)

    print "Loading Word2Vec"
    word2vec = Word2Vec.load_word2vec_format(WORD2VEC_FILENAME, binary=True) # C binary format

    print "Calculate word embeddings"
    embeddings = 0.2 * np.random.uniform(-1.0, 1.0, (vocsize + 1, conf['emb_dimension'])).astype(theano.config.floatX) # add one for PADDING at the end @UndefinedVariable
    for idx, word in idx2word.iteritems():
        try:
            embedding = word2vec[word]
        except KeyError:
            try:
                embedding = word2vec[word.capitalize()]
            except KeyError:
                embedding = embeddings[idx] # Keep it random
        embeddings[idx] = embedding

    del word2vec # It is huge

    print "Create a Neural Network"
    rnn = elman2vec(nh=conf['nhidden'],
                nc=nclasses,
                ne=vocsize,
                de=conf['emb_dimension'],
                cs=conf['win'],
                embeddings=embeddings)

    # train with early stopping on validation set
    best_f1 = -np.inf
    conf['clr'] = conf['lr']
    print "Start training"
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
#                 rnn.normalize()
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

    print 'BEST RESULT: epoch', epoch, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], 'with the model', folder

def play_with_splitting_sentences():
    """Play with splitting sentences"""
    conf = { # 'fold': 3, # 5 folds 0,1,2,3,4
        'lr': 0.0627142536696559,
        'verbose': False,
        'decay': True, # decay on the learning rate if improvement stops
        'win': 15, # number of characters in the context window
        'bs': 5, # number of back-propagation through time steps
        'nhidden': 100, # number of hidden units
        'seed': 345,
        'emb_dimension': 30, # dimension of character embedding
        'nepochs': 10}
    number_of_files = 50000
    np.random.seed(conf['seed'])
    random.seed(conf['seed'])
    print "Calculate output"
    session_files = get_session_files(number_of_files=number_of_files, random_seed=conf['seed']) # Limit the scope To speed things up...
    labels2idx = {"O": 0, "X": 1}
    sentences = []
    idxes = []
    labels_idxes = []
    labels = []
    char2idx = get_char_to_idx(session_files)
    for session_file in session_files:
        session = json.loads(open(session_file, "rb").read())
        sentence = session_to_text0(session)
        if not sentence.strip():
            continue
        sentence_out, label = create_test(sentence, probability=0.2)
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
    nclasses = 2  #len(set(reduce(lambda x, y: list(x) + list(y), train_y + test_y + valid_y)))
    nsentences = len(train_lex)
    print "Some file os calls"
    folder = os.path.basename(__file__).split('.')[0] + "_3"
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
    words = sentence.split()
    if words:
        output, labels = words[0], "X" + "O" * (len(words[0]) - 1)
        for word in words[1:]:
            if random.random() > probability:
                output += "_" # Replace space with underline
                labels += "O"
            labels += "X" + "O" * (len(word) - 1)
            output += word
    else:
        output = labels = ""
    return output, labels


if __name__ == '__main__':
    prepare_data()
    # play_with_splitting_sentences()
