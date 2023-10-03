import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from hmm_utils import HMM
from params import *

import os
import re
import pickle

from tqdm import tqdm
import random

#some other libraries
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from typing import List

from solvers.nn_RS.nn_RS import nn_RS
from attackers.decoding_attacker import dec_attacker

def pre_processing(text_column):
    # lowercase all text in the column
    text_column = text_column.str.lower()

    # replacing numbers with NUM token
    text_column = text_column.str.replace(r'\d+', 'NUM')

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    text_column = text_column.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return text_column

class NER():

    '''
    Perform NER Experiment

    '''
    
    def __init__(self, N_w=300):
        
        data = pd.read_csv(data_path, 
            encoding = "latin1")
        data = data.fillna(method="ffill")
        data = data.rename(columns={'Sentence #': 'sentence'})

        data_pre_precessed = pre_processing(data.Word)
        #creating new dataframe with preprocessed word as a column
        data_processed = data
        data_processed['Word'] = data_pre_precessed

        #removing the rows where word is empty
        data_processed = data_processed[(data_processed['Word'] != '') | (data_processed['Word'].isna())]

        # Most common words
        common_words = data_processed['Word'].value_counts().sort_values(ascending=False)[:N_w].index
        self.data_reduced = data_processed[data_processed['Word'].isin(common_words)]

        tags = list(set(self.data_reduced.POS.values))  # Unique POS tags in the dataset
        words = list(set(self.data_reduced.Word.values))  # Unique words in the dataset
        tags = np.sort(tags)

        # Convert words and tags into numbers
        self.word2id = {w: i for i, w in enumerate(words)}
        self.tag2id = {t: i for i, t in enumerate(tags)}
        self.id2tag = {i: t for i, t in enumerate(tags)}
        self.id2word = {}
        for key in self.word2id:
            self.id2word[self.word2id[key]] = key

        count_tags = dict(self.data_reduced.POS.value_counts())  # Total number of POS tags in the dataset
        # Now let's create the tags to words count
        count_tags_to_words = self.data_reduced.groupby(['POS']).apply(
            lambda grp: grp.groupby('Word')['POS'].count().to_dict()).to_dict()
        # We shall also collect the counts for the first tags in the sentence
        count_init_tags = dict(self.data_reduced.groupby('sentence').first().POS.value_counts())

        # Create a mapping that stores the frequency of transitions in tags to it's next tags
        count_tags_to_next_tags = np.zeros((len(tags), len(tags)), dtype=int)
        sentences = list(self.data_reduced.sentence)
        pos = list(self.data_reduced.POS)
        for i in tqdm(range(len(sentences)), position=0, leave=True):
            if (i > 0) and (sentences[i] == sentences[i - 1]):
                prevtagid = self.tag2id[pos[i - 1]]
                nexttagid = self.tag2id[pos[i]]
                count_tags_to_next_tags[prevtagid][nexttagid] += 1

        # Build HMM
        startprob = np.zeros((len(tags),))
        transmat = np.zeros((len(tags), len(tags)))
        emissionprob = np.zeros((len(tags), len(words)))
        num_sentences = sum(count_init_tags.values())
        sum_tags_to_next_tags = np.sum(count_tags_to_next_tags, axis=1)
        for tag, tagid in tqdm(self.tag2id.items(), position=0, leave=True):
            floatCountTag = float(count_tags.get(tag, 0))
            startprob[tagid] = count_init_tags.get(tag, 0) / num_sentences
            for word, wordid in self.word2id.items():
                emissionprob[tagid][wordid] = count_tags_to_words.get(tag, {}).get(word, 0) / floatCountTag
            for tag2, tagid2 in self.tag2id.items():
                transmat[tagid][tagid2] = count_tags_to_next_tags[tagid][tagid2] / sum_tags_to_next_tags[tagid]

        cutoff = 0.001
        startprob = startprob +  cutoff
        startprob = startprob/ np.sum(startprob)
        ##
        transmat =  transmat + cutoff
        transmat = transmat / np.sum(transmat, axis=1)
        ##
        emissionprob =  emissionprob + cutoff
        emissionprob = emissionprob / np.sum(emissionprob, axis=1).reshape(-1,1)

        self.hmm_n = HMM(len(tags), len(words))
        self.hmm_n.startprob_ = startprob
        self.hmm_n.transmat_ = transmat
        self.hmm_n.emissionprob_ = emissionprob

        self.n_obs = len(words)
        self.n_hidden = len(tags)

    def seq2word(self, X):
        l = []
        for i in range(len(X)):
            l.append(self.id2word[X[i]])
        return l

    def attack_seq(self, X, attack_tag):
        l = []
        for i in range(len(X)):
            l.append(attack_tag[X[i]])
        return l

    def init_attacker(self, X, y, attack_tag, w1=1.0, w2=5.0, k_value=10e6, rho=1.0):

        self.rho_probs = rho*np.ones(self.n_obs)
        self.k_value = k_value
        self.w1 = w1
        self.w2 = w2
        self.X = X
        self.y = y

        _, self.y_pred = self.hmm_n.nu(X[0])

        self.target_seq = target_seq = self.attack_seq(self.y_pred.astype(int), attack_tag)
        self.T = len(target_seq)

        self.att = dec_attacker(self.hmm_n.startprob_ , self.hmm_n.transmat_,
         self.hmm_n.emissionprob_, self.rho_probs,
         self.X.T, self.w1, self.w2, self.target_seq, self.k_value)

    def find_attack(self, seconds=10):
        find_sol = nn_RS(self.att, "SA", RS_iters=5000, mcts_iters=10, sa_iters=10, 
        eps=0.05, lr=0.005, verbose=True)
        self.sol, _ = find_sol.iterate(simulation_seconds=seconds)

    def get_info(self, X, y, fname, n_exp, attack_tag, w1=1.0, w2=5.0,
     k_value=10e6, rho=1.0, seconds=10):

        
        self.init_attacker(X, y, attack_tag, w1, w2, k_value, rho)

        self.find_attack(seconds)
        self.attack_obs = self.att.attack_X(np.ones_like(self.sol), self.sol)
        self.attack_obs = self.attack_obs.squeeze().astype(int)
        V, self.att_seq = self.hmm_n.nu(self.attack_obs)
    

        results = {'n_exp': n_exp, 'w1' : w1, 'w2' : w2, 'k_value' : k_value, 'rho' : rho,
                   'n_obs': self.n_obs, 'n_hidden': self.n_hidden, 'T': self.T,
                   'original_phrase':self.seq2word(X[0]),
                   'attacked_phrase':self.seq2word(self.attack_obs),
                   'hamming_d2target': np.mean(self.att_seq == self.target_seq),
                   'd2original': np.mean( self.attack_obs == X[0] ),
                   'exp_util' : self.att.expected_utility(self.sol),
                   'original_acc': np.mean(self.y_pred == self.y), 
                   'attacked_acc': np.mean(self.att_seq == self.y)
                   }

        with open(f'{fname}', 'wb') as fp:
            pickle.dump(results, fp)
            print('dictionary saved successfully to file')


def make_exp(n_exp, dirname, fname, w1, w2, seconds, sentence, attack):

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if 'ner' in globals():
        del ner

    ner = NER()

    k_value = 10e6
    rho = 1.0

    ###
    sentence = ner.data_reduced[ner.data_reduced.sentence == sentence]

    X = np.zeros( (1, sentence.shape[0] ) )
    for i in range(sentence.shape[0]):
        X[0, i] = ner.word2id[sentence.Word.values[i]]
        
    X =  X.astype(int)

    y = np.zeros(len(sentence.POS))
    for i in range(len(sentence.POS)):
        y[i] = ner.tag2id[sentence.POS.iloc[i]]
    ###

    ner.get_info(X, y, fname, n_exp, attack, w1, w2,
        k_value, rho, seconds)




if __name__ == "__main__":

    w1 = 1.0
    w2 = 5.0
    seconds = 9000

    sentence_num = 41785
    sentence = f'Sentence: {sentence_num}'
    # sentence = "Sentence: 41785"
    # sentence = "Sentence: 44516"
    n_exp = 10
    dirname = f'{results_path}ner_path_attraction/w1_{w1}_w2_{w2}_sentence_{sentence_num}_{seconds}/'

    for i in range(n_exp):
        fname = f'{dirname}exp{i}_w1_{w1}_w2_{w2}_sentence_{sentence_num}_{seconds}_seconds.pkl'
        make_exp(i, dirname, fname, w1, w2, seconds, sentence, attack1)
        print(f'Finished Experiment {i}')

    
