from __future__ import division

import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from nltk import FreqDist, MLEProbDist
import spacy
tokenizer=spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
from wordplay.utils import *

class Vector():
    def __init__(self):
        pass

    def load(self,fname):
        self.df=pd.read_pickle(fname)

    def build(self,index,data,columns=None):
        if not hasattr(columns,'__iter__'):
            columns=[columns]
        self.df=pd.DataFrame(data, index=index, columns=columns)

    def _indices_match(self,df_):
        assert len(self.df.index)==len(df_.index)
        assert all([ i in df_.index for i in self.df.index ])
        return True

    def entropy(self,column=None):
        if isinstance(column,type(None)):
            assert len(self.df.columns)==1
            column=self.df.columns[0]
        ent=0
        for i in self.df[column]:
            if i==0:
                continue
            ent+=i*math.log(i,2)
        return -1*ent

    # Computes the KL-divergence of this instance of a Vector w.r.t. another
    # vector
    def kl_divergence(self,q,column=None):  
        assert hasattr(q,'df')
        assert self._indices_match(q.df)
        if isinstance(column,type(None)):
            assert len(self.df.columns)==len(q.df.columns)==1
            assert self.df.columns==q.df.columns
            column=self.df.columns[0]
        kld=0
        for i in self.df.index:
            p_i=self.df.loc[i][column]
            q_i=q.df.loc[i][column]
            kld+=q_i*math.log((q_i/p_i),2)
        return kld

    def mixture(self,q,column=None):
        assert hasattr(q,'df')
        assert self._indices_match(q.df)
        if isinstance(column,type(None)):
            assert len(self.df.columns)==len(q.df.columns)==1
            assert self.df.columns==q.df.columns
            column=self.df.columns[0]
        mvec=Vector()
        mvec.df=.5*(self.df+q.df)
        return mvec

    # Calculate the JSD between this Vector and a Vector q
    def jensen_shannon_distance(self,q,column=None):
        assert hasattr(q,'df')
        assert self._indices_match(q.df)
        if isinstance(column,type(None)):
            assert len(self.df.columns)==len(q.df.columns)==1
            assert self.df.columns==q.df.columns
            column=self.df.columns[0]
        m=self.mixture(q)
        Hm=m.entropy()
        Hp=self.entropy()
        Hq=q.entropy()
        return Hm-.5*(Hp+Hq)

    def partial_kl(self,q,which,column=None):  
        assert hasattr(q,'df')
        if isinstance(column,type(None)):
            assert len(self.df.columns)==len(q.df.columns)==1
            assert self.df.columns==q.df.columns
            column=self.df.columns[0]
        p_i=self.df.loc[which][column]
        q_i=q.df.loc[which][column]
        return p_i*math.log(2*p_i/(p_i+q_i),2)

    def plot(self,column=None):
        if isinstance(column,type(None)):
            assert len(self.df.columns)==1
            column=self.df.columns[0]
        lab=list(self.df.index)
        y=[]
        for tok in lab:
            y.append(self.df.loc[tok][column])
        x=range(len(y))
        PLT=plt.figure()
        plt.plot(x,y)
        return PLT

    def save(self,fname):
        self.df.to_pickle(fname)

class sentences():
    """A handler to generate tokens from the words in a tsv file.
    """

    def __init__(self, phrase_model=None):
        startLog(log=False)

        # Initialize phrase-detector
        self.phrases=self.get_phraser(fn=phrase_model)
        self.phraser=Phraser(self.phrases)

    # Declare and initialize handler for tsv file to be read from.
    def open_file(self,fn,column=None):
        self.fn=open(fn,'r')
        self.headers=self.fn.readline().strip('\n').split('\t')
        # Reset file handler's line count
        self.fn.seek(0)
        self.column=column

    def close_file(self):
        self.fn.close()

    # Generates "sentences" from the tsv file of the form sentence =
    # [['word0','word1','word2'],['word3','word4','word5']]. The class
    # attribute column specifies the column of the file to be read from.
    # If training is set to True, sentences will be passed to the class's
    # phrase-detection model to train it to generate phrasegrams.
    def tokens_from_tsv(self):
        self.fn.seek(0)
        
        # Discard header row
        self.fn.readline()
        assert self.column in self.headers
        col_ix=self.headers.index(self.column)
        line=self.fn.readline().split('\t')
        while any([ l.strip() for l in line ]):
            yield self.tokenize_(to_ascii(line[col_ix].strip()))
            line=self.fn.readline().split('\t')

    def tokens_from_plain_text(self,training=False):
        self.fn.seek(0)
        line=self.fn.readline()
        while len(line)>0:
            if line.strip():
                yield self.tokenize_(to_ascii(line.strip()),
                                    training=training)
            line=self.fn.readline()

    def __iter__(self):
        return self.tokens_from_plain_text()

    # Generate and remove extraneous punctuation from tokens. If training =
    # True, tokens are used to train a phrase-detection model.
    def tokenize_(self, sentence, training=False, lemmatize=True,
                  html_elements_to_exclude=[]):
        # Much of this taken from 
        # https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%e2%80%8bin-python/

        # Remove HTML formatting and hyperlinks
        soup=BeautifulSoup(sentence, "html5lib")
        sentence=soup.get_text()
        for element in html_elements_to_exclude:
            for el in soup.find_all(element):
                sentence=sentence.replace(el.get_text(),'')

        # Return the lemma of each token. Exclude pronouns, stopwords and
        # punctuation.
        if lemmatize:
            words=tokenizer(sentence)
            lemmas=[ word.lemma_ for word in words 
                     if word.lemma_ != '-PRON-' ]
            lemmas=[ re.sub('[%s]'%re.escape(string.punctuation),'',lemma)
                     for lemma in lemmas ]
        else:
            sentence=re.sub('[%s]'%re.escape(string.punctuation),' ',sentence)
            lemmas=[ word.lower() for word in 
                     sentence.replace('\n',' ').split() ]

        tokens=[ token for token in lemmas if ( token not in stop_words and
                 token.strip() )]

        if training:
            # Add new words to phrase-detector
            self.phrases.add_vocab([tokens])

        return self.phraser[tokens]

    # Look for a phrase detection model saved at path fn. If not found or fn
    # is None, return an untrained phrase detection model.
    def get_phraser(self,fn=None):
        if isinstance(fn,type(None)):
            return Phrases()
        if os.path.exists(fn):
            logging.info('Found {}. Loading phrase detection model.\
                         '.format(fn))
            return Phrases().load(fn)
        else:
            logging.info('Can\'t find {}. Loading empty phrase detection model.\
                         '.format(fn))
            return Phrases()

    def train_phrase_detector(self,fn=None):
        for sentence in self.tokens_from_plain_text(training=True):
            pass

    def save_phrase_model(self,fn):
        self.phrases.save(fn)

class token_distributions():
    def __init__(self,corpus=None,files=None,phrase_model=None):
        assert ( not isinstance(corpus,type(None)) or 
                 not isinstance(files,type(None)) )
        if not isinstance(corpus,type(None)):
            self.load_corpus(corpus)
        if isinstance(files,basestring):
            self.files=[files]
        else:
            self.files=files
        self.set_phrase_model(phrase_model)

    def set_phrase_model(self,phrase_model=None):
        self.filehandler=sentences(phrase_model=phrase_model)
       
    def build_corpus(self,tsv=True,column='TEXT'):
        self.corpus=[]
        if tsv:
            for f in self.files:
                self.filehandler.open_file(f,column)
                for sentence in self.filehandler.tokens_from_tsv():
                    self.corpus.append(sentence)
        else:
            for f in self.files:
                self.filehandler.open_file(f)
                for sentence in self.filehandler.tokens_from_plain_text():
                    self.corpus.append(sentence)
        self.filehandler.close_file()

    # Save this instance's corpus to disk so it can be quickly accessed
    # later. The file is saved in plain text, with words separated by a 
    # space and sentences separated by a new line. It is by default saved in
    # VOCAB_DIR and is included in the package's recognized vocabulary.
    def save_corpus(self,fn):
        os.system('mkdir -p {}'.format(os.path.dirname(fn)))
        fh=open(fn,'w')
        logging.info('Writing to {}.'.format(fn))
        fh.write('\n'.join([ ' '.join(s) for s in self.corpus ]))
        fh.close()

    # Load a saved corpus. Set from_vocab to False if fn is not stored in
    # VOCAB_DIR (default).
    def load_corpus(self,fn):
        fh=open(fn,'r')
        logging.info('Reading from {}.'.format(fn))
        self.corpus=[ s.split() for s in fh.read().split('\n') ]
        fh.close()

    def calculate_freq_dist(self):
        self.freq_dist=FreqDist()
        for token in [ t for tokens in self.corpus for t in tokens ]:
            self.freq_dist[token]+=1

    def estimate_prob_dist(self):
        self.prob_dist=MLEProbDist(self.freq_dist)

    def load_vector(self,fname):
        self.vector=Vector()
        self.vector.load(fname)

    def set_vector(self,vocab=None):
        if isinstance(vocab,type(None)):
            vocab=self.prob_dist.samples()
        assert hasattr(vocab, 'index')
        data_=map(lambda x:self.prob_dist.prob(x) if x in
                  self.prob_dist.samples() else 0,
                  vocab)
        self.vector=Vector()
        self.vector.build(index=vocab, data=data_, columns='est_prob')

    def word2vec_(self,fname=None):
        model=Word2Vec(self.corpus)
        if isinstance(fname,basestring):
            model.save(fname)
        return model

if __name__=='__main__':
    pass
