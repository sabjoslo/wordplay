from __future__ import division

import os
import numpy as np
from wordplay.core import Vector,sentences,token_distributions

def _almost_equal(x,y):
    return abs(x-y)<1e-10

def test_vector():
    index=['a','b','c','d']
    data0=[.25,.25,.25,.25]
    data1=[.5,.25,.125,.125]
    v0=Vector()
    v0.build(index=index, data=data0, columns='est_prob')
    v1=Vector()
    v1.build(index=index, data=data1, columns='est_prob')

    # Test format of Vector().df
    assert v0.df.columns==v1.df.columns
    assert len(v0.df.columns)==1
    column=v0.df.columns[0]
    data_=zip(data0,data1)
    for ix in range(len(index)):
        assert v0.df.loc[index[ix]][column]==data_[ix][0]
        assert v1.df.loc[index[ix]][column]==data_[ix][1]

    # Test partial_kl
    assert _almost_equal(v0.partial_kl(v1,'c'),0.10375937481)

    # Test entropy
    assert _almost_equal(v0.entropy(),2)
    assert _almost_equal(v1.entropy(),1.75)

    # Test KL-divergence
    assert _almost_equal(v0.kl_divergence(v1),.25)
    assert _almost_equal(v1.kl_divergence(v0),.25)

    # Test mixture
    truth=np.array([.375,.25,.1875,.1875])
    for nparr in ( v0.mixture(v1).df['est_prob'].values,
                    v1.mixture(v0).df['est_prob'].values ):
        assert truth.shape==nparr.shape
        assert all([ truth[i]==nparr[i] for i in range(len(truth)) ])

    # Test Jensen-Shannon distance
    assert _almost_equal(v0.jensen_shannon_distance(v1),0.06127812445)
    assert _almost_equal(v1.jensen_shannon_distance(v0),0.06127812445)

def test_tokenize_():
    sentence='<p>This is the user\'s test sentence.</p>'
    tokens=sentences().tokenize_(sentence)
    assert tokens==['user','test','sentence']

    sentence='<a>test</a> <code>this is a test</code> <a>that is a test</a> also a test <blockquote>this is a test</blockquote>'
    tokens=sentences().tokenize_(sentence, html_elements_to_exclude=['a'])
    assert tokens==['test','also','test','test']
    tokens=sentences().tokenize_(sentence, html_elements_to_exclude=['a','code'])
    assert tokens==['also','test','test']
    tokens=sentences().tokenize_(sentence, html_elements_to_exclude=['a','code','blockquote'])
    assert tokens==['also','test']

def test_token_distributions():
    testTD=token_distributions(files=os.path.dirname(os.path.realpath(__file__))+'/.fake_data')
    # Test build_corpus
    testTD.build_corpus(column='Message')
    assert testTD.corpus==[ ['hello','world'],['talk'],['try','friendly'],
                            ['thank','try'],['hello','world']
                          ]

    # Test freq_dist
    testTD.calculate_freq_dist()
    assert set(testTD.freq_dist.keys())==set([ t for tokens in testTD.corpus
                                               for t in tokens ]) 
    assert testTD.freq_dist.values().count(1)==3
    assert testTD.freq_dist.values().count(2)==3
    assert testTD.freq_dist['hello']==2
    assert testTD.freq_dist['world']==2
    assert testTD.freq_dist['try']==2

    # Test prob_dist
    testTD.estimate_prob_dist()
    assert testTD.prob_dist.prob('hello')==2/9
