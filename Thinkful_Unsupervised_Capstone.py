#!/home/keithlaskay/anaconda3/envs/nlp
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:37:42 2018

@author: keithlaskay
"""
# %%
import pandas as pd
import numpy as np
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords
import spacy
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim

from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
df = pd.read_csv('./Data/yelp_rest.csv')
df.drop(['is_restaurants', 'name', 'review_id'], inplace=True, axis=1)

# %%
state_clean = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
               'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
               'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
               'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
               'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

# %%
df['in_USA'] = df.state.apply(lambda x: 1 if x in state_clean else 0)
filtered = df[df['in_USA'] == 1]

X, holdout = train_test_split(filtered[:200000], test_size=.25)

# %%
# update the stopwords list
nltk.download('stopwords')

# %%
# create the corpus, clean the text
X['clean_text'] = X['text'].str.lower()
X['clean_text'] = X['clean_text'].apply(lambda z: re.sub(r'\W', ' ', z))
X['clean_text'] = X['clean_text'].apply(lambda z: re.sub(r'^br$', ' ', z))
X['clean_text'] = X['clean_text'].apply(lambda z: re.sub(r'\s+br\s+', ' ', z))
X['clean_text'] = X['clean_text'].apply(lambda z: re.sub(r'\s+[a-z]\s+', ' ',
                                                         z))
X['clean_text'] = X['clean_text'].apply(lambda z: re.sub(r'^b\s+', '', z))
X['clean_text'] = X['clean_text'].apply(lambda z: re.sub(r'\s+', ' ', z))

# %%
stop = set(stopwords.words('english') + ['good', 'order', 'ordered', 'like',
                                         'get', 'go', 'place', 'one', 'great',
                                         'really', 'rostauraunt',
                                         'chicken', 'came', 'got', 'service',
                                         'took', 'food', 'meal', 'while',
                                         'would'])

# %%
X['tokens'] = X.clean_text.str.split()
X['tokens'] = X.tokens.apply(lambda x: [word.lower() for word in x if
                                        word not in stop])
X['token_string'] = X.tokens.str.join(sep=' ')

# %%
# Creating Bag of Words
vectorizer = CountVectorizer(max_features=3000, min_df=3,
                             stop_words=stop)
bag_of_words = vectorizer.fit_transform(X.token_string)

# %%
# Creating Tf-idf
tfidf = TfidfVectorizer(max_features=3000, min_df=3, max_df=0.6,
                        stop_words=stop)
vects = tfidf.fit_transform(X.token_string)

# %%
# Using Spacy
nlp = spacy.load('en')

text = []
for i in range(0, 1500):
    try:
        text.append(X.text[i])
    except:
        continue

doc = nlp(str(text))

# %%
# LDA topic modeling with gensim
tokens = [w for w in doc if w.text != '\n' and not w.is_stop and not
          w.is_punct and not w.like_num]
sents = [sent for sent in doc.sents]
# %%
dictionary = Dictionary(X.tokens)
corpus = [dictionary.doc2bow(sent) for sent in X.tokens]

# %%
ldamodel = LdaModel(corpus, num_topics=8, id2word=dictionary, passes=20)
ldamodel.print_topics(20, 5)

# %%
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

# %%
review_topic_gensim = []
for text in X.tokens:
        doc_bow = dictionary.doc2bow(text)
        topics = sorted(ldamodel[doc_bow], key=lambda x: x[1], reverse=True)
        review_topic_gensim.append(topics[0][0])

X['review_topic_gensim'] = review_topic_gensim
# %%
sklearn_lda = LatentDirichletAllocation(n_components=8, max_iter=20,
                                        learning_method='batch')

# %%
sklearn_lda.fit(vects)

sklearn_lda_array = sklearn_lda.transform(vects)
sklearn_lda_topic = [np.argmax(i) for i in sklearn_lda_array]
X['sklearn_lda_topic'] = sklearn_lda_topic

# %%
nmf = NMF(n_components=8, beta_loss='kullback-leibler', solver='mu',
          max_iter=1000, alpha=.1, l1_ratio=.5)
nmf.fit(vects)
nmf_arrays = nmf.transform(vects)
nmf_topic = [np.argmax(i) for i in nmf_arrays]
X['nmf_topic'] = nmf_topic

# %%
X['text_length'] = X.text.str.len()
X['token_length'] = X.tokens.apply(lambda x: len(x))
X['token_percent'] = X.token_length / X.text_length
X['num_categories'] = X.categories.str.split(';').apply(lambda x: len(x))
X['cool'] = X.cool.astype('int64')
X['state'] = X.state.astype('category')
X['stars_x'] = X.stars_x.astype('int64')
X['useful'] = X.useful.astype('int64')
X['funny'] = X.funny.astype('int64')

# %%
chart1 = sns.relplot(x='funny', y='useful', data=X, alpha=.5)

# %%
chart2 = sns.countplot(X.review_topic_gensim)

# %%
chart3 = sns.countplot(X.state)

# %%
chart4 = sns.distplot(X.token_percent)

# %%
chart5 = sns.countplot(X.num_categories)

# %%
chart6 = sns.countplot(X.token_length)

# %%
chart7 = sns.scatterplot(X.text_length, X.token_percent, hue=X.stars_x)
# %%
chart8 = sns.countplot(X.stars_x)

# %%
chart9 = sns.countplot(X.stars_x, hue=X.num_categories)

# %%
chart10 = sns.kdeplot(X.token_length)

# %%
chart11 = sns.heatmap(X.corr())

# %%
chart12 = sns.countplot(X.stars_x, hue=X.review_topic_gensim)

# %%
holdout['clean_text'] = holdout['text'].str.lower()
holdout['clean_text'] = holdout['clean_text'].apply(lambda z:
                                                    re.sub(r'\W', ' ', z))
holdout['clean_text'] = holdout['clean_text'].apply(lambda z:
                                                    re.sub(r'^br$', ' ', z))
holdout['clean_text'] = holdout['clean_text'].apply(lambda z:
                                                    re.sub(r'\s+br\s+',
                                                           ' ', z))
holdout['clean_text'] = holdout['clean_text'].apply(lambda z:
                                                    re.sub(r'\s+[a-z]\s+',
                                                           ' ', z))
holdout['clean_text'] = holdout['clean_text'].apply(lambda z:
                                                    re.sub(r'^b\s+', '', z))
holdout['clean_text'] = holdout['clean_text'].apply(lambda z:
                                                    re.sub(r'\s+', ' ', z))

holdout['tokens'] = holdout.clean_text.str.split()
holdout['tokens'] = holdout.tokens.apply(lambda x: [word.lower() for word in x
                                                    if word not in stop])
holdout['token_string'] = holdout.tokens.str.join(sep=' ')

holdout_bag_of_words = vectorizer.transform(holdout.token_string)
holdout_tfidf = tfidf.transform(holdout.token_string)

# %%
dictionary_holdout = Dictionary(holdout.tokens)
corpus_holdout = [dictionary_holdout.doc2bow(sent) for sent in holdout.tokens]

# %%
ldamodel_holdout = LdaModel(corpus_holdout, num_topics=8,
                            id2word=dictionary_holdout, passes=20)
ldamodel_holdout.print_topics(8, 5)

# %%
sklearn_lda_array_holdout = sklearn_lda.transform(holdout_bag_of_words)
sklearn_lda_topic_holdout = [np.argmax(i) for i in sklearn_lda_array_holdout]
holdout['sklearn_lda_topic'] = sklearn_lda_topic_holdout

# %%
vects_holdout = tfidf.transform(holdout.token_string)
nmf_arrays_holdout = nmf.transform(vects_holdout)
nmf_topic_holdout = [np.argmax(i) for i in nmf_arrays_holdout]
holdout['nmf_topic'] = nmf_topic_holdout

holdout['text_length'] = holdout.text.str.len()
holdout['token_length'] = holdout.tokens.apply(lambda x: len(x))
holdout['token_percent'] = holdout.token_length / X.text_length
holdout['num_categories'] = holdout.categories.str.split(';').apply(lambda x:
                                                                    len(x))
holdout['cool'] = holdout.cool.astype('int64')
holdout['state'] = holdout.state.astype('category')
holdout['stars_x'] = holdout.stars_x.astype('int64')
holdout['useful'] = holdout.useful.astype('int64')
holdout['funny'] = holdout.funny.astype('int64')

# %%
holdout1 = sns.relplot(x='funny', y='useful', data=X, alpha=.5)

# %%
holdout2 = sns.countplot(X.review_topic_gensim)

# %%
holdout3 = sns.countplot(X.state)

# %%
holdout4 = sns.distplot(X.token_percent)

# %%
holdout5 = sns.countplot(X.num_categories)

# %%
holdout6 = sns.countplot(X.token_length)

# %%
holdout7 = sns.scatterplot(X.text_length, X.token_percent, hue=X.stars_x)

# %%
holdout8 = sns.countplot(X.stars_x)

# %%
holdout9 = sns.countplot(X.stars_x, hue=X.num_categories)

# %%
haldout10 = sns.kdeplot(X.token_length)

# %%
holdout11 = sns.heatmap(X.corr())

# %%
holdout12 = sns.countplot(X.stars_x, hue=X.review_topic_gensim)
