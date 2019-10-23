#!/usr/bin/env python
# coding: utf-8

import json
import pickle

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_article(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def read_articles(data_dir):
    articles = {}
    for f in listdir(data_dir):
        path = join(data_dir, f)
        if isfile(path):
            articles[f] = read_article(path)
    return articles


def read_metadata(path):
    with open(path, 'r') as f:
        metadata = json.load(f)
    return metadata


def extact_sentences(metadata, articles):
    top_num = 5
    cv = CountVectorizer()
    transformer = TfidfTransformer()

    for i, m in enumerate(metadata):
        # get all related articles
        related_articles = []
        for art_id in m['related_articles']:
            related_articles.append(articles['{}.txt'.format(art_id)])
            
        # convert all related articles to sentences
        sentences = [m['claim']]
        for a in related_articles:
            sentences.extend(sent_tokenize(a))
        
        # convert all sentences and claim to a tf-idf representation
        tfidf = transformer.fit_transform(cv.fit_transform(sentences))
        # compute cosine similarity between the claim and all sentences
        similarity = cosine_similarity(tfidf[0], tfidf[1:])
        
        # get five most related sentences
        if len(similarity[0]) >= top_num:
            ind = similarity[0].argsort()[-top_num:][::-1]
        else:
            ind = similarity[0].argsort()[::-1]
        
        m['related_sentences'] = []
        for index in ind:
            m['related_sentences'].append(sentences[1:][index])


def make_data(articles_dir, metadata_path, output_dir, type='dev'):
    print("Processing claims...")
    metadata = read_metadata(metadata_path)
    print("Total claims: {}".format(len(metadata)))

    print("Processing articles...")
    articles = read_articles(articles_dir)
    print('Total articles: {}'.format(len(articles)))

    extact_sentences(metadata, articles)

    with open(join(output_dir, '{}.pickle'.format(type)), 'wb') as f:
        pickle.dump(metadata, f)
    print("Processed data saved at: {}".format(join(output_dir, '{}.pickle'.format(type))))
