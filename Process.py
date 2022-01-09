import builtins
import math
from inverted_index_gcp import InvertedIndex
# import search_frontend as se
# !pip install pyspark
# !pip install graphframes

# import pyspark
# import sys
# from collections import Counter, OrderedDict, defaultdict
# import itertools
# from itertools import islice, count, groupby
# import pandas as pd
# import os
# import re
# from operator import itemgetter
# import nltk
# from nltk.stem.porter import *
from nltk.corpus import stopwords
# from time import time
# from pathlib import Path
# import pickle
# import pandas as pd
# from google.cloud import storage
# import operator
# import hashlib

# from pyspark.sql import *
# from pyspark.sql.functions import *
# from pyspark import SparkContext, SparkConf, SparkFiles
# from pyspark.sql import SQLContext
# from graphframes import *
from google.cloud import storage
# from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import nltk
import pickle
import numpy as np

import pickle

# TODO: Check about 3th tab in colab

# !pip install -q pyspark
# !pip install -U -q PyDrive
# !apt-get update -qq
# !apt install openjdk-8-jdk-headless -qq
import ast

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    if isinstance(text, list):
        text = ' '.join(text)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


class Process:

    def __init__(self):
        bucket_name = '313371858'
        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'postings_gcp/index.pkl':
                with b.open("rb") as f:
                    self.index_body = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'index_title.pkl':
                with b.open("rb") as f:
                    self.index_title = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'postings_gcp/index_anchor.pkl':
                with b.open("rb") as f:
                    self.index_anchor = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'DL.pkl':
                with b.open("rb") as f:
                    dl_body = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'dl_title.pickle':
                with b.open("rb") as f:
                    dl_title = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'dl_anchor.pickle':
                with b.open("rb") as f:
                    dl_anchor = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'tfidf_dict.pkl':
                with b.open("rb") as f:
                    tfidf_dict_body = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'tfidf_title_dict.pickle':
                with b.open("rb") as f:
                    tfidf_title_dict = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'tfidf_anchor_dict.pickle':
                with b.open("rb") as f:
                    tfidf_anchor_dict = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'page_rank_dict.pckl':
                with b.open("rb") as f:
                    self.page_rank_dict = pickle.load(f)

        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'id_title_dict.pkl':
                with b.open("rb") as f:
                    self.id_title_dict = pickle.load(f)

        self.index_body.DL = dl_body
        self.index_title.DL = dl_title
        self.index_anchor.DL = dl_anchor
        self.index_body.tfidf_dict = tfidf_dict_body
        self.index_title.tfidf_dict = tfidf_title_dict
        self.index_anchor.tfidf_dict = tfidf_anchor_dict

    def generate_query_tfidf_vector(self, query_to_search, index, words):
        epsilon = .0000001
        # total_vocab_size = len(index.term_total)
        Q = np.zeros((len(query_to_search)))
        term_vector = query_to_search
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in index.df:  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def get_candidate_documents_and_scores(self, query_to_search, index, words, pls):
        candidates = {}
        N = len(index.DL)
        for term in np.unique(query_to_search):
            if term in index.df:
                list_of_doc = pls[words.index(term)]
                for doc_id, freq in list_of_doc:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + (
                            (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10))

        return candidates

    def generate_document_tfidf_matrix(self, query_to_search, index, words, pls, Q):
        cosine_dict = {}
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words,pls)
        unique_candidates = pd.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        queries_amount = len(query_to_search)
        for doc_id in unique_candidates:
            single_tfidf_list = np.zeros(queries_amount)
            i = 0
            for query in query_to_search:
                if (doc_id, query) in candidates_scores:
                    single_tfidf_list[i] = candidates_scores[(doc_id, query)]
                i += 1
            try:
                pr_score = self.page_rank_dict[doc_id]
                cosine_score = np.dot(single_tfidf_list, Q) / (index.tfidf_dict[doc_id] * np.linalg.norm(Q))
                cosine_dict[doc_id] = 2 * (cosine_score * pr_score) / (cosine_score + pr_score)
            except:
                cosine_dict[doc_id] = np.dot(single_tfidf_list, Q) / (index.tfidf_dict[doc_id] * np.linalg.norm(Q))

        return cosine_dict

    def get_top_n(self, sim_dict, N=3):
        return builtins.sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()],
                               key=lambda x: x[1], reverse=True)[:N]

    def get_topN_score_for_queries(self, queries_to_search, index, N=3):
        fin = {}
        for query in queries_to_search.keys():
            queries_to_search[query] = tokenize(queries_to_search[query])
            # clock = time()
            # overall = time()
            query_words, query_pls = zip(*index.posting_lists_iter_query_specified(queries_to_search[query]))
            Q = self.generate_query_tfidf_vector(queries_to_search[query], index, query_words)
            cosine_dict = self.generate_document_tfidf_matrix(queries_to_search[query], index, query_words, query_pls, Q)
            fin[query] = self.get_top_n(cosine_dict, N)
            # print(f' OVERALL took: {time() - overall}')
        return fin

    def search(self,queries_to_search,N=3):
        results_body = self.get_topN_score_for_queries(queries_to_search, self.index_body, N)
        results_title = self.get_topN_score_for_queries(queries_to_search, self.index_title, N)
        results_anchor = self.get_topN_score_for_queries(queries_to_search, self.index_anchor, N)
        title_body = self.merge_results(results_title, results_body, title_weight=0.5, text_weight=0.5, N=N)
        # body_anchor = merge_results(results_body, results_anchor, title_weight=0.5, text_weight=0.5, N=100)
        merged = self.merge_results(title_body, results_anchor, title_weight=0.5, text_weight=0.5, N=N)
        final = [(i[0], self.id_title_dict[i[0]]) for i in merged[0]]
        return final

    def merge_results(self,title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
        merged_dict = {}

        for query in title_scores:
            merged_list = []
            title_list = title_scores[query]
            if query in body_scores:
                body_list = body_scores[query]
                for i in title_list:
                    flag = False
                    for j in body_list:
                        if i[0] == j[0]:
                            merged_list.append((i[0], (title_weight * i[1] + text_weight * j[1])))
                            flag = True
                    if not flag:
                        merged_list.append((i[0], (title_weight * i[1] + text_weight * 0)))

                for j in body_list:
                    flag = False
                    for i in title_list:
                        if j[0] == i[0]:
                            flag = True
                            break
                    if not flag:
                        merged_list.append((j[0], (title_weight * 0 + text_weight * j[1])))

                merged_list = sorted(merged_list, key=lambda x: x[1], reverse=True)
                merged_dict[query] = merged_list[0:N]

            # if query not in body_scores
            else:
                for i in title_list:
                    i[1] = (title_weight * i[1] + text_weight * 0)
                title_list = sorted(title_list, key=lambda x: x[1], reverse=True)
                merged_dict[query] = title_list[0:N]

        for query in body_scores:
            merged_list = []
            body_list = body_scores[query]
            if query not in merged_dict:
                body_list = body_scores[query]
                for i in body_list:
                    i[1] = (title_weight * 0 + text_weight * i[1])

                body_list = sorted(body_list, key=lambda x: x[1], reverse=True)
                merged_dict[query] = body_list[0:N]

        return merged_dict
