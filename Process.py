import builtins
import math
import InvertedIndex
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
# from nltk.corpus import stopwords
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

# from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import nltk
import pickle
import numpy as np

nltk.download('stopwords')
import pickle
from InvertedIndex import InvertedIndex

# TODO: Check about 3th tab in colab

# !pip install -q pyspark
# !pip install -U -q PyDrive
# !apt-get update -qq
# !apt install openjdk-8-jdk-headless -qq
import stopwords
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

        index_file = open('./index_text.pkl','rb')
        self.index = pickle.load(index_file)
        DL_file = open('./dl.pckl','rb')
        self.index.DL = pickle.load(DL_file)
        pr_file = open('./page_rank_dict.pckl', 'rb')
        self.page_rank = pickle.load(pr_file)


        DL_file.close()
        index_file.close()
        pr_file.close()

        bucket_name = '313371858'
        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name == 'postings_gcp/index.pkl':
                with b.open("rb") as f:
                    self.index_body = pickle.load(f)

        for b in blobs:
            if b.name == 'index_title.pkl':
                with b.open("rb") as f:
                    self.index_title = pickle.load(f)

        for b in blobs:
            if b.name == 'postings_gcp/index_anchor.pkl':
                with b.open("rb") as f:
                    self.index_anchor = pickle.load(f)

        for b in blobs:
            if b.name == 'DL.pkl':
                with b.open("rb") as f:
                    dl_body = pickle.load(f)

        for b in blobs:
            if b.name == 'dl_title.pickle':
                with b.open("rb") as f:
                    dl_title = pickle.load(f)

        for b in blobs:
            if b.name == 'dl_anchor.pickle':
                with b.open("rb") as f:
                    dl_anchor = pickle.load(f)

        for b in blobs:
            if b.name == 'tfidf_dict.pkl':
                with b.open("rb") as f:
                    tfidf_dict_body = pickle.load(f)

        for b in blobs:
            if b.name == 'tfidf_title_dict.pickle':
                with b.open("rb") as f:
                    tfidf_title_dict = pickle.load(f)

        for b in blobs:
            if b.name == 'tfidf_anchor_dict.pickle':
                with b.open("rb") as f:
                    tfidf_anchor_dict = pickle.load(f)

        for b in blobs:
            if b.name == 'page_rank_dict.pckl':
                with b.open("rb") as f:
                    self.page_rank_dict = pickle.load(f)

        self.index_body.DL = dl_body
        self.index_title.DL = dl_title
        self.index_anchor.DL = dl_anchor
        self.index_body.tfidf_dict = tfidf_dict_body
        self.index_title.tfidf_dict = tfidf_title_dict
        self.index_anchor.tfidf_dict = tfidf_anchor_dict

    def generate_query_tfidf_vector(self,query_to_search, index, words):
        epsilon = .0000001
        # total_vocab_size = len(index.term_total)
        Q = np.zeros((len(query_to_search)))
        term_vector = query_to_search
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in words:  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def get_candidate_documents_and_scores(self,query_to_search, index, words, pls):
        candidates = {}
        N = len(index.DL)
        for term in np.unique(query_to_search):
            if term in words:
                list_of_doc = pls[words.index(term)]
                for doc_id, freq in list_of_doc:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + (
                                (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10))

        return candidates

    def generate_document_tfidf_matrix(self,query_to_search, index, words, pls, Q):
        cosine_dict = {}
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words,pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = pd.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        queries_amount = len(query_to_search)
        for doc_id in unique_candidates:
            single_tfidf_list = np.zeros(queries_amount)
            i = 0
            for query in query_to_search:
                if (doc_id, query) in candidates_scores:
                    single_tfidf_list[i] = candidates_scores[(doc_id, query)]
                i += 1
            cosine_dict[doc_id] = np.dot(single_tfidf_list, Q) / (index.tfidf_dict[doc_id] * np.linalg.norm(Q))
        return cosine_dict

    def get_top_n(self,sim_dict, N=3):
        return builtins.sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()],
                               key=lambda x: x[1], reverse=True)[:N]

    def get_topN_score_for_queries(self,queries_to_search, index, N=3):
        fin = {}
        for query in queries_to_search.keys():
            queries_to_search[query] = tokenize(queries_to_search[query])
            query_words, query_pls = zip(*index.posting_lists_iter_query_specified(queries_to_search[query]))
            Q = self.generate_query_tfidf_vector(queries_to_search[query], index, query_words)
            cosine_dict = self.generate_document_tfidf_matrix(queries_to_search[query], index, query_words, query_pls, Q)
            fin[query] = self.get_top_n(cosine_dict, N)
        return fin

