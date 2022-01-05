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

import ast
NUM_BUCKETS = 124
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

class Process:

    def __init__(self):

        file = open('./index_text.pkl','rb')
        self.index = pickle.load(file)
        file.close()
        file = open('./dl.pckl','rb')
        self.index.DL = pickle.load(file)


    def generate_query_tfidf_vector(self, query_to_search, index, words):
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

    def get_candidate_documents_and_scores(self, query_to_search, index, words, pls):

        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words,
                                                                    pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), len(query_to_search)))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = query_to_search
        t_start = time()
        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf
        return D

    def generate_document_tfidf_matrix(self, query_to_search, index, words, pls):
        # total_vocab_size = len(index.term_total)
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words,
                                                               pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), len(query_to_search)))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = query_to_search

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf

        return D

    def cosine_similarity(self, D, Q):
        d_transpose = np.transpose(D)
        result = np.dot(Q, d_transpose) / (np.linalg.norm(Q) * (np.linalg.norm(D, axis=1)))
        fin = {}
        for i in range(result.shape[0]):
            fin[D.index[i]] = result[i]
        return fin

    def get_top_n(self,sim_dict, N=3):
        return builtins.sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()],
                               key=lambda x: x[1], reverse=True)[:N]

    def get_topN_score_for_queries(self,queries_to_search, index, N=3):
        fin = {}
        for query in queries_to_search.keys():
            query_words, query_pls = zip(*index.posting_lists_iter_query_specified(queries_to_search[query]))
            matrix = self.generate_document_tfidf_matrix(queries_to_search[query], index, query_words, query_pls)
            vector = self.generate_query_tfidf_vector(queries_to_search[query], index,query_words)
            cosine_dict = self.cosine_similarity(matrix, vector)
            fin[query] = self.get_top_n(cosine_dict, N)
        return fin


