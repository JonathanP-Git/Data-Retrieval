import builtins
import math

import search_frontend as se
# !pip install pyspark
# !pip install graphframes

import pyspark
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
import pandas as pd
from google.cloud import storage
import operator
import hashlib

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SQLContext
from graphframes import *

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import nltk
import pickle
import numpy as np

nltk.download('stopwords')

from nltk.corpus import stopwords
from tqdm import tqdm
import operator
from itertools import islice, count
from contextlib import closing

import json
from io import StringIO
from pathlib import Path
from operator import itemgetter
import pickle
import matplotlib.pyplot as plt
import InvertedIndex

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

        conf = SparkConf().set("spark.ui.port", "4050")
        sc = pyspark.SparkContext(conf=conf)
        sc.addPyFile(str(Path(spark_jars) / Path(graphframes_jar).name))
        self.spark = SparkSession.builder.getOrCreate()

        # import ast import literal_eval
        pandasDF = pd.read_csv("/content/smallCourpus.csv")
        df_data_spark = spark.createDataFrame(pandasDF)
        self.doc_text_pairs_rdd = df_data_spark.select("text", "id").rdd
        self.doc_title_pairs_rdd = df_data_spark.select("title", "id").rdd
        self.doc_anchor_text_pairs_rdd = df_data_spark.select("id", "anchor_text").rdd

        # TODO: Add lines

        self.text_dict = dict(zip(pandasDF.id, pandasDF.text))
        self.title_dict = dict(zip(pandasDF.id, pandasDF.title))
        self.anchor_dict = dict(zip(pandasDF.id, pandasDF.anchor_text))

        self.id_text = {}
        self.id_title = {}
        self.id_anchor = {}

        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = english_stopwords.union(corpus_stopwords)

    def _hash(self, s):
        return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

    def tokenize(self, text):
        """
        This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

        Parameters:
        -----------
        text: string , represting the text to tokenize.

        Returns:
        -----------
        list of tokens (e.g., list of tokens).
        """
        stopwords_frozen = frozenset(stopwords.words('english'))
        list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                          token.group() not in stopwords_frozen]
        return list_of_tokens


    def tokenize_on_dicts(self):
        self.id_text = {k: self.tokenize(v) for k, v in self.text_dict.items()}
        self.id_title = {k: self.tokenize(v) for k, v in self.title_dict.items()}
        self.id_anchor = {k: self.tokenize(v) for k, v in self.anchor_dict.items()}


    def token2bucket_id(self, token):
        return int(self._hash(token), 16) % NUM_BUCKETS


    def word_count(self, text, id):
        tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
        finalList = []
        dict_count = {}
        for token in tokens:
            if token not in dict_count and token not in self.all_stopwords:
                dict_count[token] = 1
            elif token not in self.all_stopwords:
                dict_count[token] += 1
        for k, v in dict_count.items():
            finalList.append((k, (id, v)))
        return finalList


    def reduce_word_counts(self, unsorted_pl):
        return sorted(unsorted_pl, key=lambda x: x[0])


    def calculate_df(self, postings):
        return postings.map(lambda x: (x[0], len(x[1])))


    def partition_postings_and_write(self, postings, name=''):
        bucket_name = "Test"
        buckets = postings.map(lambda x: (self.token2bucket_id(x[0]), x)).groupByKey()
        return buckets.map(lambda x: (InvertedIndex.write_a_posting_list(x, name)))


    def creation_inverted_index(self):

        # # word counts map
        word_counts_text = self.doc_text_pairs_rdd.flatMap(lambda x: self.word_count(x[0], x[1]))
        word_counts_anchor_text = self.doc_anchor_text_pairs_rdd.flatMap(lambda x: self.word_count(x[0], x[1]))
        word_counts_title = self.doc_title_pairs_rdd.flatMap(lambda x: self.word_count(x[0], x[1]))

        postings_text = word_counts_text.groupByKey().mapValues(self.reduce_word_counts)
        postings_text_anchor_text = word_counts_text.groupByKey().mapValues(self.reduce_word_counts)
        postings_title = word_counts_title.groupByKey().mapValues(self.reduce_word_counts)

        # # # filtering postings and calculate df
        postings_filtered_text = postings_text.filter(lambda x: len(x[1]) > 50)
        postings_filtered_anchor_text = postings_text_anchor_text.filter(lambda x: len(x[1]) > 50)
        postings_filtered_title = postings_title.filter(lambda x: len(x[1]) >= 1)

        w2df_text = self.calculate_df(postings_filtered_text)
        w2df_anchor_text = self.calculate_df(postings_filtered_anchor_text)
        w2df_title = self.calculate_df(postings_filtered_title)

        w2df_text_dict = w2df_text.collectAsMap()
        w2df_anchor_text_dict = w2df_anchor_text.collectAsMap()
        w2df_title_dict = w2df_title.collectAsMap()

        # partition posting lists and write out
        # type(postings_filtered)
        posting_locs_list_text = self.partition_postings_and_write(postings_filtered_text, 'text').collect()
        posting_locs_list_anchor_text = self.partition_postings_and_write(postings_filtered_anchor_text, 'anchor_text').collect()
        posting_locs_list_title = self.partition_postings_and_write(postings_filtered_title, 'title').collect()

        super_posting_locs_text = defaultdict(list)
        for posting_loc in posting_locs_list_text:
            for k, v in posting_loc.items():
                super_posting_locs_text[k].extend(v)

        super_posting_locs_anchor = defaultdict(list)
        for posting_loc in posting_locs_list_anchor_text:
            for k, v in posting_loc.items():
                super_posting_locs_anchor[k].extend(v)

        super_posting_locs_title = defaultdict(list)
        for posting_loc in posting_locs_list_title:
            for k, v in posting_loc.items():
                super_posting_locs_title[k].extend(v)


        # Create inverted index instance
        inverted_text = InvertedIndex(docs=self.id_text)
        inverted_anchor_text = InvertedIndex(docs=self.id_anchor)
        inverted_title = InvertedIndex(docs=self.id_title)

        # Adding the posting locations dictionary to the inverted index
        inverted_text.posting_locs = self.super_posting_locs_text
        inverted_anchor_text.posting_locs = self.super_posting_locs_anchor
        inverted_title.posting_locs = self.super_posting_locs_title

        # Add the token - df dictionary to the inverted index
        inverted_text.df = w2df_text_dict
        inverted_anchor_text.df = w2df_anchor_text_dict
        inverted_title.df = w2df_title_dict

        # write the global stats out
        inverted_text.write_index('.', 'index_text')
        inverted_anchor_text.write_index('.', 'index_anchor')
        inverted_title.write_index('.', 'index_title')


    def generate_query_tfidf_vector(self, query_to_search, index):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """

        epsilon = .0000001
        total_vocab_size = len(index.term_total)
        Q = np.zeros((total_vocab_size))
        term_vector = list(index.term_total.keys())
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log((len(InvertedIndex.DL)) / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def get_candidate_documents_and_scores(self, query_to_search, index, words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = {}
        N = len(InvertedIndex.DL)
        for term in np.unique(query_to_search):
            if term in words:
                list_of_doc = pls[words.index(term)]
                for doc_id, freq in list_of_doc:
                    normlized_tfidf = [(doc_id, (freq / InvertedIndex.DL[(doc_id)]) * math.log(N / index.df[term], 10)) for
                                       doc_id, freq in list_of_doc]
                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates


    def generate_document_tfidf_matrix(self, query_to_search, index, words, pls):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        DataFrame of tfidf scores.
        """

        total_vocab_size = len(index.term_total)
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words,
                                                               pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = index.term_total.keys()

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf

        return


    def cosine_similarity(self, D, Q):
        """
        Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
        Generate a dictionary of cosine similarity scores
        key: doc_id
        value: cosine similarity score

        Parameters:
        -----------
        D: DataFrame of tfidf scores.

        Q: vectorized query with tfidf scores

        Returns:
        -----------
        dictionary of cosine similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: cosine similarty score.
        """
        d_transpose = np.transpose(D)
        result = np.dot(Q, d_transpose) / (np.linalg.norm(Q) * (np.linalg.norm(D, axis=1)))
        fin = {}
        for i in range(result.shape[0]):
            fin[D.index[i]] = result[i]
        return fin


    def get_top_n(self, sim_dict, N=3):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """
        print(sim_dict.items())
        return builtins.sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()],
                               key=lambda x: x[1], reverse=True)[:N]

    def get_topN_score_for_queries(self, queries_to_search, index, N=3):
        """
        Generate a dictionary that gathers for every query its topN score.

        Parameters:
        -----------
        queries_to_search: a dictionary of queries as follows:
                                                            key: query_id
                                                            value: list of tokens.
        index:           inverted index loaded from the corresponding files.
        N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        return: a dictionary of queries and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id, score).
        """

        fin = {}
        words, pls = zip(*index.posting_lists_iter())
        for query in queries_to_search.keys():
            matrix = self.generate_document_tfidf_matrix(queries_to_search[query], index, words, pls)
            vector = self.generate_query_tfidf_vector(queries_to_search[query], index)
            cosine_dict = cosine_similarity(matrix, vector)
            fin[query] = self.get_top_n(cosine_dict, N)
        return fin

    def merge_results(self, title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
        """
        This function merge and sort documents retrieved by its weighte score (e.g., title and body).

        Parameters:
        -----------
        title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)

        body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)
        title_weight: float, for weigted average utilizing title and body scores
        text_weight: float, for weigted average utilizing title and body scores
        N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        dictionary of querires and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id,score).
        """
        #Yoni
        def weights(dictionary, weight):
            for k in dictionary.keys():
                new_list = []
                for item in dictionary[k]:
                    new_list.append((item[0], item[1] * weight))
                dictionary[k] = new_list

        titles = title_scores.copy()
        body = body_scores.copy()
        weights(titles, title_weight)
        weights(body, text_weight)
        merged_dict = {}
        db = {}

        for k_title, v_title in titles.items():
            merged_dict[k_title] = []
            db[k_title] = []
            for title_item in titles[k_title]:
                merged_dict[k_title].append((title_item[0], title_item[1]))
                db[k_title].append(title_item[0])

        for k_body, v_body in body.items():
            for body_item in body[k_body]:
                if body_item[0] in db[k_body]:
                    items = []
                    for item in merged_dict[k_body]:
                        if item[0] == body_item[0]:
                            items.append((item[0], item[1] + body_item[1]))
                        else:
                            items.append(item)
                    merged_dict[k_body] = items
                else:
                    merged_dict[k_body].append((body_item[0], body_item[1]))

        for k, v in merged_dict.items():
            merged_dict[k] = sorted(v, key=lambda tup: tup[1], reverse=True)
            merged_dict[k] = merged_dict[k][:N]
        return merged_dict

        #Yuval
        # merged_dict = {}
        #
        # for query in title_scores:
        #     merged_list = []
        #     title_list = title_scores[query]
        #     if query in body_scores:
        #         body_list = body_scores[query]
        #         for i in title_list:
        #             flag = False
        #             for j in body_list:
        #                 if i[0] == j[0]:
        #                     merged_list.append((i[0], (title_weight * i[1] + text_weight * j[1])))
        #                     flag = True
        #             if not flag:
        #                 merged_list.append((i[0], (title_weight * i[1] + text_weight * 0)))
        #
        #         for j in body_list:
        #             flag = False
        #             for i in title_list:
        #                 if j[0] == i[0]:
        #                     flag = True
        #                     break
        #             if not flag:
        #                 merged_list.append((j[0], (title_weight * 0 + text_weight * j[1])))
        #
        #         merged_list = sorted(merged_list, key=lambda x: x[1], reverse=True)
        #         merged_dict[query] = merged_list[0:N]
        #
        #     # if query not in body_scores
        #     else:
        #         for i in title_list:
        #             i[1] = (title_weight * i[1] + text_weight * 0)
        #         title_list = sorted(title_list, key=lambda x: x[1], reverse=True)
        #         merged_dict[query] = title_list[0:N]
        #
        # for query in body_scores:
        #     merged_list = []
        #     body_list = body_scores[query]
        #     if query not in merged_dict:
        #         body_list = body_scores[query]
        #         for i in body_list:
        #             i[1] = (title_weight * 0 + text_weight * i[1])
        #
        #         body_list = sorted(body_list, key=lambda x: x[1], reverse=True)
        #         merged_dict[query] = body_list[0:N]
        #
        # return merged_dict