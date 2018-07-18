import numpy as np
import pickle
from os.path import join

class ContentReader():
    """
    Reads content of a given doc/query content
    """
    def __init__(self, relations_list, token2id, external_doc_ids, queries_list, q_max_len, d_max_len,
                 train_queries,
                 input_files,
                 index
                 ):
        self.index = index
        self.relations_list = relations_list
        self.token2id = token2id
        self.external_doc_ids = external_doc_ids
        self.queries_list = queries_list
        self.q_max_len = q_max_len
        self.d_max_len = d_max_len
        self.train_queries = train_queries
        self.input_files = input_files

    def get_query(self, query):
        """
        Reads the q_id query content
        :param query: id
        :return: list
        """
        if self.index and not self.input_files:
            q_words = [self.token2id[qi] if qi in self.token2id else 0 for qi in self.train_queries[query].split()]
            if len(q_words) < self.q_max_len:
                q_words = list(np.pad(q_words, (self.q_max_len - len(q_words), 0), "constant", constant_values=0))
            elif len(q_words) > self.q_max_len:
                q_words = q_words[:self.q_max_len]
        else:
            q_words = [int(e) for e in open(join(join(self.input_files, "queries"), query)).read().strip().split()]
        return q_words

    def get_document(self, d_id):
        """
        Reads the q_id query content
        :param d_id: str
        :return: list
        """
        doc = self.external_doc_ids[d_id]
        if self.index and not self.input_files:
            doc_words = list(self.index.document(doc)[1])
            if len(doc_words) < self.d_max_len:
                doc_words = list(np.pad(doc_words, (self.d_max_len - len(doc_words), 0), "constant", constant_values=0))
            elif len(doc_words) > self.d_max_len:
                doc_words = doc_words[:self.d_max_len]
        else:
            doc_words = [int(e) for e in open(join(join(self.input_files, "documents"), d_id)).read().strip().split()]
        return doc_words


class ContentPickleReader(ContentReader):
    """
    Reads content of a given doc/query content
    """
    def __init__(self, relations_list, token2id, external_doc_ids, queries_list, q_max_len, d_max_len,
                 train_queries,
                 input_files=None,
                 index=None):
        ContentReader.__init__(self, relations_list, token2id, external_doc_ids, queries_list, q_max_len, d_max_len,
                 train_queries,
                 input_files,
                 index)

    def pickle_data(self):
        """
        Reads the list of data inputs and corresponding dictionary
        :param query: self
        :return: tuple
        """
        data_list = pickle.load(open(join(self.input_files, "data_list.pickle"), 'rb'))
        index_dict = pickle.load(open(join(self.input_files, "index_dict.pickle"), 'rb'))
        return data_list, index_dict

