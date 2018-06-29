# -*- coding: utf-8 -*-
from __future__ import unicode_literals

"""
Tools for data extraction and analysis
"""

import collections
import re
import nltk
import gzip
import ntpath
import six
import array
import numpy as np
import pickle
import pyndri
import codecs
import math
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from os import listdir
from os.path import join
from nltk.stem.porter import PorterStemmer
from krovetzstemmer import Stemmer

LABELS = ["bin", "official", "multi_scale"]

"""
It return the file name extracted from a path
"""
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


"""
removes the file extension:
example: file.txt becomes file
return: file name without extension
"""
def remove_extension(file):
    if len(file.split('.')) > 1:
        return ".".join(file.split('.')[:-1])
    return file


"""
Cleans the input text of special characters
return cleaned text
"""
def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('^'): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('!'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('€'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
        ord('#'): ' ',
        ord('_'): ' ',
        ord('@'): ' ',
        ord('~'): ' ',
        ord('='): None,
        ord('*'): None,
    })


"""
Performs stemming according to the selected algo
return stemed text
"""
def stem(algo, text):
    if algo == "krovetz":
        stemmer = Stemmer()
        return stemmer.stem(text)
    elif algo == "porter":
        stm = PorterStemmer()
        return stm.stem(text)
    print("ERROR STEMMING: {t} unkown.".format(t=algo))


"""
Performs cleaning and stemming 
return cleaned and stemmed text
"""
def clean(text_to_clean, steming, stoplist):
    prog = re.compile("[_\-\(]*([A-Z]\.)*[_\-\(]*")
    tex = []
    for w in text_to_clean.split():
        if prog.match(w):
            w = w.replace('.', '')
        tex.append(w)
    text = " ".join(tex)
    text = ' '.join(escape(text).split())
    text = " ".join(nltk.word_tokenize(text))
    text = " ".join([stem(steming, w) for w in text.split() if w not in stoplist])
    return text


""" 
Extract TREC topics on the pathTop parameter as dictionnary. 
return dictionnary of queries.
ex: {0:"this is a text of the topic"}
"""
def extract_topics(path_top):
    print("Extraction de : %s" % path_top)
    nb = 0
    topics = {}
    for f in listdir(path_top):
        f = open(join(path_top, f), 'r')  # Reading file
        l = f.readline().lower()
        # extracting topics
        while l != "":
            if l != "":
                num = 0
                while (l.startswith("<num>") == False) and (l != ""):
                    l = f.readline().lower()
                num = l.replace("<num>", "").replace("number:", "").replace("\n", "").replace(" ", "")
                while (l.startswith("<title>") == False) and (l != ""):
                    l = f.readline().lower()
                titre = ""
                while (not l.startswith("</top>")) and (not l.startswith("<desc>")) and (l != ""):
                    titre = titre + " " + l.replace("<title>", "")
                    l = f.readline().lower()
                if titre != "" and num != 0:
                    topics[str(int(num))] = titre.replace("\n", "").replace("topic:", "").replace("\t", " ")
                    nb += 1
            else:
                print("Fin.\n ")
        f.close()
    return OrderedDict(sorted(topics.items()))


""" 
Extract TREC million queries on the path_top parameter as dictionnary. 
return: dictionnary of queries.
ex: {0:"this is a text of the query"}
"""
def extract_trec_million_queries(path_top):
    topics = {}
    for f in listdir(path_top):
        print("Processing file ", f)
        if ".gz" not in f:
            input = open(join(path_top, f), 'r')  # Reading file
        else:
            input = gzip.open(join(path_top, f))
        for line in tqdm(input.readlines()):
            l = line.decode("iso-8859-15")
            query = l.strip().split(":")
            q = str(int(query[0]))
            q_text = query[-1]  # last token string
            topics[q] = q_text
    return OrderedDict(sorted(topics.items()))


"""
Read the qrels file to a dictionary.
Return dictionary of: {(q_id, d_id):rel} 
"""
def get_qrels(qrels_file):
    print("Reading Qrels ... ")
    qdr = {}
    with open(qrels_file, 'r') as qrels:
        for line in tqdm(qrels):
            if line is not None:
                q = str(int(line.strip().split()[0]))
                doc = line.strip().split()[2]
                rel = int(line.strip().split()[3])
                qdr[(q, doc)] = rel
    print("Qrels ok.")
    return collections.OrderedDict(sorted(qdr.items()))


"""
Read document list from a trec like run_file.
return: a set of ranked distinct documents
"""
def get_docs_from_run(run_file):
    print("Reading run_file: ", run_file)
    docs = []
    with open(run_file) as rf:
        for l in rf:
            if l is not None:
                docs.append(l.strip().split()[2])
    return set(docs)


"""
construct a list of relevance judgements associated to each rank interval, 
then gives the corresponding relevance judgement to the given rank
"""
def rank_to_relevance(rank, scales=3, ranks=((1, 10), (11, 30), (31, 50))):
    relevance = {(ranks[i][0], ranks[i][1]): scales - i for i in range(len(ranks))}
    for interval in relevance:
        if rank in range(interval[0], interval[1] + 1):
            return relevance[interval]


"""
Create relations from a run file. Same as rank2relations but with another format
Return: list of relations [((q, doc), rel)]
"""
def run2relations(run_file, labels, scales, ranks, qrels=[], k=1000):
    relations = []
    with open(run_file, "r") as rank:
        i = 0
        j = -1
        queries_rank = []
        for line in tqdm(rank):
            if line is not None:
                j += 1
                q = str(int(line.strip().split()[0]))
                if q in queries_rank:
                    i += 1
                else:
                    queries_rank.append(q)
                    i = 1
                doc = line.strip().split()[2]
                rel = -1
                assert labels in LABELS
                if labels == "multi_scale":
                    x = rank_to_relevance(i, scales, ranks)
                    rel = x if x is not None else 0  # multiscale relevance
                if labels == "bin":
                    rel = 1 if i <= 10 else 0  # binary relevance
                if labels == "official":
                    try:
                        rel = qrels[(q, doc)]
                    except:
                        rel = 0
                if i in range(k + 1):
                    relations.insert(j, ((q, doc), rel))
                else:
                    continue
    return relations


"""
parse files of a given query directory
"""
def write_queries_to_file(queries_dir, out,  _format):
    queries = {}
    if _format not in {'trec', 'mq'}:
        raise ("Unknown query file format {}".format(_format))
    if _format == "trec":
        queries = extract_topics(queries_dir)
    if _format == "mq":
        queries = extract_trec_million_queries(queries_dir)
    queries_text = {}
    q_times = defaultdict(int)
    print("Preprocess queries ...")
    for q in tqdm(queries):
        q_text = clean(queries[q], "krovetz", {})
        q_times[q_text] += 1
        queries_text[q] = q_text if q_times[q_text] == 1 else ' '.join([q_text, str(q_times[q_text])])
    print("Saving to file ...")
    with open(out, 'w') as out_f:
        for q in tqdm(queries_text):
            out_f.write("{q_id}\t{q_txt}\n".format(q_id=q, q_txt=queries_text[q]))


"""
read unique values from column n in the file f
"""
def read_values(f, n):
    inf = open(f, "r")
    lines = inf.readlines()
    result = []
    for x in lines:
        result.append(x.split(' ')[n])
    inf.close()
    return set(result)


"""
devide list seq into num different sub-lists
return: list of folds
"""
def chunkIt(seq, num=5):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


"""
Pros:
Save the oov words in oov.p for further analysis.
Refs:
class Vectors, https://github.com/pytorch/text/blob/master/torchtext/vocab.py
Args:
vocab: dict,
w2v_file: file, path to file of pre-trained word2vec/glove/fasttext
Returns:
vectors
"""
def load_word_embedding(vocab, w2v_file):
    pre_trained = {}
    n_words = len(vocab) + 1  # for unknown word label
    embeddings = None
    vectors, dim = array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        with open(w2v_file, encoding="utf8") as f:
            lines = [line for line in f]
    except:
        print("Could not read {} as UTF8 file, "
              "reading file as bytes and skipping "
              "words with malformed UTF8.".format(w2v_file))
        with open(w2v_file, 'rb') as f:
            lines = [line for line in f]
        binary_lines = True

    print("Loading vectors from {}".format(w2v_file))
    unk = []

    for line in tqdm(lines):
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(b" " if binary_lines else " ")

        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)
            # init the embeddings with zeros
            embeddings = np.zeros((n_words, dim), dtype='float32')

        elif len(entries) == 1:
            print("Skipping token {} with 1-dimensional "
                  "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                "Vector for token {} has {} dimensions, but previously "
                "read vectors have {} dimensions. All vectors must have "
                "the same number of dimensions.".format(word, len(entries), dim))

        if binary_lines:
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print("Skipping non-UTF8 token {}".format(repr(word)))
                continue

        if word in vocab and word not in pre_trained and word != "<unk>":
            w_id = vocab[word]
            try:
                embeddings[w_id] = [float(x) for x in entries]
            except Exception as error:
                print('Error saving this word : {}\n'.format(word) + repr(error))
                print(entries)

            pre_trained[word] = 1
        if word == "<unk>":
            unk = [float(x) for x in entries]

    # init tht OOV word embeddings
    if len(unk) == 0:
        unk = np.zeros([dim], dtype='float32')
    for word in vocab:
        if word not in pre_trained:
            """
            alpha = 0.5 * (2.0 * np.random.random() - 1.0)
            curr_embed = (2.0 * np.random.random_sample([dim]) - 1.0) * alpha
            """
            curr_embed = unk
            embeddings[vocab[word]] = curr_embed
    embeddings[0] = unk

    pre_trained_len = len(pre_trained)
    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))

    oov_word_list = [w for w in vocab if w not in pre_trained]
    print('oov word list example (30): ', oov_word_list[:30])
    pickle.dump(oov_word_list, open('oov.p', 'wb'), protocol=2)

    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings


""" indri index -> {word: index} """
def load_word_dict(data_index):
    print("Reading index ...")
    index = pyndri.Index(data_index)
    token2id, _, _ = index.get_dictionary()
    return token2id


"""
Normalize the input embeddings
"""
def normalize(infile, outfile):
    fout = codecs.open(outfile, 'w', encoding='utf8')
    with codecs.open(infile, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            r = line.split()
            w = r[0]
            try:
                vec = [float(k) for k in r[1:]]
            except:
                print(line)
            sum = 0.001
            for k in vec:
                sum += k * k
            sum = math.sqrt(sum)
            for i, k in enumerate(vec):
                vec[i] = (vec[i] + 0.001)/sum
            print(w, ' '.join(['%f' % k for k in vec]), file=fout)
    fout.close()
