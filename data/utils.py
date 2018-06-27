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
    return collections.OrderedDict(sorted(topics.items()))


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
            q = "mq" + str(int(query[0]))
            q_text = query[-1]  # last token string
            topics[q] = q_text
    return collections.OrderedDict(sorted(topics.items()))


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
