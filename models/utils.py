import numpy as np
import os
import sys
from os.path import join
from tqdm import tqdm

sys.path.append('../data/utils')
from utils import *


""" 
read embed_norm to dictionary
return: dict
"""
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
    return embed


"""
Convert Embedding Dict 2 numpy array
return: numpy matrix
"""
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros((max_size, feat_size), dtype=np.float32)

    if len(embed_dict) > len(embed):
        raise Exception("vocab_size %d is larger than embed_size %d, change the vocab_size in the config!"
                        % (len(embed_dict), len(embed)))

    for k in embed_dict:
        embed[k] = np.array(embed_dict[k])
    print('Generate numpy embed:', str(embed.shape), end='\n')
    return embed


"""
read label files to relations
return: (set, dict)
"""
def read_lablers_to_relations(labelers_dir):
    relation_labeler = {}
    relations = set()
    for labeler_file in os.listdir(labelers_dir):
        relation_labeler[labeler_file] = {}
        with open(join(labelers_dir, labeler_file), 'r') as labeler:
            for l in tqdm(labeler.readlines()):
                relation = (l.strip().split()[0], l.strip().split()[1])  # (q, d)
                relations.add(relation)
                relation_labeler[labeler_file][relation] = l.strip().split()[2]  # rel
    return relations, relation_labeler


"""
read query files where each line is in the format: "q_id    q_text"
return: dict
"""
def get_queries(query_file):
    with open(query_file, "r") as f:
        return {l.strip().split("\t")[0]: l.strip().split("\t")[1] for l in f}
