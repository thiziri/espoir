import sys
import os
import pyndri
import pickle
from os.path import join
from tqdm import tqdm
sys.path.append('../models')
from content_reader import ContentReader

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


if __name__ == "__main__":
    print("[First]:\nRead label files to relations...")
    relations, _ = read_lablers_to_relations(sys.argv[1])  # relation .label file
    queries = get_queries(sys.argv[2])  # extracted queries
    out = sys.argv[3]  # output folder
    index = pyndri.Index(sys.argv[4])  # index
    print("Reading data index ...")
    token2id, _, _ = index.get_dictionary()
    print(len(token2id))
    externalDocId = {}
    for doc_id in range(index.document_base(), index.maximum_document()):  # type: int
        extD_id, _ = index.document(doc_id)
        externalDocId[extD_id] = doc_id
    q_max_len, d_max_len = int(sys.argv[5]), int(sys.argv[6])  # query and document max length respectively

    relations_list = list(relations)
    queries_list = list(queries.keys())
    reader = ContentReader(relations_list, token2id, externalDocId, queries_list, q_max_len, d_max_len, queries,
                           index=index)

    uniq_queries = set()
    uniq_documents = set()
    all_data_list = []
    data_dict = {}

    for rel in relations:
        uniq_queries.add(rel[0])
        uniq_documents.add(rel[1])

    print("computing data ...")
    for q_id in tqdm(uniq_queries):
        q = reader.get_query(q_id)
        all_data_list.append(q)
        data_dict[q_id] = len(all_data_list)-1

    for d_id in tqdm(uniq_documents):
        d = reader.get_document(d_id)
        all_data_list.append(d)
        data_dict[d_id] = len(all_data_list) - 1

    print("Save to files ...")
    with open(join(out, "data_list.pickle"), 'wb') as output:
        pickle.dump(all_data_list, output)
    with open(join(out, "index_dict.pickle"), 'wb') as output:
        pickle.dump(data_dict, output, protocol=pickle.HIGHEST_PROTOCOL)

    # use pickle.load for reading

    print("Done")
