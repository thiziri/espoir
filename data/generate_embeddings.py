# coding: utf8
from __future__ import print_function

import sys
from utils import load_word_dict, load_word_embedding, normalize
from tqdm import tqdm
import json


if __name__ == '__main__':

    config_file = sys.argv[1]
    configure = json.load(open(config_file))
    config = configure["main_configuration"]
    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    w2v_file = config["pretrained_embedding"]  # w2v_file
    data_index = config["index"]  # Indri index
    mapped_w2v_file = config["output_embedding"]  # output shared w2v dict

    print('load word dict ...')
    word_dict = load_word_dict(data_index)
    print("Dictionary length: {}".format(len(word_dict)))

    print('load word vectors ...')
    embeddings = load_word_embedding(word_dict, w2v_file)

    print('save word vectors ...')
    with open(mapped_w2v_file, 'w') as fw:
        # assert word_dict
        for w, idx in tqdm(word_dict.items()):
            try:
                print(word_dict[w], ' '.join(map(str, embeddings[idx])), file=fw)
            except Exception as error:
                print('Error saving this word : {}\n'.format(word_dict[w]) + repr(error))
                # print(embeddings[idx])

    print('Map word vectors finished ...')

    if config["normalize"]:
        print("Normalization ...")
        normalize(mapped_w2v_file, config["normalized_embedding"])

    print("Embeddings OK.")


