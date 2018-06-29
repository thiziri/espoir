import sys
import json
import pyndri
import numpy as np
from keras.layers import Input, Dense, Embedding
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model
from utils import read_lablers_to_relations, convert_embed_2_numpy, read_embedding, get_queries
from keras import backend as K
from keras.layers import Lambda
from os.path import join
from tqdm import tqdm


if __name__ == '__main__':
    config_file = sys.argv[1]
    configure = json.load(open(config_file))
    config = configure["main_configuration"]
    config_data = config["data_sets"]
    config_model = config["model"]
    config_model_param = config_model["parameters"]
    config_model_train = config_model["train"]
    config_model_test = config_model["test"]
    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Read embeddings ...")
    embed_tensor = convert_embed_2_numpy(read_embedding(config_data["embed"]), config_data["vocab_size"])

    print("Create a model...")

    query = Input(name="in_query", shape=(config_data['query_maxlen'], ), dtype='int32')  # ex: query vector of 10 words
    doc = Input(name="in_doc", shape=(config_data['doc_maxlen'], ), dtype='int32')

    embedding = Embedding(config_data['vocab_size'], config_data['embed_size'], weights=[embed_tensor],
                          trainable=config_model_train['train_embed'], name="embeddings")  # load and/or train embeddings
    q_embed = embedding(query)
    d_embed = embedding(doc)
    print("Embedded inputs: \nq_embed: {qe}\nd_embed: {de}".format(qe=q_embed, de=d_embed))
    sum_dim1 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(config_data['embed_size'],), name="sum_vectors")
    q_vector = sum_dim1(q_embed)  # (1 x embed_size)
    d_vector = sum_dim1(d_embed)  # (1 x embed_size)
    print("Added vectors\nq_vector: {qv}\nd_vector: {dv}".format(qv=q_vector, dv=d_vector))

    q_d_labels = Input(name="labels_vector", shape=(config_data['labelers_num'], ), dtype='float32')

    input_vector = concatenate([q_vector, d_vector, q_d_labels])
    print("Concatenated vector: {iv}".format(iv=input_vector))
    dense = Dense(config_model_param["layers_size"][0], activation=config_model_param['hidden_activation'],
                  name="MLP_combine_0")(input_vector)
    for i in range(config_model_param["num_layers"]-2):
        dense = Dense(config_model_param["layers_size"][i], activation=config_model_param['hidden_activation'],
                      name="MLP_combine_"+str(i+1))(dense)
    dense = Dense(1, activation=config_model_param['output_activation'], name="MLP_out"+str(i+1))(dense)
    model = Model(inputs=[query, doc, q_d_labels], outputs=dense)
    model.compile(optimizer=config_model_param["optimizer"], loss=config_model_train["loss_function"],
                  metrics=config_model_train["metrics"])
    print(model.summary())
    plot_model(model, to_file=join(config_model_train["train_details"], 'collaborative.png'))
    # save model and resume

    print("Reading training data ...")
    x_train = []
    print("Reading train instances ...")
    print("[First]:\nRead label files to relations...")
    relations, relation_labeler = read_lablers_to_relations(config_data["labels"])

    print("[Second]:\nSet relations as train instances...")

    print("Reading data index ...")
    index = pyndri.Index(config_data["index"])
    token2id, _, _ = index.get_dictionary()
    externalDocId = {}
    for doc_id in range(index.document_base(), index.maximum_document()):  # type: int
        extD_id, _ = index.document(doc_id)
        externalDocId[extD_id] = doc_id
    train_queries = get_queries(config_data["train_queries"])

    print("x_train preparation...")
    # the model needs list of 3 input arrays :
    v_q_words = []
    v_d_words = []
    v_rel_labels = []

    # print(train_queries)
    # print(relations)

    for relation in tqdm(relations):
        # get q_word_ids from index
        q_words = [token2id[qi] if qi in token2id else 0 for qi in train_queries[relation[0]].strip().split()]
        # ############## pad/truncate q_words to a query_maxlen
        d_words = list(index.document(externalDocId[relation[1]])[1])  # get d_word_ids from index
        # ############## pad/truncate d_words to a doc_maxlen
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
        rel_labels = []
        for labeler in relation_labeler:
            try:
                rel_labels.append(int(relation_labeler[labeler][relation]))  # read the label given by labeler
            except:
                rel_labels.append(-1)  # set to -1 if the labeler doesn't gave any label
        v_q_words.append(q_words)
        v_d_words.append(d_words)
        v_rel_labels.append([int(relation_labeler[labeler][relation]) if relation in relation_labeler[labeler] else -1
                             for labeler in relation_labeler])

        x_train_i = [np.array(q_words), np.array(d_words), np.array(rel_labels)]
        x_train.append(np.array(x_train_i))
    # print(x_train[0])

    print("y_train preparation...")
    y_train = [np.average(x_train_i[2]) for x_train_i in x_train]  # output [array]
    print(y_train[0])

    # print(v_q_words)

    print("Model training...")
    model.fit(x=[np.array(v_q_words), np.array(v_d_words), np.array(v_rel_labels)], y=np.array(y_train),
              batch_size=5, epochs=1, verbose=1, shuffle=True)




