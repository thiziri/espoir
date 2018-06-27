import sys
import json
from keras.layers import Input, Dense, Embedding, Concatenate
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model
from utils import *
from keras import backend as K
from keras.layers import Lambda


if __name__ == '__main__':
    config_file = sys.argv[1]
    configure = json.load(open(config_file))
    config = configure["main_configuration"]
    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Read embeddings ...")
    embed_tensor = convert_embed_2_numpy(read_embedding(config["embed"]), config["vocab_size"])

    print("Create a model...")

    query = Input(name="in_query", shape=(config['query_maxlen'], ), dtype='int32')  # ex: query vector of 10 words
    doc = Input(name="in_doc", shape=(config['doc_maxlen'], ), dtype='int32')

    embedding = Embedding(config['vocab_size'], config['embed_size'], weights=[embed_tensor],
                          trainable=config['train_embed'], name="embeddings")  # load and/or train embeddings
    q_embed = embedding(query)
    d_embed = embedding(doc)
    print("Embedded inputs: \nq_embed: {qe}\nd_embed: {de}".format(qe=q_embed, de=d_embed))
    sum_dim1 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(config['embed_size'],), name="sum_vectors")
    q_vector = sum_dim1(q_embed)  # (1 x embed_size)
    d_vector = sum_dim1(d_embed)  # (1 x embed_size)
    print("Added vectors\nq_vector: {qv}\nd_vector: {dv}".format(qv=q_vector, dv=d_vector))

    q_d_labels = Input(name="labels_vector", shape=(config['labelers_num'], ))

    input_vector = concatenate([q_vector, d_vector, q_d_labels])
    print("Concatenated vector: {iv}".format(iv=input_vector))
    dense = Dense(config["layers_size"][0], activation=config['hidden_activation'], name="MLP_combine_0")(input_vector)
    for i in range(config["num_layers"]-2):
        dense = Dense(config["layers_size"][i], activation=config['hidden_activation'],
                      name="MLP_combine_"+str(i+1))(dense)
    dense = Dense(1, activation=config['output_activation'], name="MLP_out"+str(i+1))(dense)
    model = Model(inputs=[query, doc, q_d_labels], outputs=dense)
    model.compile(optimizer=config["optimizer"], loss=config["loss_function"], metrics=config["metrics"])
    print(model.summary())
    plot_model(model, to_file='collaborative.png')
    # save model and resume

    print("Reading training data ...")
    x_train = None
    # input [arrays] each entry in x_train is a list of 3 lists [[],[], []]:
    # list of q_word_ids, list of d_word_ids and list of automatically generated labels
    # details : https://keras.io/getting-started/functional-api-guide/#shared-layers
    y_train = None  # output [array]



