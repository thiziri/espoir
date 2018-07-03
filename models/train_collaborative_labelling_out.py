import sys
import json
import pyndri
import numpy as np
from keras.layers import Input, Dense, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model
from utils import read_lablers_to_relations, convert_embed_2_numpy, read_embedding, get_queries
from utils import get_input_label_size, get_mask, plot_history, get_optimizer
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

    input_vector = concatenate([q_vector, d_vector])
    print("Concatenated vector: {iv}".format(iv=input_vector))
    dense = Dense(config_model_param["layers_size"][0], activation=config_model_param['hidden_activation'],
                  name="MLP_combine_0")(input_vector)
    i = 0
    for i in range(config_model_param["num_layers"]-2):
        # dense = Dropout(0.25)(dense)
        dense = Dense(config_model_param["layers_size"][i+1], activation=config_model_param['hidden_activation'],
                      name="MLP_combine_"+str(i+1))(dense)

    # dense = Dropout(0.5)(dense)
    out_size = get_input_label_size(config_data)
    out_labels = Dense(out_size, activation=config_model_param['output_activation'], name="MLP_out")(dense)
    model = Model(inputs=[query, doc], outputs=out_labels)
    model2 = Model(inputs=[query, doc], outputs=input_vector)
    optimizer = get_optimizer(config_model_param["optimizer"])(lr=config_model_param["learning_rate"])
    print(optimizer)
    model.compile(optimizer=optimizer, loss=config_model_train["loss_function"],
                  metrics=config_model_train["metrics"])
    print(model.summary())
    plot_model(model, to_file=join(config_model_train["train_details"], config_model_param['model_name']+".png"))
    # save model and resume
    # serialize model to JSON
    model_json = model.to_json()
    with open(join(config_model_train["train_details"], config_model_param["model_name"] + ".json"), "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk.")

    print("Reading training data:")
    print("[First]:\nRead label files to relations...")
    relations, relation_labeler = read_lablers_to_relations(config_data["labels"])

    print("[Second]:\nSet relations as train instances...")

    print("Reading data index ...")
    index = pyndri.Index(config_data["index"])
    token2id, _, _ = index.get_dictionary()
    print(len(token2id))
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
    print(list(relations)[0])

    for relation in tqdm(relations):
        # get q_word_ids from index
        q_words = [token2id[qi] if qi in token2id else 0 for qi in train_queries[relation[0]].strip().split()]
        print(len(q_words), len([x for x in q_words if x > 0]))
        # ############## pad/truncate q_words to a query_maxlen
        if len(q_words) < config_data['query_maxlen']:
            q_words = list(np.pad(q_words, (config_data['query_maxlen']-len(q_words), 0), "constant", constant_values=0))
        elif len(q_words) > config_data['query_maxlen']:
            q_words = q_words[:config_data['query_maxlen']]

        d_words = list(index.document(externalDocId[relation[1]])[1])  # get d_word_ids from index
        # ############## pad/truncate d_words to a doc_maxlen
        # print(len(d_words))
        if len(d_words) < config_data['doc_maxlen']:
            d_words = list(np.pad(d_words, (config_data['doc_maxlen']-len(d_words), 0), "constant", constant_values=0))
        elif len(d_words) > config_data['doc_maxlen']:
            d_words = d_words[:config_data['doc_maxlen']]
        # print(len(d_words))

        rel_labels = [int(relation_labeler[labeler][relation]) if relation in relation_labeler[labeler] else -1
                      for labeler in relation_labeler]

        v_q_words.append(q_words)
        v_d_words.append(d_words)
        v_rel_labels.append(rel_labels)

    print("y_train preparation...")
    y_train = get_mask(v_rel_labels, config_data)

    print("Model training...")
    print(np.array(v_q_words).shape, np.array(v_d_words).shape, np.array(v_rel_labels).shape)
    x_train = [np.array(v_q_words), np.array(v_d_words)]
    # print(np.array(x_train).shape)
    history = model.fit(x=x_train, y=np.array(y_train),
              batch_size=config_model_train["batch_size"],
              epochs=config_model_train["epochs"],
              verbose=config_model_train["verbose"],
              shuffle=config_model_train["shuffle"])

    # save trained model
    print("Saving model and its weights ...")
    model.save_weights(config_model_train["weights"]+".h5")

    print("Plotting history ...")
    plot_history(history, config_model_train["train_details"], config_model_param["model_name"])
    print("Done.")
