import sys
import json
import pyndri
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model
from utils import read_lablers_to_relations, convert_embed_2_numpy, read_embedding, get_queries
from utils import get_input_label_size, get_mask, plot_history, get_optimizer
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Embedding, Dot
from keras.layers.normalization import BatchNormalization
from os.path import join
from keras import callbacks
from content_reader import ContentReader, ContentPickleReader
from data_generator import DataGenerator


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
                          trainable=config_model_train['train_embed'], name="embeddings")
    del embed_tensor
    q_embed = embedding(query)
    d_embed = embedding(doc)
    print("Embedded inputs: \nq_embed: {qe}\nd_embed: {de}".format(qe=q_embed, de=d_embed))
    lstm_layer = Bidirectional(LSTM(config_model_param["number_lstm_units"], dropout=config_model_param["lstm_dropout"],
                                    recurrent_dropout=config_model_param["lstm_dropout"]))
    q_vector = lstm_layer(q_embed)  # (1 x embed_size)
    d_vector = lstm_layer(d_embed)  # (1 x embed_size)

    print("biLstm vectors\nq_vector: {qv}\nd_vector: {dv}".format(qv=q_vector, dv=d_vector))

    input_vector = concatenate([q_vector, d_vector])
    # merge_layer = Dot(axes=1,normalize=False)([q_vector, d_vector]) ######## similarity dotprod between the 2 vectors
    # c = concatenate([q_vector, merge_layer, d_vector], axis=1)
    print("Concatenated vector: {iv}".format(iv=input_vector))
    merged = BatchNormalization()(input_vector)
    merged = Dropout(config_model_param["dropout_rate"])(merged)
    dense = Dense(config_model_param["layers_size"][0], activation=config_model_param['hidden_activation'],
                  name="MLP_combine_0")(merged)
    i = 0
    for i in range(config_model_param["num_layers"]-2):
        dense = BatchNormalization()(dense)
        dense = Dropout(config_model_param["dropout_rate"])(dense)
        dense = Dense(config_model_param["layers_size"][i+1], activation=config_model_param['hidden_activation'],
                      name="MLP_combine_"+str(i+1))(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(config_model_param["dropout_rate"])(dense)
    if config_model_param["predict_labels"]:
        out_size = get_input_label_size(config_data)
    else:
        out_size = 1
    out_labels = Dense(out_size, activation=config_model_param['output_activation'], name="MLP_out")(dense)
    model = Model(inputs=[query, doc], outputs=out_labels)
    # model2 = Model(inputs=[query, doc], outputs=input_vector)
    optimizer = get_optimizer(config_model_param["optimizer"])(lr=config_model_param["learning_rate"])
    # print(optimizer)
    model.compile(optimizer=optimizer, loss=config_model_train["loss_function"],
                  metrics=config_model_train["metrics"])

    print(model.summary())
    plot_model(model, to_file=join(config_model_train["train_details"], config_model_param['model_name']+".png"))
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

    relations_list = list(relations)
    queries_list = list(train_queries.keys())
    """
    reader = ContentReader(relations_list, token2id, externalDocId, queries_list, config_data['query_maxlen'],
                           config_data['doc_maxlen'], train_queries,
                           input_files=config_data['input_files'],
                           index=index)
    """
    reader = ContentPickleReader(relations_list, token2id, externalDocId, queries_list, config_data['query_maxlen'],
                           config_data['doc_maxlen'], train_queries,
                           input_files=config_data['input_files'],
                           index=index)
    list_IDs = [i for i, _ in enumerate(relations_list)]

    rel_label = {relation: int(relation_labeler[labeler][relation]) for labeler in relation_labeler
                 for relation in relations if relation in relation_labeler[labeler]}
    labels = {idx: rel_label[relation] for idx, relation in enumerate(relations_list)}

    # print(train_queries)
    print(list(relations)[0])
    params = {'relations_list': relations_list,
              'batch_size': config_model_train["batch_size"],
              'shuffle': config_model_train["shuffle"]}

    # Generators
    data_list, index_dict = reader.pickle_data()
    # training_generator = DataGenerator(reader, list_IDs, labels, **params)
    training_generator = DataGenerator(data_list, index_dict, list_IDs, labels, **params)

    steps_per_epoch = int(len(relations)/config_model_train["batch_size"])+1

    del relations
    del index
    del relation_labeler

    print("Model training...")
    mc = callbacks.ModelCheckpoint(config_model_train["weights"]+'_iter_{epoch:04d}.h5', save_weights_only=True,
                                   period=config_model_train["save_period"])

    history = model.fit_generator(generator=training_generator,
                                  epochs=config_model_train["epochs"],
                                  verbose=config_model_train["verbose"],
                                  steps_per_epoch=steps_per_epoch,
                                  callbacks=[mc])

    # save trained model
    print("Saving model and its weights ...")
    model.save_weights(config_model_train["weights"]+".h5")

    print("Plotting history ...")
    plot_history(history, config_model_train["train_details"], config_model_param["model_name"])
    print("Done.")
