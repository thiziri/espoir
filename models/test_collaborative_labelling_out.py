import sys
import json
import pyndri
import numpy as np
from os.path import join
from keras.models import model_from_json
from utils import get_input_label_size, get_queries, run_test_data
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

    print("Loading model a model...")

    # load json and create model
    json_file = open(join(config_model_train["train_details"], config_model_param["model_name"]+'.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    model_weights = config_model_train["weights"]+'_iter_{x:04d}.h5'.format(x=config_model_test["test_period"])
    loaded_model.load_weights(model_weights)
    print("Loaded model from disk: ", model_weights)

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=config_model_param["optimizer"], loss=config_model_train["loss_function"],
                  metrics=config_model_train["metrics"])
    print("Compiled.")

    print("Reading data index ...")
    index = pyndri.Index(config_data["index"])
    token2id, _, _ = index.get_dictionary()
    externalDocId = {}
    for doc_id in range(index.document_base(), index.maximum_document()):  # type: int
        extD_id, _ = index.document(doc_id)
        externalDocId[extD_id] = doc_id
    test_queries = get_queries(config_data["train_queries"])

    # label_len = get_input_label_size(config_data)
    if config_model_test["if_reranking"]:
        out = open(join(config_model_test["save_rank"], config_model_param["model_name"]+"predict.txt"), 'w')
        relations = run_test_data(config_model_test["rank"], config_model_test["top_rank"])  # [(q, doc)]
        print("Please, wait while predicting ...")
        for relation in tqdm(relations):
            # read and pad data:
            query = [token2id[qi] if qi in token2id else 0 for qi in test_queries[relation[0]].strip().split()]  # get query terms
            if len(query) < config_data['query_maxlen']:
                query = list(np.pad(query, (config_data['query_maxlen'] - len(query), 0), "constant", constant_values=0))
            elif len(query) > config_data['query_maxlen']:
                query = query[:config_data['query_maxlen']]

            doc = list(index.document(externalDocId[relation[1]])[1])   # get document terms
            if len(doc) < config_data['doc_maxlen']:
                doc = list(np.pad(doc, (config_data['doc_maxlen'] - len(doc), 0), "constant", constant_values=0))
            elif len(doc) > config_data['doc_maxlen']:
                doc = doc[:config_data['doc_maxlen']]

            x_test = [np.array([query]), np.array([doc])]  # , np.array([[0 for i in range(label_len)]])]
            rank = loaded_model.predict(x_test, verbose=0)
            score = np.average(np.array(rank[0]))
            out.write("{q}\tQ0\t{d}\t1\t{s}\t{m}\n".format(q=relation[0], d=relation[1], s=score,
                                                           m=config_model_param["model_name"]))
            # break
        print("Prediction done.")
