# read a trec formatted run file to .label file
import os
import sys
import json
import logging
from tqdm import tqdm
from os.path import join
from utils import run2relations, get_qrels, remove_extension, write_queries_to_file

if __name__ == '__main__':
    config_file = sys.argv[1]
    configure = json.load(open(config_file))
    config = configure["main_configuration"]
    logging.info('Config: '+json.dumps(config, indent=2))
    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Reading run pool files ...")
    qrels = get_qrels(config["relevance_judgements"]) if bool(config["relevance_judgements"]) else []
    for run in os.listdir(config["run_pool"]):
        print("Run : ", run)
        relations = run2relations(join(config["run_pool"], run), config["labels"], config["scales"], config["ranks"],
                                  qrels, config["max_rank"])  # ((q, doc), rel)
        with open(join(config["output_folder"], remove_extension(run))+".label", 'w') as out:
            for r in tqdm(relations):
                out.write("{q}\t{d}\t{rel}\n".format(q=r[0][0], d=r[0][1], rel=r[1]))

    print("Parse queries ...")
    write_queries_to_file(config["train_queries"], join(config["out_queries"], "train_queries.txt"),
                          config["train_query_format"])
    write_queries_to_file(config["test_queries"], join(config["out_queries"], "test_queries.txt"),
                          config["test_query_format"])
    print("Queries ok.")
    print('Done.')


