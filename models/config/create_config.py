import sys
import json
from itertools import product
from os.path import join
from tqdm import tqdm

# generate a list of float intervals
def frange(start, stop, step=0.1):
    i = float(start)
    while i < float(stop):
        yield i
        i += float(step)


if __name__ == '__main__':
    config_file = sys.argv[1]  # basic configuration
    variables_file = sys.argv[2]  # parameters to be varied
    out = sys.argv[3]
    basic_configuration = json.load(open(config_file))
    to_variate = json.load(open(variables_file))

    combinations = []
    variable_values = {}
    print("Combine parameters ...")
    for parameter in tqdm(to_variate):
        if isinstance(to_variate[parameter][0], list):
            try:
                values = list(range(to_variate[parameter][0][0], to_variate[parameter][0][1]+to_variate[parameter][1],
                                to_variate[parameter][1]))
            except:
                values = list(frange(to_variate[parameter][0][0], to_variate[parameter][0][1] + to_variate[parameter][1],
                                    to_variate[parameter][1]))
        else:
            values = to_variate[parameter]
        variable_values[parameter] = values

    combinations = list(product(*[variable_values[variable] for variable in variable_values]))

    print("List of configurations:")
    configurations = []
    for combination in tqdm(combinations):
        config = {}
        for i, parameter in enumerate(list(variable_values.keys())):
            config[parameter] = combination[i]
        configurations.append(config)

    print(configurations[:3])

    print("Create valid configurations...")
    for configuration in tqdm(configurations):
        valid_config = basic_configuration.copy()
        # print(valid_config)
        for category in valid_config["main_configuration"]["model"]:
            # print(category)
            for parameter in valid_config["main_configuration"]["model"][category]:
                # print(parameter)
                if parameter in configuration:
                    valid_config["main_configuration"]["model"][category][parameter] = configuration[parameter]
                    # print(json.dumps(valid_config, indent=2))
                    print(parameter)

        valid_file = "_".join([valid_config["main_configuration"]["model"]["parameters"]["model_name"],
                              "hActiv", valid_config["main_configuration"]["model"]["parameters"]["hidden_activation"],
                              "oActiv", valid_config["main_configuration"]["model"]["parameters"]["output_activation"],
                              "optimizer", valid_config["main_configuration"]["model"]["parameters"]["optimizer"],
                              "lstm", str(valid_config["main_configuration"]["model"]["parameters"]["number_lstm_units"]),
                              "lstm_dropout", str(valid_config["main_configuration"]["model"]["parameters"]["lstm_dropout"]),
                              "layers", str(valid_config["main_configuration"]["model"]["parameters"]["num_layers"]),
                              "dense_dropout", str(valid_config["main_configuration"]["model"]["parameters"]["dropout_rate"]),
                              "loss", valid_config["main_configuration"]["model"]["train"]["loss_function"],
                              "test", str(valid_config["main_configuration"]["model"]["test"]["test_period"])
                              ])
        conf_file = open(join(out, valid_file), 'w')
        conf_file.write(json.dumps(valid_config, indent=2))

    print("{%d} configurations" % len(configurations))



