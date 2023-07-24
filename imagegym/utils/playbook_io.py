import datetime
import json
import os
import shutil
import ast

import yaml

from yacs.config import CfgNode as CN

import pickle


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_info(my_str):
    print(f"{bcolors.OKBLUE}[INFO] {my_str}{bcolors.ENDC}")


def print_warning(my_str):
    print(f"{bcolors.WARNING}[WARNING] {my_str}{bcolors.ENDC}")


def print_debug(my_str):
    print(f"{bcolors.OKCYAN}[DEBUG] {my_str}{bcolors.ENDC}")


def string_to_python(string):
    try:
        return ast.literal_eval(string)
    except:
        return string


def dict_to_cn(my_dict):
    my_cn = CN()

    for key, value in my_dict.items():
        my_cn[key] = value

    return my_cn


def dict_to_json(dict, fname):
    with open(fname, 'a') as f:
        json.dump(dict, f)
        f.write('\n')


def dict_list_to_json(dict_list, fname):
    with open(fname, 'a') as f:
        for dict in dict_list:
            json.dump(dict, f)
            f.write('\n')


def json_to_dict_list(fname, check_for_epoch=True, split=None):
    dict_list = []
    epoch_set = set()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            dict = json.loads(line)
            if isinstance(split, str):
                keys = list(dict.keys())
                append_line = False
                for key in list(dict.keys()):
                    if split in key:
                        append_line = True
                        break

                if not append_line: continue

            if not check_for_epoch:

                dict_list.append(dict)
            else:
                if dict['epoch'] not in epoch_set:
                    dict_list.append(dict)
            epoch_set.add(dict['epoch'])
    return dict_list


def dict_to_tb(dict, writer, epoch):
    for key in dict:
        writer.add_scalar(key, dict[key], epoch)


def dict_list_to_tb(dict_list, writer):
    for dict in dict_list:
        assert 'epoch' in dict, 'Key epoch must exist in stats dict'
        dict_to_tb(dict, writer, dict['epoch'])


def makedirs(dir, only_if_not_exists=False):
    if only_if_not_exists:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    else:
        os.makedirs(dir, exist_ok=True)


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def load_json(json_file):
    with open(json_file) as file:
        info = json.load(file)
    return info


def save_pickle(obj, pickle_file):
    with open(pickle_file, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_yaml(my_dict, yaml_file):
    with open(yaml_file, 'w') as f:
        documents = yaml.dump(my_dict, f)


def load_yaml(yaml_file, flatten=False, as_cn=False):
    with open(yaml_file) as file:
        my_yaml = yaml.load(file, Loader=yaml.FullLoader)

    if flatten:
        my_yaml_flat = {}
        for key1, value1 in my_yaml.items():
            assert '__' not in key1
            if isinstance(value1, dict):
                for key2, value2 in value1.items():
                    assert '__' not in key2
                    my_yaml_flat[f"{key1}__{key2}"] = value2
            else:
                my_yaml_flat[f"{key1}"] = value1
        return my_yaml_flat

    if as_cn:
        my_cn = CN()
        for key, values in my_yaml.items():
            if isinstance(values, dict):
                my_cn[key] = dict_to_cn(values)
            else:
                my_cn[key] = values

        return my_cn
    else:
        return my_yaml


def create_yaml(base_grid, keys, option):
    cfg_i = base_grid.copy()
    for j, key in enumerate(keys):
        if '__' not in key:
            cfg_i[key] = option[j]
        else:
            key1, key2 = key.split('__')
            cfg_i[key1][key2] = option[j]
    return cfg_i
