import json
import os
import shutil
import ast



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


def dict_to_json(dict, fname):
    with open(fname, 'a') as f:
        json.dump(dict, f)
        f.write('\n')


def dict_list_to_json(dict_list, fname):
    with open(fname, 'a') as f:
        for dict in dict_list:
            json.dump(dict, f)
            f.write('\n')


def json_to_dict_list(fname):
    dict_list = []
    epoch_set = set()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            dict = json.loads(line)
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


def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
