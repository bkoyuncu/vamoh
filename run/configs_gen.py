from traitlets import default
import yaml
import os
import sys
import argparse
import csv
import random
import copy
import numpy as np
from yacs.config import CfgNode as CN
from imagegym.utils.io import makedirs_rm_exist, string_to_python
random.seed(123)
from imagegym.config import cfg, set_cfg
import subprocess
print(os.getcwd())
# os.chdir("./run")
def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        dest='config',
        help='the base configuration file used for edit',
        default='configs/models/MNIST_overfit_base.yaml',
        type=str
    )
    parser.add_argument(
        '--main_file',
        dest='main_file',
        help='the base configuration file used for edit',
        default='main',
        type=str
    )
    parser.add_argument(
        '--grid',
        dest='grid',
        help='configuration file for grid search',
        required=False,
        type=str,
        default='grids/mnist_local.txt'
    )
    parser.add_argument(
        '--sample_alias',
        dest='sample_alias',
        help='configuration file for sample alias',
        default=None,
        required=False,
        type=str
    )
    parser.add_argument(
        '--sample_num',
        dest='sample_num',
        help='Number of random samples in the space',
        default=10,
        type=int
    )
    parser.add_argument(
        '--repeat',
        dest='repeat',
        help='Number of random seeds for each of the experiments in the grid',
        default=2,
        type=int
    )
    parser.add_argument(
        '--out_dir',
        dest='out_dir',
        help='output directory for generated config files',
        default='configs',
        type=str
    )
    parser.add_argument(
        '--config_budget',
        dest='config_budget',
        help='the base configuration file used for matching computation',
        default=None,
        type=str
    )
    parser.add_argument(
        '--config_cluster',
        dest='config_cluster',
        help='the base configuration file used for the cluster',
        type=str,
        default='configs/cluster_gpu.yaml'
    )
    parser.add_argument(
        '--version',
        dest='version',
        help='Version of the batch of experiments',
        default=None,
        type=str
    )
    return parser.parse_args()


def get_fname(string):
    if string is not None:
        return string.split('/')[-1].split('.')[0]
    else:
        return 'default'


def grid2list(grid):
    list_in = [[]]
    for grid_temp in grid:
        list_out = []
        for val in grid_temp:
            for list_temp in list_in:
                list_out.append(list_temp + [val])
        list_in = list_out
    return list_in


def lists_distance(l1, l2):
    assert len(l1) == len(l2)
    dist = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            dist += 1
    return dist


def grid2list_sample(grid, sample=10):
    configs = []
    while len(configs) < sample:
        config = []
        for grid_temp in grid:
            config.append(random.choice(grid_temp))
        if config not in configs:
            configs.append(config)
    return configs


def load_config(fname):
    if fname is not None:
        with open(fname) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}


def load_search_file(fname):
    with open(fname, 'r') as f:
        out_raw = csv.reader(f, delimiter=' ')
        outs = []
        out = []
        for row in out_raw:
            if '#' in row:
                continue
            elif len(row) > 0:
                print(row)
                assert len(row) == 3, \
                    'Exact 1 space between each grid argument file' \
                    'And no spaces within each argument is allowed'
                out.append(row)
            else:
                if len(out) > 0:
                    outs.append(out)
                out = []
        if len(out) > 0:
            outs.append(out)
    return outs


def load_alias_file(fname):
    with open(fname, 'r') as f:
        file = csv.reader(f, delimiter=' ')
        for line in file:
            break
    return line


def exclude_list_id(list, id):
    return [list[i] for i in range(len(list)) if i != id]



def create_cluster_submission_files(config_cluster, task_name):
    job_file = f'c{task_name}.sub'
    job_file_test = f'c{task_name}_test.sub'
    for job_f in [job_file, job_file_test]:
        with open(job_f, "w") as f:
            for key, value in config_cluster.items():
                f.write(f'{key} = {value} \n')


error_file = lambda a: f"error = {a}.err\n"
output_file = lambda a: f"output = {a}.out\n"
log_file = lambda a: f"log = {a}.log\n"

def add_experiment(config_file,
                   task_name,
                   i,
                   seed):

    main_str = f'{args.main_file}.py --cfg  {config_file} --seed {seed}'
    arguments = f"\narguments = \"{main_str}\"\n"
    exper_path = os.path.join('_cluster',  task_name, f"permutation_{str(i)}")
    job_file = f'c{task_name}.sub'
    job_file_test = f'c{task_name}_test.sub'
    with open(job_file, "a") as f:
        f.write(arguments)
        f.write(error_file(exper_path))
        f.write(output_file(exper_path))
        f.write(log_file(exper_path))
        f.write('queue\n')
    if i == 0:
        with open(job_file_test, "a") as f:
            f.write(arguments)
            f.write(error_file(exper_path))
            f.write(output_file(exper_path))
            f.write(log_file(exper_path))
            f.write('queue\n')

def gen_grid(args,
             config,
             repeat,
             config_budget={},
             config_cluster={},
             dataset=''):


    datasets_dict = {}

    if dataset is None:dataset = ''

    task_name = '{}_grid_{}'.format(get_fname(args.config),
                                    get_fname(args.grid))
    fname_start = get_fname(args.config)
    out_dir = os.path.join(args.out_dir, task_name)
    makedirs_rm_exist(out_dir)
    config['out_dir'] = os.path.join(config['out_dir'], task_name)

    if len(config_cluster)>0:
        exper_path = os.path.join('_cluster',  task_name)
        if not os.path.exists(exper_path):
            os.makedirs(exper_path)
            subprocess.call(f"touch {os.path.join(exper_path, '.file')}", shell=True)


    create_cluster_submission_files(config_cluster,
                                    task_name)


    outs = load_search_file(args.grid)
    idx = 0
    for i, out in enumerate(outs):
        vars_label = [row[0].split('.') for row in out]
        vars_alias = [row[1] for row in out]
        vars_value = grid2list([string_to_python(row[2]) for row in out])
        if i == 0:
            print('Variable label: {}'.format(vars_label))
            print('Variable alias: {}'.format(vars_alias))

        for j, vars in enumerate(vars_value):
            print(f"Experiment {i}/{len(outs)} {j}/{len(vars_value)}")
            config_out = config.copy()
            fname_out = fname_start
            for id, var in enumerate(vars):
                if len(vars_label[id]) == 1:
                    config_out[vars_label[id][0]] = var
                elif len(vars_label[id]) == 2:
                    if vars_label[id][0] in config_out:  # if key1 exist
                        config_out[vars_label[id][0]][vars_label[id][1]] = var
                    else:
                        config_out[vars_label[id][0]] = {vars_label[id][1]: var}
                else:
                    raise ValueError('Only 2-level config files are supported')
                # fname_out += '-{}={}'.format(vars_alias[id],
                #                              str(var).strip("[]").strip("''"))
                fname_out += '-{}={}'.format(vars_alias[id],
                                             str(var).strip("''").replace(" ", ""))
            if  config_out['dataset']['name']  not in datasets_dict:
                set_cfg(cfg)
                cfg_new = CN(config_out)
                cfg.merge_from_other_cfg(cfg_new)
                datasets_dict[config_out['dataset']['name']] = create_dataset_fn()
            if dataset not in ['', 'all']:
                if config_out['dataset']['name'] != dataset:
                    print(f"Creating experiments only for {dataset}: {config_out['dataset']['name']}")
                    continue
            if len(config_budget) > 0:
                raise NotImplementedError
                config_out = dict_match_baseline(config_out, config_budget,
                                                 datasets=datasets_dict[config_out['dataset']['name']])
                _ = datasets_dict[config_out['dataset']['name']][0].num_labels

            filename = '{}/{}.yaml'.format(out_dir, fname_out)

            for i in range(repeat):
                add_experiment(config_file=filename,
                               task_name=task_name,
                               i=idx,
                               seed=i)
                idx +=1
            with open(filename, "w") as f:
                yaml.dump(config_out, f, default_flow_style=False)
        print('{} configurations saved to: {}'.format(
            len(vars_value), out_dir))


def gen_grid_sample(args, config, config_budget={}, compare_alias_list=[]):
    raise NotImplementedError
    task_name = '{}_grid_{}'.format(get_fname(args.config),
                                    get_fname(args.grid))
    fname_start = get_fname(args.config)
    out_dir = '{}/{}'.format(args.out_dir, task_name)
    makedirs_rm_exist(out_dir)
    config['out_dir'] = os.path.join(config['out_dir'], task_name)
    outs = load_search_file(args.grid)

    counts = []
    for out in outs:
        vars_grid = [string_to_python(row[2]) for row in out]
        count = 1
        for var in vars_grid:
            count *= len(var)
        counts.append(count)
    counts = np.array(counts)
    print('Total size of each chunk of experiment space:', counts)
    counts = counts / np.sum(counts)
    counts = np.round(counts * args.sample_num)
    counts[0] += args.sample_num - np.sum(counts)
    print('Total sample size of each chunk of experiment space:', counts)

    for i, out in enumerate(outs):
        vars_label = [row[0].split('.') for row in out]
        vars_alias = [row[1] for row in out]
        if i == 0:
            print('Variable label: {}'.format(vars_label))
            print('Variable alias: {}'.format(vars_alias))
        vars_grid = [string_to_python(row[2]) for row in out]
        for alias in compare_alias_list:
            alias_id = vars_alias.index(alias)
            vars_grid_select = copy.deepcopy(vars_grid[alias_id])
            vars_grid[alias_id] = [vars_grid[alias_id][0]]
            vars_value = grid2list_sample(vars_grid, counts[i])

            vars_value_new = []
            for vars in vars_value:
                for grid in vars_grid_select:
                    vars[alias_id] = grid
                    vars_value_new.append(copy.deepcopy(vars))
            vars_value = vars_value_new

            vars_grid[alias_id] = vars_grid_select
            for vars in vars_value:
                config_out = config.copy()
                fname_out = fname_start + '-sample={}'.format(
                    vars_alias[alias_id])
                for id, var in enumerate(vars):
                    if len(vars_label[id]) == 1:
                        config_out[vars_label[id][0]] = var
                    elif len(vars_label[id]) == 2:
                        if vars_label[id][0] in config_out:  # if key1 exist
                            config_out[vars_label[id][0]][
                                vars_label[id][1]] = var
                        else:
                            config_out[vars_label[id][0]] = {
                                vars_label[id][1]: var}
                    else:
                        raise ValueError(
                            'Only 2-level config files are supported')
                    fname_out += '-{}={}'.format(vars_alias[id],
                                                 str(var).strip("[]").strip(
                                                     "''"))
                if len(config_budget) > 0:
                    config_out = dict_match_baseline(
                        config_out, config_budget, verbose=False)

                filaname = '{}/{}.yaml'.format(out_dir, fname_out)
                with open(filaname, "w") as f:
                    print(f'Config: {filaname}')
                    yaml.dump(config_out, f, default_flow_style=False)
            print('Chunk {}/{}: Perturbing design dimension {}, '
                  '{} configurations saved to: {}'.format(i + 1, len(outs),
                                                          alias,
                                                          len(vars_value),
                                                          out_dir))


args = parse_args()
config = load_config(args.config)
repeat = args.repeat
config_budget = load_config(args.config_budget)
config_cluster = load_config(args.config_cluster)


from imagegym.loader import create_dataset


create_dataset_fn = create_dataset
if args.sample_alias is None:
    gen_grid(args, config, repeat, config_budget, config_cluster, args.version)
else:
    alias_list = load_alias_file(args.sample_alias)
    gen_grid_sample(args, config, config_budget, alias_list)
