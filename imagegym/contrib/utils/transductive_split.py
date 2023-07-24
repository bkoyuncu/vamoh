from copy import copy
def transductive_split(dataset, split_sizes, task):
    assert task in ['node', 'edge', 'link']
    if task == 'node':
        pass
    datasets = [copy(dataset) for _ in split_sizes]
    return datasets